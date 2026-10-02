"""Application pre-expiry flatten guard for futures paper trading.

A dated futures contract must not be carried into its expiry. The guard decides,
for one frozen decision, whether the portfolio has to be taken flat in that
contract now, so that it is already flat when the protected pre-expiry window
opens. It decides nothing else: it never selects another contract, never rolls,
and never reads a clock.

What ``sessions_before_expiry`` means
-------------------------------------
With ``K = sessions_before_expiry`` and expiry session ``E``:

    the portfolio must be flat by the OPEN of session E-K,
    and stays flat in this contract from E-K through E.

Paper execution fills an intent at the OPEN of the first stored session
strictly after its decision. To be flat by E-K's open, the flatten therefore
has to be decided at the close of E-(K+1) at the latest. For K = 5:

    decision at E-7 close   no flatten yet; a fill would land at E-6
    decision at E-6 close   flatten required; it fills at the E-5 open
    E-5 .. E                flat; no signal may reopen or increase the contract

Sessions are counted only through the FuturesTradingSessionResolver, so
weekends, holidays and special sessions shift the window exactly as the
venue's calendar does. Calendar days are never subtracted from the expiration
date: five sessions before a Tuesday expiry across a weekend and a holiday is
not five days before it.

Which session a decision belongs to
-----------------------------------
A decision instant is a completed daily bar's instant, the close of its
session. Its session is found with the resolver's own membership rule,
``opens_at < instant <= closes_at``, never by taking the UTC date of the
instant, because a session label need not equal that date.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta

from northstar_core.foundation.value_objects import PointInTime
from northstar_core.futures import FuturesContract

from northstar_application.ports import (
    FuturesTradingSession,
    FuturesTradingSessionResolver,
)

# Widens only the search for the session containing a decision instant; which
# session contains it is decided by resolved session boundaries alone. Never
# applied to the expiration date or used to measure distance to expiry.
_MEMBERSHIP_SEARCH_MARGIN = timedelta(days=2)


class InvalidFuturesExpiryFlattenPolicyError(ValueError):
    """Raised when a FuturesExpiryFlattenPolicy value is invalid."""


class FuturesExpiryWindowError(ValueError):
    """Raised when a decision's expiry window cannot be assessed from the calendar.

    The decision instant falls inside no trading session, or the contract's
    expiration date is not a trading session of its venue. Neither is guessed
    around: a countdown measured from the wrong session is worse than none.
    """


@dataclass(frozen=True, slots=True)
class FuturesExpiryFlattenPolicy:
    """How many full pre-expiry sessions a portfolio must already be flat for.

    ``sessions_before_expiry`` is K: the portfolio is flat by the OPEN of the
    session K trading sessions before the expiry session. K is at least 1, so
    the expiry session itself is always protected.
    """

    sessions_before_expiry: int

    def __post_init__(self) -> None:
        value = self.sessions_before_expiry
        # bool is an int subclass; True must not pass as one session.
        if isinstance(value, bool) or not isinstance(value, int):
            raise InvalidFuturesExpiryFlattenPolicyError(
                "FuturesExpiryFlattenPolicy sessions before expiry must be an integer."
            )
        if value < 1:
            raise InvalidFuturesExpiryFlattenPolicyError(
                "FuturesExpiryFlattenPolicy sessions before expiry must be at least 1."
            )


@dataclass(frozen=True, slots=True)
class FuturesExpiryWindowAssessment:
    """The audited pre-expiry position of one decision on one exact contract.

    ``sessions_after_decision_through_expiry`` counts the trading sessions
    labelled strictly after ``decision_trading_date`` up to and including
    ``expiry_trading_date``. The first of them, ``next_trading_date``, is the
    session at whose OPEN an intent decided now would fill. It is zero, and
    ``next_trading_date`` is None, when the decision is on or after expiry.

    ``flatten_required`` is true when that fill session would already lie in
    the protected window E-K .. E, which is exactly when the count is at most
    ``sessions_before_expiry + 1``. The extra one is the next-open execution
    delay, not a looser policy.
    """

    contract: FuturesContract
    decision_instant: PointInTime
    decision_trading_date: date
    expiry_trading_date: date
    sessions_before_expiry: int
    sessions_after_decision_through_expiry: int
    next_trading_date: date | None

    def __post_init__(self) -> None:
        subject = "FuturesExpiryWindowAssessment"
        if not isinstance(self.contract, FuturesContract):
            raise TypeError(f"{subject} contract must be a FuturesContract.")
        if not isinstance(self.decision_instant, PointInTime):
            raise TypeError(f"{subject} decision instant must be a PointInTime.")
        for name in ("decision_trading_date", "expiry_trading_date"):
            value = getattr(self, name)
            if isinstance(value, datetime) or not isinstance(value, date):
                raise TypeError(f"{subject} {name} must be a plain date.")
        if self.expiry_trading_date.isoformat() != self.contract.expiration_date.value:
            raise ValueError(f"{subject} expiry trading date must be the contract's expiration.")
        # Rejects a non-positive K through the policy's own rule.
        FuturesExpiryFlattenPolicy(self.sessions_before_expiry)
        count = self.sessions_after_decision_through_expiry
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(
                f"{subject} sessions after decision through expiry must be a non-negative integer."
            )
        following = self.next_trading_date
        if following is not None and (
            isinstance(following, datetime) or not isinstance(following, date)
        ):
            raise TypeError(f"{subject} next trading date must be a plain date or None.")
        if (count == 0) != (following is None):
            raise ValueError(
                f"{subject} has a next trading date exactly when sessions remain after "
                "the decision."
            )
        if following is not None and not (
            self.decision_trading_date < following <= self.expiry_trading_date
        ):
            raise ValueError(
                f"{subject} next trading date must follow the decision and not pass expiry."
            )
        if count and self.decision_trading_date >= self.expiry_trading_date:
            raise ValueError(f"{subject} counts no sessions for a decision on or after expiry.")

    @property
    def flatten_required(self) -> bool:
        """Return whether the flatten must be decided now to be flat by E-K's open."""
        return self.sessions_after_decision_through_expiry <= self.sessions_before_expiry + 1


class FuturesExpiryFlattenGuard:
    """Assess one decision's distance to its contract's expiry in trading sessions."""

    def __init__(
        self, resolver: FuturesTradingSessionResolver, policy: FuturesExpiryFlattenPolicy
    ) -> None:
        if not isinstance(resolver, FuturesTradingSessionResolver):
            raise TypeError(
                "FuturesExpiryFlattenGuard resolver must be a FuturesTradingSessionResolver."
            )
        if not isinstance(policy, FuturesExpiryFlattenPolicy):
            raise TypeError(
                "FuturesExpiryFlattenGuard policy must be a FuturesExpiryFlattenPolicy."
            )
        self._resolver = resolver
        self._policy = policy

    @property
    def policy(self) -> FuturesExpiryFlattenPolicy:
        """Return the policy this guard applies."""
        return self._policy

    def assess(
        self, contract: FuturesContract, decision_instant: PointInTime
    ) -> FuturesExpiryWindowAssessment:
        """Return the audited expiry position of a decision made at ``decision_instant``.

        Resolver failures propagate unchanged. A decision instant inside no
        session, or an expiration date that is not a session, raises
        FuturesExpiryWindowError.
        """
        if not isinstance(contract, FuturesContract):
            raise TypeError("FuturesExpiryFlattenGuard contract must be a FuturesContract.")
        if not isinstance(decision_instant, PointInTime):
            raise TypeError("FuturesExpiryFlattenGuard decision instant must be a PointInTime.")

        decision_session = self._containing_session(contract, decision_instant)
        expiry = date.fromisoformat(contract.expiration_date.value)
        if self._resolver.resolve(contract.product, expiry) is None:
            raise FuturesExpiryWindowError(
                f"Expiration date {expiry.isoformat()} of {contract} is not a trading session "
                "of its venue; the pre-expiry window cannot be counted from it."
            )

        decided = decision_session.trading_date
        following: tuple[FuturesTradingSession, ...] = ()
        if decided < expiry:
            following = tuple(
                session
                for session in self._resolver.sessions_in_range(contract.product, decided, expiry)
                if session.trading_date > decided
            )

        return FuturesExpiryWindowAssessment(
            contract=contract,
            decision_instant=decision_instant,
            decision_trading_date=decided,
            expiry_trading_date=expiry,
            sessions_before_expiry=self._policy.sessions_before_expiry,
            sessions_after_decision_through_expiry=len(following),
            next_trading_date=following[0].trading_date if following else None,
        )

    def _containing_session(
        self, contract: FuturesContract, instant: PointInTime
    ) -> FuturesTradingSession:
        """Return the one session whose window ``opens_at < instant <= closes_at`` holds."""
        around = _utc(instant).date()
        candidates = self._resolver.sessions_in_range(
            contract.product,
            around - _MEMBERSHIP_SEARCH_MARGIN,
            around + _MEMBERSHIP_SEARCH_MARGIN,
        )
        containing = [
            session
            for session in candidates
            if session.opens_at.compare(instant) < 0 and instant.compare(session.closes_at) <= 0
        ]
        if len(containing) != 1:
            raise FuturesExpiryWindowError(
                f"Decision instant {instant} for {contract} lies inside "
                f"{'no' if not containing else len(containing)} trading session(s); "
                "its pre-expiry countdown cannot be placed."
            )
        return containing[0]


def _utc(instant: PointInTime) -> datetime:
    return datetime.fromisoformat(instant.value.replace("Z", "+00:00")).astimezone(UTC)
