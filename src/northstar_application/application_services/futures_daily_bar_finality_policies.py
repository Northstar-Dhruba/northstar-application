"""Deterministic Application implementations of FuturesDailyBarFinalityPolicy.

Neither policy observes a provider or reads a clock. They exist so that
operational acquisition can be gated before any provider-specific finality rule
has evidence behind it:

- OperatorApprovedFuturesDailyBarFinalityPolicy treats a session as final only
  if an operator has explicitly approved data through its trading date.
- DisabledFuturesDailyBarFinalityPolicy never establishes finality, so nothing
  can be acquired under it. It is the fail-closed default for unattended
  operation until a validated rule exists.

How an operator's approval is stored, updated or supplied is a composition
concern; these policies only apply the value they are given.
"""

from __future__ import annotations

from datetime import date, datetime

from northstar_core.futures import FuturesContract

from northstar_application.ports import (
    FuturesDailyBarFinality,
    FuturesDailyBarFinalityOutcome,
    FuturesDailyBarFinalityPolicy,
    FuturesTradingSession,
)


class InvalidFuturesDailyBarFinalityPolicyError(ValueError):
    """Raised when a futures daily-bar finality policy is configured invalidly."""


def _validate_inputs(subject: str, contract: object, session: object) -> None:
    if not isinstance(contract, FuturesContract):
        raise TypeError(f"{subject} contract must be a FuturesContract.")
    if not isinstance(session, FuturesTradingSession):
        raise TypeError(f"{subject} session must be a FuturesTradingSession.")


class OperatorApprovedFuturesDailyBarFinalityPolicy(FuturesDailyBarFinalityPolicy):
    """Daily bars are final through an operator-approved trading date, inclusive.

    A session labelled on or before ``final_through`` is FINAL; any later
    session is NOT_YET_FINAL. The approval is a session label, compared with the
    resolved session's ``trading_date``, so it means "data through this trading
    session", never an instant or a time of day.
    """

    def __init__(self, final_through: date) -> None:
        subject = "OperatorApprovedFuturesDailyBarFinalityPolicy"
        if final_through is None:
            raise InvalidFuturesDailyBarFinalityPolicyError(
                f"{subject} final-through date cannot be None."
            )
        # A datetime carries a clock; an approved session label has none.
        if isinstance(final_through, datetime) or not isinstance(final_through, date):
            raise InvalidFuturesDailyBarFinalityPolicyError(
                f"{subject} final-through must be a plain date."
            )
        self._final_through = final_through

    @property
    def final_through(self) -> date:
        """Return the last trading date the operator approved as final."""
        return self._final_through

    def __repr__(self) -> str:
        return (
            "OperatorApprovedFuturesDailyBarFinalityPolicy("
            f"final_through={self._final_through.isoformat()!r})"
        )

    def assess(
        self, contract: FuturesContract, session: FuturesTradingSession
    ) -> FuturesDailyBarFinality:
        """Return FINAL on or before the approved date, NOT_YET_FINAL after it."""
        _validate_inputs("OperatorApprovedFuturesDailyBarFinalityPolicy", contract, session)
        label = session.trading_date.isoformat()
        approved = self._final_through.isoformat()
        if session.trading_date <= self._final_through:
            return FuturesDailyBarFinality(
                contract=contract,
                session=session,
                outcome=FuturesDailyBarFinalityOutcome.FINAL,
                reason=(
                    f"Session {label} is on or before the operator-approved "
                    f"final-through date {approved}."
                ),
            )
        return FuturesDailyBarFinality(
            contract=contract,
            session=session,
            outcome=FuturesDailyBarFinalityOutcome.NOT_YET_FINAL,
            reason=(
                f"Session {label} is after the operator-approved final-through date "
                f"{approved}; it has not been approved as final."
            ),
        )


class DisabledFuturesDailyBarFinalityPolicy(FuturesDailyBarFinalityPolicy):
    """Never establishes finality: every session is UNKNOWN. Fails closed."""

    _REASON = (
        "Daily-bar finality is not established: automatic finality is disabled "
        "and no operator approval applies."
    )

    def __repr__(self) -> str:
        return "DisabledFuturesDailyBarFinalityPolicy()"

    def assess(
        self, contract: FuturesContract, session: FuturesTradingSession
    ) -> FuturesDailyBarFinality:
        """Return UNKNOWN for any valid contract and session."""
        _validate_inputs("DisabledFuturesDailyBarFinalityPolicy", contract, session)
        return FuturesDailyBarFinality(
            contract=contract,
            session=session,
            outcome=FuturesDailyBarFinalityOutcome.UNKNOWN,
            reason=self._REASON,
        )
