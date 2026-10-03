"""Application port for deciding whether a session's daily bar is final.

Two questions are easy to conflate and must not be:

- does a trading session exist, and has it ended?
- is the market-data provider's daily bar for that session safe to treat as
  final, so that acquiring it now will not later be contradicted?

The first is a calendar fact and is the FuturesTradingSessionResolver's answer.
The second is a property of a data provider's publication behaviour, and it is
this port's. A session can have ended hours ago while its daily bar is still
being revised; acquiring it then would freeze a value the provider later
changes, which the store refuses as a conflict rather than overwriting.

The policy assesses one already resolved session. It never resolves a calendar,
decides whether a date trades, computes a session boundary, queries a provider
or reads a clock. A future policy that needs an observation instant must have
that instant supplied by its composition, never read internally.

Three outcomes, kept distinct
-----------------------------
FINAL means the session's daily bar is explicitly approved for operational
acquisition. NOT_YET_FINAL means the policy knows the session is not yet
approved. UNKNOWN means the policy cannot establish finality at all. A caller
treats both non-final outcomes as "do not acquire", but they are not the same
fact: one is an expected wait, the other an absence of any basis for deciding,
and collapsing them would hide which of the two an operator is looking at.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum

from northstar_core.futures import FuturesContract

from northstar_application.ports.futures_trading_session_resolver import (
    FuturesTradingSession,
)


class InvalidFuturesDailyBarFinalityError(ValueError):
    """Raised when a FuturesDailyBarFinality value is invalid."""


class FuturesDailyBarFinalityOutcome(StrEnum):
    """Closed vocabulary for whether one session's daily bar is final."""

    FINAL = "FINAL"
    NOT_YET_FINAL = "NOT_YET_FINAL"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class FuturesDailyBarFinality:
    """One policy's assessment of one exact contract's daily bar for one session.

    ``contract`` and ``session`` are the exact inputs assessed, kept so an
    assessment can be audited without the call that produced it. ``reason`` is
    a deterministic, human-readable explanation; the same inputs to the same
    policy always produce an equal assessment.
    """

    contract: FuturesContract
    session: FuturesTradingSession
    outcome: FuturesDailyBarFinalityOutcome
    reason: str

    def __post_init__(self) -> None:
        subject = "FuturesDailyBarFinality"
        if not isinstance(self.contract, FuturesContract):
            raise InvalidFuturesDailyBarFinalityError(
                f"{subject} contract must be a FuturesContract value."
            )
        if not isinstance(self.session, FuturesTradingSession):
            raise InvalidFuturesDailyBarFinalityError(
                f"{subject} session must be a FuturesTradingSession value."
            )
        if not isinstance(self.outcome, FuturesDailyBarFinalityOutcome):
            raise InvalidFuturesDailyBarFinalityError(
                f"{subject} outcome must be a FuturesDailyBarFinalityOutcome value."
            )
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise InvalidFuturesDailyBarFinalityError(f"{subject} reason must be non-empty text.")
        object.__setattr__(self, "reason", self.reason.strip())

    @property
    def is_final(self) -> bool:
        """Return whether the daily bar may be acquired for operation."""
        return self.outcome is FuturesDailyBarFinalityOutcome.FINAL

    def __str__(self) -> str:
        return (
            f"{self.contract} {self.session.trading_date.isoformat()} "
            f"{self.outcome.value}: {self.reason}"
        )


class FuturesDailyBarFinalityPolicy(ABC):
    """Decides whether one exact contract's daily bar for one session is final.

    Implementations must honour these semantics:

    - The session is already resolved; the policy neither confirms that it
      exists nor computes its boundaries.
    - The returned assessment carries exactly the ``contract`` and ``session``
      it was asked about.
    - The same inputs always produce an equal assessment. No clock is read; an
      implementation that depends on time takes its instant explicitly.
    - When finality cannot be established, the answer is UNKNOWN, never FINAL.

    A FuturesTradingSession carries no instrument identity, so a policy cannot
    check that the session belongs to the contract's venue; resolving the right
    calendar for the contract remains the resolver's caller's responsibility.

    The abstract contract documents these obligations but cannot enforce them
    at runtime.
    """

    @abstractmethod
    def assess(
        self, contract: FuturesContract, session: FuturesTradingSession
    ) -> FuturesDailyBarFinality:
        """Return whether ``contract``'s daily bar for ``session`` is final."""
