"""Application port for resolving equity trading sessions and their trading phases.

An intraday equity bar belongs to one continuous-trading phase of one session,
so the phases a session trades in, and their boundaries, come from one
authoritative port rather than from a provider's candle timestamps or from
weekday arithmetic.

Equity sessions are deliberately separate from futures and option sessions.
The derivative ports carry a single window per session, which is enough to
stamp a daily bar but not to lay out intraday intervals: a session interrupted
by a break would be one window, and an interval grid built over it would place
candles inside the break. An equity session therefore lists its continuous
trading phases explicitly, and nothing between two phases is ever eligible.

Which venue facts hold -- hours, holidays, special sessions, breaks -- is the
adapter's reference data and is not asserted here. No pre-open, call-auction
or closing-session phase is implied: a phase exists only when the reference
states it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum

from northstar_core.foundation.value_objects import ExchangeCode, PointInTime


class EquitySessionResolutionError(RuntimeError):
    """Raised when equity trading sessions cannot be resolved reliably."""


class InvalidEquityTradingSessionError(ValueError):
    """Raised when an EquityTradingSession or EquityTradingPhase value is invalid."""


class EquitySessionKind(StrEnum):
    """Whether a session follows the reference's normal hours or its own notified hours."""

    NORMAL = "NORMAL"
    SPECIAL = "SPECIAL"


def _validate_instant(value: PointInTime, subject: str, field_name: str) -> PointInTime:
    if value is None:
        raise InvalidEquityTradingSessionError(f"{subject} {field_name} cannot be None.")
    if not isinstance(value, PointInTime):
        raise InvalidEquityTradingSessionError(
            f"{subject} {field_name} must be a PointInTime value."
        )
    return value


@dataclass(frozen=True, slots=True)
class EquityTradingPhase:
    """One continuous-trading phase, half-open on the left and closed on the right.

    An interval belongs to the phase when ``opens_at < instant <= closes_at``,
    which is where a bar stamped at its interval end falls.
    """

    opens_at: PointInTime
    closes_at: PointInTime

    def __post_init__(self) -> None:
        opens_at = _validate_instant(self.opens_at, "EquityTradingPhase", "opens_at")
        closes_at = _validate_instant(self.closes_at, "EquityTradingPhase", "closes_at")
        # Compared semantically: text ordering of canonical instants is unreliable.
        if opens_at.compare(closes_at) >= 0:
            raise InvalidEquityTradingSessionError(
                "EquityTradingPhase opens_at must be before closes_at."
            )

    def __str__(self) -> str:
        return f"[{self.opens_at} .. {self.closes_at}]"


@dataclass(frozen=True, slots=True)
class EquityTradingSession:
    """One equity trading session: its label, venue, kind, phases and reference.

    ``trading_date`` is the exchange session label. ``phases`` are the
    continuous-trading phases in strictly ascending order; consecutive phases
    must be separated by a real break, because two touching phases would be one
    phase written twice. ``reference`` names the reference data the session was
    resolved from, so every consumer can say which evidence it relied on.

    This is calendar-derived Application data. It carries no instrument
    identity and no trading policy such as entry or exit cutoffs.
    """

    exchange_code: ExchangeCode
    trading_date: date
    kind: EquitySessionKind
    phases: tuple[EquityTradingPhase, ...]
    reference: str

    def __post_init__(self) -> None:
        if not isinstance(self.exchange_code, ExchangeCode):
            raise InvalidEquityTradingSessionError(
                "EquityTradingSession exchange_code must be an ExchangeCode value."
            )
        if isinstance(self.trading_date, datetime) or not isinstance(self.trading_date, date):
            raise InvalidEquityTradingSessionError(
                "EquityTradingSession trading date must be a plain date."
            )
        if not isinstance(self.kind, EquitySessionKind):
            raise InvalidEquityTradingSessionError(
                "EquityTradingSession kind must be an EquitySessionKind value."
            )
        if not isinstance(self.phases, tuple) or not self.phases:
            raise InvalidEquityTradingSessionError(
                "EquityTradingSession phases must be a non-empty tuple."
            )
        for phase in self.phases:
            if not isinstance(phase, EquityTradingPhase):
                raise InvalidEquityTradingSessionError(
                    "EquityTradingSession phases must be EquityTradingPhase values."
                )
        for earlier, later in zip(self.phases, self.phases[1:], strict=False):
            if earlier.closes_at.compare(later.opens_at) >= 0:
                raise InvalidEquityTradingSessionError(
                    "EquityTradingSession phases must be ascending and separated by a break."
                )
        if not isinstance(self.reference, str) or not self.reference.strip():
            raise InvalidEquityTradingSessionError(
                "EquityTradingSession reference must be a non-empty string."
            )

    @property
    def opens_at(self) -> PointInTime:
        """Return the opening instant of the first continuous-trading phase."""
        return self.phases[0].opens_at

    @property
    def closes_at(self) -> PointInTime:
        """Return the closing instant of the last continuous-trading phase."""
        return self.phases[-1].closes_at

    def __str__(self) -> str:
        phases = ", ".join(str(phase) for phase in self.phases)
        return f"{self.exchange_code} {self.trading_date.isoformat()} {self.kind} {phases}"


class EquityTradingSessionResolver(ABC):
    """Resolves the equity trading sessions of a venue from reference data.

    Implementations must honour these semantics for ``resolve``:

    - A date the reference states is a session returns one
      EquityTradingSession whose ``trading_date`` and ``exchange_code`` equal
      the request.
    - A date the reference states is closed -- a holiday or any other stated
      closure -- returns None.
    - A date the reference does not state, a year the reference has not
      loaded, a special session whose timings are not established, an
      unsupported venue, and a reference that cannot answer reliably all raise
      EquitySessionResolutionError. A missing answer and "closed that day" are
      different facts and must not share a representation, and no session is
      ever inferred from the day of the week.

    The method takes a plain date: a session label has no clock. Nothing is
    read from a wall clock, a provider or the network. The abstract contract
    documents these obligations but cannot enforce them at runtime.
    """

    @abstractmethod
    def resolve(
        self, exchange_code: ExchangeCode, trading_date: date
    ) -> EquityTradingSession | None:
        """Return the session labelled ``trading_date``, or None if the venue is closed."""
