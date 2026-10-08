"""Application port for resolving option trading-session windows.

A canonical option daily bar is stamped at its trading session's close, so the
sessions an option trades in, and their boundaries, come from one authoritative
port rather than from a provider's candle timestamps.

Option sessions are deliberately separate from futures sessions. Futures open
earlier, with a pre-open session that NSE applies to futures only, so a futures
window would stamp an option bar at the wrong instants. Only the day-level
calendar of the derivatives segment is shared, and that sharing is an
implementation detail of the adapter.

The port has one operation, the one native daily acquisition needs: every
session labelled within an inclusive range.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, datetime

from northstar_core.foundation.value_objects import PointInTime
from northstar_core.options import OptionProductReference


class OptionTradingSessionResolutionError(RuntimeError):
    """Raised when option trading sessions cannot be resolved reliably."""


class InvalidOptionTradingSessionError(ValueError):
    """Raised when an OptionTradingSession value is invalid."""


def _validate_trading_date(value: date) -> date:
    if value is None:
        raise InvalidOptionTradingSessionError("OptionTradingSession trading date cannot be None.")
    if isinstance(value, datetime) or not isinstance(value, date):
        raise InvalidOptionTradingSessionError(
            "OptionTradingSession trading date must be a plain date."
        )
    return value


def _validate_instant(value: PointInTime, field_name: str) -> PointInTime:
    if value is None:
        raise InvalidOptionTradingSessionError(f"OptionTradingSession {field_name} cannot be None.")
    if not isinstance(value, PointInTime):
        raise InvalidOptionTradingSessionError(
            f"OptionTradingSession {field_name} must be a PointInTime value."
        )
    return value


@dataclass(frozen=True, slots=True)
class OptionTradingSession:
    """One option trading session, named by its label and bounded by instants.

    ``trading_date`` is the exchange session label. ``opens_at`` and
    ``closes_at`` are the option normal market's open and close; ``closes_at``
    is the instant a daily option bar is stamped at. The window is half-open on
    the left and closed on the right: ``opens_at < instant <= closes_at``.

    This is calendar-derived Application data and carries no instrument
    identity.
    """

    trading_date: date
    opens_at: PointInTime
    closes_at: PointInTime

    def __post_init__(self) -> None:
        trading_date = _validate_trading_date(self.trading_date)
        opens_at = _validate_instant(self.opens_at, "opens_at")
        closes_at = _validate_instant(self.closes_at, "closes_at")

        # Compared semantically: text ordering of canonical instants is unreliable.
        if opens_at.compare(closes_at) >= 0:
            raise InvalidOptionTradingSessionError(
                "OptionTradingSession opens_at must be before closes_at."
            )

        object.__setattr__(self, "trading_date", trading_date)
        object.__setattr__(self, "opens_at", opens_at)
        object.__setattr__(self, "closes_at", closes_at)

    def __str__(self) -> str:
        return f"{self.trading_date.isoformat()} [{self.opens_at} .. {self.closes_at}]"

    def __repr__(self) -> str:
        return (
            "OptionTradingSession("
            f"trading_date={self.trading_date!r}, "
            f"opens_at={self.opens_at!r}, "
            f"closes_at={self.closes_at!r}"
            ")"
        )


class OptionTradingSessionResolver(ABC):
    """Resolves the trading sessions of one option product's venue.

    Implementations must honour these semantics for ``sessions_in_range``:

    - ``start_date`` and ``end_date`` are inclusive session labels, and
      ``start_date`` must not be after ``end_date``.
    - Non-session dates -- weekends and published trading holidays -- are
      simply absent.
    - Results are ordered strictly ascending by ``trading_date``, with no
      duplicates, and every label falls inside the requested range.
    - A range containing no session returns an empty tuple.
    - An unsupported product or venue, a date outside the sourced option
      session regimes, a date whose option session timings are not
      established, and a calendar that cannot answer reliably all raise
      OptionTradingSessionResolutionError. A range is never partly answered.

    The method takes plain dates: a session label has no clock. Nothing is read
    from a wall clock, a provider or the network. The abstract contract
    documents these obligations but cannot enforce them at runtime.
    """

    @abstractmethod
    def sessions_in_range(
        self, product: OptionProductReference, start_date: date, end_date: date
    ) -> tuple[OptionTradingSession, ...]:
        """Return every option session labelled within the inclusive range, ascending."""
