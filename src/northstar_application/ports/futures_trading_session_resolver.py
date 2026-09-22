"""Application port for resolving futures trading-session windows.

Daily futures bars are not supplied ready-made by any provider Northstar has
evidence for: a CME daily bar has to be aggregated from sub-daily data, and
that aggregation needs both ends of the session, not just its completion. This
port is the one authoritative source of those boundaries.

It is deliberately a single port returning a single window value rather than
two ports answering "when does it open" and "when does it close" separately.
Two ports could disagree about whether a date is a session at all, and there is
no correct way to reconcile that disagreement after the fact.

Why both boundaries are stored rather than one derived from the other
---------------------------------------------------------------------
Within a trading week a CME session opens exactly when the previous one closed,
which makes it tempting to keep only closes and chain them. Weekends and
holidays break that:

    2026-07-03  closes 2026-07-03T17:00Z   (an early close)
    2026-07-06  opens  2026-07-05T22:00Z   (two days and five hours later)

Chaining would make Monday's window start at Friday lunchtime and swallow the
whole weekend, including the Friday afternoon when the market was already shut.
Any trade falling in that gap would be attributed to the wrong session, and
nothing downstream could detect it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, datetime

from northstar_core.foundation.value_objects import PointInTime
from northstar_core.futures import FuturesProductReference


class FuturesSessionResolutionError(RuntimeError):
    """Raised when futures session resolution cannot be performed reliably."""


class InvalidFuturesTradingSessionError(ValueError):
    """Raised when a FuturesTradingSession value is invalid."""


def _validate_trading_date(value: date) -> date:
    if value is None:
        raise InvalidFuturesTradingSessionError(
            "FuturesTradingSession trading date cannot be None."
        )
    if isinstance(value, datetime):
        raise InvalidFuturesTradingSessionError(
            "FuturesTradingSession trading date must be a plain date, not a datetime."
        )
    if not isinstance(value, date):
        raise InvalidFuturesTradingSessionError(
            "FuturesTradingSession trading date must be a date value."
        )
    return value


def _validate_instant(value: PointInTime, field_name: str) -> PointInTime:
    if value is None:
        raise InvalidFuturesTradingSessionError(
            f"FuturesTradingSession {field_name} cannot be None."
        )
    if not isinstance(value, PointInTime):
        raise InvalidFuturesTradingSessionError(
            f"FuturesTradingSession {field_name} must be a PointInTime value."
        )
    return value


@dataclass(frozen=True, slots=True)
class FuturesTradingSession:
    """One exchange trading session, named by its label and bounded by instants.

    ``trading_date`` is the exchange's *session label*. It is not a UTC civil
    day over which to aggregate, and it is not the date the session begins on.
    A CME session labelled 2026-09-15 opens at 2026-09-14T22:00Z and completes
    at 2026-09-15T22:00Z, so the label and the opening civil date routinely
    differ.

    ``opens_at`` and ``closes_at`` are the actual exchange session boundaries.
    Three properties follow, and code consuming this value must not assume any
    of them away:

    - The session may open on ``trading_date`` minus one civil day. For CME
      this is not an edge case; it is every session.
    - The session may be shorter than twenty-four hours. An early close ends
      roughly five hours sooner while the sessions around it are unaffected.
    - There may be a gap from the previous session across weekends and
      holidays. ``opens_at`` is never guaranteed to equal the previous
      session's ``closes_at``, and treating it as such misattributes trades.

    The window is used as half-open on the left and closed on the right when
    assigning completion-stamped bars to a session -- ``opens_at < instant <=
    closes_at`` -- because a bar stamped at its interval completion represents
    the interval that began one timeframe earlier.

    This is calendar-derived Application data. It carries no instrument
    identity and belongs to no Core bounded context.
    """

    trading_date: date
    opens_at: PointInTime
    closes_at: PointInTime

    def __post_init__(self) -> None:
        trading_date = _validate_trading_date(self.trading_date)
        opens_at = _validate_instant(self.opens_at, "opens_at")
        closes_at = _validate_instant(self.closes_at, "closes_at")

        # Compared semantically. A canonical instant drops zero fractional
        # seconds, so text ordering disagrees with chronology.
        if opens_at.compare(closes_at) >= 0:
            raise InvalidFuturesTradingSessionError(
                "FuturesTradingSession opens_at must be before closes_at."
            )

        object.__setattr__(self, "trading_date", trading_date)
        object.__setattr__(self, "opens_at", opens_at)
        object.__setattr__(self, "closes_at", closes_at)

    def __str__(self) -> str:
        return f"{self.trading_date.isoformat()} [{self.opens_at} .. {self.closes_at}]"

    def __repr__(self) -> str:
        return (
            "FuturesTradingSession("
            f"trading_date={self.trading_date!r}, "
            f"opens_at={self.opens_at!r}, "
            f"closes_at={self.closes_at!r}"
            ")"
        )


class FuturesTradingSessionResolver(ABC):
    """Resolves the trading-session windows of one futures product's venue.

    Implementations must honour these semantics for ``resolve``:

    - A date that is a trading session for the product's venue returns one
      FuturesTradingSession whose ``trading_date`` equals the requested label.
    - A date that is not a session for that venue -- a weekend, an exchange
      holiday or a scheduled closure -- returns None.
    - An unsupported or unknown venue, or a calendar that cannot be resolved
      reliably, raises FuturesSessionResolutionError rather than returning
      None. A missing answer and "no session that day" are different facts and
      must not share a representation.

    And for ``sessions_in_range``:

    - ``start_date`` and ``end_date`` are inclusive session labels, and
      ``start_date`` must not be after ``end_date``.
    - Non-session dates are simply absent; they are not represented by None.
    - Results are ordered strictly ascending by ``trading_date``, with no
      duplicates.
    - Every returned session's label falls inside the requested inclusive
      range.
    - A range containing no session returns an empty tuple.
    - An unsupported venue or an unresolvable calendar raises
      FuturesSessionResolutionError.

    Both methods take a plain ``date``. A datetime carries a clock, and a
    session label has none.

    No calendar-library concept crosses this boundary: implementations return
    domain values only, and Application never learns which calendar source, if
    any, produced them.

    The abstract contract documents these obligations but cannot enforce them
    at runtime.
    """

    @abstractmethod
    def resolve(
        self, product: FuturesProductReference, trading_date: date
    ) -> FuturesTradingSession | None:
        """Return the session labelled ``trading_date``, or None if not a session.

        ``product`` is a FuturesProductReference rather than a bare
        ExchangeCode. Session schedules resolve at venue level today, so an
        implementation is expected to read only ``product.exchange_code``. The
        product is nevertheless the smallest stable input: if a venue ever
        needs a per-product schedule, that arrives without changing this
        signature.
        """

    @abstractmethod
    def sessions_in_range(
        self, product: FuturesProductReference, start_date: date, end_date: date
    ) -> tuple[FuturesTradingSession, ...]:
        """Return every session labelled within the inclusive range, ascending.

        Provided so that a caller enumerating a period never has to walk dates
        itself and discard non-sessions, which would mean reimplementing a
        calendar walk in Application.
        """
