"""Application port for acquiring futures historical market data.

No provider Northstar has evidence for supplies a session-aligned daily futures
bar, so a daily bar is folded from minute bars. This port is where those minute
bars enter Application, one trading session at a time.

Why a session rather than an instant range
-------------------------------------------
Acquisition deliberately does not reuse FuturesHistoricalMarketDataQuery, whose
bounds are inclusive at both ends. Consecutive sessions abut -- the previous
session's ``closes_at`` is the next session's ``opens_at`` -- while a canonical
bar is stamped at its interval *completion*. An inclusive window
``[opens_at, closes_at]`` therefore admits the bar completing exactly at
``opens_at``, which belongs to the previous session:

    session A closes   2026-09-15T22:00:00Z
    session B opens    2026-09-15T22:00:00Z
    a bar completing at 22:00:00Z opened at 21:59 and is session A's last minute

That single instant is the whole difference between the inclusive query rule
and the session rule ``opens_at < point_in_time <= closes_at``. Passing the
session itself removes the ambiguity rather than encoding a half-open window in
a contract whose bounds are inclusive, and it leaves the repository query --
which is correct for reading stored bars by instant range -- untouched.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, datetime

from northstar_core.futures import FuturesContract, FuturesOHLCVBar

from northstar_application.ports.futures_trading_session_resolver import (
    FuturesTradingSession,
)


class InvalidFuturesDailyHistoricalAcquisitionQueryError(ValueError):
    """Raised when a futures daily historical acquisition query is invalid."""


@dataclass(frozen=True, slots=True)
class FuturesDailyHistoricalAcquisitionQuery:
    """Request to acquire one contract's daily history over a range of sessions.

    ``start_trading_date`` and ``end_trading_date`` are inclusive exchange
    *session labels*, not civil days over which to aggregate and not instants.
    Which labels are sessions is the calendar's answer, not this query's: a
    range may name weekends and holidays, and those simply yield no sessions.

    The query carries no timeframe. Daily acquisition folds minute bars into a
    daily bar, and both timeframes are fixed by that purpose.

    It deliberately does not check the range against the contract's tradable
    life. A range outside it enumerates sessions that return no bars, which is
    already the correct outcome, and an expiry-aware rule here would duplicate
    knowledge the data itself supplies.
    """

    contract: FuturesContract
    start_trading_date: date
    end_trading_date: date

    def __post_init__(self) -> None:
        if not isinstance(self.contract, FuturesContract):
            raise InvalidFuturesDailyHistoricalAcquisitionQueryError(
                "FuturesDailyHistoricalAcquisitionQuery contract must be a FuturesContract value."
            )
        _validate_label(self.start_trading_date, "start trading date")
        _validate_label(self.end_trading_date, "end trading date")

        if self.start_trading_date > self.end_trading_date:
            raise InvalidFuturesDailyHistoricalAcquisitionQueryError(
                "FuturesDailyHistoricalAcquisitionQuery start trading date "
                "must not be after the end trading date."
            )

    def __str__(self) -> str:
        return (
            f"{self.contract} "
            f"[{self.start_trading_date.isoformat()} .. {self.end_trading_date.isoformat()}]"
        )

    def __repr__(self) -> str:
        return (
            "FuturesDailyHistoricalAcquisitionQuery("
            f"contract={self.contract!r}, "
            f"start_trading_date={self.start_trading_date!r}, "
            f"end_trading_date={self.end_trading_date!r}"
            ")"
        )


def _validate_label(value: date, field_name: str) -> None:
    if value is None:
        raise InvalidFuturesDailyHistoricalAcquisitionQueryError(
            f"FuturesDailyHistoricalAcquisitionQuery {field_name} cannot be None."
        )
    if isinstance(value, datetime):
        raise InvalidFuturesDailyHistoricalAcquisitionQueryError(
            f"FuturesDailyHistoricalAcquisitionQuery {field_name} "
            "must be a plain date, not a datetime."
        )
    if not isinstance(value, date):
        raise InvalidFuturesDailyHistoricalAcquisitionQueryError(
            f"FuturesDailyHistoricalAcquisitionQuery {field_name} must be a date value."
        )


class FuturesHistoricalMarketDataSource(ABC):
    """Acquires one futures contract's minute observations for one session.

    Implementations must return an immutable tuple of canonical Northstar
    FuturesOHLCVBar values, ordered oldest to newest, for exactly the requested
    contract and session. A session in which nothing traded returns an empty
    tuple; that is a fact, not a failure.

    Every returned bar carries:

    - the requested FuturesContract
    - ``Timeframe("1m")``
    - a ``point_in_time`` that is the interval's **completion** instant, and
      that satisfies ``session.opens_at < point_in_time <= session.closes_at``

    Sparse output is expected and valid. A minute in which nothing traded
    produces no bar at all, so consecutive bars need not be a minute apart and
    a session may return only a handful.

    What Infrastructure owns
    ------------------------
    Providers timestamp a bar at the interval's *open*, not its completion, and
    they are queried in those terms. An implementation therefore requests

        session.opens_at <= provider interval-open timestamp < session.closes_at

    and converts each provider timestamp to the canonical completion instant by
    adding exactly one timeframe. That addition is legitimate only because a
    one-minute timeframe *defines* a sixty-second interval; it must never be
    generalised to calendar-dependent timeframes, and it must never be used to
    derive a session boundary, whose length varies.

    Provider symbols, instrument identifiers, payload formats and authentication
    are likewise Infrastructure's alone. Nothing provider-shaped crosses this
    boundary.

    There is no timeframe parameter. This port exists to feed daily-session
    aggregation, which requires minute bars; accepting a coarser timeframe
    would let a caller request data the aggregation must then reject.

    The abstract contract documents these obligations but cannot enforce them
    at runtime. They are enforced centrally by
    AggregateFuturesDailySessionBarUseCase, which the acquisition use case
    applies to every source result.
    """

    @abstractmethod
    def fetch_session_bars(
        self, contract: FuturesContract, session: FuturesTradingSession
    ) -> tuple[FuturesOHLCVBar, ...]:
        """Return one session's minute observations, oldest to newest."""
