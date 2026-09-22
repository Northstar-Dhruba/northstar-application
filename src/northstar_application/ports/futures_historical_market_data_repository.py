"""Application port for deterministic futures historical market data retrieval."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from northstar_core.foundation.value_objects import PointInTime, Timeframe
from northstar_core.futures import FuturesContract, FuturesOHLCVBar


class InvalidFuturesHistoricalMarketDataQueryError(ValueError):
    """Raised when a futures historical market data query is invalid."""


@dataclass(frozen=True, slots=True)
class FuturesHistoricalMarketDataQuery:
    """Request for one concrete expiring contract's observations over a window.

    The query names a FuturesContract, which is ``(product reference,
    expiration date)`` and therefore already carries the product code, the
    exchange and the expiry. Nothing narrower and nothing broader identifies a
    series of bars: ES and MES on one exchange with one expiry are different
    instruments, and March and June of one product are different instruments
    too, so a query that could not tell them apart would silently blend two
    price series into one.

    That is why the query deliberately cannot be made by underlying, by product
    alone, by a provider symbol or by a listing reference. An underlying is
    shared by every product written on it, a product spans every expiry it has
    ever listed, and a provider symbol is a vendor's spelling rather than a
    domain identity. Continuous contracts are not queryable here at all: a
    continuous series is constructed from many contracts by a method this
    domain has not chosen, so it is not a thing this port can return.

    ``start`` and ``end`` are both optional and, when present, both inclusive.
    Equal endpoints are valid and select the single instant. Omitting a bound
    leaves that side open, which is meaningful for futures in a way it is not
    for equities: a contract's life is already bounded by its own listing and
    expiry, so "every bar of this contract" is a finite request rather than an
    unbounded scan.

    Chronological validation uses PointInTime.compare(), which compares the
    normalized temporal instant rather than its serialized text. The two
    disagree: a canonical instant omits fractional seconds when they are zero,
    so ``...T21:00:00.5Z`` sorts before ``...T21:00:00Z`` as text while being
    the later instant, and two spellings at different UTC offsets can name one
    instant while comparing unequal as strings.
    """

    contract: FuturesContract
    timeframe: Timeframe
    start: PointInTime | None = None
    end: PointInTime | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.contract, FuturesContract):
            raise InvalidFuturesHistoricalMarketDataQueryError(
                "FuturesHistoricalMarketDataQuery contract must be a FuturesContract value."
            )
        if not isinstance(self.timeframe, Timeframe):
            raise InvalidFuturesHistoricalMarketDataQueryError(
                "FuturesHistoricalMarketDataQuery timeframe must be a Timeframe value."
            )
        if self.start is not None and not isinstance(self.start, PointInTime):
            raise InvalidFuturesHistoricalMarketDataQueryError(
                "FuturesHistoricalMarketDataQuery start must be a PointInTime value."
            )
        if self.end is not None and not isinstance(self.end, PointInTime):
            raise InvalidFuturesHistoricalMarketDataQueryError(
                "FuturesHistoricalMarketDataQuery end must be a PointInTime value."
            )
        if self.start is not None and self.end is not None and self.start.compare(self.end) > 0:
            raise InvalidFuturesHistoricalMarketDataQueryError(
                "FuturesHistoricalMarketDataQuery start must be before or equal to end."
            )

    def covers(self, point_in_time: PointInTime) -> bool:
        """Return whether an instant falls inside this query's inclusive window.

        Provided so that every implementation applies one window rule rather
        than re-deriving inclusivity, and so the comparison is semantic. An
        adapter that filters on a persisted timestamp column may only do so if
        that ordering is proven to agree with PointInTime.compare().
        """
        if self.start is not None and point_in_time.compare(self.start) < 0:
            return False
        return not (self.end is not None and point_in_time.compare(self.end) > 0)


class FuturesHistoricalMarketDataRepository(ABC):
    """Retrieves one futures contract's observations in chronological order.

    Implementations must return an immutable tuple containing exactly those
    bars whose contract and timeframe equal the query's and whose instant falls
    inside the inclusive window, ordered oldest to newest.

    Ordering needs no tie-break. A bar's natural key is ``(FuturesContract,
    PointInTime, Timeframe)``, and a query fixes the contract and the
    timeframe, so at most one bar can occupy any instant in a conforming
    result. Two bars sharing an instant in one result means the underlying
    store holds conflicting evidence, which is the condition the store port
    exists to prevent.

    The chronological requirement is semantic, not textual, for the reasons
    given on the query. An implementation must not encode its storage engine's
    collation into this contract.

    Matching is by value equality on the contract, so a query for ES March
    never returns MES March or ES June even though those share a venue, an
    underlying and, in the first case, an expiry.

    A valid query matching no observation returns an empty tuple. This port
    performs no acquisition: it reads what has already been stored and reaches
    no provider or network. The abstract contract documents these obligations
    but cannot enforce ordering at runtime.
    """

    @abstractmethod
    def get_bars(self, query: FuturesHistoricalMarketDataQuery) -> tuple[FuturesOHLCVBar, ...]:
        """Return one contract's observations within the window, oldest to newest."""
