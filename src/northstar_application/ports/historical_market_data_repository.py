"""Application port for deterministic historical market data retrieval."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from northstar_core.foundation.value_objects import ExchangeCode, PointInTime, Symbol, Timeframe
from northstar_core.market_data import HistoricalOHLCVBar


class InvalidHistoricalMarketDataQueryError(ValueError):
    """Raised when a historical market data query has an invalid range."""


@dataclass(frozen=True, slots=True)
class HistoricalMarketDataQuery:
    """Inclusive request for one market identity and time range.

    Both ``start`` and ``end`` are inclusive. Equal endpoints are valid.

    Chronological validation uses PointInTime.compare(), which compares the
    actual normalized temporal instant rather than its serialized value.
    """

    symbol: Symbol
    exchange_code: ExchangeCode
    timeframe: Timeframe
    start: PointInTime
    end: PointInTime

    def __post_init__(self) -> None:
        if self.start.compare(self.end) > 0:
            raise InvalidHistoricalMarketDataQueryError(
                "HistoricalMarketDataQuery start must be before or equal to end."
            )


class HistoricalMarketDataRepository(ABC):
    """Retrieves factual historical observations in chronological order.

    Implementations must return an immutable tuple ordered oldest to newest,
    bounded by the inclusive query range. A valid query with no observations
    returns an empty tuple. The abstract contract documents these obligations
    but cannot enforce ordering at runtime.
    """

    @abstractmethod
    def get_history(self, query: HistoricalMarketDataQuery) -> tuple[HistoricalOHLCVBar, ...]:
        """Return observations from start through end, oldest to newest."""
