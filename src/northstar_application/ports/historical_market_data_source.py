"""Application port for external historical market data acquisition."""

from __future__ import annotations

from abc import ABC, abstractmethod

from northstar_core.market_data import HistoricalOHLCVBar

from northstar_application.ports.historical_market_data_repository import (
    HistoricalMarketDataQuery,
)


class HistoricalMarketDataSource(ABC):
    """Acquires factual historical observations from an external data source.

    Implementations must return an immutable tuple ordered oldest to newest,
    bounded by the inclusive query range. A valid query with no available
    observations must return an empty tuple ``()``.
    """

    @abstractmethod
    def fetch_history(self, query: HistoricalMarketDataQuery) -> tuple[HistoricalOHLCVBar, ...]:
        """Fetch historical observations from start through end, oldest to newest."""
