"""Application port for persisting historical market data observations."""

from __future__ import annotations

from abc import ABC, abstractmethod

from northstar_core.market_data import HistoricalOHLCVBar


class HistoricalMarketDataStore(ABC):
    """Persists factual historical observations into research storage.

    Implementations must ensure deterministic and idempotent persistence.
    If an observation with the same logical identity (Symbol, ExchangeCode,
    Timeframe, PointInTime) already exists, the newly acquired observation
    replaces or updates the stored state to accommodate provider adjustments
    and corrections without introducing duplicate records.

    Persisting an empty tuple must be a safe no-op.

    On success, store() must return the integer count of input observations
    accepted and applied as part of the logical batch (i.e. len(observations)).
    Partial batch success is not permitted; if any observation cannot be
    persisted, the operation must fail by raising an exception rather than
    returning a partial count.
    """

    @abstractmethod
    def store(self, observations: tuple[HistoricalOHLCVBar, ...]) -> int:
        """Persist a batch of observations and return the number of records accepted."""
