"""Application port for persisting canonical option daily market data."""

from __future__ import annotations

from abc import ABC, abstractmethod

from northstar_core.options import OptionOHLCVBar


class OptionHistoricalMarketDataConflictError(ValueError):
    """Raised when a stored option bar would be replaced by a different bar."""


class OptionHistoricalMarketDataStore(ABC):
    """Persists canonical option bars as immutable evidence.

    A bar's natural identity is ``(OptionContract, PointInTime, Timeframe)``,
    and an OptionContract is product, exchange, expiration, strike and right,
    so a call and a put, two strikes and two expiries at one instant occupy
    different keys.

    Implementations must honour these semantics:

    - A bar whose natural key is not yet stored is persisted.
    - Re-storing a bar whose natural key exists with an equal value is an
      idempotent success.
    - Storing a bar whose natural key exists with a different value raises
      OptionHistoricalMarketDataConflictError and does not overwrite the stored
      bar. A differing bar may be a correction or a disagreement between
      sources; deciding which is a reconciliation step, never a silent write.
    - A batch containing two bars sharing one natural key raises, even when
      they are equal.
    - An empty batch is a safe no-op returning zero.
    - On success, store() returns len(bars), counting idempotent re-stores as
      accepted. A batch is stored completely or not at all.

    The abstract contract documents these obligations but cannot enforce them
    at runtime.
    """

    @abstractmethod
    def store(self, bars: tuple[OptionOHLCVBar, ...]) -> int:
        """Persist a batch of option bars and return the number of bars accepted."""
