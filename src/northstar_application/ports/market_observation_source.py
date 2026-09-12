"""Application abstraction for acquiring one asset's market observations."""

from __future__ import annotations

from abc import ABC, abstractmethod

from northstar_core.foundation.value_objects import Symbol
from northstar_core.strategy import MarketObservationContext


class MarketObservationSource(ABC):
    """Acquires factual market observations for one requested asset."""

    @abstractmethod
    def get_observation_context(self, symbol: Symbol) -> MarketObservationContext:
        """Return complete factual market observations for one symbol."""
