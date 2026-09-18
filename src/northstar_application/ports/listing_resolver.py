"""Application port for resolving static listing reference data."""

from __future__ import annotations

from abc import ABC, abstractmethod

from northstar_core.domain.listing import Listing
from northstar_core.foundation.value_objects import ExchangeCode, PointInTime, Symbol


class ListingResolver(ABC):
    """Resolves a canonical Listing independently of market-data acquisition."""

    @abstractmethod
    def resolve_listing(
        self,
        symbol: Symbol,
        exchange_code: ExchangeCode,
        as_of: PointInTime,
    ) -> Listing:
        """Return listing reference facts valid at the supplied replay instant."""
