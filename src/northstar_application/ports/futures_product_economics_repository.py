"""Application port for looking up futures product economics."""

from __future__ import annotations

from abc import ABC, abstractmethod

from northstar_core.futures import FuturesProductEconomics, FuturesProductReference


class FuturesProductEconomicsRepository(ABC):
    """Retrieves historical product-level economics of one futures product.

    Historical reference only: this port reads the product-level economics of
    the original ES reference MVP. It is never the profit and loss authority;
    P&L resolves FuturesContractEconomicsRepository by the complete contract,
    because expiries of one product can carry different point values. No P&L
    use case accepts this port, and nothing may read it as a fallback.

    The lookup key is a FuturesProductReference, never a FuturesContract.

    Implementations must return the economics whose ``reference`` equals the
    requested reference, or None when none are configured. They must never
    fabricate a default point value or currency: an unknown product is None,
    and deciding what that means belongs to the caller. The port reaches no
    provider or network. The abstract contract documents these obligations but
    cannot enforce them at runtime.
    """

    @abstractmethod
    def get_economics(self, reference: FuturesProductReference) -> FuturesProductEconomics | None:
        """Return one product's economics, or None when none are configured."""
