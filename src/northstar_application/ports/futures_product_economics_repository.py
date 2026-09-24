"""Application port for looking up futures product economics."""

from __future__ import annotations

from abc import ABC, abstractmethod

from northstar_core.futures import FuturesProductEconomics, FuturesProductReference


class FuturesProductEconomicsRepository(ABC):
    """Retrieves the economics of one exchange-defined futures product.

    Economics are product-level: every expiry of a product shares them, so the
    lookup key is a FuturesProductReference, never a FuturesContract.

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
