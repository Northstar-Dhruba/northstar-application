"""Application port for looking up futures contract economics."""

from __future__ import annotations

from abc import ABC, abstractmethod

from northstar_core.futures import FuturesContract, FuturesContractEconomics


class FuturesContractEconomicsRepository(ABC):
    """Retrieves the economics of one individual futures contract.

    Economics are contract-level: expiries of one product can carry different
    point values, so the lookup key is the complete FuturesContract -- product,
    exchange and expiration -- never its FuturesProductReference.

    Implementations must return the economics whose ``contract`` equals the
    requested contract, or None when none are configured for exactly that
    contract. They must never fall back to another expiration of the product,
    to product-level economics or to a default point value or currency: an
    unconfigured contract is None, and deciding what that means belongs to the
    caller. The port reaches no provider or network. The abstract contract
    documents these obligations but cannot enforce them at runtime.
    """

    @abstractmethod
    def get_economics(self, contract: FuturesContract) -> FuturesContractEconomics | None:
        """Return one contract's economics, or None when none are configured."""
