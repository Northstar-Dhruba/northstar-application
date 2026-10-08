"""Application port for looking up option contract economics."""

from __future__ import annotations

from abc import ABC, abstractmethod

from northstar_core.options import OptionContract, OptionContractEconomics


class OptionContractEconomicsRepository(ABC):
    """Retrieves the economics of one individual option contract.

    Economics are contract-level: every expiry, strike and right of a product
    is its own contract and can carry its own point value, so the lookup key is
    the complete OptionContract -- product, exchange, expiration, strike and
    right -- never its OptionProductReference.

    Implementations must return the economics whose ``contract`` equals the
    requested contract, or None when none are configured for exactly that
    contract. They must never fall back to a neighbouring strike, the other
    right, another expiration, another product, product-level economics or a
    default point value or currency: an unconfigured contract is None, and
    deciding what that means belongs to the caller. The port reaches no provider
    or network. The abstract contract documents these obligations but cannot
    enforce them at runtime.
    """

    @abstractmethod
    def get_economics(self, contract: OptionContract) -> OptionContractEconomics | None:
        """Return one contract's economics, or None when none are configured."""
