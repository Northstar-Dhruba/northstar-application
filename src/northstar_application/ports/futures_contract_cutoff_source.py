"""Application port for the verified last tradable session of a futures contract.

A contract's expiration date is a calendar date. It is not proof that the
contract's last tradable session falls on that date: the date can be a holiday,
an exchange can move an expiry, and a product can stop trading before the date
it is named for. A recommendation's contract cutoff is therefore never derived
from ``FuturesContract.expiration_date``; it comes from this port, whose
implementation must resolve it from an authoritative contract reference.

The port returns ``None`` when no verified cutoff is known. A caller must then
fail closed rather than guess. Nothing here computes exchange rules.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from northstar_core.futures import FuturesContract

from northstar_application.ports.futures_trading_session_resolver import FuturesTradingSession


class InvalidFuturesContractCutoffError(ValueError):
    """Raised when a FuturesContractCutoff value is invalid."""


@dataclass(frozen=True, slots=True)
class FuturesContractCutoff:
    """The verified last tradable session of one contract and who verified it.

    ``last_session`` is the contract's final eligible session; its ``closes_at``
    is the recommendation contract cutoff. ``authority`` names the contract
    reference that established it -- an exchange circular, a reviewed contract
    master -- so the provenance of every cutoff stays visible. It is a trusted
    statement by the port's implementation, not something Application verifies.
    """

    contract: FuturesContract
    last_session: FuturesTradingSession
    authority: str

    def __post_init__(self) -> None:
        if not isinstance(self.contract, FuturesContract):
            raise InvalidFuturesContractCutoffError(
                "FuturesContractCutoff contract must be a FuturesContract value."
            )
        if not isinstance(self.last_session, FuturesTradingSession):
            raise InvalidFuturesContractCutoffError(
                "FuturesContractCutoff last session must be a FuturesTradingSession value."
            )
        if not isinstance(self.authority, str) or not self.authority.strip():
            raise InvalidFuturesContractCutoffError(
                "FuturesContractCutoff authority must name the contract reference."
            )
        object.__setattr__(self, "authority", self.authority.strip())

    def __str__(self) -> str:
        return f"{self.contract} last session {self.last_session} per {self.authority}"


class FuturesContractCutoffSource(ABC):
    """Resolves the verified last tradable session of one exact contract.

    Implementations must honour these semantics:

    - The answer is for exactly the ``contract`` asked about.
    - ``None`` means no verified cutoff is known; it is never a guess.
    - No cutoff is computed from the expiration date alone.

    The abstract contract documents these obligations but cannot enforce them
    at runtime.
    """

    @abstractmethod
    def resolve(self, contract: FuturesContract) -> FuturesContractCutoff | None:
        """Return the verified cutoff of ``contract``, or None when none is known."""
