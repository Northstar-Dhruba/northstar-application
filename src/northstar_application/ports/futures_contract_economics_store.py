"""Application port for persisting futures contract economics."""

from __future__ import annotations

from abc import ABC, abstractmethod

from northstar_core.futures import FuturesContractEconomics


class FuturesContractEconomicsConflictError(ValueError):
    """Raised when stored contract economics would be replaced by different economics."""


class FuturesContractEconomicsStore(ABC):
    """Persists futures contract economics as immutable reference facts.

    Economics are contract-level, keyed by their complete FuturesContract, and
    assumed constant for the life of that contract. Different expiries of one
    product are different keys and coexist, whatever their point values. The
    economics are supplied by an operator, never fabricated or loaded from a
    provider here.

    Implementations must honour these semantics for each contract:

    - absent: the economics are persisted.
    - present and equal to the economics being stored: an idempotent success,
      so a retried write is safe. Equality is full value equality.
    - present and different, in point-value amount or currency: raise
      FuturesContractEconomicsConflictError and leave the stored economics
      exactly as they were.

    Input must be a tuple of FuturesContractEconomics. A batch containing two
    entries for one contract is a contract violation and must raise
    FuturesContractEconomicsConflictError, even when the two are equal.

    Persisting an empty tuple must be a safe no-op returning zero.

    On success, store() must return len(economics), counting an idempotent
    re-store as accepted. Partial batch success is not permitted: if any entry
    cannot be persisted, the operation must raise and leave stored economics
    unchanged.

    The abstract contract documents these obligations but cannot enforce them
    at runtime.
    """

    @abstractmethod
    def store(self, economics: tuple[FuturesContractEconomics, ...]) -> int:
        """Persist a batch of contract economics and return the number accepted."""
