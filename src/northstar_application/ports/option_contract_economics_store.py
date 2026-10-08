"""Application port for persisting option contract economics."""

from __future__ import annotations

from abc import ABC, abstractmethod

from northstar_core.options import OptionContractEconomics


class OptionContractEconomicsConflictError(ValueError):
    """Raised when stored option economics would be replaced by different economics."""


class OptionContractEconomicsStore(ABC):
    """Persists option contract economics as immutable reference facts.

    Economics are contract-level, keyed by their complete OptionContract, and
    assumed constant for the life of that contract. Different expiries, strikes
    and rights of one product are different keys and coexist, whatever their
    point values. The economics are supplied by an operator, never fabricated
    or loaded from a provider here.

    Implementations must honour these semantics for each contract:

    - absent: the economics are persisted.
    - present and equal to the economics being stored: an idempotent success,
      so a retried write is safe. Equality is full value equality.
    - present and different, in point-value amount or currency: raise
      OptionContractEconomicsConflictError and leave the stored economics
      exactly as they were.

    Input must be a tuple of OptionContractEconomics. A batch containing two
    entries for one contract is a contract violation and must raise
    OptionContractEconomicsConflictError, even when the two are equal.

    Persisting an empty tuple must be a safe no-op returning zero.

    On success, store() must return len(economics), counting an idempotent
    re-store as accepted. Partial batch success is not permitted: if any entry
    cannot be persisted, the operation must raise and leave stored economics
    unchanged.

    The abstract contract documents these obligations but cannot enforce them
    at runtime.
    """

    @abstractmethod
    def store(self, economics: tuple[OptionContractEconomics, ...]) -> int:
        """Persist a batch of option contract economics and return the number accepted."""
