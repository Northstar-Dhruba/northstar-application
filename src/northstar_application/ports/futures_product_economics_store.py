"""Application port for persisting futures product economics."""

from __future__ import annotations

from abc import ABC, abstractmethod

from northstar_core.futures import FuturesProductEconomics


class FuturesProductEconomicsConflictError(ValueError):
    """Raised when stored product economics would be replaced by different economics."""


class FuturesProductEconomicsStore(ABC):
    """Persists futures product economics as immutable reference facts.

    Economics are product-level, keyed by their FuturesProductReference -- never
    by a contract or expiry -- and assumed constant for that reference. They are
    supplied by an operator, never fabricated or loaded from a provider here.

    Implementations must honour these semantics for each product reference:

    - absent: the economics are persisted.
    - present and equal to the economics being stored: an idempotent success,
      so a retried write is safe. Equality is full value equality.
    - present and different, in point-value amount or currency: raise
      FuturesProductEconomicsConflictError and leave the stored economics
      exactly as they were.

    Input must be a tuple of FuturesProductEconomics. A batch containing two
    entries for one product reference is a contract violation and must raise
    FuturesProductEconomicsConflictError, even when the two are equal.

    Persisting an empty tuple must be a safe no-op returning zero.

    On success, store() must return len(economics), counting an idempotent
    re-store as accepted. Partial batch success is not permitted: if any entry
    cannot be persisted, the operation must raise and leave stored economics
    unchanged.

    The abstract contract documents these obligations but cannot enforce them
    at runtime.
    """

    @abstractmethod
    def store(self, economics: tuple[FuturesProductEconomics, ...]) -> int:
        """Persist a batch of product economics and return the number accepted."""
