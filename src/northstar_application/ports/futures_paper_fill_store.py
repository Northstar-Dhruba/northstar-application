"""Application port for persisting completed futures paper fills."""

from __future__ import annotations

from abc import ABC, abstractmethod

from northstar_core.paper_trading import FuturesPaperFill


class FuturesPaperFillConflictError(ValueError):
    """Raised when a stored futures paper fill would be replaced or re-attributed."""


class FuturesPaperFillStore(ABC):
    """Persists completed futures paper fills as immutable history.

    One FuturesPaperOrder has zero or one FuturesPaperFill. Fills are full
    fills only, so a second, different fill for an order is a conflict, never a
    partial fill.

    Implementations must honour these semantics:

    - A fill whose PaperFillIdentity is not yet stored is persisted.
    - Re-storing a fill whose identity exists and whose stored value is equal
      is an idempotent success.
    - Storing a fill whose identity exists but whose value differs must raise
      FuturesPaperFillConflictError and must NOT overwrite the stored fill.
    - Storing a fill whose order identity is already attached to a different
      PaperFillIdentity must raise FuturesPaperFillConflictError.

    A batch containing two fills sharing one PaperFillIdentity, or two fills
    sharing one PaperOrderIdentity, is a contract violation and must raise
    FuturesPaperFillConflictError, even when the two fills are equal.

    Persisting an empty tuple must be a safe no-op returning zero.

    On success, store() must return len(fills), counting an idempotent re-store
    as accepted. Partial batch success is not permitted: if any fill cannot be
    persisted, the operation must raise and leave the stored history unchanged.

    The abstract contract documents these obligations but cannot enforce them
    at runtime.
    """

    @abstractmethod
    def store(self, fills: tuple[FuturesPaperFill, ...]) -> int:
        """Persist a batch of fills and return the number of fills accepted."""
