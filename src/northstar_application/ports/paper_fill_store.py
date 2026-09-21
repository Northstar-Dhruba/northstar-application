"""Application port for persisting completed paper fills."""

from __future__ import annotations

from abc import ABC, abstractmethod

from northstar_core.paper_trading import PaperFill


class PaperFillConflictError(ValueError):
    """Raised when a stored paper fill would be replaced or re-attributed."""


class PaperFillStore(ABC):
    """Persists completed paper fills as immutable history.

    A fill is a record of something that already happened, so persistence is
    append-only. Holdings are never stored: a portfolio is folded from this
    history on demand, which is why the history itself must never change under
    a reader's feet.

    Implementations must honour these semantics:

    - A fill whose PaperFillIdentity is not yet stored is persisted.
    - Re-storing a fill whose identity exists and whose stored value is equal
      is an idempotent success, so a retried write is safe.
    - Storing a fill whose identity exists but whose value differs must raise
      PaperFillConflictError and must NOT overwrite the stored fill.
    - The current model permits exactly one fill per PaperOrderIdentity.
      Storing a fill whose order identity is already attached to a different
      PaperFillIdentity must raise PaperFillConflictError. Relaxing this is
      what partial fills would require, and it is deliberately not permitted
      yet.

    A batch containing two fills that share one PaperFillIdentity, or two
    fills that share one PaperOrderIdentity, is a contract violation and must
    raise rather than silently collapsing them.

    Persisting an empty tuple must be a safe no-op returning zero.

    On success, store() must return the integer count of input fills accepted
    as part of the logical batch (i.e. len(fills)), counting an idempotent
    re-store as accepted. Partial batch success is not permitted: if any fill
    cannot be persisted, the operation must fail by raising rather than
    returning a partial count, leaving the stored history unchanged.

    The abstract contract documents these obligations but cannot enforce them
    at runtime.
    """

    @abstractmethod
    def store(self, fills: tuple[PaperFill, ...]) -> int:
        """Persist a batch of fills and return the number of fills accepted."""
