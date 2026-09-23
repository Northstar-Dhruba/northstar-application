"""Application port for persisting futures paper orders."""

from __future__ import annotations

from abc import ABC, abstractmethod

from northstar_core.paper_trading import FuturesPaperOrder


class FuturesPaperOrderConflictError(ValueError):
    """Raised when a stored futures paper order would be replaced by a different one."""


class FuturesPaperOrderStore(ABC):
    """Persists futures paper orders as immutable history.

    A futures paper order is stored when a decision is taken, before the next
    stored bar that will fill it exists. The order carries no status and none is
    persisted: whether it is pending or filled is derived later from whether a
    FuturesPaperFill exists for its identity.

    Implementations must honour these semantics for each PaperOrderIdentity:

    - absent: the order is persisted.
    - present and equal to the order being stored: an idempotent success, so a
      retried write is safe. Equality is full value equality.
    - present and different: raise FuturesPaperOrderConflictError and leave the
      stored order exactly as it was. Order identities are derived from the
      decision, so a retry whose side or size changed is a conflict, never a
      second order.

    A batch containing two orders sharing one PaperOrderIdentity is a contract
    violation and must raise FuturesPaperOrderConflictError, even when the two
    orders are equal.

    Persisting an empty tuple must be a safe no-op returning zero.

    On success, store() must return len(orders), counting an idempotent re-store
    as accepted. Partial batch success is not permitted: if any order cannot be
    persisted, the operation must raise and leave the stored history unchanged.

    The abstract contract documents these obligations but cannot enforce them
    at runtime.
    """

    @abstractmethod
    def store(self, orders: tuple[FuturesPaperOrder, ...]) -> int:
        """Persist a batch of orders and return the number of orders accepted."""
