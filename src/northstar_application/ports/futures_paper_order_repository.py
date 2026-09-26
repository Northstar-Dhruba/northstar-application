"""Application port for retrieving futures paper orders."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from northstar_core.paper_trading import FuturesPaperOrder, PaperPortfolioIdentity


class InvalidFuturesPaperOrderQueryError(ValueError):
    """Raised when a futures paper order query is invalid."""


@dataclass(frozen=True, slots=True)
class FuturesPaperOrderQuery:
    """Request for every futures paper order of one paper portfolio.

    The query names a portfolio and nothing narrower. Discovering which orders
    still await a fill means comparing a portfolio's orders against its fills,
    and both are retrieved by portfolio, so a contract filter would add a second
    way to ask the same question without serving any caller.

    It deliberately carries no status: pending and filled are derived, never
    stored. It carries no as-of boundary either; temporal filtering is applied
    by the caller with PointInTime semantics.
    """

    portfolio_identity: PaperPortfolioIdentity

    def __post_init__(self) -> None:
        if not isinstance(self.portfolio_identity, PaperPortfolioIdentity):
            raise InvalidFuturesPaperOrderQueryError(
                "FuturesPaperOrderQuery portfolio identity must be a PaperPortfolioIdentity value."
            )


class FuturesPaperOrderRepository(ABC):
    """Retrieves one paper portfolio's futures orders in deterministic order.

    Implementations must return an immutable tuple containing every order whose
    intent belongs to the queried portfolio, and no other, ordered:

    1. chronologically by ``intent.decided_at``, compared with PointInTime
       semantics
    2. by ``PaperOrderIdentity.identity`` ascending for orders sharing an instant

    Chronological order is semantic, not textual: ``...T21:00:00.1Z`` sorts
    before ``...T21:00:00Z`` as text while being the later instant. Ordering by
    a stored timestamp column is acceptable only if it is proven to agree with
    PointInTime.compare().

    A valid query matching no order returns an empty tuple. The abstract
    contract documents these obligations but cannot enforce ordering at runtime.
    """

    @abstractmethod
    def get_orders(self, query: FuturesPaperOrderQuery) -> tuple[FuturesPaperOrder, ...]:
        """Return one portfolio's futures orders, oldest decision first."""
