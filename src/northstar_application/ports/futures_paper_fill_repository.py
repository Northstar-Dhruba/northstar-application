"""Application port for retrieving completed futures paper fill history."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from northstar_core.paper_trading import FuturesPaperFill, PaperPortfolioIdentity


class InvalidFuturesPaperFillQueryError(ValueError):
    """Raised when a futures paper fill query is invalid."""


@dataclass(frozen=True, slots=True)
class FuturesPaperFillQuery:
    """Request for the complete futures fill history of one paper portfolio.

    Rebuilding a portfolio, and knowing which orders already have a fill, both
    need every fill the portfolio has ever received. The query therefore
    carries no contract filter, no status and no as-of boundary: as-of
    filtering is applied by the caller over the ordered history with
    PointInTime semantics.
    """

    portfolio_identity: PaperPortfolioIdentity

    def __post_init__(self) -> None:
        if not isinstance(self.portfolio_identity, PaperPortfolioIdentity):
            raise InvalidFuturesPaperFillQueryError(
                "FuturesPaperFillQuery portfolio identity must be a PaperPortfolioIdentity value."
            )


class FuturesPaperFillRepository(ABC):
    """Retrieves one paper portfolio's futures fill history in deterministic order.

    Implementations must return an immutable tuple containing every fill whose
    intent belongs to the queried portfolio, and no other, ordered:

    1. chronologically by ``filled_at``, compared with PointInTime semantics
    2. by ``order_identity.identity`` ascending for fills sharing an instant

    Chronological order is semantic, not textual: ``...T21:00:00.1Z`` sorts
    before ``...T21:00:00Z`` as text while being the later instant. Ordering by
    a stored timestamp column is acceptable only if it is proven to agree with
    PointInTime.compare().

    A valid query matching no fill returns an empty tuple. The abstract
    contract documents these obligations but cannot enforce ordering at runtime.
    """

    @abstractmethod
    def get_fills(self, query: FuturesPaperFillQuery) -> tuple[FuturesPaperFill, ...]:
        """Return one portfolio's complete futures fill history, oldest to newest."""
