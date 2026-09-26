"""Application port for retrieving completed paper fill history."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from northstar_core.paper_trading import PaperFill, PaperPortfolioIdentity


class InvalidPaperFillQueryError(ValueError):
    """Raised when a paper fill query is invalid."""


@dataclass(frozen=True, slots=True)
class PaperFillQuery:
    """Request for the complete fill history of one paper portfolio.

    The query identifies a portfolio and nothing narrower. Folding a portfolio
    requires every fill that has ever affected it, because a holding in one
    listing can only be known from the full sequence of buys and sells against
    it; asking callers to name listings first would require them to know the
    answer before they could ask the question.

    The query deliberately carries no listing filter, no as-of boundary and no
    status. As-of filtering belongs to the fold, which applies it semantically
    over an already-ordered history. Status is never stored at all: a fill is
    complete by definition.
    """

    portfolio_identity: PaperPortfolioIdentity

    def __post_init__(self) -> None:
        if not isinstance(self.portfolio_identity, PaperPortfolioIdentity):
            raise InvalidPaperFillQueryError(
                "PaperFillQuery portfolio identity must be a PaperPortfolioIdentity value."
            )


class PaperFillRepository(ABC):
    """Retrieves one paper portfolio's fill history in deterministic order.

    Implementations must return an immutable tuple containing every fill
    belonging to the queried portfolio, ordered:

    1. chronologically by ``filled_at``, compared with PointInTime semantics
    2. by ``PaperFillIdentity.identity`` ascending for fills sharing an instant

    The chronological requirement is a semantic one, not a textual one. A
    canonical instant omits fractional seconds when they are zero, so
    ``...T16:00:00.1Z`` sorts before ``...T16:00:00Z`` as text while being the
    later instant. An implementation must not encode its storage engine's
    collation into this contract: ordering by a persisted timestamp column is
    only acceptable if that ordering is proven to agree with
    PointInTime.compare().

    Ordering is part of the contract rather than a convenience, because the
    fold refuses to reorder what it is given: an out-of-order history cannot be
    folded without guessing which trade happened first.

    A valid query matching no fill returns an empty tuple. The abstract
    contract documents these obligations but cannot enforce ordering at
    runtime.
    """

    @abstractmethod
    def get_fills(self, query: PaperFillQuery) -> tuple[PaperFill, ...]:
        """Return one portfolio's complete fill history, oldest to newest."""
