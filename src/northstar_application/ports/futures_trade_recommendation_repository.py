"""Application port for reading recorded futures trade recommendations.

Two reads, both of internally recorded recommendations only:

- one recommendation by its identity, so an identical retry returns what was
  recorded instead of generating again;
- every recorded recommendation of one futures *product*, across all of its
  expiries, as lifecycles, so recommendation generation sees the complete
  product-level state it needs for one open recommendation per product
  (ADR-018, P-1).

The product query deliberately cannot be narrowed to one contract: exclusivity
is per product, and a contract-scoped read would hide an open recommendation on
another expiry. A recorded recommendation is not a published one, and a
lifecycle is recommendation state, never a paper or broker position.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from northstar_core.futures import FuturesProductReference
from northstar_core.recommendations.futures import (
    FuturesTradeRecommendation,
    FuturesTradeRecommendationIdentity,
    FuturesTradeRecommendationLifecycle,
)


class InvalidFuturesTradeRecommendationQueryError(ValueError):
    """Raised when a futures trade recommendation query is invalid."""


@dataclass(frozen=True, slots=True)
class FuturesTradeRecommendationQuery:
    """Every recorded recommendation of one futures product, on any expiry."""

    product: FuturesProductReference

    def __post_init__(self) -> None:
        if not isinstance(self.product, FuturesProductReference):
            raise InvalidFuturesTradeRecommendationQueryError(
                "FuturesTradeRecommendationQuery product must be a FuturesProductReference."
            )


class FuturesTradeRecommendationRepository(ABC):
    """Reads internally recorded futures trade recommendations.

    Implementations must honour these semantics:

    - ``get_recommendation`` returns the recommendation recorded under exactly
      that identity, or None when none is.
    - ``get_lifecycles`` returns the current lifecycle of every recommendation
      recorded for the query's product, terminal or not, on every expiry. The
      view must be complete: an omitted open recommendation would let a second
      one be generated for the product. Ordering carries no meaning.

    The abstract contract documents these obligations but cannot enforce them
    at runtime; completeness in particular is a trusted property of the
    implementation.
    """

    @abstractmethod
    def get_recommendation(
        self, identity: FuturesTradeRecommendationIdentity
    ) -> FuturesTradeRecommendation | None:
        """Return the recommendation recorded under ``identity``, if any."""

    @abstractmethod
    def get_lifecycles(
        self, query: FuturesTradeRecommendationQuery
    ) -> tuple[FuturesTradeRecommendationLifecycle, ...]:
        """Return every recorded recommendation lifecycle of the query's product."""
