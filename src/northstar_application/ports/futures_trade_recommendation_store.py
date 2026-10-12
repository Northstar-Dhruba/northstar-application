"""Application port for internally recording futures trade recommendations.

Recording is internal: a recorded recommendation is not published, delivered,
notified or executed, and this port models none of those states.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from northstar_core.recommendations.futures import FuturesTradeRecommendation


class FuturesTradeRecommendationConflictError(ValueError):
    """Raised when a recorded recommendation would be replaced by a different one."""


class FuturesTradeRecommendationExclusivityError(ValueError):
    """Raised when recording would leave two open recommendations on one product."""


class FuturesTradeRecommendationStore(ABC):
    """Records futures trade recommendations as immutable history.

    Implementations must honour these semantics for each recommendation:

    - identity absent, and no open recommendation on its product: recorded.
    - identity present with an equal recommendation: an idempotent success.
    - identity present with a different recommendation: raise
      FuturesTradeRecommendationConflictError; the record is unchanged.
    - another open (non-terminal) recommendation already on the product, on any
      expiry: raise FuturesTradeRecommendationExclusivityError (ADR-018, P-1).

    A batch is all-or-nothing and returns the number of recommendations
    accepted. A durable implementation must check exclusivity and insert in one
    transaction, so that concurrent writers cannot both record an open
    recommendation for one product; an in-memory implementation cannot prove
    that.

    The abstract contract documents these obligations but cannot enforce them
    at runtime.
    """

    @abstractmethod
    def record(self, recommendations: tuple[FuturesTradeRecommendation, ...]) -> int:
        """Record a batch of recommendations and return the number accepted."""
