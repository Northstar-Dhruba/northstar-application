"""Result contract for asset analysis application services."""

from __future__ import annotations

from dataclasses import dataclass

from northstar_core.strategy import (
    MarketObservationContext,
    Recommendation,
    RecommendationExplanation,
)


@dataclass(frozen=True, slots=True)
class AnalyzeAssetResult:
    """Complete application outcome for analyzing one asset."""

    recommendation: Recommendation
    explanation: RecommendationExplanation
    market_observation_context: MarketObservationContext
