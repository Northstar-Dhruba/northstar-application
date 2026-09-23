"""Result contract for futures replay analysis.

The futures parallel of AnalyzeAssetResult, without the explanation: the
equity RecommendationExplanation accepts only an equity Recommendation, and the
reason it would carry is already the analysis's own summarized signals.

A result pairs a directional view with the observation it was decided on. The
observation is kept, rather than only the recommendation, because what a view
is later measured against starts from the evidence the decision saw -- above
all the latest quote -- and that fact is not part of the recommendation.

Neither the contract nor the decision instant is stored a second time. Both are
owned by the recommendation and by the context; the result only requires that
the two agree, so no consumer can read an identity or an instant from one view
that the other contradicts.
"""

from __future__ import annotations

from dataclasses import dataclass

from northstar_core.strategy import FuturesMarketObservationContext, FuturesRecommendation


@dataclass(frozen=True, slots=True)
class FuturesAnalysisResult:
    """A directional futures research view and the observation it was decided on.

    The recommendation is a directional research classification only. Nothing
    here is a quantity, a position, a margin, a multiplier or an order.
    """

    recommendation: FuturesRecommendation
    market_observation_context: FuturesMarketObservationContext

    def __post_init__(self) -> None:
        if not isinstance(self.recommendation, FuturesRecommendation):
            raise TypeError("FuturesAnalysisResult recommendation must be a FuturesRecommendation.")
        if not isinstance(self.market_observation_context, FuturesMarketObservationContext):
            raise TypeError(
                "FuturesAnalysisResult market observation context "
                "must be a FuturesMarketObservationContext."
            )

        context = self.market_observation_context
        if self.recommendation.contract != context.contract:
            raise ValueError(
                f"FuturesAnalysisResult recommendation for {self.recommendation.contract} "
                f"does not describe the observed contract {context.contract}."
            )
        # Compared semantically: two spellings of one instant are one instant.
        if self.recommendation.point_in_time.compare(context.observed_at) != 0:
            raise ValueError(
                f"FuturesAnalysisResult recommendation instant "
                f"{self.recommendation.point_in_time} must equal the observed instant "
                f"{context.observed_at}."
            )
