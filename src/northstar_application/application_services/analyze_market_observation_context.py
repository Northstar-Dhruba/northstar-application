"""Reusable provider-independent analysis of canonical market context."""

from __future__ import annotations

from northstar_core.strategy import (
    AssetAnalysisGenerator,
    ExplanationReason,
    MarketObservationContext,
    RecommendationExplanation,
    Strategy,
)

from northstar_application.application_services.analyze_asset_result import AnalyzeAssetResult


class AnalyzeMarketObservationContextService:
    """Apply existing analysis and recommendation logic to one factual context."""

    def __init__(
        self,
        strategy: Strategy,
        analysis_generator: AssetAnalysisGenerator,
    ) -> None:
        if strategy is None:
            raise TypeError("AnalyzeMarketObservationContextService strategy cannot be None.")
        if not isinstance(strategy, Strategy):
            raise TypeError("AnalyzeMarketObservationContextService strategy must be a Strategy.")
        if analysis_generator is None:
            raise TypeError(
                "AnalyzeMarketObservationContextService analysis_generator cannot be None."
            )
        if not isinstance(analysis_generator, AssetAnalysisGenerator):
            raise TypeError(
                "AnalyzeMarketObservationContextService analysis_generator "
                "must be an AssetAnalysisGenerator."
            )
        self._strategy = strategy
        self._analysis_generator = analysis_generator

    def execute(self, context: MarketObservationContext) -> AnalyzeAssetResult:
        if context is None:
            raise TypeError("AnalyzeMarketObservationContextService context cannot be None.")
        if not isinstance(context, MarketObservationContext):
            raise TypeError(
                "AnalyzeMarketObservationContextService context must be a MarketObservationContext."
            )

        asset_analysis = self._analysis_generator.generate(context)
        recommendation = self._strategy.evaluate(asset_analysis)
        explanation = RecommendationExplanation(
            recommendation=recommendation,
            reasons=(
                ExplanationReason(
                    rationale="Recommendation is supported by the analyzed market signals.",
                    supporting_signals=asset_analysis.summarized_signals,
                ),
            ),
        )
        return AnalyzeAssetResult(
            recommendation=recommendation,
            explanation=explanation,
            market_observation_context=context,
        )
