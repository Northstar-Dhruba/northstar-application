"""Application use case for producing one recommendation from one asset symbol."""

from __future__ import annotations

from northstar_core.foundation.value_objects import Symbol
from northstar_core.strategy import (
    AssetAnalysisGenerator,
    Strategy,
)

from northstar_application.application_services.analyze_asset_result import AnalyzeAssetResult
from northstar_application.application_services.analyze_market_observation_context import (
    AnalyzeMarketObservationContextService,
)
from northstar_application.ports import MarketObservationSource


class AnalyzeAssetUseCase:
    """Coordinates one asset analysis without defining recommendation policy."""

    def __init__(
        self,
        strategy: Strategy,
        observation_source: MarketObservationSource,
        analysis_generator: AssetAnalysisGenerator,
    ) -> None:
        if strategy is None:
            raise TypeError("AnalyzeAssetUseCase strategy cannot be None.")
        if not isinstance(strategy, Strategy):
            raise TypeError("AnalyzeAssetUseCase strategy must be a Strategy instance.")
        if observation_source is None:
            raise TypeError("AnalyzeAssetUseCase observation_source cannot be None.")
        if not isinstance(observation_source, MarketObservationSource):
            raise TypeError(
                "AnalyzeAssetUseCase observation_source must be a MarketObservationSource."
            )
        if analysis_generator is None:
            raise TypeError("AnalyzeAssetUseCase analysis_generator cannot be None.")
        if not isinstance(analysis_generator, AssetAnalysisGenerator):
            raise TypeError(
                "AnalyzeAssetUseCase analysis_generator must be an AssetAnalysisGenerator."
            )

        self._observation_source = observation_source
        self._context_analyzer = AnalyzeMarketObservationContextService(
            strategy=strategy,
            analysis_generator=analysis_generator,
        )

    def execute(self, symbol: Symbol) -> AnalyzeAssetResult:
        """Analyze one symbol and return its recommendation with an explanation."""
        if symbol is None:
            raise TypeError("AnalyzeAssetUseCase symbol cannot be None.")
        if not isinstance(symbol, Symbol):
            raise TypeError("AnalyzeAssetUseCase symbol must be a Symbol value.")

        observation_context = self._observation_source.get_observation_context(symbol)
        return self._context_analyzer.execute(observation_context)
