"""Application use case for producing one recommendation from one asset symbol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from northstar_core.domain.listing import Listing
from northstar_core.foundation.value_objects import PointInTime, Symbol
from northstar_core.strategy import AssetAnalysis, Recommendation, Strategy


@dataclass(frozen=True, slots=True)
class AssetAnalysisInput:
    """Market context needed to construct an AssetAnalysis for one asset."""

    listing: Listing
    point_in_time: PointInTime
    summarized_signals: tuple[str, ...]


class MarketObservationProvider(Protocol):
    """Provides single-asset market context to the Analyze Asset use case."""

    def get_analysis_input(self, symbol: Symbol) -> AssetAnalysisInput:
        """Return the current market context required to analyze one symbol."""


class AnalyzeAssetUseCase:
    """Coordinates one asset analysis without defining recommendation policy."""

    def __init__(self, strategy: Strategy, observation_provider: MarketObservationProvider) -> None:
        if strategy is None:
            raise TypeError("AnalyzeAssetUseCase strategy cannot be None.")
        if not isinstance(strategy, Strategy):
            raise TypeError("AnalyzeAssetUseCase strategy must be a Strategy instance.")
        if observation_provider is None:
            raise TypeError("AnalyzeAssetUseCase observation_provider cannot be None.")
        if not hasattr(observation_provider, "get_analysis_input"):
            raise TypeError(
                "AnalyzeAssetUseCase observation_provider must provide get_analysis_input(symbol)."
            )

        self._strategy = strategy
        self._observation_provider = observation_provider

    def execute(self, symbol: Symbol) -> Recommendation:
        """Analyze one symbol and return the Strategy-produced recommendation."""
        if symbol is None:
            raise TypeError("AnalyzeAssetUseCase symbol cannot be None.")
        if not isinstance(symbol, Symbol):
            raise TypeError("AnalyzeAssetUseCase symbol must be a Symbol value.")

        analysis_input = self._observation_provider.get_analysis_input(symbol)
        asset_analysis = AssetAnalysis(
            listing=analysis_input.listing,
            point_in_time=analysis_input.point_in_time,
            summarized_signals=analysis_input.summarized_signals,
        )
        return self._strategy.evaluate(asset_analysis)
