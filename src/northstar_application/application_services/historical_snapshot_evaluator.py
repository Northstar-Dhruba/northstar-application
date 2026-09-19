"""Concrete historical replay evaluator using existing analysis logic."""

from __future__ import annotations

from northstar_core.domain.value_objects import ListingReference
from northstar_core.foundation.value_objects import Timeframe
from northstar_core.market_data import HistoricalReplaySnapshot
from northstar_core.strategy import AssetAnalysisGenerator, MarketObservationContext, Strategy

from northstar_application.application_services.analyze_asset import AnalyzeAssetResult
from northstar_application.application_services.analyze_market_observation_context import (
    AnalyzeMarketObservationContextService,
)
from northstar_application.ports import HistoricalSnapshotEvaluator


class HistoricalSnapshotEvaluatorService(HistoricalSnapshotEvaluator):
    """Evaluate daily historical snapshots without live market-data acquisition.

    Analysis identity is derived from the legally visible observations
    themselves. No reference-data resolution, provider lookup, or current-state
    acquisition participates in historical evaluation.
    """

    _SUPPORTED_TIMEFRAME = Timeframe("1d")
    _REQUIRED_HISTORY_LENGTH = 20

    def __init__(
        self,
        strategy: Strategy,
        analysis_generator: AssetAnalysisGenerator,
    ) -> None:
        self._context_analyzer = AnalyzeMarketObservationContextService(
            strategy=strategy,
            analysis_generator=analysis_generator,
        )

    def evaluate(self, snapshot: HistoricalReplaySnapshot) -> AnalyzeAssetResult:
        if snapshot is None:
            raise TypeError("HistoricalSnapshotEvaluatorService snapshot cannot be None.")
        if not isinstance(snapshot, HistoricalReplaySnapshot):
            raise TypeError(
                "HistoricalSnapshotEvaluatorService snapshot must be a HistoricalReplaySnapshot."
            )

        bars = snapshot.observations
        if not bars:
            raise ValueError(
                "HistoricalSnapshotEvaluatorService cannot evaluate an empty snapshot."
            )
        if any(bar.timeframe != self._SUPPORTED_TIMEFRAME for bar in bars):
            raise ValueError("HistoricalSnapshotEvaluatorService supports only daily bars.")
        if len(bars) < self._REQUIRED_HISTORY_LENGTH:
            raise ValueError("HistoricalSnapshotEvaluatorService requires at least 20 daily bars.")
        if any(bar.point_in_time.compare(snapshot.replay_instant) > 0 for bar in bars):
            raise ValueError(
                "Historical snapshot contains an observation after its replay instant."
            )

        first_bar = bars[0]
        if any(
            bar.symbol != first_bar.symbol or bar.exchange_code != first_bar.exchange_code
            for bar in bars
        ):
            raise ValueError("Historical snapshot contains incompatible market identities.")

        listing_reference = ListingReference(
            symbol=first_bar.symbol,
            exchange_code=first_bar.exchange_code,
        )
        latest_bar = bars[-1]
        previous_bar = bars[-2]
        recent_bars = bars[-self._REQUIRED_HISTORY_LENGTH :]
        context = MarketObservationContext(
            listing_reference=listing_reference,
            observed_at=latest_bar.point_in_time,
            latest_price=latest_bar.close,
            previous_close=previous_bar.close,
            latest_volume=latest_bar.volume,
            daily_high=latest_bar.high,
            daily_low=latest_bar.low,
            recent_closes=tuple(bar.close for bar in recent_bars),
            recent_volumes=tuple(bar.volume for bar in recent_bars),
        )
        return self._context_analyzer.execute(context)
