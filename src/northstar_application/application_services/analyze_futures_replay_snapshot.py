"""Application analysis of one look-ahead-safe futures replay snapshot.

Replay (ReplayFuturesHistoricalMarketDataUseCase) decides what evidence exists
at each instant; this service evaluates one snapshot it is handed. It reads
nothing but ``snapshot.observations`` -- no repository, no provider, no
calendar -- so the replay stays the single boundary through which evidence
enters research, and nothing this service does can see past the snapshot's own
replay instant.

Warm-up is absence, not failure
-------------------------------
The directional signal compares a five-observation average with a twenty-
observation one, so a snapshot holding fewer than twenty observations cannot be
analysed. Every replay begins with nineteen such snapshots, which makes them the
ordinary start of research rather than an error: the service returns None for
them, and the first analysis appears at the twentieth observation.

The arithmetic belongs to Core. This service only arranges facts: it builds the
observation context from the latest twenty bars, asks the generator for the
analysis and the strategy for the directional view, and performs no Decimal
arithmetic of its own.
"""

from __future__ import annotations

from northstar_core.foundation.value_objects import Timeframe
from northstar_core.futures import FuturesReplaySnapshot
from northstar_core.strategy import (
    FuturesAssetAnalysisGenerator,
    FuturesMarketObservationContext,
    Strategy,
)

from northstar_application.application_services.futures_analysis_result import (
    FuturesAnalysisResult,
)
from northstar_application.application_services.replay_futures_historical_market_data import (
    UnsupportedFuturesReplayTimeframeError,
)

_SUPPORTED_TIMEFRAME = Timeframe("1d")
_MINIMUM_HISTORY = 20


class AnalyzeFuturesReplaySnapshotService:
    """Evaluate one session-daily futures replay snapshot into a directional view."""

    def __init__(
        self,
        strategy: Strategy,
        analysis_generator: FuturesAssetAnalysisGenerator,
    ) -> None:
        if not isinstance(strategy, Strategy):
            raise TypeError("AnalyzeFuturesReplaySnapshotService strategy must be a Strategy.")
        if not isinstance(analysis_generator, FuturesAssetAnalysisGenerator):
            raise TypeError(
                "AnalyzeFuturesReplaySnapshotService analysis_generator "
                "must be a FuturesAssetAnalysisGenerator."
            )
        self._strategy = strategy
        self._analysis_generator = analysis_generator

    def execute(self, snapshot: FuturesReplaySnapshot) -> FuturesAnalysisResult | None:
        """Return the directional view at this snapshot, or None during warm-up."""
        if not isinstance(snapshot, FuturesReplaySnapshot):
            raise TypeError(
                "AnalyzeFuturesReplaySnapshotService snapshot must be a FuturesReplaySnapshot."
            )
        if snapshot.timeframe != _SUPPORTED_TIMEFRAME:
            raise UnsupportedFuturesReplayTimeframeError(
                f"AnalyzeFuturesReplaySnapshotService supports session-daily research only; "
                f"timeframe {snapshot.timeframe} is not {_SUPPORTED_TIMEFRAME}."
            )
        if len(snapshot.observations) < _MINIMUM_HISTORY:
            return None

        context = self._build_context(snapshot)
        analysis = self._analysis_generator.generate(context)
        recommendation = self._strategy.evaluate_futures(analysis)
        return FuturesAnalysisResult(
            recommendation=recommendation,
            market_observation_context=context,
        )

    @staticmethod
    def _build_context(snapshot: FuturesReplaySnapshot) -> FuturesMarketObservationContext:
        """Arrange the latest twenty observations as the decision-time context.

        Only the trailing window is carried, not the whole visible history: it
        is all the signal reads, and it keeps each context the same size however
        long the replay has run.
        """
        recent = snapshot.observations[-_MINIMUM_HISTORY:]
        latest = snapshot.observations[-1]
        return FuturesMarketObservationContext(
            contract=snapshot.contract,
            timeframe=snapshot.timeframe,
            observed_at=latest.point_in_time,
            latest_quote=latest.close,
            previous_close=snapshot.observations[-2].close,
            latest_volume=latest.volume,
            session_high=latest.high,
            session_low=latest.low,
            recent_closes=tuple(bar.close for bar in recent),
            recent_volumes=tuple(bar.volume for bar in recent),
        )
