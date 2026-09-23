"""Application orchestration for measuring one futures recommendation outcome.

Measurement records what the market did after a directional research view. It
never interprets BUY, SELL or HOLD and models no order, position, multiplier,
margin or profit and loss.

This use case selects evidence and nothing more. Every judgement about that
evidence -- whether a percentage return is defined, which unavailable reason
applies, and the arithmetic itself -- belongs to FuturesRecommendationOutcome,
so the use case hands Core the facts it found and returns the value Core
builds. There is no Application measurement wrapper: the Core outcome already
represents every state, including a horizon not yet reached.

Decision evidence is frozen
---------------------------
The decision quote is the analysis result's own ``latest_quote``: the evidence
that actually produced the recommendation. The inclusive query may return a
stored bar at the decision instant, and that bar is never used to replace the
quote. Measurement answers "what happened after this decision", which only
makes sense against the evidence the decision saw.

Only strictly later bars count
------------------------------
The query's start bound is inclusive, so the decision bar itself may come back.
It is valid repository output, and it is not a future observation. A horizon of
N counts the first N stored bars strictly after the decision instant, compared
with PointInTime.compare(). Bars are counted, not calendar days: a session with
no bar is not an observation, so sparse history needs no calendar.

The evidence window is the caller's
-----------------------------------
``available_through`` is the last instant whose evidence the measurement may
use, and the query ends there inclusively. Without it an outcome would depend
on how much history happened to be stored when it was measured: an
insufficient outcome would silently become a measured one after the next
acquisition. With it, the same result, horizon and window always measure the
same outcome, and extending the window is an explicit act. The window is never
derived from a clock.

Session-daily research only
---------------------------
Like the replay and the snapshot analysis, measurement serves session-daily
research and refuses any other timeframe before reading the repository: a
horizon counted in minute bars is not the horizon this research defines.
"""

from __future__ import annotations

from northstar_core.foundation.value_objects import PointInTime, Timeframe
from northstar_core.strategy import FuturesRecommendationOutcome, ResearchHorizon

from northstar_application.application_services._futures_repository_output import (
    validate_futures_repository_bars,
)
from northstar_application.application_services.futures_analysis_result import (
    FuturesAnalysisResult,
)
from northstar_application.application_services.replay_futures_historical_market_data import (
    UnsupportedFuturesReplayTimeframeError,
)
from northstar_application.ports import (
    FuturesHistoricalMarketDataQuery,
    FuturesHistoricalMarketDataRepository,
)

_SUPPORTED_TIMEFRAME = Timeframe("1d")


class MeasureFuturesRecommendationOutcomeUseCase:
    """Measure one futures analysis result at one research horizon."""

    def __init__(self, repository: FuturesHistoricalMarketDataRepository) -> None:
        if not isinstance(repository, FuturesHistoricalMarketDataRepository):
            raise TypeError(
                "MeasureFuturesRecommendationOutcomeUseCase repository "
                "must be a FuturesHistoricalMarketDataRepository."
            )
        self._repository = repository

    def execute(
        self,
        result: FuturesAnalysisResult,
        horizon: ResearchHorizon,
        available_through: PointInTime,
    ) -> FuturesRecommendationOutcome:
        """Return the outcome of ``result`` after ``horizon`` observations.

        Only evidence up to and including ``available_through`` is read. A
        window ending at the decision instant is valid and can hold no future
        observation, so it measures as insufficient.
        """
        if not isinstance(result, FuturesAnalysisResult):
            raise TypeError(
                "MeasureFuturesRecommendationOutcomeUseCase result must be a FuturesAnalysisResult."
            )
        if not isinstance(horizon, ResearchHorizon):
            raise TypeError(
                "MeasureFuturesRecommendationOutcomeUseCase horizon must be a ResearchHorizon."
            )
        if not isinstance(available_through, PointInTime):
            raise TypeError(
                "MeasureFuturesRecommendationOutcomeUseCase available-through "
                "must be a PointInTime."
            )

        context = result.market_observation_context
        recommendation = result.recommendation
        decision_instant = recommendation.point_in_time

        if context.timeframe != _SUPPORTED_TIMEFRAME:
            raise UnsupportedFuturesReplayTimeframeError(
                f"MeasureFuturesRecommendationOutcomeUseCase supports session-daily research "
                f"only; timeframe {context.timeframe} is not {_SUPPORTED_TIMEFRAME}."
            )
        if available_through.compare(decision_instant) < 0:
            raise ValueError(
                f"MeasureFuturesRecommendationOutcomeUseCase available-through "
                f"{available_through} precedes the decision instant {decision_instant}."
            )

        decision_quote = context.latest_quote
        query = FuturesHistoricalMarketDataQuery(
            contract=recommendation.contract,
            timeframe=context.timeframe,
            start=decision_instant,
            end=available_through,
        )
        bars = validate_futures_repository_bars(self._repository.get_bars(query), query)

        future = tuple(bar for bar in bars if bar.point_in_time.compare(decision_instant) > 0)
        if len(future) < horizon.observations:
            return FuturesRecommendationOutcome(
                recommendation=recommendation,
                horizon=horizon,
                decision_quote=decision_quote,
                evaluation_instant=None,
                evaluation_quote=None,
            )

        selected = future[horizon.observations - 1]
        return FuturesRecommendationOutcome(
            recommendation=recommendation,
            horizon=horizon,
            decision_quote=decision_quote,
            evaluation_instant=selected.point_in_time,
            evaluation_quote=selected.close,
        )
