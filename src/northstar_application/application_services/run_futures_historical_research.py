"""Application orchestration for one reproducible futures historical research run.

A run replays one concrete contract's persisted session-daily history, analyses
every snapshot that has enough history, and measures each resulting decision at
every requested horizon. It records facts only: it never interprets BUY, SELL or
HOLD as a position, a profit or an order, and it aggregates nothing.

The evidence cutoff is the run's
--------------------------------
``available_through`` bounds everything the run reads. The replay query ends
there, so no decision is taken on evidence after it, and every outcome is
measured against that same instant, so no horizon is resolved by evidence after
it either. It is supplied by the caller and stored on the run; it is never read
from a clock or from the latest stored bar. Two runs with equal inputs over the
same stored history are therefore equal, and storing more history after the
cutoff changes nothing until a caller deliberately asks for a later one.

One evidence source
-------------------
The use case is built from one repository and composes replay, analysis and
measurement over it. Injecting the replay and the measurement separately would
allow a run whose decisions and outcomes were read from different stores, which
would make "the same run" an ill-defined idea.

Why the run stores its subject
------------------------------
The equity run derives its listing and strategy from its evaluations. A futures
run cannot: a contract whose history is shorter than the warm-up produces no
decision at all, and that run must still say which contract, strategy and
cutoff it describes. So contract, strategy identity, timeframe and cutoff are
fields, and every stored decision and outcome is checked against them.
"""

from __future__ import annotations

from dataclasses import dataclass

from northstar_core.foundation.value_objects import PointInTime, Timeframe
from northstar_core.futures import FuturesContract
from northstar_core.strategy import (
    FuturesAssetAnalysisGenerator,
    FuturesRecommendationOutcome,
    ResearchHorizon,
    Strategy,
    StrategyIdentity,
)

from northstar_application.application_services.analyze_futures_replay_snapshot import (
    AnalyzeFuturesReplaySnapshotService,
)
from northstar_application.application_services.futures_analysis_result import (
    FuturesAnalysisResult,
)
from northstar_application.application_services.measure_futures_recommendation_outcome import (
    MeasureFuturesRecommendationOutcomeUseCase,
)
from northstar_application.application_services.replay_futures_historical_market_data import (
    ReplayFuturesHistoricalMarketDataUseCase,
    UnsupportedFuturesReplayTimeframeError,
)
from northstar_application.ports import (
    FuturesHistoricalMarketDataQuery,
    FuturesHistoricalMarketDataRepository,
)

_SUPPORTED_TIMEFRAME = Timeframe("1d")


def _validate_horizons(horizons: object, subject: str) -> None:
    """Require a non-empty tuple of distinct horizons, kept in the caller's order.

    Duplicates are refused rather than collapsed: two equal horizons would
    produce two identical outcome facts for one decision. Order is not
    canonicalised, matching the equity run; the caller's order is the order in
    which each decision's outcomes appear.
    """
    if not isinstance(horizons, tuple):
        raise TypeError(f"{subject} horizons must be a tuple.")
    if not horizons:
        raise ValueError(f"{subject} requires at least one horizon.")
    if not all(isinstance(horizon, ResearchHorizon) for horizon in horizons):
        raise TypeError(f"{subject} horizons must contain ResearchHorizon values.")
    if len(set(horizons)) != len(horizons):
        raise ValueError(f"{subject} horizons cannot contain duplicates.")


def _require_daily(timeframe: Timeframe, subject: str) -> None:
    if timeframe != _SUPPORTED_TIMEFRAME:
        raise UnsupportedFuturesReplayTimeframeError(
            f"{subject} supports session-daily research only; "
            f"timeframe {timeframe} is not {_SUPPORTED_TIMEFRAME}."
        )


@dataclass(frozen=True, slots=True)
class FuturesHistoricalResearchRun:
    """Complete factual record of one futures historical research run.

    ``analysis_results`` holds one result per analysable replay snapshot, oldest
    decision first. Warm-up snapshots produced none, so a run over fewer than
    twenty stored bars has no results and no outcomes, and is still a valid
    record of having looked.

    ``outcomes`` holds one outcome for every result and horizon pair, ordered
    by result first and horizon second, horizons in the run's own order. An
    unmeasurable pair is present as an outcome carrying its unavailable reason,
    so no decision is silently dropped from the run.
    """

    contract: FuturesContract
    timeframe: Timeframe
    strategy_identity: StrategyIdentity
    available_through: PointInTime
    horizons: tuple[ResearchHorizon, ...]
    analysis_results: tuple[FuturesAnalysisResult, ...]
    outcomes: tuple[FuturesRecommendationOutcome, ...]

    def __post_init__(self) -> None:
        subject = "FuturesHistoricalResearchRun"
        if not isinstance(self.contract, FuturesContract):
            raise TypeError(f"{subject} contract must be a FuturesContract.")
        if not isinstance(self.timeframe, Timeframe):
            raise TypeError(f"{subject} timeframe must be a Timeframe.")
        if not isinstance(self.strategy_identity, StrategyIdentity):
            raise TypeError(f"{subject} strategy identity must be a StrategyIdentity.")
        if not isinstance(self.available_through, PointInTime):
            raise TypeError(f"{subject} available-through must be a PointInTime.")
        _require_daily(self.timeframe, subject)
        _validate_horizons(self.horizons, subject)
        if not isinstance(self.analysis_results, tuple):
            raise TypeError(f"{subject} analysis results must be a tuple.")
        if not isinstance(self.outcomes, tuple):
            raise TypeError(f"{subject} outcomes must be a tuple.")

        self._validate_analysis_results()
        self._validate_outcomes()

    def _validate_analysis_results(self) -> None:
        previous: PointInTime | None = None
        for result in self.analysis_results:
            if not isinstance(result, FuturesAnalysisResult):
                raise TypeError(
                    "FuturesHistoricalResearchRun analysis results must contain "
                    "FuturesAnalysisResult values."
                )
            recommendation = result.recommendation
            decision = recommendation.point_in_time
            if recommendation.contract != self.contract:
                raise ValueError(
                    f"FuturesHistoricalResearchRun analysis result for {recommendation.contract} "
                    f"does not belong to the run contract {self.contract}."
                )
            if recommendation.strategy_identity != self.strategy_identity:
                raise ValueError(
                    f"FuturesHistoricalResearchRun analysis result by "
                    f"{recommendation.strategy_identity} does not belong to the run strategy "
                    f"{self.strategy_identity}."
                )
            if result.market_observation_context.timeframe != self.timeframe:
                raise ValueError(
                    "FuturesHistoricalResearchRun analysis result timeframe must match "
                    "the run timeframe."
                )
            if decision.compare(self.available_through) > 0:
                raise ValueError(
                    f"FuturesHistoricalResearchRun decision at {decision} is after the run "
                    f"cutoff {self.available_through}."
                )
            if previous is not None and previous.compare(decision) >= 0:
                raise ValueError(
                    "FuturesHistoricalResearchRun analysis results must be strictly "
                    f"chronological; {decision} does not follow {previous}."
                )
            previous = decision

    def _validate_outcomes(self) -> None:
        if len(self.outcomes) != len(self.analysis_results) * len(self.horizons):
            raise ValueError(
                "FuturesHistoricalResearchRun must hold one outcome for every analysis "
                "result and horizon pair."
            )

        pairs = ((result, horizon) for result in self.analysis_results for horizon in self.horizons)
        for outcome, (result, horizon) in zip(self.outcomes, pairs, strict=True):
            if not isinstance(outcome, FuturesRecommendationOutcome):
                raise TypeError(
                    "FuturesHistoricalResearchRun outcomes must contain "
                    "FuturesRecommendationOutcome values."
                )
            if outcome.recommendation != result.recommendation:
                raise ValueError(
                    "FuturesHistoricalResearchRun outcomes must follow result order, "
                    "measuring each result's own recommendation."
                )
            if outcome.horizon != horizon:
                raise ValueError(
                    "FuturesHistoricalResearchRun outcomes must follow the run horizon "
                    "order within each result."
                )
            if outcome.decision_quote != result.market_observation_context.latest_quote:
                raise ValueError(
                    "FuturesHistoricalResearchRun outcome decision quote must be the "
                    "decision evidence of its analysis result."
                )
            evaluation = outcome.evaluation_instant
            if evaluation is not None and evaluation.compare(self.available_through) > 0:
                raise ValueError(
                    f"FuturesHistoricalResearchRun outcome evaluated at {evaluation} is after "
                    f"the run cutoff {self.available_through}."
                )


class RunFuturesHistoricalResearchUseCase:
    """Replay, analyse and measure one futures contract up to one evidence cutoff."""

    def __init__(
        self,
        repository: FuturesHistoricalMarketDataRepository,
        analysis_generator: FuturesAssetAnalysisGenerator,
    ) -> None:
        if not isinstance(repository, FuturesHistoricalMarketDataRepository):
            raise TypeError(
                "RunFuturesHistoricalResearchUseCase repository "
                "must be a FuturesHistoricalMarketDataRepository."
            )
        if not isinstance(analysis_generator, FuturesAssetAnalysisGenerator):
            raise TypeError(
                "RunFuturesHistoricalResearchUseCase analysis_generator "
                "must be a FuturesAssetAnalysisGenerator."
            )
        self._replay = ReplayFuturesHistoricalMarketDataUseCase(repository)
        self._measure = MeasureFuturesRecommendationOutcomeUseCase(repository)
        self._analysis_generator = analysis_generator

    def execute(
        self,
        contract: FuturesContract,
        timeframe: Timeframe,
        strategy: Strategy,
        horizons: tuple[ResearchHorizon, ...],
        available_through: PointInTime,
    ) -> FuturesHistoricalResearchRun:
        """Return the run for ``contract`` using evidence up to ``available_through``."""
        subject = "RunFuturesHistoricalResearchUseCase"
        if not isinstance(contract, FuturesContract):
            raise TypeError(f"{subject} contract must be a FuturesContract.")
        if not isinstance(timeframe, Timeframe):
            raise TypeError(f"{subject} timeframe must be a Timeframe.")
        if not isinstance(strategy, Strategy):
            raise TypeError(f"{subject} strategy must be a Strategy.")
        if not isinstance(available_through, PointInTime):
            raise TypeError(f"{subject} available-through must be a PointInTime.")
        _require_daily(timeframe, subject)
        _validate_horizons(horizons, subject)

        snapshots = self._replay.execute(
            FuturesHistoricalMarketDataQuery(
                contract=contract,
                timeframe=timeframe,
                start=None,
                end=available_through,
            )
        )

        analyzer = AnalyzeFuturesReplaySnapshotService(strategy, self._analysis_generator)
        analysis_results = tuple(
            result
            for result in (analyzer.execute(snapshot) for snapshot in snapshots)
            if result is not None
        )

        outcomes = tuple(
            self._measure.execute(result, horizon, available_through)
            for result in analysis_results
            for horizon in horizons
        )

        return FuturesHistoricalResearchRun(
            contract=contract,
            timeframe=timeframe,
            strategy_identity=strategy.strategy_identity,
            available_through=available_through,
            horizons=horizons,
            analysis_results=analysis_results,
            outcomes=outcomes,
        )
