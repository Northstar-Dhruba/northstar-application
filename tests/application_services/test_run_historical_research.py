"""Tests for historical research run orchestration."""

from __future__ import annotations

import pytest
from northstar_core.domain.value_objects import ListingReference
from northstar_core.foundation.value_objects import (
    Currency,
    ExchangeCode,
    Percentage,
    PointInTime,
    Price,
    Quantity,
    Symbol,
    Timeframe,
)
from northstar_core.market_data import HistoricalOHLCVBar
from northstar_core.strategy import (
    AssetAnalysisGenerator,
    MarketObservationContext,
    ResearchHorizon,
    Strategy,
    StrategyIdentity,
)

from northstar_application.application_services import (
    AnalyzeMarketObservationContextService,
    HistoricalResearchEvaluation,
    HistoricalResearchRun,
    MeasureRecommendationOutcomeUseCase,
    RecommendationOutcomeMeasurement,
    RecommendationOutcomeUnavailableReason,
    RunHistoricalResearchUseCase,
)
from northstar_application.ports import (
    HistoricalMarketDataQuery,
    HistoricalMarketDataRepository,
)

_USD = Currency("USD")
_DAILY = Timeframe("1d")
_SYMBOL = Symbol("AAPL")
_EXCHANGE = ExchangeCode("NASDAQ")
_AVAILABLE_THROUGH = PointInTime("2026-03-01T16:00:00Z")
_INSUFFICIENT = RecommendationOutcomeUnavailableReason.INSUFFICIENT_FUTURE_OBSERVATIONS


def _instant(day: int) -> PointInTime:
    return PointInTime(f"2026-01-{day:02d}T16:00:00Z")


def _bar(day: int, close: str, *, currency: Currency = _USD) -> HistoricalOHLCVBar:
    return HistoricalOHLCVBar(
        _SYMBOL,
        _EXCHANGE,
        _instant(day),
        _DAILY,
        Price("50", currency),
        Price("500", currency),
        Price("1", currency),
        Price(close, currency),
        Quantity("1000"),
    )


class StubRepository(HistoricalMarketDataRepository):
    """Returns preconfigured observations and records every query received."""

    def __init__(self, observations: tuple[HistoricalOHLCVBar, ...] = ()) -> None:
        self.observations = observations
        self.queries: list[HistoricalMarketDataQuery] = []

    def get_history(self, query: HistoricalMarketDataQuery) -> tuple[HistoricalOHLCVBar, ...]:
        self.queries.append(query)
        return tuple(
            bar
            for bar in self.observations
            if query.start.compare(bar.point_in_time) <= 0
            and query.end.compare(bar.point_in_time) >= 0
        )


def _evaluation(day: int, *, latest_close: str = "100") -> HistoricalResearchEvaluation:
    context = MarketObservationContext(
        ListingReference(_SYMBOL, _EXCHANGE),
        _instant(day),
        Price(latest_close, _USD),
        Price("99", _USD),
        Quantity("1000"),
        Price("101", _USD),
        Price("98", _USD),
        tuple(Price("100", _USD) for _ in range(20)),
        tuple(Quantity("1000") for _ in range(20)),
    )
    result = AnalyzeMarketObservationContextService(
        strategy=Strategy(StrategyIdentity("run-test")),
        analysis_generator=AssetAnalysisGenerator(),
    ).execute(context)
    return HistoricalResearchEvaluation(replay_instant=_instant(day), result=result)


class RecordingMeasureUseCase(MeasureRecommendationOutcomeUseCase):
    """Wraps the real measurement use case and records every delegated call."""

    def __init__(self, repository: HistoricalMarketDataRepository) -> None:
        super().__init__(repository)
        self.calls: list[tuple[PointInTime, ResearchHorizon, Timeframe, PointInTime]] = []

    def execute(
        self,
        evaluation: HistoricalResearchEvaluation,
        horizon: ResearchHorizon,
        timeframe: Timeframe,
        available_through: PointInTime,
    ) -> RecommendationOutcomeMeasurement:
        self.calls.append((evaluation.replay_instant, horizon, timeframe, available_through))
        return super().execute(evaluation, horizon, timeframe, available_through)


def _use_case(
    observations: tuple[HistoricalOHLCVBar, ...] = (),
) -> RunHistoricalResearchUseCase:
    return RunHistoricalResearchUseCase(
        MeasureRecommendationOutcomeUseCase(StubRepository(observations))
    )


def _series() -> tuple[HistoricalOHLCVBar, ...]:
    return tuple(_bar(20 + offset, str(100 + offset)) for offset in range(6))


def _measurements_for(
    evaluations: tuple[HistoricalResearchEvaluation, ...],
    horizons: tuple[ResearchHorizon, ...],
) -> tuple[RecommendationOutcomeMeasurement, ...]:
    """Build the coherent measurement tuple the use case would produce."""
    measure = MeasureRecommendationOutcomeUseCase(StubRepository(_series()))
    return tuple(
        measure.execute(evaluation, horizon, _DAILY, _AVAILABLE_THROUGH)
        for evaluation in evaluations
        for horizon in horizons
    )


# ---------------------------------------------------------------------------
# Dependency validation
# ---------------------------------------------------------------------------


def test_use_case_requires_a_measurement_use_case() -> None:
    with pytest.raises(TypeError, match="cannot be None"):
        RunHistoricalResearchUseCase(None)

    with pytest.raises(TypeError, match="must be a MeasureRecommendationOutcomeUseCase"):
        RunHistoricalResearchUseCase("measure")


def test_execute_validates_timeframe_and_available_through() -> None:
    use_case = _use_case()

    with pytest.raises(TypeError, match="timeframe must be a Timeframe"):
        use_case.execute((), (ResearchHorizon(1),), "1d", _AVAILABLE_THROUGH)
    with pytest.raises(TypeError, match="available-through must be a PointInTime"):
        use_case.execute((), (ResearchHorizon(1),), _DAILY, "2026-03-01T16:00:00Z")


# ---------------------------------------------------------------------------
# Empty input
# ---------------------------------------------------------------------------


def test_empty_evaluations_return_an_empty_run() -> None:
    run = _use_case().execute((), (ResearchHorizon(1),), _DAILY, _AVAILABLE_THROUGH)

    assert isinstance(run, HistoricalResearchRun)
    assert run.evaluations == ()
    assert run.horizons == (ResearchHorizon(1),)
    assert run.measurements == ()


def test_empty_evaluations_do_not_query_the_repository() -> None:
    repository = StubRepository(_series())
    use_case = RunHistoricalResearchUseCase(MeasureRecommendationOutcomeUseCase(repository))

    use_case.execute((), (ResearchHorizon(1),), _DAILY, _AVAILABLE_THROUGH)

    assert repository.queries == []


# ---------------------------------------------------------------------------
# Single and multiple pairs
# ---------------------------------------------------------------------------


def test_one_evaluation_and_one_horizon_produces_one_measurement() -> None:
    run = _use_case(_series()).execute(
        (_evaluation(20),), (ResearchHorizon(1),), _DAILY, _AVAILABLE_THROUGH
    )

    assert len(run.measurements) == 1
    measurement = run.measurements[0]
    assert measurement.is_measured
    assert measurement.horizon == ResearchHorizon(1)
    assert measurement.outcome.evaluation_instant == _instant(21)
    assert measurement.outcome.forward_return == Percentage(1)


def test_multiple_evaluations_and_horizons_produce_the_cartesian_product() -> None:
    evaluations = (_evaluation(20), _evaluation(21), _evaluation(22))
    horizons = (ResearchHorizon(1), ResearchHorizon(2))

    run = _use_case(_series()).execute(evaluations, horizons, _DAILY, _AVAILABLE_THROUGH)

    assert len(run.measurements) == len(evaluations) * len(horizons) == 6
    assert run.evaluations == evaluations
    assert run.horizons == horizons


# ---------------------------------------------------------------------------
# Deterministic ordering
# ---------------------------------------------------------------------------


def test_measurements_are_ordered_by_evaluation_then_horizon() -> None:
    evaluations = (_evaluation(20), _evaluation(21))
    horizons = (ResearchHorizon(1), ResearchHorizon(2))

    run = _use_case(_series()).execute(evaluations, horizons, _DAILY, _AVAILABLE_THROUGH)

    assert [
        (measurement.recommendation.point_in_time, measurement.horizon)
        for measurement in run.measurements
    ] == [
        (_instant(20), ResearchHorizon(1)),
        (_instant(20), ResearchHorizon(2)),
        (_instant(21), ResearchHorizon(1)),
        (_instant(21), ResearchHorizon(2)),
    ]


def test_horizon_order_follows_the_requested_order_and_is_not_sorted() -> None:
    horizons = (ResearchHorizon(3), ResearchHorizon(1), ResearchHorizon(2))

    run = _use_case(_series()).execute((_evaluation(20),), horizons, _DAILY, _AVAILABLE_THROUGH)

    assert run.horizons == horizons
    assert [measurement.horizon for measurement in run.measurements] == list(horizons)


def test_evaluation_replay_order_is_preserved_in_the_run() -> None:
    evaluations = (_evaluation(20), _evaluation(21), _evaluation(22))

    run = _use_case(_series()).execute(
        evaluations, (ResearchHorizon(1),), _DAILY, _AVAILABLE_THROUGH
    )

    assert [measurement.recommendation.point_in_time for measurement in run.measurements] == [
        _instant(20),
        _instant(21),
        _instant(22),
    ]


# ---------------------------------------------------------------------------
# Unavailable measurements are retained
# ---------------------------------------------------------------------------


def test_unavailable_measurements_are_retained_not_dropped() -> None:
    evaluations = (_evaluation(20), _evaluation(25))
    horizons = (ResearchHorizon(1),)

    run = _use_case(_series()).execute(evaluations, horizons, _DAILY, _AVAILABLE_THROUGH)

    assert len(run.measurements) == 2
    assert run.measurements[0].is_measured
    assert run.measurements[1].outcome is None
    assert run.measurements[1].unavailable is _INSUFFICIENT


def test_mixed_measured_and_unavailable_results_keep_their_positions() -> None:
    evaluations = (_evaluation(20),)
    horizons = (ResearchHorizon(1), ResearchHorizon(20))

    run = _use_case(_series()).execute(evaluations, horizons, _DAILY, _AVAILABLE_THROUGH)

    assert run.measurements[0].is_measured
    assert run.measurements[1].unavailable is _INSUFFICIENT


def test_every_evaluation_and_horizon_pair_is_represented() -> None:
    evaluations = (_evaluation(20), _evaluation(24))
    horizons = (ResearchHorizon(1), ResearchHorizon(5))

    run = _use_case(_series()).execute(evaluations, horizons, _DAILY, _AVAILABLE_THROUGH)

    assert len(run.measurements) == 4
    assert all(
        measurement.is_measured or measurement.unavailable is not None
        for measurement in run.measurements
    )


# ---------------------------------------------------------------------------
# Input rejection
# ---------------------------------------------------------------------------


def test_rejects_duplicate_horizons() -> None:
    with pytest.raises(ValueError, match="cannot contain duplicates"):
        _use_case().execute(
            (), (ResearchHorizon(1), ResearchHorizon(1)), _DAILY, _AVAILABLE_THROUGH
        )


def test_rejects_empty_horizons() -> None:
    with pytest.raises(ValueError, match="at least one horizon"):
        _use_case().execute((), (), _DAILY, _AVAILABLE_THROUGH)


def test_rejects_none_horizons() -> None:
    with pytest.raises(TypeError, match="horizons cannot be None"):
        _use_case().execute((), None, _DAILY, _AVAILABLE_THROUGH)


def test_rejects_non_tuple_horizons() -> None:
    with pytest.raises(TypeError, match="horizons must be a tuple"):
        _use_case().execute((), [ResearchHorizon(1)], _DAILY, _AVAILABLE_THROUGH)


def test_rejects_wrong_horizon_element_type() -> None:
    with pytest.raises(TypeError, match="must contain ResearchHorizon values"):
        _use_case().execute((), (1,), _DAILY, _AVAILABLE_THROUGH)


def test_rejects_unordered_evaluations() -> None:
    evaluations = (_evaluation(22), _evaluation(20))

    with pytest.raises(ValueError, match="chronologically ordered"):
        _use_case(_series()).execute(evaluations, (ResearchHorizon(1),), _DAILY, _AVAILABLE_THROUGH)


def test_rejects_evaluations_repeating_the_same_replay_instant() -> None:
    evaluations = (_evaluation(20), _evaluation(20))

    with pytest.raises(ValueError, match="chronologically ordered"):
        _use_case(_series()).execute(evaluations, (ResearchHorizon(1),), _DAILY, _AVAILABLE_THROUGH)


def test_rejects_none_evaluations() -> None:
    with pytest.raises(TypeError, match="evaluations cannot be None"):
        _use_case().execute(None, (ResearchHorizon(1),), _DAILY, _AVAILABLE_THROUGH)


def test_rejects_non_tuple_evaluations() -> None:
    with pytest.raises(TypeError, match="evaluations must be a tuple"):
        _use_case().execute([_evaluation(20)], (ResearchHorizon(1),), _DAILY, _AVAILABLE_THROUGH)


def test_rejects_wrong_evaluation_element_type() -> None:
    with pytest.raises(TypeError, match="must contain HistoricalResearchEvaluation values"):
        _use_case().execute(("evaluation",), (ResearchHorizon(1),), _DAILY, _AVAILABLE_THROUGH)


# ---------------------------------------------------------------------------
# Delegation to the measurement use case
# ---------------------------------------------------------------------------


def test_measurement_is_delegated_once_per_evaluation_and_horizon_pair() -> None:
    recording = RecordingMeasureUseCase(StubRepository(_series()))
    evaluations = (_evaluation(20), _evaluation(21))
    horizons = (ResearchHorizon(1), ResearchHorizon(2))

    RunHistoricalResearchUseCase(recording).execute(
        evaluations, horizons, _DAILY, _AVAILABLE_THROUGH
    )

    assert recording.calls == [
        (_instant(20), ResearchHorizon(1), _DAILY, _AVAILABLE_THROUGH),
        (_instant(20), ResearchHorizon(2), _DAILY, _AVAILABLE_THROUGH),
        (_instant(21), ResearchHorizon(1), _DAILY, _AVAILABLE_THROUGH),
        (_instant(21), ResearchHorizon(2), _DAILY, _AVAILABLE_THROUGH),
    ]


def test_run_results_match_direct_measurement_results() -> None:
    repository = StubRepository(_series())
    measure = MeasureRecommendationOutcomeUseCase(repository)
    evaluation = _evaluation(20)
    horizon = ResearchHorizon(2)

    direct = measure.execute(evaluation, horizon, _DAILY, _AVAILABLE_THROUGH)
    run = RunHistoricalResearchUseCase(measure).execute(
        (evaluation,), (horizon,), _DAILY, _AVAILABLE_THROUGH
    )

    assert run.measurements[0] == direct


def test_timeframe_and_available_through_are_passed_through_unchanged() -> None:
    recording = RecordingMeasureUseCase(StubRepository(_series()))
    available_through = PointInTime("2026-02-10T16:00:00Z")

    RunHistoricalResearchUseCase(recording).execute(
        (_evaluation(20),), (ResearchHorizon(1),), _DAILY, available_through
    )

    assert recording.calls[0][2] == _DAILY
    assert recording.calls[0][3] == available_through


# ---------------------------------------------------------------------------
# Determinism and result semantics
# ---------------------------------------------------------------------------


def test_repeated_runs_are_equal() -> None:
    evaluations = (_evaluation(20), _evaluation(21))
    horizons = (ResearchHorizon(1), ResearchHorizon(3))

    first = _use_case(_series()).execute(evaluations, horizons, _DAILY, _AVAILABLE_THROUGH)
    second = _use_case(_series()).execute(evaluations, horizons, _DAILY, _AVAILABLE_THROUGH)

    assert first == second
    assert first.measurements == second.measurements


def test_run_is_immutable() -> None:
    run = _use_case(_series()).execute(
        (_evaluation(20),), (ResearchHorizon(1),), _DAILY, _AVAILABLE_THROUGH
    )

    with pytest.raises(AttributeError):
        run.measurements = ()


def test_run_requires_one_measurement_per_pair() -> None:
    evaluation = _evaluation(20)

    with pytest.raises(ValueError, match="one measurement for every"):
        HistoricalResearchRun(
            evaluations=(evaluation,),
            horizons=(ResearchHorizon(1), ResearchHorizon(2)),
            measurements=(),
        )


def test_run_accepts_a_coherent_multi_evaluation_multi_horizon_construction() -> None:
    evaluations = (_evaluation(20), _evaluation(21))
    horizons = (ResearchHorizon(1), ResearchHorizon(2))
    measurements = _measurements_for(evaluations, horizons)

    run = HistoricalResearchRun(
        evaluations=evaluations, horizons=horizons, measurements=measurements
    )

    assert run.measurements == measurements
    assert [
        (measurement.recommendation.point_in_time, measurement.horizon)
        for measurement in run.measurements
    ] == [
        (_instant(20), ResearchHorizon(1)),
        (_instant(20), ResearchHorizon(2)),
        (_instant(21), ResearchHorizon(1)),
        (_instant(21), ResearchHorizon(2)),
    ]


def test_run_rejects_measurement_for_the_wrong_recommendation() -> None:
    evaluations = (_evaluation(20), _evaluation(21))
    horizons = (ResearchHorizon(1),)
    measurements = _measurements_for(evaluations, horizons)
    wrong = (measurements[0], measurements[0])

    with pytest.raises(ValueError, match="must follow evaluation order"):
        HistoricalResearchRun(evaluations=evaluations, horizons=horizons, measurements=wrong)


def test_run_rejects_measurement_for_the_wrong_horizon() -> None:
    evaluations = (_evaluation(20),)
    horizons = (ResearchHorizon(1), ResearchHorizon(2))
    measurements = _measurements_for(evaluations, horizons)
    wrong = (measurements[1], measurements[1])

    with pytest.raises(ValueError, match="must follow horizon order"):
        HistoricalResearchRun(evaluations=evaluations, horizons=horizons, measurements=wrong)


def test_run_rejects_correct_measurements_in_the_wrong_order() -> None:
    evaluations = (_evaluation(20), _evaluation(21))
    horizons = (ResearchHorizon(1), ResearchHorizon(2))
    measurements = _measurements_for(evaluations, horizons)
    reordered = (measurements[1], measurements[0], measurements[2], measurements[3])

    with pytest.raises(ValueError, match="must follow horizon order"):
        HistoricalResearchRun(evaluations=evaluations, horizons=horizons, measurements=reordered)


def test_run_rejects_evaluation_blocks_swapped_as_a_whole() -> None:
    evaluations = (_evaluation(20), _evaluation(21))
    horizons = (ResearchHorizon(1), ResearchHorizon(2))
    measurements = _measurements_for(evaluations, horizons)
    swapped = measurements[2:] + measurements[:2]

    with pytest.raises(ValueError, match="must follow evaluation order"):
        HistoricalResearchRun(evaluations=evaluations, horizons=horizons, measurements=swapped)


def test_run_rejects_non_tuple_members() -> None:
    with pytest.raises(TypeError, match="evaluations must be a tuple"):
        HistoricalResearchRun(evaluations=[], horizons=(), measurements=())
    with pytest.raises(TypeError, match="horizons must be a tuple"):
        HistoricalResearchRun(evaluations=(), horizons=[], measurements=())
    with pytest.raises(TypeError, match="measurements must be a tuple"):
        HistoricalResearchRun(evaluations=(), horizons=(), measurements=[])


def test_run_applies_no_metrics_or_action_interpretation() -> None:
    run = _use_case(_series()).execute(
        (_evaluation(20),), (ResearchHorizon(1),), _DAILY, _AVAILABLE_THROUGH
    )

    for forbidden in (
        "accuracy",
        "win_rate",
        "hit_rate",
        "average_return",
        "total_return",
        "pnl",
        "profit",
        "trades",
        "positions",
        "equity_curve",
        "is_correct",
    ):
        assert not hasattr(run, forbidden)
