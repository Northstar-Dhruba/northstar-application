"""Tests for factual historical research metrics calculation."""

from __future__ import annotations

from decimal import ROUND_DOWN, Decimal, getcontext, localcontext

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
    CalculateHistoricalResearchMetricsUseCase,
    HistoricalResearchEvaluation,
    HistoricalResearchHorizonMetrics,
    HistoricalResearchRun,
    MeasureRecommendationOutcomeUseCase,
    RunHistoricalResearchUseCase,
)
from northstar_application.ports import (
    HistoricalMarketDataQuery,
    HistoricalMarketDataRepository,
)

_USD = Currency("USD")
_EUR = Currency("EUR")
_DAILY = Timeframe("1d")
_SYMBOL = Symbol("AAPL")
_EXCHANGE = ExchangeCode("NASDAQ")
_AVAILABLE_THROUGH = PointInTime("2026-03-01T16:00:00Z")


def _instant(day: int) -> PointInTime:
    return PointInTime(f"2026-01-{day:02d}T16:00:00Z")


def _bar(day: int, close: str, *, currency: Currency = _USD) -> HistoricalOHLCVBar:
    return HistoricalOHLCVBar(
        _SYMBOL,
        _EXCHANGE,
        _instant(day),
        _DAILY,
        Price("50", currency),
        Price("900", currency),
        Price("1", currency),
        Price(close, currency),
        Quantity("1000"),
    )


class StubRepository(HistoricalMarketDataRepository):
    """Returns bounded observations exactly as a conforming repository would."""

    def __init__(self, observations: tuple[HistoricalOHLCVBar, ...] = ()) -> None:
        self.observations = observations

    def get_history(self, query: HistoricalMarketDataQuery) -> tuple[HistoricalOHLCVBar, ...]:
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
        Price("900", _USD),
        Price("1", _USD),
        tuple(Price("100", _USD) for _ in range(20)),
        tuple(Quantity("1000") for _ in range(20)),
    )
    result = AnalyzeMarketObservationContextService(
        strategy=Strategy(StrategyIdentity("metrics-test")),
        analysis_generator=AssetAnalysisGenerator(),
    ).execute(context)
    return HistoricalResearchEvaluation(replay_instant=_instant(day), result=result)


def _run(
    evaluations: tuple[HistoricalResearchEvaluation, ...],
    horizons: tuple[ResearchHorizon, ...],
    observations: tuple[HistoricalOHLCVBar, ...],
) -> HistoricalResearchRun:
    use_case = RunHistoricalResearchUseCase(
        MeasureRecommendationOutcomeUseCase(StubRepository(observations))
    )
    return use_case.execute(evaluations, horizons, _DAILY, _AVAILABLE_THROUGH)


def _metrics(run: HistoricalResearchRun) -> tuple[HistoricalResearchHorizonMetrics, ...]:
    return CalculateHistoricalResearchMetricsUseCase().execute(run)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_execute_rejects_none_run() -> None:
    with pytest.raises(TypeError, match="run cannot be None"):
        CalculateHistoricalResearchMetricsUseCase().execute(None)


def test_execute_rejects_wrong_run_type() -> None:
    with pytest.raises(TypeError, match="must be a HistoricalResearchRun"):
        CalculateHistoricalResearchMetricsUseCase().execute("run")


# ---------------------------------------------------------------------------
# Empty runs
# ---------------------------------------------------------------------------


def test_empty_evaluations_produce_metrics_with_zero_counts_and_no_statistics() -> None:
    run = _run((), (ResearchHorizon(1),), ())

    metrics = _metrics(run)

    assert len(metrics) == 1
    only = metrics[0]
    assert only.horizon == ResearchHorizon(1)
    assert only.total_count == 0
    assert only.measured_count == 0
    assert only.insufficient_future_observations_count == 0
    assert only.currency_mismatch_count == 0
    assert only.average_forward_return is None
    assert only.median_forward_return is None
    assert only.minimum_forward_return is None
    assert only.maximum_forward_return is None


def test_empty_run_returns_one_metrics_object_per_horizon() -> None:
    run = _run((), (ResearchHorizon(1), ResearchHorizon(5)), ())

    metrics = _metrics(run)

    assert [entry.horizon for entry in metrics] == [ResearchHorizon(1), ResearchHorizon(5)]


# ---------------------------------------------------------------------------
# Single measured result
# ---------------------------------------------------------------------------


def test_one_measured_result_reports_identical_statistics() -> None:
    run = _run((_evaluation(20),), (ResearchHorizon(1),), (_bar(20, "100"), _bar(21, "105")))

    only = _metrics(run)[0]

    assert only.total_count == 1
    assert only.measured_count == 1
    assert only.average_forward_return == Percentage(5)
    assert only.median_forward_return == Percentage(5)
    assert only.minimum_forward_return == Percentage(5)
    assert only.maximum_forward_return == Percentage(5)


# ---------------------------------------------------------------------------
# Mixed measured and unavailable results
# ---------------------------------------------------------------------------


def test_mixed_measured_insufficient_and_currency_mismatch_are_counted() -> None:
    observations = (
        _bar(20, "100"),
        _bar(21, "105"),
        _bar(22, "100"),
        _bar(23, "110", currency=_EUR),
    )
    evaluations = (_evaluation(20), _evaluation(22), _evaluation(23))

    only = _metrics(_run(evaluations, (ResearchHorizon(1),), observations))[0]

    assert only.total_count == 3
    assert only.measured_count == 1
    assert only.currency_mismatch_count == 1
    assert only.insufficient_future_observations_count == 1
    assert only.average_forward_return == Percentage(5)


def test_unavailable_measurements_are_excluded_from_return_statistics() -> None:
    observations = (_bar(20, "100"), _bar(21, "105"), _bar(22, "100"))
    evaluations = (_evaluation(20), _evaluation(22))

    only = _metrics(_run(evaluations, (ResearchHorizon(1),), observations))[0]

    assert only.total_count == 2
    assert only.measured_count == 1
    assert only.average_forward_return == Percentage(5)
    assert only.minimum_forward_return == Percentage(5)
    assert only.maximum_forward_return == Percentage(5)


def test_count_partition_invariant_holds_for_every_horizon() -> None:
    observations = (_bar(20, "100"), _bar(21, "105"), _bar(22, "95"))
    evaluations = (_evaluation(20), _evaluation(21), _evaluation(22))

    horizons = (ResearchHorizon(1), ResearchHorizon(4))

    for entry in _metrics(_run(evaluations, horizons, observations)):
        assert (
            entry.measured_count
            + entry.insufficient_future_observations_count
            + entry.currency_mismatch_count
            == entry.total_count
        )


def test_no_measured_outcomes_produce_none_statistics_not_zero() -> None:
    run = _run((_evaluation(20),), (ResearchHorizon(10),), (_bar(20, "100"), _bar(21, "105")))

    only = _metrics(run)[0]

    assert only.measured_count == 0
    assert only.insufficient_future_observations_count == 1
    assert only.average_forward_return is None
    assert only.median_forward_return is None
    assert only.minimum_forward_return is None
    assert only.maximum_forward_return is None
    assert only.average_forward_return != Percentage(0)


# ---------------------------------------------------------------------------
# Horizon isolation and ordering
# ---------------------------------------------------------------------------


def test_horizons_remain_isolated_and_ordered() -> None:
    observations = (_bar(20, "100"), _bar(21, "110"), _bar(22, "120"))
    horizons = (ResearchHorizon(2), ResearchHorizon(1))

    metrics = _metrics(_run((_evaluation(20),), horizons, observations))

    assert [entry.horizon for entry in metrics] == list(horizons)
    assert metrics[0].average_forward_return == Percentage(20)
    assert metrics[1].average_forward_return == Percentage(10)


def test_measurements_are_never_mixed_across_horizons() -> None:
    observations = (_bar(20, "100"), _bar(21, "110"), _bar(22, "120"))
    evaluations = (_evaluation(20), _evaluation(21))

    metrics = _metrics(_run(evaluations, (ResearchHorizon(1), ResearchHorizon(2)), observations))

    assert metrics[0].total_count == 2
    assert metrics[1].total_count == 2
    assert metrics[0].measured_count == 2
    assert metrics[1].measured_count == 1


# ---------------------------------------------------------------------------
# Return statistics
# ---------------------------------------------------------------------------


def test_positive_negative_and_zero_returns_are_recorded_factually() -> None:
    observations = (_bar(20, "100"), _bar(21, "110"), _bar(22, "90"), _bar(23, "100"))
    evaluations = (_evaluation(20), _evaluation(21, latest_close="110"), _evaluation(22))

    only = _metrics(_run(evaluations, (ResearchHorizon(1),), observations))[0]

    # returns are +10 (100 -> 110), -18.18... (110 -> 90) and 0 (100 -> 100)
    assert only.measured_count == 3
    assert only.minimum_forward_return == Percentage(Decimal("-18.18181818181818181818181818"))
    assert only.maximum_forward_return == Percentage(10)
    assert only.median_forward_return == Percentage(0)


def test_odd_sized_median_selects_the_middle_value() -> None:
    observations = (_bar(20, "100"), _bar(21, "110"), _bar(22, "220"), _bar(23, "242"))
    evaluations = (
        _evaluation(20),
        _evaluation(21, latest_close="110"),
        _evaluation(22, latest_close="220"),
    )

    only = _metrics(_run(evaluations, (ResearchHorizon(1),), observations))[0]

    assert only.measured_count == 3
    assert only.median_forward_return == Percentage(Decimal("10"))


def test_even_sized_median_averages_the_two_middle_values() -> None:
    observations = (_bar(20, "100"), _bar(21, "120"), _bar(22, "100"), _bar(23, "140"))
    evaluations = (_evaluation(20), _evaluation(22))

    only = _metrics(_run(evaluations, (ResearchHorizon(1),), observations))[0]

    assert only.measured_count == 2
    assert only.median_forward_return == Percentage(30)
    assert only.average_forward_return == Percentage(30)


def test_minimum_and_maximum_span_the_measured_returns() -> None:
    observations = (_bar(20, "100"), _bar(21, "120"), _bar(22, "100"), _bar(23, "110"))
    evaluations = (_evaluation(20), _evaluation(22))

    only = _metrics(_run(evaluations, (ResearchHorizon(1),), observations))[0]

    assert only.minimum_forward_return == Percentage(10)
    assert only.maximum_forward_return == Percentage(20)
    assert only.average_forward_return == Percentage(15)


# ---------------------------------------------------------------------------
# Decimal determinism
# ---------------------------------------------------------------------------


def _recurring_run() -> HistoricalResearchRun:
    observations = (_bar(20, "100"), _bar(21, "110"), _bar(22, "300"), _bar(23, "400"))
    evaluations = (_evaluation(20), _evaluation(22, latest_close="300"))
    return _run(evaluations, (ResearchHorizon(1),), observations)


@pytest.mark.parametrize("precision", [6, 28, 50])
def test_metrics_are_identical_under_any_ambient_precision(precision: int) -> None:
    run = _recurring_run()
    expected = _metrics(run)[0]

    with localcontext() as context:
        context.prec = precision

        assert _metrics(run)[0] == expected


def test_metrics_agree_across_every_ambient_precision() -> None:
    run = _recurring_run()

    results = []
    for precision in (6, 28, 50):
        with localcontext() as context:
            context.prec = precision
            results.append(_metrics(run)[0])

    assert results[0] == results[1] == results[2]


def test_calculation_does_not_modify_the_caller_decimal_context() -> None:
    run = _recurring_run()
    caller_context = getcontext()
    precision_before = caller_context.prec
    rounding_before = caller_context.rounding

    _metrics(run)

    assert getcontext() is caller_context
    assert getcontext().prec == precision_before
    assert getcontext().rounding == rounding_before


def test_calculation_restores_a_customised_caller_context() -> None:
    run = _recurring_run()
    expected = _metrics(run)[0]

    with localcontext() as context:
        context.prec = 6
        context.rounding = ROUND_DOWN

        assert _metrics(run)[0] == expected
        assert getcontext().prec == 6
        assert getcontext().rounding == ROUND_DOWN


def test_statistics_remain_decimal_backed_percentages() -> None:
    only = _metrics(_recurring_run())[0]

    assert isinstance(only.average_forward_return, Percentage)
    assert isinstance(only.average_forward_return.value, Decimal)
    assert not isinstance(only.average_forward_return.value, float)


# ---------------------------------------------------------------------------
# Determinism and contract invariants
# ---------------------------------------------------------------------------


def test_repeated_calculation_is_deterministic() -> None:
    run = _recurring_run()

    assert _metrics(run) == _metrics(run)


def test_metrics_are_immutable() -> None:
    only = _metrics(_recurring_run())[0]

    with pytest.raises(AttributeError):
        only.total_count = 99


def test_metrics_reject_a_broken_count_partition() -> None:
    with pytest.raises(ValueError, match="must equal the total count"):
        HistoricalResearchHorizonMetrics(
            horizon=ResearchHorizon(1),
            total_count=3,
            measured_count=1,
            insufficient_future_observations_count=1,
            currency_mismatch_count=0,
            average_forward_return=Percentage(1),
            median_forward_return=Percentage(1),
            minimum_forward_return=Percentage(1),
            maximum_forward_return=Percentage(1),
        )


def test_metrics_reject_statistics_without_any_measured_outcome() -> None:
    with pytest.raises(ValueError, match="must be None when"):
        HistoricalResearchHorizonMetrics(
            horizon=ResearchHorizon(1),
            total_count=1,
            measured_count=0,
            insufficient_future_observations_count=1,
            currency_mismatch_count=0,
            average_forward_return=Percentage(0),
            median_forward_return=None,
            minimum_forward_return=None,
            maximum_forward_return=None,
        )


def test_metrics_require_statistics_when_an_outcome_was_measured() -> None:
    with pytest.raises(ValueError, match="are required when"):
        HistoricalResearchHorizonMetrics(
            horizon=ResearchHorizon(1),
            total_count=1,
            measured_count=1,
            insufficient_future_observations_count=0,
            currency_mismatch_count=0,
            average_forward_return=None,
            median_forward_return=None,
            minimum_forward_return=None,
            maximum_forward_return=None,
        )


def test_metrics_reject_negative_counts() -> None:
    with pytest.raises(ValueError, match="cannot be negative"):
        HistoricalResearchHorizonMetrics(
            horizon=ResearchHorizon(1),
            total_count=-1,
            measured_count=0,
            insufficient_future_observations_count=0,
            currency_mismatch_count=0,
            average_forward_return=None,
            median_forward_return=None,
            minimum_forward_return=None,
            maximum_forward_return=None,
        )


def test_metrics_apply_no_action_interpretation_or_trading_semantics() -> None:
    only = _metrics(_recurring_run())[0]

    for forbidden in (
        "accuracy",
        "win_rate",
        "hit_rate",
        "success_count",
        "failure_count",
        "correct_count",
        "pnl",
        "profit",
        "trades",
        "positions",
        "equity_curve",
        "aligned_return",
    ):
        assert not hasattr(only, forbidden)
