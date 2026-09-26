"""Tests for factual forward research metrics calculation."""

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
    AnalyzeAssetResult,
    AnalyzeMarketObservationContextService,
    CalculateForwardResearchMetricsUseCase,
    ForwardResearchRecord,
    ForwardResearchRun,
    ForwardResearchStrategyHorizonMetrics,
    MeasureForwardResearchRecordUseCase,
    MeasureRecommendationOutcomeUseCase,
    RunForwardResearchUseCase,
)
from northstar_application.ports import (
    ForwardResearchRecordQuery,
    ForwardResearchRecordRepository,
    HistoricalMarketDataQuery,
    HistoricalMarketDataRepository,
)

_USD = Currency("USD")
_EUR = Currency("EUR")
_DAILY = Timeframe("1d")
_SYMBOL = Symbol("AAPL")
_EXCHANGE = ExchangeCode("NASDAQ")
_AVAILABLE_THROUGH = PointInTime("2026-03-01T16:00:00Z")
_ALPHA = StrategyIdentity("alpha")
_ZETA = StrategyIdentity("zeta")


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


class StubMarketData(HistoricalMarketDataRepository):
    def __init__(self, observations: tuple[HistoricalOHLCVBar, ...] = ()) -> None:
        self.observations = observations

    def get_history(self, query: HistoricalMarketDataQuery) -> tuple[HistoricalOHLCVBar, ...]:
        return tuple(
            bar
            for bar in self.observations
            if bar.symbol == query.symbol
            and bar.exchange_code == query.exchange_code
            and bar.timeframe == query.timeframe
            and query.start.compare(bar.point_in_time) <= 0
            and query.end.compare(bar.point_in_time) >= 0
        )


class StubRecordRepository(ForwardResearchRecordRepository):
    def __init__(self, records: tuple[ForwardResearchRecord, ...] = ()) -> None:
        self.records = records

    def get_records(self, query: ForwardResearchRecordQuery) -> tuple[ForwardResearchRecord, ...]:
        return self.records


def _result(day: int, *, strategy: str = "alpha", latest_close: str = "100") -> AnalyzeAssetResult:
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
    return AnalyzeMarketObservationContextService(
        strategy=Strategy(StrategyIdentity(strategy)),
        analysis_generator=AssetAnalysisGenerator(),
    ).execute(context)


def _record(day: int, **kwargs: str) -> ForwardResearchRecord:
    return ForwardResearchRecord(result=_result(day, **kwargs), timeframe=_DAILY)


def _run(
    records: tuple[ForwardResearchRecord, ...] = (),
    horizons: tuple[ResearchHorizon, ...] = (ResearchHorizon(1),),
    observations: tuple[HistoricalOHLCVBar, ...] = (),
    *,
    available_through: PointInTime = _AVAILABLE_THROUGH,
) -> ForwardResearchRun:
    use_case = RunForwardResearchUseCase(
        StubRecordRepository(records),
        MeasureForwardResearchRecordUseCase(
            MeasureRecommendationOutcomeUseCase(StubMarketData(observations))
        ),
    )
    return use_case.execute(
        ForwardResearchRecordQuery(_SYMBOL, _EXCHANGE, _DAILY), horizons, available_through
    )


def _metrics(run: ForwardResearchRun) -> tuple[ForwardResearchStrategyHorizonMetrics, ...]:
    return CalculateForwardResearchMetricsUseCase().execute(run)


# ---------------------------------------------------------------------------
# Two-strategy fixture
#
# Observations:  d20=100  d21=110  d22=200  d23=260  d24=400  d25=380
# alpha freezes at d20 (price 100) and d24 (price 400)
# zeta  freezes at d22 (price 200) and d24 (price 400)
#
# Horizon 1 returns:  alpha +10 and -5   zeta +30 and -5
# ---------------------------------------------------------------------------


def _two_strategy_observations() -> tuple[HistoricalOHLCVBar, ...]:
    return (
        _bar(20, "100"),
        _bar(21, "110"),
        _bar(22, "200"),
        _bar(23, "260"),
        _bar(24, "400"),
        _bar(25, "380"),
    )


def _two_strategy_records() -> tuple[ForwardResearchRecord, ...]:
    return (
        _record(20, strategy="alpha", latest_close="100"),
        _record(22, strategy="zeta", latest_close="200"),
        _record(24, strategy="alpha", latest_close="400"),
        _record(24, strategy="zeta", latest_close="400"),
    )


def _two_strategy_run(
    horizons: tuple[ResearchHorizon, ...] = (ResearchHorizon(1),),
) -> ForwardResearchRun:
    return _run(_two_strategy_records(), horizons, _two_strategy_observations())


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_calculation_requires_a_forward_research_run() -> None:
    use_case = CalculateForwardResearchMetricsUseCase()

    with pytest.raises(TypeError, match="run cannot be None"):
        use_case.execute(None)
    with pytest.raises(TypeError, match="must be a ForwardResearchRun"):
        use_case.execute("run")


# ---------------------------------------------------------------------------
# Empty run
# ---------------------------------------------------------------------------


def test_a_run_without_records_produces_no_metrics() -> None:
    assert _metrics(_run()) == ()


def test_a_run_without_records_produces_no_metrics_for_any_horizon_count() -> None:
    run = _run((), (ResearchHorizon(1), ResearchHorizon(5), ResearchHorizon(20)))

    assert run.records == ()
    assert _metrics(run) == ()


# ---------------------------------------------------------------------------
# One strategy
# ---------------------------------------------------------------------------


def test_one_strategy_is_summarised_as_one_entry_per_horizon() -> None:
    records = (
        _record(20, strategy="alpha", latest_close="100"),
        _record(22, strategy="alpha", latest_close="200"),
    )
    observations = (_bar(20, "100"), _bar(21, "110"), _bar(22, "200"), _bar(23, "260"))

    metrics = _metrics(_run(records, (ResearchHorizon(1),), observations))

    assert len(metrics) == 1
    only = metrics[0]
    assert only.strategy_identity == _ALPHA
    assert only.horizon == ResearchHorizon(1)
    assert only.total_count == 2
    assert only.measured_count == 2
    assert only.pending_count == 0
    assert only.unavailable_count == 0
    assert only.minimum_forward_return == Percentage(10)
    assert only.maximum_forward_return == Percentage(30)
    assert only.average_forward_return == Percentage(20)
    assert only.median_forward_return == Percentage(20)


# ---------------------------------------------------------------------------
# Strategy separation
# ---------------------------------------------------------------------------


def test_two_strategies_are_summarised_separately() -> None:
    metrics = _metrics(_two_strategy_run())

    assert len(metrics) == 2
    alpha, zeta = metrics
    assert alpha.strategy_identity == _ALPHA
    assert zeta.strategy_identity == _ZETA
    assert alpha.total_count == 2
    assert zeta.total_count == 2
    assert alpha.minimum_forward_return == Percentage(-5)
    assert alpha.maximum_forward_return == Percentage(10)
    assert zeta.minimum_forward_return == Percentage(-5)
    assert zeta.maximum_forward_return == Percentage(30)


def test_strategies_deciding_at_the_same_instant_remain_separated() -> None:
    """Both strategies freeze at day 24; neither absorbs the other's decision."""
    records = (
        _record(24, strategy="alpha", latest_close="400"),
        _record(24, strategy="zeta", latest_close="400"),
    )
    observations = (_bar(24, "400"), _bar(25, "380"))

    metrics = _metrics(_run(records, (ResearchHorizon(1),), observations))

    assert [entry.strategy_identity for entry in metrics] == [_ALPHA, _ZETA]
    assert all(entry.total_count == 1 for entry in metrics)
    assert all(entry.measured_count == 1 for entry in metrics)
    assert all(entry.average_forward_return == Percentage(-5) for entry in metrics)


def test_pooled_average_differs_from_every_strategy_average() -> None:
    """The decisive guard: pooling would produce a number no strategy earned."""
    run = _two_strategy_run()
    alpha, zeta = _metrics(run)

    assert alpha.average_forward_return == Percentage(Decimal("2.5"))
    assert zeta.average_forward_return == Percentage(Decimal("12.5"))

    pooled_returns = tuple(
        measurement.measurement.outcome.forward_return
        for measurement in run.measurements
        if measurement.is_measured
    )
    assert len(pooled_returns) == 4
    pooled_average = Percentage(
        sum((forward_return.value for forward_return in pooled_returns), Decimal(0))
        / Decimal(len(pooled_returns))
    )

    assert pooled_average == Percentage(Decimal("7.5"))
    assert pooled_average != alpha.average_forward_return
    assert pooled_average != zeta.average_forward_return


def test_one_strategys_measurements_never_enter_another_summary() -> None:
    run = _two_strategy_run()
    alpha, zeta = _metrics(run)

    assert alpha.maximum_forward_return != zeta.maximum_forward_return
    assert alpha.total_count + zeta.total_count == len(run.measurements)


# ---------------------------------------------------------------------------
# Grouping and ordering
# ---------------------------------------------------------------------------


def test_metrics_are_strategy_major_and_horizon_minor() -> None:
    horizons = (ResearchHorizon(1), ResearchHorizon(2))

    metrics = _metrics(_two_strategy_run(horizons))

    assert [(entry.strategy_identity, entry.horizon) for entry in metrics] == [
        (_ALPHA, ResearchHorizon(1)),
        (_ALPHA, ResearchHorizon(2)),
        (_ZETA, ResearchHorizon(1)),
        (_ZETA, ResearchHorizon(2)),
    ]


def test_run_horizon_order_is_preserved_within_each_strategy() -> None:
    horizons = (ResearchHorizon(2), ResearchHorizon(1))

    metrics = _metrics(_two_strategy_run(horizons))

    assert [entry.horizon for entry in metrics] == [
        ResearchHorizon(2),
        ResearchHorizon(1),
        ResearchHorizon(2),
        ResearchHorizon(1),
    ]


def test_strategies_are_ordered_ascending_by_identity_not_first_appearance() -> None:
    """Zeta decides first, yet alpha is still summarised first."""
    records = (
        _record(20, strategy="zeta", latest_close="100"),
        _record(22, strategy="alpha", latest_close="200"),
    )
    observations = (_bar(20, "100"), _bar(21, "110"), _bar(22, "200"), _bar(23, "260"))

    metrics = _metrics(_run(records, (ResearchHorizon(1),), observations))

    assert [entry.strategy_identity for entry in metrics] == [_ALPHA, _ZETA]


def test_every_measurement_is_counted_exactly_once() -> None:
    run = _two_strategy_run((ResearchHorizon(1), ResearchHorizon(2)))

    metrics = _metrics(run)

    assert sum(entry.total_count for entry in metrics) == len(run.measurements)


# ---------------------------------------------------------------------------
# Measured, pending and unavailable counts
# ---------------------------------------------------------------------------


def test_a_horizon_without_enough_future_observations_is_pending() -> None:
    horizons = (ResearchHorizon(1), ResearchHorizon(2))

    metrics = _metrics(_two_strategy_run(horizons))

    alpha_first, alpha_second, zeta_first, zeta_second = metrics
    assert alpha_first.measured_count == 2
    assert alpha_first.pending_count == 0
    assert alpha_second.measured_count == 1
    assert alpha_second.pending_count == 1
    assert alpha_second.unavailable_count == 0
    assert zeta_second.measured_count == 1
    assert zeta_second.pending_count == 1
    assert alpha_second.average_forward_return == Percentage(100)
    assert zeta_second.average_forward_return == Percentage(100)


def test_a_currency_transition_is_unavailable_not_pending() -> None:
    records = (_record(20, strategy="alpha", latest_close="100"),)
    observations = (_bar(20, "100"), _bar(21, "110", currency=_EUR))

    only = _metrics(_run(records, (ResearchHorizon(1),), observations))[0]

    assert only.total_count == 1
    assert only.measured_count == 0
    assert only.pending_count == 0
    assert only.unavailable_count == 1


def test_measured_pending_and_unavailable_partition_the_total() -> None:
    records = (
        _record(20, strategy="alpha", latest_close="100"),
        _record(22, strategy="alpha", latest_close="200"),
        _record(24, strategy="alpha", latest_close="400"),
    )
    observations = (
        _bar(20, "100"),
        _bar(21, "110"),
        _bar(22, "200"),
        _bar(23, "260", currency=_EUR),
        _bar(24, "400"),
    )

    only = _metrics(_run(records, (ResearchHorizon(1),), observations))[0]

    assert only.total_count == 3
    assert only.measured_count == 1
    assert only.unavailable_count == 1
    assert only.pending_count == 1
    assert only.measured_count + only.pending_count + only.unavailable_count == only.total_count


def test_statistics_are_none_when_nothing_was_measured() -> None:
    records = (_record(24, strategy="alpha", latest_close="400"),)

    only = _metrics(_run(records, (ResearchHorizon(1),), (_bar(24, "400"),)))[0]

    assert only.total_count == 1
    assert only.measured_count == 0
    assert only.pending_count == 1
    assert only.average_forward_return is None
    assert only.median_forward_return is None
    assert only.minimum_forward_return is None
    assert only.maximum_forward_return is None


def test_a_pending_horizon_is_never_reported_as_a_neutral_movement() -> None:
    records = (_record(24, strategy="alpha", latest_close="400"),)

    only = _metrics(_run(records, (ResearchHorizon(1),), (_bar(24, "400"),)))[0]

    assert only.average_forward_return is not Percentage(0)
    assert only.average_forward_return is None


# ---------------------------------------------------------------------------
# Decimal determinism
# ---------------------------------------------------------------------------


def _recurring_run() -> ForwardResearchRun:
    records = (
        _record(20, strategy="alpha", latest_close="300"),
        _record(22, strategy="zeta", latest_close="300"),
    )
    observations = (_bar(20, "300"), _bar(21, "400"), _bar(22, "300"), _bar(23, "100"))
    return _run(records, (ResearchHorizon(1),), observations)


@pytest.mark.parametrize("precision", [6, 28, 50])
def test_metrics_are_identical_under_any_ambient_precision(precision: int) -> None:
    run = _recurring_run()
    expected = _metrics(run)

    with localcontext() as context:
        context.prec = precision

        assert _metrics(run) == expected


def test_metrics_agree_across_every_ambient_precision() -> None:
    run = _recurring_run()

    results = []
    for precision in (6, 28, 50):
        with localcontext() as context:
            context.prec = precision
            results.append(_metrics(run))

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
    expected = _metrics(run)

    with localcontext() as context:
        context.prec = 6
        context.rounding = ROUND_DOWN

        assert _metrics(run) == expected
        assert getcontext().prec == 6
        assert getcontext().rounding == ROUND_DOWN


def test_repeated_calculation_of_one_run_is_equal() -> None:
    run = _two_strategy_run((ResearchHorizon(1), ResearchHorizon(2)))

    assert _metrics(run) == _metrics(run)


# ---------------------------------------------------------------------------
# Metrics value object invariants
# ---------------------------------------------------------------------------


def _valid_metrics(**overrides: object) -> ForwardResearchStrategyHorizonMetrics:
    values: dict[str, object] = {
        "strategy_identity": _ALPHA,
        "horizon": ResearchHorizon(1),
        "total_count": 2,
        "measured_count": 1,
        "pending_count": 1,
        "unavailable_count": 0,
        "average_forward_return": Percentage(10),
        "median_forward_return": Percentage(10),
        "minimum_forward_return": Percentage(10),
        "maximum_forward_return": Percentage(10),
    }
    values.update(overrides)
    return ForwardResearchStrategyHorizonMetrics(**values)


def test_metrics_reject_invalid_member_types() -> None:
    with pytest.raises(TypeError, match="strategy identity must be a StrategyIdentity"):
        _valid_metrics(strategy_identity="alpha")
    with pytest.raises(TypeError, match="horizon must be a ResearchHorizon"):
        _valid_metrics(horizon=1)
    with pytest.raises(TypeError, match="average_forward_return must be a Percentage or None"):
        _valid_metrics(average_forward_return=10)


@pytest.mark.parametrize(
    "name", ["total_count", "measured_count", "pending_count", "unavailable_count"]
)
def test_metrics_reject_non_integer_counts(name: str) -> None:
    with pytest.raises(TypeError, match=f"{name} must be an integer"):
        _valid_metrics(**{name: "2"})


@pytest.mark.parametrize(
    "name", ["total_count", "measured_count", "pending_count", "unavailable_count"]
)
def test_metrics_reject_boolean_counts(name: str) -> None:
    with pytest.raises(TypeError, match=f"{name} must be an integer"):
        _valid_metrics(**{name: True})


def test_metrics_reject_negative_counts() -> None:
    with pytest.raises(ValueError, match="pending_count cannot be negative"):
        _valid_metrics(total_count=0, measured_count=1, pending_count=-1)


def test_metrics_require_counts_to_partition_the_total() -> None:
    with pytest.raises(ValueError, match="must equal the total count"):
        _valid_metrics(total_count=3)


def test_metrics_reject_statistics_without_a_measurement() -> None:
    with pytest.raises(ValueError, match="must be None when"):
        _valid_metrics(total_count=1, measured_count=0, pending_count=1)


def test_metrics_require_statistics_when_something_was_measured() -> None:
    with pytest.raises(ValueError, match="are required when"):
        _valid_metrics(median_forward_return=None)


def test_metrics_are_immutable() -> None:
    metrics = _valid_metrics()

    with pytest.raises(AttributeError):
        metrics.total_count = 5


def test_statistics_remain_decimal_backed_percentages() -> None:
    only = _metrics(_recurring_run())[0]

    assert isinstance(only.average_forward_return, Percentage)
    assert isinstance(only.average_forward_return.value, Decimal)
