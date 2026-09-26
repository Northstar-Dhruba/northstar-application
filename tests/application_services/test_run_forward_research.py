"""Tests for forward research monitoring runs."""

from __future__ import annotations

import pytest
from northstar_core.domain.value_objects import ListingReference
from northstar_core.foundation.value_objects import (
    Currency,
    ExchangeCode,
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
    ForwardResearchMeasurementState,
    ForwardResearchRecord,
    ForwardResearchRun,
    HistoricalDataContractViolationError,
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
    """Returns preconfigured records and counts how often it was asked."""

    def __init__(self, records: tuple[ForwardResearchRecord, ...] = ()) -> None:
        self.records = records
        self.queries: list[ForwardResearchRecordQuery] = []

    def get_records(self, query: ForwardResearchRecordQuery) -> tuple[ForwardResearchRecord, ...]:
        self.queries.append(query)
        return self.records


def _result(
    day: int,
    *,
    strategy: str = "mvp",
    symbol: Symbol = _SYMBOL,
    exchange_code: ExchangeCode = _EXCHANGE,
) -> AnalyzeAssetResult:
    context = MarketObservationContext(
        ListingReference(symbol, exchange_code),
        _instant(day),
        Price("100", _USD),
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


def _record(day: int, timeframe: Timeframe = _DAILY, **kwargs: object) -> ForwardResearchRecord:
    return ForwardResearchRecord(result=_result(day, **kwargs), timeframe=timeframe)


def _query(
    symbol: Symbol = _SYMBOL,
    exchange_code: ExchangeCode = _EXCHANGE,
    timeframe: Timeframe = _DAILY,
) -> ForwardResearchRecordQuery:
    return ForwardResearchRecordQuery(symbol, exchange_code, timeframe)


def _series() -> tuple[HistoricalOHLCVBar, ...]:
    return tuple(_bar(20 + offset, str(100 + offset * 10)) for offset in range(6))


def _use_case(
    records: tuple[ForwardResearchRecord, ...] = (),
    observations: tuple[HistoricalOHLCVBar, ...] = (),
    repository: ForwardResearchRecordRepository | None = None,
) -> RunForwardResearchUseCase:
    return RunForwardResearchUseCase(
        repository or StubRecordRepository(records),
        MeasureForwardResearchRecordUseCase(
            MeasureRecommendationOutcomeUseCase(StubMarketData(observations))
        ),
    )


def _run(
    records: tuple[ForwardResearchRecord, ...] = (),
    horizons: tuple[ResearchHorizon, ...] = (ResearchHorizon(1),),
    observations: tuple[HistoricalOHLCVBar, ...] = (),
    *,
    available_through: PointInTime = _AVAILABLE_THROUGH,
    repository: ForwardResearchRecordRepository | None = None,
) -> ForwardResearchRun:
    return _use_case(records, observations, repository).execute(
        _query(), horizons, available_through
    )


# ---------------------------------------------------------------------------
# Dependency and input validation
# ---------------------------------------------------------------------------


def test_use_case_requires_both_dependencies() -> None:
    measure = MeasureForwardResearchRecordUseCase(
        MeasureRecommendationOutcomeUseCase(StubMarketData())
    )

    with pytest.raises(TypeError, match="repository cannot be None"):
        RunForwardResearchUseCase(None, measure)
    with pytest.raises(TypeError, match="must be a ForwardResearchRecordRepository"):
        RunForwardResearchUseCase("repository", measure)
    with pytest.raises(TypeError, match="measure_forward_research_record cannot be None"):
        RunForwardResearchUseCase(StubRecordRepository(), None)
    with pytest.raises(TypeError, match="must be a MeasureForwardResearchRecordUseCase"):
        RunForwardResearchUseCase(StubRecordRepository(), "measure")


def test_execute_validates_query_and_available_through() -> None:
    use_case = _use_case()

    with pytest.raises(TypeError, match="query must be a ForwardResearchRecordQuery"):
        use_case.execute("query", (ResearchHorizon(1),), _AVAILABLE_THROUGH)
    with pytest.raises(TypeError, match="available-through must be a PointInTime"):
        use_case.execute(_query(), (ResearchHorizon(1),), "2026-03-01T16:00:00Z")


def test_execute_validates_horizons() -> None:
    use_case = _use_case()

    with pytest.raises(TypeError, match="horizons cannot be None"):
        use_case.execute(_query(), None, _AVAILABLE_THROUGH)
    with pytest.raises(TypeError, match="horizons must be a tuple"):
        use_case.execute(_query(), [ResearchHorizon(1)], _AVAILABLE_THROUGH)
    with pytest.raises(ValueError, match="at least one horizon"):
        use_case.execute(_query(), (), _AVAILABLE_THROUGH)
    with pytest.raises(ValueError, match="cannot contain duplicates"):
        use_case.execute(_query(), (ResearchHorizon(1), ResearchHorizon(1)), _AVAILABLE_THROUGH)
    with pytest.raises(TypeError, match="must contain ResearchHorizon values"):
        use_case.execute(_query(), (1,), _AVAILABLE_THROUGH)


# ---------------------------------------------------------------------------
# Empty run
# ---------------------------------------------------------------------------


def test_empty_repository_produces_a_valid_empty_run() -> None:
    run = _run()

    assert isinstance(run, ForwardResearchRun)
    assert run.records == ()
    assert run.measurements == ()
    assert run.horizons == (ResearchHorizon(1),)
    assert run.query == _query()
    assert run.available_through == _AVAILABLE_THROUGH


def test_repository_is_called_exactly_once() -> None:
    repository = StubRecordRepository((_record(20), _record(21)))

    _run(horizons=(ResearchHorizon(1), ResearchHorizon(2)), repository=repository)

    assert len(repository.queries) == 1
    assert repository.queries[0] == _query()


# ---------------------------------------------------------------------------
# Records and horizons
# ---------------------------------------------------------------------------


def test_one_record_and_one_horizon_produces_one_measurement() -> None:
    run = _run((_record(20),), (ResearchHorizon(1),), _series())

    assert len(run.records) == 1
    assert len(run.measurements) == 1
    assert run.measurements[0].state is ForwardResearchMeasurementState.MEASURED


def test_multiple_records_and_horizons_produce_the_cartesian_product() -> None:
    records = (_record(20), _record(21), _record(22))
    horizons = (ResearchHorizon(1), ResearchHorizon(2))

    run = _run(records, horizons, _series())

    assert len(run.measurements) == len(records) * len(horizons) == 6
    assert run.records == records
    assert run.horizons == horizons


def test_measurements_are_ordered_record_major_horizon_minor() -> None:
    records = (_record(20), _record(21))
    horizons = (ResearchHorizon(1), ResearchHorizon(2))

    run = _run(records, horizons, _series())

    assert [
        (measurement.record.decision_instant, measurement.horizon)
        for measurement in run.measurements
    ] == [
        (_instant(20), ResearchHorizon(1)),
        (_instant(20), ResearchHorizon(2)),
        (_instant(21), ResearchHorizon(1)),
        (_instant(21), ResearchHorizon(2)),
    ]


def test_horizon_order_follows_the_requested_order() -> None:
    horizons = (ResearchHorizon(3), ResearchHorizon(1), ResearchHorizon(2))

    run = _run((_record(20),), horizons, _series())

    assert run.horizons == horizons
    assert [measurement.horizon for measurement in run.measurements] == list(horizons)


def test_record_order_follows_repository_order() -> None:
    records = (_record(20), _record(21), _record(22))

    run = _run(records, (ResearchHorizon(1),), _series())

    assert [record.decision_instant for record in run.records] == [
        _instant(20),
        _instant(21),
        _instant(22),
    ]


def test_mixed_strategies_at_one_instant_keep_repository_order() -> None:
    records = (_record(20, strategy="alpha"), _record(20, strategy="zeta"))

    run = _run(records, (ResearchHorizon(1),), _series())

    assert [record.strategy_identity.identity for record in run.records] == ["alpha", "zeta"]
    assert [measurement.record.strategy_identity.identity for measurement in run.measurements] == [
        "alpha",
        "zeta",
    ]


# ---------------------------------------------------------------------------
# As-of filtering
# ---------------------------------------------------------------------------


def test_records_after_available_through_are_excluded() -> None:
    records = (_record(20), _record(21), _record(25))

    run = _run(records, (ResearchHorizon(1),), _series(), available_through=_instant(22))

    assert [record.decision_instant for record in run.records] == [_instant(20), _instant(21)]
    assert len(run.measurements) == 2


def test_record_exactly_at_available_through_is_included() -> None:
    records = (_record(20), _record(21))

    run = _run(records, (ResearchHorizon(1),), _series(), available_through=_instant(21))

    assert len(run.records) == 2
    assert run.records[-1].decision_instant == _instant(21)


def test_all_records_after_the_boundary_produce_an_empty_run() -> None:
    records = (_record(24), _record(25))

    run = _run(records, (ResearchHorizon(1),), _series(), available_through=_instant(21))

    assert run.records == ()
    assert run.measurements == ()


def test_as_of_filtering_is_compared_semantically() -> None:
    records = (_record(20),)
    boundary = PointInTime("2026-01-20T21:30:00+05:30")

    run = _run(records, (ResearchHorizon(1),), _series(), available_through=boundary)

    assert len(run.records) == 1


def test_available_through_is_passed_to_measurement() -> None:
    records = (_record(20),)

    run = _run(records, (ResearchHorizon(3),), _series(), available_through=_instant(21))

    assert run.available_through == _instant(21)
    assert run.measurements[0].state is ForwardResearchMeasurementState.PENDING


# ---------------------------------------------------------------------------
# State retention
# ---------------------------------------------------------------------------


def test_pending_measurements_are_retained() -> None:
    run = _run((_record(20),), (ResearchHorizon(20),), _series())

    assert len(run.measurements) == 1
    assert run.measurements[0].state is ForwardResearchMeasurementState.PENDING


def test_unavailable_measurements_are_retained() -> None:
    observations = (_bar(20, "100"), _bar(21, "110", currency=_EUR))

    run = _run((_record(20),), (ResearchHorizon(1),), observations)

    assert run.measurements[0].state is ForwardResearchMeasurementState.UNAVAILABLE


def test_mixed_states_are_all_retained_in_position() -> None:
    run = _run((_record(20),), (ResearchHorizon(1), ResearchHorizon(20)), _series())

    assert [measurement.state for measurement in run.measurements] == [
        ForwardResearchMeasurementState.MEASURED,
        ForwardResearchMeasurementState.PENDING,
    ]


def test_every_record_and_horizon_pair_is_represented() -> None:
    records = (_record(20), _record(24))
    horizons = (ResearchHorizon(1), ResearchHorizon(5))

    run = _run(records, horizons, _series())

    assert len(run.measurements) == 4
    assert all(measurement.state is not None for measurement in run.measurements)


# ---------------------------------------------------------------------------
# Defensive repository validation
# ---------------------------------------------------------------------------


def test_rejects_repository_symbol_mismatch() -> None:
    records = (_record(20, symbol=Symbol("MSFT")),)

    with pytest.raises(HistoricalDataContractViolationError, match="does not match query symbol"):
        _run(records, (ResearchHorizon(1),), _series())


def test_rejects_repository_exchange_mismatch() -> None:
    records = (_record(20, exchange_code=ExchangeCode("NYSE")),)

    with pytest.raises(HistoricalDataContractViolationError, match="does not match query exchange"):
        _run(records, (ResearchHorizon(1),), _series())


def test_rejects_repository_timeframe_mismatch() -> None:
    records = (_record(20, timeframe=Timeframe("1h")),)

    with pytest.raises(
        HistoricalDataContractViolationError, match="does not match query timeframe"
    ):
        _run(records, (ResearchHorizon(1),), _series())


def test_rejects_unordered_repository_records() -> None:
    records = (_record(22), _record(20))

    with pytest.raises(HistoricalDataContractViolationError, match="must be ordered"):
        _run(records, (ResearchHorizon(1),), _series())


def test_rejects_records_unordered_by_strategy_at_one_instant() -> None:
    records = (_record(20, strategy="zeta"), _record(20, strategy="alpha"))

    with pytest.raises(HistoricalDataContractViolationError, match="must be ordered"):
        _run(records, (ResearchHorizon(1),), _series())


def test_rejects_duplicate_natural_keys() -> None:
    record = _record(20)

    with pytest.raises(HistoricalDataContractViolationError, match="duplicate natural keys"):
        _run((record, record), (ResearchHorizon(1),), _series())


def test_rejects_non_tuple_repository_result() -> None:
    class ListRepository(ForwardResearchRecordRepository):
        def get_records(
            self, query: ForwardResearchRecordQuery
        ) -> tuple[ForwardResearchRecord, ...]:
            return [_record(20)]

    with pytest.raises(HistoricalDataContractViolationError, match="must return a tuple"):
        _run(repository=ListRepository())


def test_rejects_non_record_repository_entries() -> None:
    class InvalidRepository(ForwardResearchRecordRepository):
        def get_records(
            self, query: ForwardResearchRecordQuery
        ) -> tuple[ForwardResearchRecord, ...]:
            return ("record",)

    with pytest.raises(
        HistoricalDataContractViolationError, match="must be a ForwardResearchRecord"
    ):
        _run(repository=InvalidRepository())


# ---------------------------------------------------------------------------
# Run coherence
# ---------------------------------------------------------------------------


def _measurements_for(
    records: tuple[ForwardResearchRecord, ...],
    horizons: tuple[ResearchHorizon, ...],
) -> tuple:
    measure = MeasureForwardResearchRecordUseCase(
        MeasureRecommendationOutcomeUseCase(StubMarketData(_series()))
    )
    return tuple(
        measure.execute(record, horizon, _AVAILABLE_THROUGH)
        for record in records
        for horizon in horizons
    )


def _direct_run(**kwargs: object) -> ForwardResearchRun:
    return ForwardResearchRun(
        query=kwargs.get("query", _query()),
        horizons=kwargs["horizons"],
        available_through=kwargs.get("available_through", _AVAILABLE_THROUGH),
        records=kwargs["records"],
        measurements=kwargs["measurements"],
    )


def test_run_rejects_a_wrong_measurement_count() -> None:
    records = (_record(20),)
    horizons = (ResearchHorizon(1), ResearchHorizon(2))
    measurements = _measurements_for(records, horizons)

    with pytest.raises(ValueError, match="one measurement for every record"):
        _direct_run(records=records, horizons=horizons, measurements=measurements[:1])


def test_run_rejects_measurement_for_the_wrong_record() -> None:
    records = (_record(20), _record(21))
    horizons = (ResearchHorizon(1),)
    measurements = _measurements_for(records, horizons)

    with pytest.raises(ValueError, match="must follow record order"):
        _direct_run(
            records=records,
            horizons=horizons,
            measurements=(measurements[0], measurements[0]),
        )


def test_run_rejects_measurement_for_the_wrong_horizon() -> None:
    records = (_record(20),)
    horizons = (ResearchHorizon(1), ResearchHorizon(2))
    measurements = _measurements_for(records, horizons)

    with pytest.raises(ValueError, match="must follow horizon order"):
        _direct_run(
            records=records,
            horizons=horizons,
            measurements=(measurements[1], measurements[1]),
        )


def test_run_rejects_reordered_measurements() -> None:
    records = (_record(20), _record(21))
    horizons = (ResearchHorizon(1), ResearchHorizon(2))
    measurements = _measurements_for(records, horizons)

    with pytest.raises(ValueError, match="must follow horizon order"):
        _direct_run(
            records=records,
            horizons=horizons,
            measurements=(measurements[1], measurements[0]) + measurements[2:],
        )


def test_run_accepts_a_valid_multi_strategy_construction() -> None:
    records = (
        _record(20, strategy="alpha"),
        _record(20, strategy="zeta"),
        _record(21, strategy="alpha"),
    )
    horizons = (ResearchHorizon(1), ResearchHorizon(2))
    measurements = _measurements_for(records, horizons)

    run = _direct_run(records=records, horizons=horizons, measurements=measurements)

    assert run.records == records
    assert len(run.measurements) == 6
    assert [
        (record.decision_instant.value, record.strategy_identity.identity) for record in run.records
    ] == [
        ("2026-01-20T16:00:00Z", "alpha"),
        ("2026-01-20T16:00:00Z", "zeta"),
        ("2026-01-21T16:00:00Z", "alpha"),
    ]


def test_run_rejects_a_record_from_a_different_listing() -> None:
    records = (_record(20, symbol=Symbol("MSFT")),)
    horizons = (ResearchHorizon(1),)
    measurements = _measurements_for(records, horizons)

    with pytest.raises(ValueError, match="must match the run query listing"):
        _direct_run(records=records, horizons=horizons, measurements=measurements)


def test_run_rejects_a_record_from_a_different_exchange() -> None:
    records = (_record(20, exchange_code=ExchangeCode("NYSE")),)
    horizons = (ResearchHorizon(1),)
    measurements = _measurements_for(records, horizons)

    with pytest.raises(ValueError, match="must match the run query listing"):
        _direct_run(records=records, horizons=horizons, measurements=measurements)


def test_run_rejects_a_record_with_a_different_timeframe() -> None:
    records = (_record(20, timeframe=Timeframe("1h")),)
    horizons = (ResearchHorizon(1),)
    measurements = _measurements_for(records, horizons)

    with pytest.raises(ValueError, match="must match the run query listing"):
        _direct_run(records=records, horizons=horizons, measurements=measurements)


def test_run_rejects_a_record_frozen_after_available_through() -> None:
    records = (_record(25),)
    horizons = (ResearchHorizon(1),)
    measurements = _measurements_for(records, horizons)

    with pytest.raises(ValueError, match="cannot be frozen after"):
        _direct_run(
            records=records,
            horizons=horizons,
            measurements=measurements,
            available_through=_instant(21),
        )


def test_run_accepts_a_record_exactly_at_available_through() -> None:
    records = (_record(21),)
    horizons = (ResearchHorizon(1),)
    measurements = _measurements_for(records, horizons)

    run = _direct_run(
        records=records,
        horizons=horizons,
        measurements=measurements,
        available_through=_instant(21),
    )

    assert run.records == records


def test_run_rejects_duplicate_record_natural_keys() -> None:
    record = _record(20)
    records = (record, record)
    horizons = (ResearchHorizon(1),)
    measurements = _measurements_for(records, horizons)

    with pytest.raises(ValueError, match="duplicate natural keys"):
        _direct_run(records=records, horizons=horizons, measurements=measurements)


def test_run_rejects_records_out_of_chronological_order() -> None:
    records = (_record(22), _record(20))
    horizons = (ResearchHorizon(1),)
    measurements = _measurements_for(records, horizons)

    with pytest.raises(ValueError, match="must be ordered by decision instant"):
        _direct_run(records=records, horizons=horizons, measurements=measurements)


def test_run_rejects_a_wrong_strategy_tie_break() -> None:
    records = (_record(20, strategy="zeta"), _record(20, strategy="alpha"))
    horizons = (ResearchHorizon(1),)
    measurements = _measurements_for(records, horizons)

    with pytest.raises(ValueError, match="then strategy identity"):
        _direct_run(records=records, horizons=horizons, measurements=measurements)


def test_run_rejects_empty_horizons() -> None:
    with pytest.raises(ValueError, match="at least one horizon"):
        _direct_run(records=(), horizons=(), measurements=())


def test_run_rejects_duplicate_horizons() -> None:
    records = (_record(20),)
    horizons = (ResearchHorizon(1), ResearchHorizon(1))
    measurements = _measurements_for(records, (ResearchHorizon(1),)) * 2

    with pytest.raises(ValueError, match="horizons cannot contain duplicates"):
        _direct_run(records=records, horizons=horizons, measurements=measurements)


def test_run_rejects_invalid_member_types() -> None:
    with pytest.raises(TypeError, match="query must be a ForwardResearchRecordQuery"):
        ForwardResearchRun(
            query="query",
            horizons=(),
            available_through=_AVAILABLE_THROUGH,
            records=(),
            measurements=(),
        )
    with pytest.raises(TypeError, match="records must be a tuple"):
        _direct_run(records=[], horizons=(ResearchHorizon(1),), measurements=())
    with pytest.raises(TypeError, match="measurements must be a tuple"):
        _direct_run(records=(), horizons=(ResearchHorizon(1),), measurements=[])


# ---------------------------------------------------------------------------
# Determinism and scope
# ---------------------------------------------------------------------------


def test_repeated_runs_are_deterministic() -> None:
    records = (_record(20), _record(21))
    horizons = (ResearchHorizon(1), ResearchHorizon(2))

    first = _run(records, horizons, _series())
    second = _run(records, horizons, _series())

    assert first == second
    assert first.measurements == second.measurements


def test_run_is_immutable() -> None:
    run = _run((_record(20),), (ResearchHorizon(1),), _series())

    with pytest.raises(AttributeError):
        run.records = ()


def test_run_calculates_no_metrics_and_persists_no_state() -> None:
    run = _run((_record(20),), (ResearchHorizon(1),), _series())

    assert ForwardResearchRun.__slots__ == (
        "query",
        "horizons",
        "available_through",
        "records",
        "measurements",
    )
    for forbidden in (
        "accuracy",
        "win_rate",
        "average_return",
        "pending_count",
        "measured_count",
        "state",
        "pnl",
        "trades",
    ):
        assert not hasattr(run, forbidden)
