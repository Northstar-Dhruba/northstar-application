"""Tests for measuring frozen forward research decisions."""

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
    AnalyzeAssetResult,
    AnalyzeMarketObservationContextService,
    ForwardResearchMeasurement,
    ForwardResearchMeasurementState,
    ForwardResearchRecord,
    MeasureForwardResearchRecordUseCase,
    MeasureRecommendationOutcomeUseCase,
    RecommendationOutcomeMeasurement,
    RecommendationOutcomeUnavailableReason,
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
_DECISION_INSTANT = PointInTime("2026-01-20T16:00:00Z")
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
    """Returns bounded, identity-matched observations as a conforming repository does."""

    def __init__(self, observations: tuple[HistoricalOHLCVBar, ...] = ()) -> None:
        self.observations = observations
        self.queries: list[HistoricalMarketDataQuery] = []

    def get_history(self, query: HistoricalMarketDataQuery) -> tuple[HistoricalOHLCVBar, ...]:
        self.queries.append(query)
        return tuple(
            bar
            for bar in self.observations
            if bar.symbol == query.symbol
            and bar.exchange_code == query.exchange_code
            and bar.timeframe == query.timeframe
            and query.start.compare(bar.point_in_time) <= 0
            and query.end.compare(bar.point_in_time) >= 0
        )


def _result(strategy: str = "mvp") -> AnalyzeAssetResult:
    context = MarketObservationContext(
        ListingReference(_SYMBOL, _EXCHANGE),
        _DECISION_INSTANT,
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


def _record(timeframe: Timeframe = _DAILY, strategy: str = "mvp") -> ForwardResearchRecord:
    return ForwardResearchRecord(result=_result(strategy), timeframe=timeframe)


def _measure(
    observations: tuple[HistoricalOHLCVBar, ...],
    *,
    horizon: int = 1,
    record: ForwardResearchRecord | None = None,
    available_through: PointInTime = _AVAILABLE_THROUGH,
    repository: HistoricalMarketDataRepository | None = None,
) -> ForwardResearchMeasurement:
    use_case = MeasureForwardResearchRecordUseCase(
        MeasureRecommendationOutcomeUseCase(repository or StubRepository(observations))
    )
    return use_case.execute(record or _record(), ResearchHorizon(horizon), available_through)


# ---------------------------------------------------------------------------
# Dependency and input validation
# ---------------------------------------------------------------------------


def test_use_case_requires_a_measurement_use_case() -> None:
    with pytest.raises(TypeError, match="cannot be None"):
        MeasureForwardResearchRecordUseCase(None)

    with pytest.raises(TypeError, match="must be a MeasureRecommendationOutcomeUseCase"):
        MeasureForwardResearchRecordUseCase("measure")


def test_execute_rejects_none_record() -> None:
    use_case = MeasureForwardResearchRecordUseCase(
        MeasureRecommendationOutcomeUseCase(StubRepository())
    )

    with pytest.raises(TypeError, match="record cannot be None"):
        use_case.execute(None, ResearchHorizon(1), _AVAILABLE_THROUGH)


def test_execute_rejects_wrong_record_type() -> None:
    use_case = MeasureForwardResearchRecordUseCase(
        MeasureRecommendationOutcomeUseCase(StubRepository())
    )

    with pytest.raises(TypeError, match="must be a ForwardResearchRecord"):
        use_case.execute("record", ResearchHorizon(1), _AVAILABLE_THROUGH)


# ---------------------------------------------------------------------------
# Measured forward result
# ---------------------------------------------------------------------------


def test_measured_forward_result_reports_measured_state() -> None:
    measurement = _measure((_bar(20, "100"), _bar(21, "105")), horizon=1)

    assert measurement.state is ForwardResearchMeasurementState.MEASURED
    assert measurement.is_measured
    assert measurement.measurement.outcome is not None
    assert measurement.measurement.outcome.forward_return == Percentage(5)


def test_measured_result_preserves_record_and_horizon() -> None:
    record = _record()

    measurement = _measure((_bar(20, "100"), _bar(21, "105")), horizon=1, record=record)

    assert measurement.record is record
    assert measurement.horizon == ResearchHorizon(1)
    assert measurement.measurement.horizon == ResearchHorizon(1)


def test_negative_forward_return_is_still_measured() -> None:
    measurement = _measure((_bar(20, "100"), _bar(21, "95")), horizon=1)

    assert measurement.state is ForwardResearchMeasurementState.MEASURED
    assert measurement.measurement.outcome.forward_return == Percentage(-5)


# ---------------------------------------------------------------------------
# State derivation
# ---------------------------------------------------------------------------


def test_insufficient_future_observations_is_pending() -> None:
    measurement = _measure((_bar(20, "100"),), horizon=1)

    assert measurement.state is ForwardResearchMeasurementState.PENDING
    assert measurement.is_measured is False
    assert (
        measurement.measurement.unavailable
        is RecommendationOutcomeUnavailableReason.INSUFFICIENT_FUTURE_OBSERVATIONS
    )


def test_empty_future_data_is_pending() -> None:
    measurement = _measure((), horizon=1)

    assert measurement.state is ForwardResearchMeasurementState.PENDING


def test_currency_mismatch_is_unavailable() -> None:
    measurement = _measure((_bar(20, "100"), _bar(21, "105", currency=_EUR)), horizon=1)

    assert measurement.state is ForwardResearchMeasurementState.UNAVAILABLE
    assert measurement.is_measured is False
    assert (
        measurement.measurement.unavailable
        is RecommendationOutcomeUnavailableReason.CURRENCY_MISMATCH
    )


def test_pending_becomes_measured_once_observations_arrive() -> None:
    record = _record()
    early = _measure((_bar(20, "100"),), horizon=1, record=record)
    later = _measure((_bar(20, "100"), _bar(21, "105")), horizon=1, record=record)

    assert early.state is ForwardResearchMeasurementState.PENDING
    assert later.state is ForwardResearchMeasurementState.MEASURED


def test_state_is_derived_and_not_stored() -> None:
    measurement = _measure((_bar(20, "100"), _bar(21, "105")), horizon=1)

    assert ForwardResearchMeasurement.__slots__ == ("record", "horizon", "measurement")
    assert "state" not in ForwardResearchMeasurement.__slots__
    assert not hasattr(measurement, "__dict__")


def test_state_vocabulary_is_stable() -> None:
    assert [state.value for state in ForwardResearchMeasurementState] == [
        "PENDING",
        "MEASURED",
        "UNAVAILABLE",
    ]


def test_record_persists_no_state() -> None:
    record = _record()

    for forbidden in ("state", "status", "pending", "measured", "unavailable", "outcome"):
        assert not hasattr(record, forbidden)


# ---------------------------------------------------------------------------
# Delegation
# ---------------------------------------------------------------------------


class RecordingMeasureUseCase(MeasureRecommendationOutcomeUseCase):
    """Wraps the real measurement use case and records every delegated call."""

    def __init__(self, repository: HistoricalMarketDataRepository) -> None:
        super().__init__(repository)
        self.calls: list[tuple[AnalyzeAssetResult, ResearchHorizon, Timeframe, PointInTime]] = []

    def execute(
        self,
        result: AnalyzeAssetResult,
        horizon: ResearchHorizon,
        timeframe: Timeframe,
        available_through: PointInTime,
    ) -> RecommendationOutcomeMeasurement:
        self.calls.append((result, horizon, timeframe, available_through))
        return super().execute(result, horizon, timeframe, available_through)


def test_delegation_passes_exact_inputs() -> None:
    recording = RecordingMeasureUseCase(StubRepository((_bar(20, "100"), _bar(21, "105"))))
    record = _record()
    available_through = PointInTime("2026-02-10T16:00:00Z")

    MeasureForwardResearchRecordUseCase(recording).execute(
        record, ResearchHorizon(3), available_through
    )

    assert len(recording.calls) == 1
    delegated_result, horizon, timeframe, delegated_through = recording.calls[0]
    assert delegated_result is record.result
    assert horizon == ResearchHorizon(3)
    assert timeframe is record.timeframe
    assert delegated_through is available_through


def test_delegation_uses_the_record_timeframe() -> None:
    recording = RecordingMeasureUseCase(StubRepository())
    record = _record(timeframe=Timeframe("1h"))

    MeasureForwardResearchRecordUseCase(recording).execute(
        record, ResearchHorizon(1), _AVAILABLE_THROUGH
    )

    assert recording.calls[0][2] == Timeframe("1h")


def test_repository_query_uses_the_recorded_identity_and_decision_instant() -> None:
    repository = StubRepository((_bar(20, "100"), _bar(21, "105")))

    _measure((), horizon=1, repository=repository)

    assert len(repository.queries) == 1
    query = repository.queries[0]
    assert query.symbol == _SYMBOL
    assert query.exchange_code == _EXCHANGE
    assert query.timeframe == _DAILY
    assert query.start == _DECISION_INSTANT


def test_measurement_matches_direct_measurement() -> None:
    observations = (_bar(20, "100"), _bar(21, "105"))
    repository = StubRepository(observations)
    measure = MeasureRecommendationOutcomeUseCase(repository)
    record = _record()

    direct = measure.execute(record.result, ResearchHorizon(1), _DAILY, _AVAILABLE_THROUGH)
    forward = MeasureForwardResearchRecordUseCase(measure).execute(
        record, ResearchHorizon(1), _AVAILABLE_THROUGH
    )

    assert forward.measurement == direct


# ---------------------------------------------------------------------------
# Coherence
# ---------------------------------------------------------------------------


def test_rejects_measurement_for_a_different_horizon() -> None:
    record = _record()
    measurement = MeasureRecommendationOutcomeUseCase(
        StubRepository((_bar(20, "100"), _bar(21, "105")))
    ).execute(record.result, ResearchHorizon(1), _DAILY, _AVAILABLE_THROUGH)

    with pytest.raises(ValueError, match="same horizon"):
        ForwardResearchMeasurement(
            record=record, horizon=ResearchHorizon(5), measurement=measurement
        )


def test_rejects_measurement_for_a_different_recommendation() -> None:
    record = _record(strategy="mvp")
    other = _record(strategy="momentum")
    measurement = MeasureRecommendationOutcomeUseCase(
        StubRepository((_bar(20, "100"), _bar(21, "105")))
    ).execute(other.result, ResearchHorizon(1), _DAILY, _AVAILABLE_THROUGH)

    with pytest.raises(ValueError, match="recorded recommendation"):
        ForwardResearchMeasurement(
            record=record, horizon=ResearchHorizon(1), measurement=measurement
        )


@pytest.mark.parametrize(
    "record, horizon, measurement, expected",
    [
        ("record", ResearchHorizon(1), None, "record must be a ForwardResearchRecord"),
        (None, ResearchHorizon(1), None, "record must be a ForwardResearchRecord"),
    ],
)
def test_rejects_invalid_member_types(
    record: object, horizon: object, measurement: object, expected: str
) -> None:
    with pytest.raises(TypeError, match=expected):
        ForwardResearchMeasurement(record=record, horizon=horizon, measurement=measurement)


def test_rejects_wrong_horizon_and_measurement_types() -> None:
    record = _record()

    with pytest.raises(TypeError, match="horizon must be a ResearchHorizon"):
        ForwardResearchMeasurement(record=record, horizon=1, measurement=None)
    with pytest.raises(TypeError, match="must be a RecommendationOutcomeMeasurement"):
        ForwardResearchMeasurement(
            record=record, horizon=ResearchHorizon(1), measurement="measurement"
        )


# ---------------------------------------------------------------------------
# Determinism and value semantics
# ---------------------------------------------------------------------------


def test_repeated_measurement_is_deterministic() -> None:
    observations = (_bar(20, "100"), _bar(21, "105"))
    record = _record()

    first = _measure(observations, horizon=1, record=record)
    second = _measure(observations, horizon=1, record=record)

    assert first == second
    assert first.state == second.state


def test_measurement_is_immutable() -> None:
    measurement = _measure((_bar(20, "100"), _bar(21, "105")), horizon=1)

    with pytest.raises(AttributeError):
        measurement.horizon = ResearchHorizon(5)


def test_measurement_applies_no_trading_semantics() -> None:
    measurement = _measure((_bar(20, "100"), _bar(21, "95")), horizon=1)

    for forbidden in ("pnl", "profit", "is_correct", "verdict", "trades", "positions"):
        assert not hasattr(measurement, forbidden)
