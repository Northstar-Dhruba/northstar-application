"""Tests for measuring a frozen futures forward research decision.

The repository answers each query as a conforming repository must, and its bars
can be extended between measurements to model a forward decision maturing as
new sessions are persisted. The frozen record never changes; only what can be
measured about it does.
"""

from __future__ import annotations

import ast
from decimal import Decimal
from functools import cmp_to_key
from pathlib import Path

import pytest
from northstar_core.derivatives import ExpirationDate, QuoteValue
from northstar_core.foundation.value_objects import (
    ExchangeCode,
    Percentage,
    PointInTime,
    Quantity,
    Symbol,
    Timeframe,
)
from northstar_core.futures import FuturesContract, FuturesOHLCVBar, FuturesProductReference
from northstar_core.strategy import (
    FuturesAssetAnalysis,
    FuturesMarketObservationContext,
    FuturesRecommendationOutcome,
    FuturesRecommendationOutcomeUnavailableReason,
    ResearchHorizon,
    Strategy,
    StrategyIdentity,
)

from northstar_application.application_services import (
    ForwardResearchMeasurementState,
    FuturesAnalysisResult,
    FuturesForwardResearchMeasurement,
    FuturesForwardResearchRecord,
    MeasureFuturesForwardResearchRecordUseCase,
    UnsupportedFuturesReplayTimeframeError,
)
from northstar_application.ports import (
    FuturesHistoricalMarketDataQuery,
    FuturesHistoricalMarketDataRepository,
)

_ES_DEC = FuturesContract(
    FuturesProductReference(Symbol("ES"), ExchangeCode("CME")), ExpirationDate("2026-12-18")
)
_DAILY = Timeframe("1d")
_H1, _H2, _H3 = ResearchHorizon(1), ResearchHorizon(2), ResearchHorizon(3)

_PENDING = ForwardResearchMeasurementState.PENDING
_MEASURED = ForwardResearchMeasurementState.MEASURED
_UNAVAILABLE = ForwardResearchMeasurementState.UNAVAILABLE
_INSUFFICIENT = FuturesRecommendationOutcomeUnavailableReason.INSUFFICIENT_FUTURE_OBSERVATIONS
_UNDEFINED = FuturesRecommendationOutcomeUnavailableReason.UNDEFINED_RETURN_BASIS

_D = "2026-09-15T21:00:00Z"
_F1 = "2026-09-16T21:00:00Z"
_F2 = "2026-09-17T21:00:00Z"
_F3 = "2026-09-18T21:00:00Z"
_F4 = "2026-09-21T21:00:00Z"
_F5 = "2026-09-22T21:00:00Z"


def _quote(value: str) -> QuoteValue:
    return QuoteValue(Decimal(value))


def _record(
    decision_quote: str = "100",
    *,
    observed_at: str = _D,
    signal: str = "strong bullish",
    strategy: str = "futures-forward",
    timeframe: Timeframe = _DAILY,
) -> FuturesForwardResearchRecord:
    """A frozen decision whose latest quote is the decision quote.

    Every other quote in the context differs from it, so a measurement that
    took its decision quote from the wrong field would be visible.
    """
    instant = PointInTime(observed_at)
    quote = _quote(decision_quote)
    context = FuturesMarketObservationContext(
        contract=_ES_DEC,
        timeframe=timeframe,
        observed_at=instant,
        latest_quote=quote,
        previous_close=QuoteValue(quote.value - 1),
        latest_volume=Quantity(Decimal("1000")),
        session_high=QuoteValue(quote.value + 2),
        session_low=QuoteValue(quote.value - 2),
        recent_closes=(_quote("12345"),) * 19 + (quote,),
        recent_volumes=(Quantity(Decimal("1000")),) * 20,
    )
    recommendation = Strategy(StrategyIdentity(strategy)).evaluate_futures(
        FuturesAssetAnalysis(_ES_DEC, instant, (signal,))
    )
    result = FuturesAnalysisResult(recommendation, context)
    if timeframe != _DAILY:
        return result  # type: ignore[return-value]  # a record cannot hold it
    return FuturesForwardResearchRecord(result)


def _bar(instant: str, close: str) -> FuturesOHLCVBar:
    quote = Decimal(close)
    return FuturesOHLCVBar(
        contract=_ES_DEC,
        point_in_time=PointInTime(instant),
        timeframe=_DAILY,
        open=QuoteValue(quote),
        high=QuoteValue(quote + 1),
        low=QuoteValue(quote - 1),
        close=QuoteValue(quote),
        volume=Quantity(Decimal("1000")),
    )


class ConformingRepository(FuturesHistoricalMarketDataRepository):
    def __init__(self, bars: tuple[FuturesOHLCVBar, ...] = ()) -> None:
        self.bars = bars
        self.queries: list[FuturesHistoricalMarketDataQuery] = []

    def get_bars(self, query):
        self.queries.append(query)
        matching = [bar for bar in self.bars if query.covers(bar.point_in_time)]
        return tuple(
            sorted(matching, key=cmp_to_key(lambda a, b: a.point_in_time.compare(b.point_in_time)))
        )


def _measure(
    repository: FuturesHistoricalMarketDataRepository,
    record: FuturesForwardResearchRecord,
    horizon: ResearchHorizon,
    through: str,
) -> FuturesForwardResearchMeasurement:
    return MeasureFuturesForwardResearchRecordUseCase(repository).execute(
        record, horizon, PointInTime(through)
    )


def _outcome(
    record: FuturesForwardResearchRecord,
    horizon: ResearchHorizon,
    evaluation: tuple[str, str] | None = None,
    *,
    decision_quote: QuoteValue | None = None,
) -> FuturesRecommendationOutcome:
    quote = (
        decision_quote
        if decision_quote is not None
        else record.result.market_observation_context.latest_quote
    )
    if evaluation is None:
        return FuturesRecommendationOutcome(record.result.recommendation, horizon, quote)
    instant, close = evaluation
    return FuturesRecommendationOutcome(
        record.result.recommendation, horizon, quote, PointInTime(instant), _quote(close)
    )


# ---------------------------------------------------------------------------
# The measurement value and its state
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("decision", "evaluation", "state"),
    [
        pytest.param("100", None, _PENDING, id="pending"),
        pytest.param("100", (_F1, "110"), _MEASURED, id="measured"),
        pytest.param("0", (_F1, "10"), _UNAVAILABLE, id="unavailable-zero"),
        pytest.param("-10", (_F1, "-5"), _UNAVAILABLE, id="unavailable-negative"),
        pytest.param("0", None, _PENDING, id="pending-zero-basis"),
        pytest.param("-10", None, _PENDING, id="pending-negative-basis"),
    ],
)
def test_the_state_is_derived_from_the_outcome_alone(
    decision: str, evaluation: tuple[str, str] | None, state: ForwardResearchMeasurementState
) -> None:
    record = _record(decision)

    measurement = FuturesForwardResearchMeasurement(record, _H1, _outcome(record, _H1, evaluation))

    assert measurement.state is state
    assert measurement.is_measured is (state is _MEASURED)


def test_only_the_three_existing_states_are_used() -> None:
    assert {state.value for state in ForwardResearchMeasurementState} == {
        "PENDING",
        "MEASURED",
        "UNAVAILABLE",
    }


def test_the_measurement_holds_exactly_record_horizon_and_outcome() -> None:
    record = _record()
    measurement = FuturesForwardResearchMeasurement(record, _H1, _outcome(record, _H1))

    assert FuturesForwardResearchMeasurement.__slots__ == ("record", "horizon", "outcome")
    for absent in ("contract", "strategy_identity", "decision_instant", "evaluation_instant"):
        assert absent not in FuturesForwardResearchMeasurement.__slots__
    with pytest.raises(AttributeError):
        measurement.horizon = _H2  # type: ignore[misc]
    with pytest.raises(AttributeError):
        measurement.state = _MEASURED  # type: ignore[misc]


@pytest.mark.parametrize("position", [0, 1, 2])
def test_every_field_is_type_checked(position: int) -> None:
    record = _record()
    fields: list[object] = [record, _H1, _outcome(record, _H1)]
    fields[position] = "not a value"

    with pytest.raises(TypeError):
        FuturesForwardResearchMeasurement(*fields)  # type: ignore[arg-type]


def test_an_outcome_for_another_record_is_rejected() -> None:
    record = _record()
    other = _record(strategy="another-strategy")

    with pytest.raises(ValueError, match="recorded recommendation"):
        FuturesForwardResearchMeasurement(record, _H1, _outcome(other, _H1))


def test_an_outcome_for_another_horizon_is_rejected() -> None:
    record = _record()

    with pytest.raises(ValueError, match="same horizon"):
        FuturesForwardResearchMeasurement(record, _H1, _outcome(record, _H2))


def test_an_outcome_measured_from_another_decision_quote_is_rejected() -> None:
    record = _record("100")
    forged = _outcome(record, _H1, (_F1, "110"), decision_quote=_quote("99"))

    with pytest.raises(ValueError, match="frozen decision evidence"):
        FuturesForwardResearchMeasurement(record, _H1, forged)


# ---------------------------------------------------------------------------
# PENDING and its transitions
# ---------------------------------------------------------------------------


def test_an_unreached_horizon_is_pending_and_repeats_equally() -> None:
    repository = ConformingRepository((_bar(_D, "100"), _bar(_F1, "101"), _bar(_F2, "102")))
    record = _record()

    first = _measure(repository, record, _H3, _F2)
    second = _measure(repository, record, _H3, _F2)

    assert first.state is _PENDING
    assert first.outcome.unavailable_reason == _INSUFFICIENT
    assert first.outcome.evaluation_instant is None
    assert second == first


def test_a_pending_decision_becomes_measured_when_its_horizon_bar_is_stored() -> None:
    repository = ConformingRepository((_bar(_D, "100"), _bar(_F1, "101"), _bar(_F2, "102")))
    record = _record("100")
    frozen_copy = _record("100")
    assert _measure(repository, record, _H3, _F2).state is _PENDING

    repository.bars = (*repository.bars, _bar(_F3, "130"))
    measured = _measure(repository, record, _H3, _F3)

    assert measured.state is _MEASURED
    assert measured.record == frozen_copy
    assert measured.horizon == _H3
    assert measured.outcome.evaluation_instant == PointInTime(_F3)
    assert measured.outcome.forward_return == Percentage(Decimal("30"))


def test_the_cutoff_decides_even_when_the_horizon_bar_is_already_stored() -> None:
    """Storing F3 changes nothing until the caller's cutoff admits it."""
    repository = ConformingRepository(
        (_bar(_D, "100"), _bar(_F1, "101"), _bar(_F2, "102"), _bar(_F3, "130"))
    )

    measurement = _measure(repository, _record(), _H3, _F2)

    assert measurement.state is _PENDING
    assert repository.queries[-1].end == PointInTime(_F2)


@pytest.mark.parametrize(
    ("decision", "evaluation"),
    [
        pytest.param("0", "10", id="zero"),
        pytest.param("-10", "-5", id="negative-to-negative"),
        pytest.param("-10", "10", id="negative-to-positive"),
        pytest.param("-37.63", "-20", id="negative-wti-like"),
    ],
)
def test_a_non_positive_decision_is_pending_until_its_horizon_then_unavailable(
    decision: str, evaluation: str
) -> None:
    repository = ConformingRepository((_bar(_D, decision),))
    record = _record(decision)

    before = _measure(repository, record, _H1, _D)
    repository.bars = (*repository.bars, _bar(_F1, evaluation))
    after = _measure(repository, record, _H1, _F1)

    assert before.state is _PENDING
    assert after.state is _UNAVAILABLE
    assert after.outcome.unavailable_reason == _UNDEFINED
    assert after.outcome.decision_quote == _quote(decision)
    assert after.outcome.evaluation_quote == _quote(evaluation)
    assert after.outcome.forward_return is None


def test_a_positive_decision_into_a_negative_quote_is_measured() -> None:
    repository = ConformingRepository((_bar(_D, "10"), _bar(_F1, "-5")))

    measurement = _measure(repository, _record("10"), _H1, _F1)

    assert measurement.state is _MEASURED
    assert measurement.outcome.forward_return == Percentage(Decimal("-150"))


# ---------------------------------------------------------------------------
# Terminal stability under append-only history
# ---------------------------------------------------------------------------


def test_bars_appended_after_the_selected_horizon_cannot_change_it() -> None:
    near = (_bar(_D, "100"), _bar(_F1, "101"), _bar(_F2, "105"))
    repository = ConformingRepository(near)
    record = _record()
    at_horizon = _measure(repository, record, _H2, _F2)

    repository.bars = (*near, _bar(_F3, "-99999"), _bar(_F4, "99999"), _bar(_F5, "0"))
    later = _measure(repository, record, _H2, _F5)

    assert later == at_horizon
    assert later.state is _MEASURED
    assert later.outcome.evaluation_instant == PointInTime(_F2)
    assert later.outcome.forward_return == Percentage(Decimal("5"))


def test_those_appended_bars_do_decide_a_longer_horizon() -> None:
    """Guard: the stability test is load-bearing only if the new bars matter."""
    repository = ConformingRepository(
        (
            _bar(_D, "100"),
            _bar(_F1, "101"),
            _bar(_F2, "105"),
            _bar(_F3, "-99999"),
            _bar(_F4, "99999"),
            _bar(_F5, "0"),
        )
    )
    record = _record()

    three = _measure(repository, record, _H3, _F5)
    five = _measure(repository, record, ResearchHorizon(5), _F5)

    assert three.outcome.evaluation_quote == _quote("-99999")
    assert (
        three.outcome.forward_return
        != _measure(repository, record, _H2, _F5).outcome.forward_return
    )
    assert five.outcome.forward_return == Percentage(Decimal("-100"))


# ---------------------------------------------------------------------------
# Sparse history, actions and determinism
# ---------------------------------------------------------------------------


def test_horizons_count_stored_sessions_not_calendar_days() -> None:
    repository = ConformingRepository(
        (
            _bar("2026-07-01T21:00:00Z", "100"),
            _bar("2026-07-02T21:00:00Z", "101"),
            _bar("2026-07-06T21:00:00Z", "106"),
            _bar("2026-07-09T21:00:00Z", "109"),
        )
    )
    record = _record(observed_at="2026-07-01T21:00:00Z")

    measurement = _measure(repository, record, _H2, "2026-07-09T21:00:00Z")

    assert measurement.outcome.evaluation_instant == PointInTime("2026-07-06T21:00:00Z")


def test_buy_hold_and_sell_are_measured_identically() -> None:
    repository = ConformingRepository((_bar(_D, "100"), _bar(_F1, "90"), _bar(_F2, "95")))
    records = [
        _record(signal=signal) for signal in ("strong bullish", "neutral trend", "strong bearish")
    ]
    assert [record.result.recommendation.action.value for record in records] == [
        "BUY",
        "HOLD",
        "SELL",
    ]

    measurements = [_measure(repository, record, _H1, _F2) for record in records]

    assert {measurement.state for measurement in measurements} == {_MEASURED}
    assert {measurement.outcome.forward_return for measurement in measurements} == {
        Percentage(Decimal("-10"))
    }
    assert len({measurement.outcome.evaluation_instant for measurement in measurements}) == 1
    assert len({measurement.record for measurement in measurements}) == 3  # only the records differ


def test_the_same_inputs_measure_identically() -> None:
    bars = (_bar(_D, "100"), _bar(_F1, "101"), _bar(_F2, "102"))

    first = _measure(ConformingRepository(bars), _record(), _H2, _F2)
    second = _measure(ConformingRepository(bars), _record(), _H2, _F2)

    assert first == second
    assert repr(first) == repr(second)


# ---------------------------------------------------------------------------
# The cutoff and semantic timestamps
# ---------------------------------------------------------------------------


def test_a_cutoff_at_the_decision_instant_is_valid_and_pending() -> None:
    repository = ConformingRepository((_bar(_D, "100"),))

    measurement = _measure(repository, _record(), _H1, _D)

    assert measurement.state is _PENDING


@pytest.mark.parametrize(
    "through", ["2026-09-14T21:00:00Z", "2026-09-15T20:59:59.5Z", "2026-09-15T15:59:59-05:00"]
)
def test_a_cutoff_before_the_decision_is_rejected_before_any_read(through: str) -> None:
    repository = ConformingRepository((_bar(_D, "100"),))

    with pytest.raises(ValueError, match="precedes the decision instant"):
        _measure(repository, _record(), _H1, through)

    assert repository.queries == []


def test_an_offset_spelled_cutoff_measures_identically() -> None:
    repository = ConformingRepository((_bar(_D, "100"), _bar(_F1, "110")))

    z = _measure(repository, _record(), _H1, _F1)
    offset = _measure(repository, _record(), _H1, "2026-09-16T16:00:00-05:00")

    assert offset == z
    assert offset.state is _MEASURED


def test_a_sub_second_later_bar_is_the_first_future_observation() -> None:
    """Lexically ``...00.5Z`` < ``...00Z``; only semantic ordering selects it."""
    half_second = "2026-09-15T21:00:00.5Z"
    assert half_second < _D
    repository = ConformingRepository((_bar(half_second, "110"), _bar(_D, "100")))

    measurement = _measure(repository, _record(), _H1, half_second)

    assert measurement.state is _MEASURED
    assert measurement.outcome.evaluation_instant == PointInTime(half_second)


# ---------------------------------------------------------------------------
# Delegation, validation and boundaries
# ---------------------------------------------------------------------------


def test_the_repository_is_read_through_the_historical_measurement_path() -> None:
    repository = ConformingRepository((_bar(_D, "100"), _bar(_F1, "110")))
    record = _record()

    _measure(repository, record, _H1, _F1)

    assert repository.queries == [
        FuturesHistoricalMarketDataQuery(_ES_DEC, _DAILY, PointInTime(_D), PointInTime(_F1))
    ]


def test_the_measurement_is_exactly_the_historical_outcome_wrapped() -> None:
    from northstar_application.application_services import (
        MeasureFuturesRecommendationOutcomeUseCase,
    )

    bars = (_bar(_D, "100"), _bar(_F1, "101"), _bar(_F2, "103"))
    record = _record()

    measurement = _measure(ConformingRepository(bars), record, _H2, _F2)
    direct = MeasureFuturesRecommendationOutcomeUseCase(ConformingRepository(bars)).execute(
        record.result, _H2, PointInTime(_F2)
    )

    assert measurement.outcome == direct


@pytest.mark.parametrize("position", [0, 1, 2])
def test_every_input_is_type_checked_before_any_read(position: int) -> None:
    arguments: list[object] = [_record(), _H1, PointInTime(_F1)]
    arguments[position] = None
    repository = ConformingRepository((_bar(_D, "100"),))

    # The error must name this use case, not the historical one it delegates to.
    with pytest.raises(TypeError, match="^MeasureFuturesForwardResearchRecordUseCase "):
        MeasureFuturesForwardResearchRecordUseCase(repository).execute(*arguments)  # type: ignore[arg-type]

    assert repository.queries == []


def test_a_bare_analysis_result_is_not_a_record() -> None:
    with pytest.raises(TypeError, match="FuturesForwardResearchRecord"):
        MeasureFuturesForwardResearchRecordUseCase(ConformingRepository()).execute(
            _record().result,  # type: ignore[arg-type]
            _H1,
            PointInTime(_F1),
        )


def test_a_non_daily_result_cannot_even_be_submitted() -> None:
    """Records reject non-daily results, so measurement never sees one."""
    with pytest.raises(UnsupportedFuturesReplayTimeframeError):
        FuturesForwardResearchRecord(_record(timeframe=Timeframe("1m")))  # type: ignore[arg-type]


@pytest.mark.parametrize("repository", [None, "repository", object()])
def test_the_repository_is_type_checked(repository: object) -> None:
    with pytest.raises(TypeError, match="FuturesHistoricalMarketDataRepository"):
        MeasureFuturesForwardResearchRecordUseCase(repository)  # type: ignore[arg-type]


def _module_tree() -> ast.Module:
    import northstar_application.application_services.measure_futures_forward_research_record as m

    return ast.parse(Path(m.__file__).read_text(encoding="utf-8"))


def test_the_use_case_delegates_rather_than_reimplementing_measurement() -> None:
    tree = _module_tree()
    names = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }

    assert "MeasureFuturesRecommendationOutcomeUseCase" in names
    for reimplemented in (
        "FuturesHistoricalMarketDataQuery",
        "validate_futures_repository_bars",
        "Percentage",
        "Decimal",
        "localcontext",
    ):
        assert reimplemented not in names
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and not isinstance(node.op, ast.BitOr):
            raise AssertionError(f"arithmetic at line {node.lineno}")
        if isinstance(node, ast.Attribute):
            assert node.attr not in {"action", "value", "now", "today", "utcnow"}


def test_the_use_case_depends_on_no_provider_or_storage() -> None:
    tree = _module_tree()
    modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    } | {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    names = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }

    for forbidden in (
        "FuturesHistoricalMarketDataSource",
        "AcquireFuturesDailyHistoryUseCase",
        "FuturesTradingSessionResolver",
        "FuturesForwardResearchRecordStore",
        "FuturesForwardResearchRecordRepository",
    ):
        assert forbidden not in names
    for module_path in modules:
        assert module_path.split(".")[0] not in {
            "databento",
            "exchange_calendars",
            "northstar_infrastructure",
            "sqlite3",
            "requests",
            "socket",
            "time",
            "datetime",
            "random",
        }
        assert "paper_trading" not in module_path


def test_the_measurement_and_use_case_are_exported() -> None:
    import northstar_application.application_services as services

    for name in ("FuturesForwardResearchMeasurement", "MeasureFuturesForwardResearchRecordUseCase"):
        assert name in services.__all__
    assert "FuturesForwardResearchMeasurementState" not in services.__all__
    for private in ("_INSUFFICIENT", "_UNDEFINED_BASIS"):
        assert not hasattr(services, private)
