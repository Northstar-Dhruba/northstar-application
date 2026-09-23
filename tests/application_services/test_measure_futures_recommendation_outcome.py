"""Tests for measuring one futures recommendation outcome.

The use case selects the horizon bar and hands Core the facts. These tests pin
the selection -- strictly later bars only, counted rather than dated, never
widened past the Nth, never past the caller's evidence window -- and prove
through the Application path that Core's return and reason semantics arrive
unchanged, whatever the recommendation's action and whatever the sign of the
decision quote.

Every measurement names its evidence window explicitly. Where a test means to
measure all of the history it supplies, the window ends at that history's last
instant rather than at some distant sentinel.
"""

from __future__ import annotations

import ast
import inspect
from decimal import Decimal
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
    RecommendationAction,
    ResearchHorizon,
    Strategy,
    StrategyIdentity,
)

from northstar_application.application_services import (
    FuturesAnalysisResult,
    FuturesHistoricalDataContractViolationError,
    MeasureFuturesRecommendationOutcomeUseCase,
    UnsupportedFuturesReplayTimeframeError,
)
from northstar_application.ports import (
    FuturesHistoricalMarketDataQuery,
    FuturesHistoricalMarketDataRepository,
)

_CME = ExchangeCode("CME")
_ES = FuturesProductReference(Symbol("ES"), _CME)
_ES_DEC = FuturesContract(_ES, ExpirationDate("2026-12-18"))
_DAILY = Timeframe("1d")
_STRATEGY = Strategy(StrategyIdentity("futures-directional"))

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


def _bar(
    instant: str,
    close: str,
    *,
    contract: FuturesContract = _ES_DEC,
    timeframe: Timeframe = _DAILY,
) -> FuturesOHLCVBar:
    quote = Decimal(close)
    return FuturesOHLCVBar(
        contract=contract,
        point_in_time=PointInTime(instant),
        timeframe=timeframe,
        open=QuoteValue(quote),
        high=QuoteValue(quote + 1),
        low=QuoteValue(quote - 1),
        close=QuoteValue(quote),
        volume=Quantity(Decimal("1000")),
    )


def _result(
    decision_quote: str = "100",
    *,
    signal: str = "neutral trend",
    observed_at: str = _D,
    contract: FuturesContract = _ES_DEC,
    timeframe: Timeframe = _DAILY,
) -> FuturesAnalysisResult:
    """A coherent analysis result whose latest quote is the decision quote.

    Every other quote in the context differs from the decision quote, so a
    measurement that read the wrong context field would be visible.
    """
    instant = PointInTime(observed_at)
    quote = _quote(decision_quote)
    context = FuturesMarketObservationContext(
        contract=contract,
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
    recommendation = _STRATEGY.evaluate_futures(FuturesAssetAnalysis(contract, instant, (signal,)))
    return FuturesAnalysisResult(recommendation, context)


class StubRepository(FuturesHistoricalMarketDataRepository):
    """Returns exactly what it is given, conforming or not."""

    def __init__(self, bars: object = ()) -> None:
        self.bars = bars
        self.queries: list[FuturesHistoricalMarketDataQuery] = []

    def get_bars(self, query):
        self.queries.append(query)
        return self.bars


class ConformingRepository(FuturesHistoricalMarketDataRepository):
    """Holds stored history and answers each query as a real repository must."""

    def __init__(self, bars: tuple[FuturesOHLCVBar, ...] = ()) -> None:
        self.bars = bars
        self.queries: list[FuturesHistoricalMarketDataQuery] = []

    def get_bars(self, query):
        self.queries.append(query)
        return tuple(bar for bar in self.bars if query.covers(bar.point_in_time))


class FailingRepository(FuturesHistoricalMarketDataRepository):
    def get_bars(self, query):
        raise RuntimeError("storage unavailable")


def _measure(
    bars: object,
    observations: int,
    result: FuturesAnalysisResult | None = None,
    *,
    through: str,
) -> FuturesRecommendationOutcome:
    """Measure over ``bars`` with the evidence window ending at ``through``."""
    return MeasureFuturesRecommendationOutcomeUseCase(StubRepository(bars)).execute(
        result if result is not None else _result(),
        ResearchHorizon(observations),
        PointInTime(through),
    )


_WITH_DECISION_BAR = (_bar(_D, "100"), _bar(_F1, "101"), _bar(_F2, "102"), _bar(_F3, "103"))
_WITHOUT_DECISION_BAR = _WITH_DECISION_BAR[1:]


# ---------------------------------------------------------------------------
# Repository query
# ---------------------------------------------------------------------------


def test_the_query_is_bounded_by_the_decision_and_the_evidence_window() -> None:
    repository = StubRepository(_WITH_DECISION_BAR)
    result = _result()

    MeasureFuturesRecommendationOutcomeUseCase(repository).execute(
        result, ResearchHorizon(1), PointInTime(_F3)
    )

    (query,) = repository.queries
    assert query.contract == result.recommendation.contract
    assert query.timeframe == result.market_observation_context.timeframe
    assert query.start is not None
    assert query.end is not None
    assert query.start.compare(result.recommendation.point_in_time) == 0
    assert query.end.compare(PointInTime(_F3)) == 0


def test_the_query_follows_the_result_contract_exactly() -> None:
    mar = FuturesContract(_ES, ExpirationDate("2027-03-19"))
    repository = StubRepository(())

    MeasureFuturesRecommendationOutcomeUseCase(repository).execute(
        _result(contract=mar), ResearchHorizon(1), PointInTime(_F1)
    )

    assert repository.queries[0].contract == mar


def test_an_offset_spelled_window_bounds_the_query_at_the_same_instant() -> None:
    repository = StubRepository((_bar(_F1, "101"),))

    outcome = MeasureFuturesRecommendationOutcomeUseCase(repository).execute(
        _result(), ResearchHorizon(1), PointInTime("2026-09-16T16:00:00-05:00")
    )

    assert repository.queries[0].end.compare(PointInTime(_F1)) == 0
    assert outcome.evaluation_instant == PointInTime(_F1)


# ---------------------------------------------------------------------------
# available_through
# ---------------------------------------------------------------------------


def test_available_through_is_required() -> None:
    parameter = inspect.signature(MeasureFuturesRecommendationOutcomeUseCase.execute).parameters[
        "available_through"
    ]

    assert parameter.default is inspect.Parameter.empty


@pytest.mark.parametrize("value", [None, _F1, _DAILY], ids=["none", "text", "timeframe"])
def test_available_through_must_be_a_point_in_time(value: object) -> None:
    repository = StubRepository(())

    with pytest.raises(TypeError, match="available-through"):
        MeasureFuturesRecommendationOutcomeUseCase(repository).execute(
            _result(),
            ResearchHorizon(1),
            value,  # type: ignore[arg-type]
        )
    assert repository.queries == []


@pytest.mark.parametrize(
    "through",
    [
        pytest.param("2026-09-14T21:00:00Z", id="a-day-before"),
        pytest.param("2026-09-15T20:59:59.5Z", id="half-a-second-before"),
        pytest.param("2026-09-15T15:59:59-05:00", id="offset-spelled-one-second-before"),
    ],
)
def test_a_window_ending_before_the_decision_is_rejected_before_any_read(through: str) -> None:
    repository = StubRepository(())

    with pytest.raises(ValueError, match="precedes the decision instant"):
        MeasureFuturesRecommendationOutcomeUseCase(repository).execute(
            _result(), ResearchHorizon(1), PointInTime(through)
        )
    assert repository.queries == []


@pytest.mark.parametrize("through", [_D, "2026-09-15T16:00:00-05:00"], ids=["z", "offset"])
def test_a_window_ending_at_the_decision_is_valid_and_insufficient(through: str) -> None:
    outcome = _measure((_bar(_D, "100"),), 1, through=through)

    assert outcome.unavailable_reason == _INSUFFICIENT
    assert outcome.evaluation_quote is None


def test_a_window_ending_at_f1_reaches_horizon_one_only() -> None:
    use_case = MeasureFuturesRecommendationOutcomeUseCase(ConformingRepository(_WITH_DECISION_BAR))

    one = use_case.execute(_result(), ResearchHorizon(1), PointInTime(_F1))
    two = use_case.execute(_result(), ResearchHorizon(2), PointInTime(_F1))

    assert one.evaluation_instant == PointInTime(_F1)
    assert two.unavailable_reason == _INSUFFICIENT


def test_a_window_ending_at_f2_reaches_horizon_two() -> None:
    use_case = MeasureFuturesRecommendationOutcomeUseCase(ConformingRepository(_WITH_DECISION_BAR))

    outcome = use_case.execute(_result(), ResearchHorizon(2), PointInTime(_F2))

    assert outcome.evaluation_instant == PointInTime(_F2)
    assert outcome.evaluation_quote == _quote("102")


def test_a_bar_half_a_second_past_the_window_is_a_contract_violation() -> None:
    """A conforming repository never returns it; one that does is refused, not filtered."""
    past = "2026-09-16T21:00:00.5Z"
    assert past < _F1  # a text comparison would have admitted it

    with pytest.raises(FuturesHistoricalDataContractViolationError, match="outside"):
        _measure((_bar(_F1, "101"), _bar(past, "150")), 1, through=_F1)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def test_the_same_window_measures_the_same_outcome_after_more_history_is_stored() -> None:
    horizon = ResearchHorizon(3)
    window = PointInTime(_F2)
    result = _result()
    repository = ConformingRepository((_bar(_D, "100"), _bar(_F1, "101"), _bar(_F2, "102")))
    use_case = MeasureFuturesRecommendationOutcomeUseCase(repository)

    first = use_case.execute(result, horizon, window)
    assert first.unavailable_reason == _INSUFFICIENT

    # A later acquisition stores F3..F5, every one of them after the window.
    repository.bars = (
        *repository.bars,
        _bar(_F3, "130"),
        _bar(_F4, "140"),
        _bar(_F5, "150"),
    )
    rerun = use_case.execute(result, horizon, window)

    assert rerun == first
    assert rerun.unavailable_reason == _INSUFFICIENT

    # Extending the window is an explicit act, and only then does F3 count.
    extended = use_case.execute(result, horizon, PointInTime(_F3))

    assert extended.evaluation_instant == PointInTime(_F3)
    assert extended.forward_return == Percentage(Decimal("30"))
    assert [query.end for query in repository.queries] == [window, window, PointInTime(_F3)]


# ---------------------------------------------------------------------------
# Daily-only measurement
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("timeframe", [Timeframe("1m"), Timeframe("1h")], ids=["1m", "1h"])
def test_a_non_daily_result_is_refused_before_the_repository_is_read(
    timeframe: Timeframe,
) -> None:
    repository = StubRepository((_bar(_F1, "101", timeframe=timeframe),))

    with pytest.raises(UnsupportedFuturesReplayTimeframeError, match="session-daily") as raised:
        MeasureFuturesRecommendationOutcomeUseCase(repository).execute(
            _result(timeframe=timeframe), ResearchHorizon(1), PointInTime(_F1)
        )

    assert repository.queries == []
    assert str(timeframe) in str(raised.value)


def test_the_replay_timeframe_error_is_reused_not_redefined() -> None:
    import northstar_application.application_services.measure_futures_recommendation_outcome as m
    import northstar_application.application_services.replay_futures_historical_market_data as r

    assert m.UnsupportedFuturesReplayTimeframeError is r.UnsupportedFuturesReplayTimeframeError
    tree = ast.parse(Path(m.__file__).read_text(encoding="utf-8"))
    assert not [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name.endswith("Error")
    ]


# ---------------------------------------------------------------------------
# Strict-future selection and horizon semantics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("observations", "instant", "close"),
    [(1, _F1, "101"), (2, _F2, "102"), (3, _F3, "103")],
)
def test_horizon_n_selects_the_nth_bar_strictly_after_the_decision(
    observations: int, instant: str, close: str
) -> None:
    outcome = _measure(_WITH_DECISION_BAR, observations, through=_F3)

    assert outcome.evaluation_instant == PointInTime(instant)
    assert outcome.evaluation_quote == _quote(close)
    assert outcome.horizon == ResearchHorizon(observations)


@pytest.mark.parametrize("observations", [1, 2, 3, 4])
def test_the_decision_bar_never_counts_as_a_future_observation(observations: int) -> None:
    assert _measure(_WITH_DECISION_BAR, observations, through=_F3) == _measure(
        _WITHOUT_DECISION_BAR, observations, through=_F3
    )


def test_a_stored_decision_bar_does_not_replace_the_decision_evidence() -> None:
    """The result's latest quote produced the view; a stored bar cannot rewrite it."""
    bars = (_bar(_D, "999"), _bar(_F1, "110"))

    outcome = _measure(bars, 1, _result("100"), through=_F1)

    assert outcome.decision_quote == _quote("100")
    assert outcome.forward_return == Percentage(Decimal("10"))


def test_sparse_history_counts_bars_not_calendar_days() -> None:
    bars = (
        _bar("2026-07-01T21:00:00Z", "100"),
        _bar("2026-07-02T21:00:00Z", "101"),
        _bar("2026-07-06T21:00:00Z", "106"),
        _bar("2026-07-09T21:00:00Z", "109"),
    )
    result = _result(observed_at="2026-07-01T21:00:00Z")
    through = "2026-07-09T21:00:00Z"

    def evaluation(n: int) -> FuturesRecommendationOutcome:
        return _measure(bars, n, result, through=through)

    assert evaluation(1).evaluation_instant == PointInTime("2026-07-02T21:00:00Z")
    assert evaluation(2).evaluation_instant == PointInTime("2026-07-06T21:00:00Z")
    assert evaluation(3).evaluation_instant == PointInTime("2026-07-09T21:00:00Z")
    assert evaluation(4).unavailable_reason == _INSUFFICIENT


def test_the_decision_quote_is_the_result_latest_quote_and_the_evaluation_is_the_close() -> None:
    outcome = _measure(_WITHOUT_DECISION_BAR, 2, _result("7663.25"), through=_F3)

    assert outcome.decision_quote == _quote("7663.25")
    assert outcome.evaluation_quote == _WITHOUT_DECISION_BAR[1].close
    assert outcome.recommendation == _result("7663.25").recommendation
    assert type(outcome) is FuturesRecommendationOutcome


# ---------------------------------------------------------------------------
# Insufficient future observations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("bars", "observations", "through"),
    [
        pytest.param((), 1, _D, id="no-bars"),
        pytest.param((_bar(_D, "100"),), 1, _D, id="only-the-decision-bar"),
        pytest.param(_WITH_DECISION_BAR, 4, _F3, id="one-short"),
        pytest.param(_WITHOUT_DECISION_BAR, 30, _F3, id="far-short"),
    ],
)
def test_too_few_later_bars_is_an_insufficient_outcome(
    bars: tuple, observations: int, through: str
) -> None:
    outcome = _measure(bars, observations, through=through)

    assert outcome.unavailable_reason == _INSUFFICIENT
    assert outcome.forward_return is None
    assert outcome.evaluation_instant is None
    assert outcome.evaluation_quote is None
    assert outcome.decision_quote == _quote("100")


@pytest.mark.parametrize("decision", ["0", "-10", "-37.63"])
def test_insufficiency_takes_precedence_over_the_basis_through_the_application_path(
    decision: str,
) -> None:
    outcome = _measure((_bar(_D, decision),), 1, _result(decision), through=_D)

    assert outcome.unavailable_reason == _INSUFFICIENT


# ---------------------------------------------------------------------------
# Return basis and signed outcomes: Core semantics arriving unchanged
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("decision", "evaluation"),
    [("0", "10"), ("0", "0"), ("-10", "-5"), ("-10", "10"), ("-37.63", "-20")],
)
def test_a_non_positive_decision_quote_has_an_undefined_basis(
    decision: str, evaluation: str
) -> None:
    outcome = _measure((_bar(_F1, evaluation),), 1, _result(decision), through=_F1)

    assert outcome.unavailable_reason == _UNDEFINED
    assert outcome.forward_return is None
    assert outcome.decision_quote == _quote(decision)
    assert outcome.evaluation_quote == _quote(evaluation)
    assert outcome.evaluation_instant == PointInTime(_F1)


@pytest.mark.parametrize(
    ("decision", "evaluation", "percent"),
    [("100", "110", "10"), ("100", "90", "-10"), ("10", "0", "-100"), ("10", "-5", "-150")],
)
def test_a_positive_decision_quote_is_measured_even_into_negative_quotes(
    decision: str, evaluation: str, percent: str
) -> None:
    outcome = _measure((_bar(_F1, evaluation),), 1, _result(decision), through=_F1)

    assert outcome.forward_return == Percentage(Decimal(percent))
    assert outcome.unavailable_reason is None


def test_the_outcome_is_exactly_what_core_builds_from_the_selected_facts() -> None:
    result = _result("100")

    outcome = _measure(_WITHOUT_DECISION_BAR, 2, result, through=_F3)

    assert outcome == FuturesRecommendationOutcome(
        recommendation=result.recommendation,
        horizon=ResearchHorizon(2),
        decision_quote=_quote("100"),
        evaluation_instant=PointInTime(_F2),
        evaluation_quote=_quote("102"),
    )


# ---------------------------------------------------------------------------
# Action independence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("bars", "observations", "through"), [(_WITH_DECISION_BAR, 2, _F3), ((), 1, _D)]
)
def test_buy_hold_and_sell_are_measured_identically(
    bars: tuple, observations: int, through: str
) -> None:
    results = [
        _result("100", signal=signal)
        for signal in ("strong bullish", "neutral trend", "strong bearish")
    ]
    assert [result.recommendation.action for result in results] == [
        RecommendationAction("BUY"),
        RecommendationAction("HOLD"),
        RecommendationAction("SELL"),
    ]

    outcomes = [_measure(bars, observations, result, through=through) for result in results]

    assert len({outcome.evaluation_instant for outcome in outcomes}) == 1
    assert len({outcome.evaluation_quote for outcome in outcomes}) == 1
    assert len({outcome.forward_return for outcome in outcomes}) == 1
    assert len({outcome.unavailable_reason for outcome in outcomes}) == 1


# ---------------------------------------------------------------------------
# Look-ahead: only the first N later bars decide the outcome
# ---------------------------------------------------------------------------


def test_bars_after_the_horizon_do_not_change_the_measured_outcome() -> None:
    near = (_bar(_F1, "101"), _bar(_F2, "105"))
    divergent = (_bar(_F3, "-99999"), _bar(_F4, "99999"), _bar(_F5, "0"))

    without = _measure(near, 2, through=_F2)
    with_later = _measure(near + divergent, 2, through=_F5)

    assert with_later == without
    assert with_later.forward_return == Percentage(Decimal("5"))


def test_the_later_bars_really_would_change_a_longer_horizon() -> None:
    """Guard: the look-ahead test is load-bearing only if this holds."""
    history = (
        _bar(_F1, "101"),
        _bar(_F2, "105"),
        _bar(_F3, "-99999"),
        _bar(_F4, "99999"),
        _bar(_F5, "0"),
    )

    two = _measure(history, 2, through=_F5)
    three = _measure(history, 3, through=_F5)

    assert three.forward_return != two.forward_return
    assert _measure(history, 5, through=_F5).forward_return == Percentage(Decimal("-100"))


# ---------------------------------------------------------------------------
# Semantic timestamps
# ---------------------------------------------------------------------------


def test_a_decision_in_offset_spelling_does_not_count_its_own_z_spelled_bar() -> None:
    result = _result(observed_at="2026-09-15T16:00:00-05:00")
    bars = (_bar("2026-09-15T21:00:00Z", "100"), _bar(_F1, "101"))

    outcome = _measure(bars, 1, result, through=_F1)

    assert outcome.evaluation_instant == PointInTime(_F1)


def test_a_bar_half_a_second_after_the_decision_is_the_first_future_observation() -> None:
    """Lexically ``...00.5Z`` < ``...00Z``; a text comparison would drop this bar."""
    half_second = "2026-09-15T21:00:00.5Z"
    assert half_second < _D

    outcome = _measure((_bar(_D, "100"), _bar(half_second, "110")), 1, through=half_second)

    assert outcome.evaluation_instant == PointInTime(half_second)
    assert outcome.forward_return == Percentage(Decimal("10"))


def test_fractional_instants_are_ordered_semantically() -> None:
    bars = (
        _bar("2026-09-15T21:00:00.25Z", "101"),
        _bar("2026-09-15T21:00:00.5Z", "102"),
        _bar("2026-09-15T21:00:01Z", "103"),
    )
    through = "2026-09-15T21:00:01Z"

    quotes = [_measure(bars, n, through=through).evaluation_quote for n in (1, 2, 3)]

    assert quotes == [_quote("101"), _quote("102"), _quote("103")]


# ---------------------------------------------------------------------------
# Repository-output validation and cross-contract isolation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "intruder",
    [
        pytest.param(FuturesContract(_ES, ExpirationDate("2027-03-19")), id="es-march-2027"),
        pytest.param(
            FuturesContract(
                FuturesProductReference(Symbol("MES"), _CME), ExpirationDate("2026-12-18")
            ),
            id="mes-same-expiry",
        ),
        pytest.param(
            FuturesContract(
                FuturesProductReference(Symbol("ES"), ExchangeCode("CBOT")),
                ExpirationDate("2026-12-18"),
            ),
            id="another-exchange",
        ),
        pytest.param(FuturesContract(_ES, ExpirationDate("2026-12-17")), id="adjacent-expiry"),
    ],
)
def test_bars_for_another_contract_are_a_contract_violation(intruder: FuturesContract) -> None:
    bars = (_bar(_F1, "101"), _bar(_F2, "102", contract=intruder))

    with pytest.raises(
        FuturesHistoricalDataContractViolationError, match="not the queried contract"
    ):
        _measure(bars, 1, through=_F2)


@pytest.mark.parametrize(
    ("bars", "message"),
    [
        pytest.param([_bar(_F1, "101")], "must return a tuple", id="list"),
        pytest.param((_bar(_F1, "101"), "bar"), "must be a FuturesOHLCVBar", id="non-bar"),
        pytest.param(
            (_bar(_F1, "101", timeframe=Timeframe("1m")),), "timeframe", id="other-timeframe"
        ),
        pytest.param((_bar("2026-09-14T21:00:00Z", "99"),), "outside", id="before-the-decision"),
        pytest.param((_bar(_F3, "103"),), "outside", id="after-the-window"),
        pytest.param((_bar(_F1, "101"), _bar(_F1, "101")), "share the instant", id="duplicate"),
        pytest.param((_bar(_F2, "102"), _bar(_F1, "101")), "oldest to newest", id="unsorted"),
    ],
)
def test_malformed_repository_output_is_refused_not_repaired(bars: object, message: str) -> None:
    with pytest.raises(FuturesHistoricalDataContractViolationError, match=message):
        _measure(bars, 1, through=_F2)


def test_replay_and_measurement_share_one_repository_output_rule() -> None:
    import northstar_application.application_services.measure_futures_recommendation_outcome as m
    import northstar_application.application_services.replay_futures_historical_market_data as r
    from northstar_application.application_services._futures_repository_output import (
        validate_futures_repository_bars,
    )

    assert m.validate_futures_repository_bars is validate_futures_repository_bars
    assert r.validate_futures_repository_bars is validate_futures_repository_bars


def test_repository_failures_propagate_unchanged() -> None:
    use_case = MeasureFuturesRecommendationOutcomeUseCase(FailingRepository())

    with pytest.raises(RuntimeError, match="storage unavailable"):
        use_case.execute(_result(), ResearchHorizon(1), PointInTime(_F1))


# ---------------------------------------------------------------------------
# Determinism, construction and boundaries
# ---------------------------------------------------------------------------


def test_the_same_inputs_measure_identically() -> None:
    use_case = MeasureFuturesRecommendationOutcomeUseCase(StubRepository(_WITH_DECISION_BAR))

    first = use_case.execute(_result(), ResearchHorizon(2), PointInTime(_F3))
    second = use_case.execute(_result(), ResearchHorizon(2), PointInTime(_F3))
    fresh = _measure(_WITH_DECISION_BAR, 2, through=_F3)

    assert first == second == fresh
    assert repr(first) == repr(fresh)


@pytest.mark.parametrize("repository", [None, "repository", object()])
def test_the_repository_is_type_checked(repository: object) -> None:
    with pytest.raises(TypeError, match="FuturesHistoricalMarketDataRepository"):
        MeasureFuturesRecommendationOutcomeUseCase(repository)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [None, "result", object()])
def test_the_result_is_type_checked(value: object) -> None:
    with pytest.raises(TypeError, match="FuturesAnalysisResult"):
        MeasureFuturesRecommendationOutcomeUseCase(StubRepository()).execute(
            value,  # type: ignore[arg-type]
            ResearchHorizon(1),
            PointInTime(_F1),
        )


@pytest.mark.parametrize("value", [None, 1, "1"])
def test_the_horizon_is_type_checked(value: object) -> None:
    with pytest.raises(TypeError, match="ResearchHorizon"):
        MeasureFuturesRecommendationOutcomeUseCase(StubRepository()).execute(
            _result(),
            value,  # type: ignore[arg-type]
            PointInTime(_F1),
        )


def _module_tree() -> ast.Module:
    import northstar_application.application_services.measure_futures_recommendation_outcome as m

    return ast.parse(Path(m.__file__).read_text(encoding="utf-8"))


def test_the_use_case_reads_persisted_history_only() -> None:
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
        "FuturesHistoricalMarketDataStore",
        "FuturesTradingSessionResolver",
        "AcquireFuturesDailyHistoryUseCase",
        "AggregateFuturesDailySessionBarUseCase",
        "ReplayFuturesHistoricalMarketDataUseCase",
    ):
        assert forbidden not in names
    for module in modules:
        assert module.split(".")[0] not in {
            "databento",
            "exchange_calendars",
            "sqlite3",
            "northstar_infrastructure",
            "requests",
            "urllib",
            "socket",
            "time",
            "datetime",
            "random",
            "decimal",
        }
        assert "paper_trading" not in module
        assert "execution" not in module


def test_the_use_case_never_reads_the_action_or_a_clock_and_computes_no_return() -> None:
    for node in ast.walk(_module_tree()):
        if isinstance(node, ast.Attribute):
            assert node.attr not in {"action", "value", "forward_return", "unavailable_reason"}
        if isinstance(node, ast.Name):
            assert node.id not in {"RecommendationAction", "Percentage", "Decimal"}
        if isinstance(node, ast.BinOp):
            # ``observations - 1`` is the only arithmetic: an index, not a return.
            assert isinstance(node.op, ast.Sub)
            assert isinstance(node.right, ast.Constant)
            assert node.right.value == 1
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in {"sorted", "reversed", "set"}
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert node.func.attr not in {"sort", "now", "today", "utcnow", "time"}


def test_the_use_case_is_exported_without_a_measurement_wrapper() -> None:
    import northstar_application.application_services as services

    assert "MeasureFuturesRecommendationOutcomeUseCase" in services.__all__
    for absent in (
        "FuturesRecommendationOutcomeMeasurement",
        "FuturesRecommendationOutcomeUnavailableReason",
        "validate_futures_repository_bars",
    ):
        assert absent not in services.__all__
