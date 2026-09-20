"""Tests for factual recommendation outcome measurement."""

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
    AssetAnalysis,
    AssetAnalysisGenerator,
    ExplanationReason,
    MarketObservationContext,
    Recommendation,
    RecommendationAction,
    RecommendationExplanation,
    RecommendationOutcome,
    ResearchHorizon,
    Strategy,
    StrategyIdentity,
)

from northstar_application.application_services import (
    AnalyzeAssetResult,
    AnalyzeMarketObservationContextService,
    HistoricalDataContractViolationError,
    HistoricalResearchEvaluation,
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
_INSUFFICIENT = RecommendationOutcomeUnavailableReason.INSUFFICIENT_FUTURE_OBSERVATIONS


def _bar(
    day: int,
    close: str,
    *,
    currency: Currency = _USD,
    symbol: Symbol = _SYMBOL,
    exchange_code: ExchangeCode = _EXCHANGE,
    timeframe: Timeframe = _DAILY,
) -> HistoricalOHLCVBar:
    return HistoricalOHLCVBar(
        symbol,
        exchange_code,
        PointInTime(f"2026-01-{day:02d}T16:00:00Z"),
        timeframe,
        Price("50", currency),
        Price("500", currency),
        Price("1", currency),
        Price(close, currency),
        Quantity("1000"),
    )


class StubRepository(HistoricalMarketDataRepository):
    """Returns preconfigured observations and records the queries it received."""

    def __init__(self, observations: tuple[HistoricalOHLCVBar, ...] = ()) -> None:
        self.observations = observations
        self.queries: list[HistoricalMarketDataQuery] = []

    def get_history(self, query: HistoricalMarketDataQuery) -> tuple[HistoricalOHLCVBar, ...]:
        self.queries.append(query)
        return self.observations


def _context(
    *,
    latest_price: Price | None = None,
    observed_at: PointInTime = _DECISION_INSTANT,
    listing_reference: ListingReference | None = None,
) -> MarketObservationContext:
    currency = (latest_price or Price("100", _USD)).currency
    return MarketObservationContext(
        listing_reference or ListingReference(_SYMBOL, _EXCHANGE),
        observed_at,
        latest_price or Price("100", _USD),
        Price("99", currency),
        Quantity("1000"),
        Price("101", currency),
        Price("98", currency),
        tuple(Price("100", currency) for _ in range(20)),
        tuple(Quantity("1000") for _ in range(20)),
    )


def _evaluation(
    *,
    replay_instant: PointInTime | None = None,
    context: MarketObservationContext | None = None,
) -> HistoricalResearchEvaluation:
    market_context = context or _context()
    result = AnalyzeMarketObservationContextService(
        strategy=Strategy(StrategyIdentity("outcome-test")),
        analysis_generator=AssetAnalysisGenerator(),
    ).execute(market_context)
    return HistoricalResearchEvaluation(
        replay_instant=replay_instant or market_context.observed_at,
        result=result,
    )


def _measure(
    observations: tuple[HistoricalOHLCVBar, ...],
    *,
    horizon: int = 1,
    evaluation: HistoricalResearchEvaluation | None = None,
    timeframe: Timeframe = _DAILY,
    available_through: PointInTime = _AVAILABLE_THROUGH,
    repository: HistoricalMarketDataRepository | None = None,
) -> RecommendationOutcomeMeasurement:
    return MeasureRecommendationOutcomeUseCase(repository or StubRepository(observations)).execute(
        evaluation or _evaluation(),
        ResearchHorizon(horizon),
        timeframe,
        available_through,
    )


# ---------------------------------------------------------------------------
# Construction and dependency validation
# ---------------------------------------------------------------------------


def test_use_case_requires_a_historical_market_data_repository() -> None:
    with pytest.raises(TypeError, match="repository cannot be None"):
        MeasureRecommendationOutcomeUseCase(None)

    with pytest.raises(TypeError, match="must be a HistoricalMarketDataRepository"):
        MeasureRecommendationOutcomeUseCase("repository")


@pytest.mark.parametrize(
    "evaluation, horizon, timeframe, available_through, expected",
    [
        (None, ResearchHorizon(1), _DAILY, _AVAILABLE_THROUGH, "evaluation cannot be None"),
        ("evaluation", ResearchHorizon(1), _DAILY, _AVAILABLE_THROUGH, "must be a Historical"),
        (object(), None, _DAILY, _AVAILABLE_THROUGH, "evaluation must be a Historical"),
    ],
)
def test_execute_validates_inputs(
    evaluation: object,
    horizon: object,
    timeframe: object,
    available_through: object,
    expected: str,
) -> None:
    use_case = MeasureRecommendationOutcomeUseCase(StubRepository())

    with pytest.raises(TypeError, match=expected):
        use_case.execute(evaluation, horizon, timeframe, available_through)


def test_execute_rejects_invalid_horizon_timeframe_and_available_through() -> None:
    use_case = MeasureRecommendationOutcomeUseCase(StubRepository())
    evaluation = _evaluation()

    with pytest.raises(TypeError, match="horizon cannot be None"):
        use_case.execute(evaluation, None, _DAILY, _AVAILABLE_THROUGH)
    with pytest.raises(TypeError, match="horizon must be a ResearchHorizon"):
        use_case.execute(evaluation, 1, _DAILY, _AVAILABLE_THROUGH)
    with pytest.raises(TypeError, match="timeframe must be a Timeframe"):
        use_case.execute(evaluation, ResearchHorizon(1), "1d", _AVAILABLE_THROUGH)
    with pytest.raises(TypeError, match="available-through must be a PointInTime"):
        use_case.execute(evaluation, ResearchHorizon(1), _DAILY, "2026-03-01T16:00:00Z")


# ---------------------------------------------------------------------------
# Horizon selection
# ---------------------------------------------------------------------------


def test_horizon_one_selects_the_immediately_following_observation() -> None:
    observations = (_bar(20, "100"), _bar(21, "105"), _bar(22, "110"))

    measurement = _measure(observations, horizon=1)

    assert measurement.is_measured
    assert measurement.outcome.evaluation_instant == PointInTime("2026-01-21T16:00:00Z")
    assert measurement.outcome.evaluation_price == Price("105", _USD)


def test_horizon_n_selects_the_nth_future_observation() -> None:
    observations = (_bar(20, "100"),) + tuple(
        _bar(20 + offset, str(100 + offset)) for offset in range(1, 6)
    )

    measurement = _measure(observations, horizon=5)

    assert measurement.outcome.evaluation_instant == PointInTime("2026-01-25T16:00:00Z")
    assert measurement.outcome.evaluation_price == Price("105", _USD)


def test_decision_bar_is_excluded_from_horizon_counting() -> None:
    observations = (_bar(20, "100"), _bar(21, "105"))

    measurement = _measure(observations, horizon=1)

    assert measurement.outcome.evaluation_instant != _DECISION_INSTANT
    assert measurement.outcome.decision_instant == _DECISION_INSTANT


def test_observation_at_the_exact_decision_instant_is_not_treated_as_future() -> None:
    observations = (_bar(20, "100"),)

    measurement = _measure(observations, horizon=1)

    assert not measurement.is_measured
    assert measurement.unavailable is _INSUFFICIENT


def test_observation_before_the_decision_instant_is_a_contract_violation() -> None:
    observations = (_bar(18, "90"), _bar(19, "95"), _bar(20, "100"))

    with pytest.raises(HistoricalDataContractViolationError, match="outside query bounds"):
        _measure(observations, horizon=1)


def test_selected_observation_uses_close_and_not_other_prices() -> None:
    observations = (_bar(20, "100"), _bar(21, "105"))

    outcome = _measure(observations, horizon=1).outcome

    assert outcome.evaluation_price == observations[1].close
    assert outcome.evaluation_price != observations[1].open
    assert outcome.evaluation_price != observations[1].high
    assert outcome.evaluation_price != observations[1].low


def test_decision_price_comes_from_the_observed_market_context() -> None:
    observations = (_bar(20, "400"), _bar(21, "105"))

    outcome = _measure(observations, horizon=1).outcome

    assert outcome.decision_price == Price("100", _USD)
    assert outcome.decision_price != observations[0].close


# ---------------------------------------------------------------------------
# Insufficient future data
# ---------------------------------------------------------------------------


def test_insufficient_future_observations_is_explicit() -> None:
    observations = (_bar(20, "100"), _bar(21, "105"), _bar(22, "110"), _bar(23, "115"))

    measurement = _measure(observations, horizon=20)

    assert measurement.outcome is None
    assert measurement.unavailable is _INSUFFICIENT
    assert measurement.is_measured is False


def test_insufficient_data_never_substitutes_the_last_available_bar() -> None:
    observations = (_bar(20, "100"), _bar(21, "105"), _bar(22, "110"), _bar(23, "115"))

    measurement = _measure(observations, horizon=20)

    assert measurement.outcome is None


def test_insufficient_data_never_reports_a_zero_return() -> None:
    measurement = _measure((_bar(20, "100"),), horizon=1)

    assert measurement.outcome is None
    assert measurement.unavailable is _INSUFFICIENT


def test_insufficient_data_still_returns_the_recommendation_and_horizon() -> None:
    evaluation = _evaluation()

    measurement = _measure((_bar(20, "100"),), horizon=3, evaluation=evaluation)

    assert measurement.recommendation is evaluation.result.recommendation
    assert measurement.horizon == ResearchHorizon(3)


def test_empty_repository_result_reports_insufficient_future_observations() -> None:
    measurement = _measure(())

    assert measurement.unavailable is _INSUFFICIENT


def test_available_through_before_the_decision_instant_is_insufficient() -> None:
    repository = StubRepository((_bar(21, "105"),))

    measurement = _measure(
        (),
        horizon=1,
        repository=repository,
        available_through=PointInTime("2026-01-19T16:00:00Z"),
    )

    assert measurement.unavailable is _INSUFFICIENT
    assert repository.queries == []


def test_boundary_horizon_matching_exactly_the_available_future_bars_is_measured() -> None:
    observations = (_bar(20, "100"), _bar(21, "105"), _bar(22, "110"))

    measurement = _measure(observations, horizon=2)

    assert measurement.is_measured
    assert measurement.outcome.evaluation_price == Price("110", _USD)


# ---------------------------------------------------------------------------
# Repository query and defensive validation
# ---------------------------------------------------------------------------


def test_query_uses_listing_identity_timeframe_and_requested_bounds() -> None:
    repository = StubRepository((_bar(20, "100"), _bar(21, "105")))

    _measure((), repository=repository)

    assert len(repository.queries) == 1
    query = repository.queries[0]
    assert query.symbol == _SYMBOL
    assert query.exchange_code == _EXCHANGE
    assert query.timeframe == _DAILY
    assert query.start == _DECISION_INSTANT
    assert query.end == _AVAILABLE_THROUGH


def test_rejects_repository_symbol_mismatch() -> None:
    observations = (_bar(21, "105", symbol=Symbol("MSFT")),)

    with pytest.raises(HistoricalDataContractViolationError, match="does not match query symbol"):
        _measure(observations)


def test_rejects_repository_exchange_mismatch() -> None:
    observations = (_bar(21, "105", exchange_code=ExchangeCode("NYSE")),)

    with pytest.raises(HistoricalDataContractViolationError, match="does not match query exchange"):
        _measure(observations)


def test_rejects_repository_timeframe_mismatch() -> None:
    observations = (_bar(21, "105", timeframe=Timeframe("1h")),)

    with pytest.raises(
        HistoricalDataContractViolationError, match="does not match query timeframe"
    ):
        _measure(observations)


def test_rejects_observation_outside_requested_bounds() -> None:
    observations = (_bar(19, "95"),)

    with pytest.raises(HistoricalDataContractViolationError, match="outside query bounds"):
        _measure(observations)


def test_rejects_unordered_repository_observations() -> None:
    observations = (_bar(22, "110"), _bar(21, "105"))

    with pytest.raises(HistoricalDataContractViolationError, match="oldest to newest"):
        _measure(observations)


def test_rejects_duplicate_repository_observations() -> None:
    observations = (_bar(21, "105"), _bar(21, "105"))

    with pytest.raises(HistoricalDataContractViolationError, match="duplicate logical bars"):
        _measure(observations)


def test_rejects_non_tuple_repository_result() -> None:
    class ListRepository(HistoricalMarketDataRepository):
        def get_history(self, query: HistoricalMarketDataQuery) -> tuple[HistoricalOHLCVBar, ...]:
            return [_bar(21, "105")]

    with pytest.raises(HistoricalDataContractViolationError, match="must return a tuple"):
        _measure((), repository=ListRepository())


def test_rejects_non_bar_repository_entries() -> None:
    class InvalidRepository(HistoricalMarketDataRepository):
        def get_history(self, query: HistoricalMarketDataQuery) -> tuple[HistoricalOHLCVBar, ...]:
            return ("bar",)

    with pytest.raises(HistoricalDataContractViolationError, match="must be a HistoricalOHLCVBar"):
        _measure((), repository=InvalidRepository())


# ---------------------------------------------------------------------------
# Decision-time anchoring
# ---------------------------------------------------------------------------


def test_anchors_to_observed_at_when_replay_instant_diverges() -> None:
    evaluation = _evaluation(replay_instant=PointInTime("2026-01-20T23:59:00Z"))
    observations = (_bar(20, "100"), _bar(21, "105"))

    measurement = _measure(observations, horizon=1, evaluation=evaluation)

    assert evaluation.replay_instant != evaluation.result.market_observation_context.observed_at
    assert measurement.outcome.decision_instant == _DECISION_INSTANT
    assert measurement.outcome.evaluation_instant == PointInTime("2026-01-21T16:00:00Z")


def test_query_start_uses_observed_at_and_not_replay_instant() -> None:
    evaluation = _evaluation(replay_instant=PointInTime("2026-01-20T23:59:00Z"))
    repository = StubRepository((_bar(20, "100"), _bar(21, "105")))

    _measure((), evaluation=evaluation, repository=repository)

    assert repository.queries[0].start == _DECISION_INSTANT


def test_rejects_recommendation_instant_that_differs_from_observed_at() -> None:
    context = _context(observed_at=_DECISION_INSTANT)
    divergent_instant = PointInTime("2026-01-21T16:00:00Z")
    analysis = AssetAnalysis(context.listing_reference, divergent_instant, ("strong bullish",))
    recommendation = Recommendation(
        action=RecommendationAction("BUY"),
        asset_analysis=analysis,
        strategy_identity=StrategyIdentity("outcome-test"),
        point_in_time=divergent_instant,
    )
    explanation = RecommendationExplanation(
        recommendation=recommendation,
        reasons=(ExplanationReason(rationale="Divergent fixture."),),
    )
    evaluation = HistoricalResearchEvaluation(
        replay_instant=_DECISION_INSTANT,
        result=AnalyzeAssetResult(
            recommendation=recommendation,
            explanation=explanation,
            market_observation_context=context,
        ),
    )
    use_case = MeasureRecommendationOutcomeUseCase(StubRepository())

    with pytest.raises(ValueError, match="recommendation instant must match"):
        use_case.execute(evaluation, ResearchHorizon(1), _DAILY, _AVAILABLE_THROUGH)


def test_decision_instant_is_compared_semantically_not_textually() -> None:
    context = _context(observed_at=PointInTime("2026-01-20T21:30:00+05:30"))
    evaluation = _evaluation(context=context)
    observations = (_bar(21, "105"),)

    measurement = _measure(observations, horizon=1, evaluation=evaluation)

    assert measurement.is_measured
    assert measurement.outcome.decision_instant == PointInTime("2026-01-20T21:30:00+05:30")


# ---------------------------------------------------------------------------
# Factual return semantics
# ---------------------------------------------------------------------------


def test_positive_forward_return_is_recorded_factually() -> None:
    measurement = _measure((_bar(20, "100"), _bar(21, "105")), horizon=1)

    assert measurement.outcome.forward_return == Percentage(5)


def test_negative_forward_return_is_recorded_factually() -> None:
    measurement = _measure((_bar(20, "100"), _bar(21, "95")), horizon=1)

    assert measurement.outcome.forward_return == Percentage(-5)


def test_zero_forward_return_is_recorded_factually() -> None:
    measurement = _measure((_bar(20, "100"), _bar(21, "100")), horizon=1)

    assert measurement.outcome.forward_return == Percentage(0)


def test_measurement_applies_no_action_interpretation() -> None:
    measurement = _measure((_bar(20, "100"), _bar(21, "95")), horizon=1)

    assert not hasattr(measurement, "is_correct")
    assert not hasattr(measurement, "aligned_return")
    assert not hasattr(measurement.outcome, "is_correct")
    assert measurement.outcome.recommendation.action.value in {"BUY", "HOLD", "SELL"}


# ---------------------------------------------------------------------------
# Determinism and result semantics
# ---------------------------------------------------------------------------


def test_repeated_measurement_is_deterministic() -> None:
    observations = (_bar(20, "100"), _bar(21, "105"))
    evaluation = _evaluation()

    first = _measure(observations, horizon=1, evaluation=evaluation)
    second = _measure(observations, horizon=1, evaluation=evaluation)

    assert first == second
    assert first.outcome.forward_return == second.outcome.forward_return


def test_measurement_requires_exactly_one_of_outcome_or_unavailable() -> None:
    evaluation = _evaluation()
    recommendation = evaluation.result.recommendation

    with pytest.raises(ValueError, match="exactly one of"):
        RecommendationOutcomeMeasurement(recommendation, ResearchHorizon(1))

    outcome = RecommendationOutcome(
        recommendation=recommendation,
        horizon=ResearchHorizon(1),
        decision_price=Price("100", _USD),
        evaluation_instant=PointInTime("2026-01-21T16:00:00Z"),
        evaluation_price=Price("105", _USD),
    )
    with pytest.raises(ValueError, match="exactly one of"):
        RecommendationOutcomeMeasurement(
            recommendation, ResearchHorizon(1), outcome=outcome, unavailable=_INSUFFICIENT
        )


def test_measurement_rejects_outcome_for_a_different_horizon() -> None:
    evaluation = _evaluation()
    recommendation = evaluation.result.recommendation
    outcome = RecommendationOutcome(
        recommendation=recommendation,
        horizon=ResearchHorizon(1),
        decision_price=Price("100", _USD),
        evaluation_instant=PointInTime("2026-01-21T16:00:00Z"),
        evaluation_price=Price("105", _USD),
    )

    with pytest.raises(ValueError, match="same horizon"):
        RecommendationOutcomeMeasurement(recommendation, ResearchHorizon(5), outcome=outcome)


def test_unavailable_reason_vocabulary_is_stable() -> None:
    assert _INSUFFICIENT.value == "INSUFFICIENT_FUTURE_OBSERVATIONS"
    assert RecommendationOutcomeUnavailableReason.CURRENCY_MISMATCH.value == "CURRENCY_MISMATCH"


def test_measurement_is_immutable() -> None:
    measurement = _measure((_bar(20, "100"), _bar(21, "105")), horizon=1)

    with pytest.raises(AttributeError):
        measurement.horizon = ResearchHorizon(5)


# ---------------------------------------------------------------------------
# Currency safety
# ---------------------------------------------------------------------------


def test_same_currency_measurement_still_produces_a_factual_outcome() -> None:
    observations = (_bar(20, "100"), _bar(21, "105"))

    measurement = _measure(observations, horizon=1)

    assert measurement.is_measured
    assert measurement.unavailable is None
    assert measurement.outcome.decision_price.currency == _USD
    assert measurement.outcome.evaluation_price.currency == _USD
    assert measurement.outcome.forward_return == Percentage(5)


def test_currency_transition_reports_currency_mismatch_instead_of_raising() -> None:
    observations = (_bar(20, "100"), _bar(21, "105", currency=_EUR))

    measurement = _measure(observations, horizon=1)

    assert measurement.unavailable is RecommendationOutcomeUnavailableReason.CURRENCY_MISMATCH
    assert measurement.outcome is None


def test_currency_mismatch_preserves_the_recommendation_and_horizon() -> None:
    evaluation = _evaluation()
    observations = (_bar(20, "100"), _bar(21, "105", currency=_EUR))

    measurement = _measure(observations, horizon=1, evaluation=evaluation)

    assert measurement.recommendation is evaluation.result.recommendation
    assert measurement.horizon == ResearchHorizon(1)


def test_currency_mismatch_fabricates_no_outcome_or_return() -> None:
    observations = (_bar(20, "100"), _bar(21, "105", currency=_EUR))

    measurement = _measure(observations, horizon=1)

    assert measurement.outcome is None
    assert measurement.is_measured is False
    assert not hasattr(measurement, "forward_return")


def test_currency_mismatch_is_detected_at_the_selected_horizon_bar() -> None:
    observations = (
        _bar(20, "100"),
        _bar(21, "105"),
        _bar(22, "110", currency=_EUR),
    )

    assert _measure(observations, horizon=1).is_measured
    assert (
        _measure(observations, horizon=2).unavailable
        is RecommendationOutcomeUnavailableReason.CURRENCY_MISMATCH
    )


def test_currency_mismatch_takes_precedence_only_when_enough_bars_exist() -> None:
    observations = (_bar(20, "100"), _bar(21, "105", currency=_EUR))

    measurement = _measure(observations, horizon=5)

    assert measurement.unavailable is _INSUFFICIENT


def test_unavailable_reason_vocabulary_includes_currency_mismatch() -> None:
    assert RecommendationOutcomeUnavailableReason.CURRENCY_MISMATCH.value == "CURRENCY_MISMATCH"
    assert [reason.value for reason in RecommendationOutcomeUnavailableReason] == [
        "INSUFFICIENT_FUTURE_OBSERVATIONS",
        "CURRENCY_MISMATCH",
    ]
