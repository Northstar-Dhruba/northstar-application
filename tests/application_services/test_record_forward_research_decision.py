"""Tests for freezing forward research decisions."""

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
from northstar_core.strategy import (
    AssetAnalysis,
    AssetAnalysisGenerator,
    ExplanationReason,
    MarketObservationContext,
    Recommendation,
    RecommendationAction,
    RecommendationExplanation,
    Strategy,
    StrategyIdentity,
)

from northstar_application.application_services import (
    AnalyzeAssetResult,
    AnalyzeMarketObservationContextService,
    ForwardResearchContractViolationError,
    ForwardResearchRecord,
    RecordForwardResearchDecisionUseCase,
)
from northstar_application.ports import (
    ForwardResearchRecordConflictError,
    ForwardResearchRecordQuery,
    ForwardResearchRecordRepository,
    ForwardResearchRecordStore,
    InvalidForwardResearchRecordQueryError,
)

_USD = Currency("USD")
_DAILY = Timeframe("1d")
_SYMBOL = Symbol("AAPL")
_EXCHANGE = ExchangeCode("NASDAQ")
_DECISION_INSTANT = PointInTime("2026-01-20T16:00:00Z")


def _context(
    *,
    observed_at: PointInTime = _DECISION_INSTANT,
    symbol: Symbol = _SYMBOL,
    exchange_code: ExchangeCode = _EXCHANGE,
) -> MarketObservationContext:
    return MarketObservationContext(
        ListingReference(symbol, exchange_code),
        observed_at,
        Price("100", _USD),
        Price("99", _USD),
        Quantity("1000"),
        Price("101", _USD),
        Price("98", _USD),
        tuple(Price("100", _USD) for _ in range(20)),
        tuple(Quantity("1000") for _ in range(20)),
    )


def _result(
    *,
    observed_at: PointInTime = _DECISION_INSTANT,
    symbol: Symbol = _SYMBOL,
    exchange_code: ExchangeCode = _EXCHANGE,
    strategy: str = "mvp",
) -> AnalyzeAssetResult:
    return AnalyzeMarketObservationContextService(
        strategy=Strategy(StrategyIdentity(strategy)),
        analysis_generator=AssetAnalysisGenerator(),
    ).execute(_context(observed_at=observed_at, symbol=symbol, exchange_code=exchange_code))


def _incoherent_result(
    *,
    recommendation_instant: PointInTime = _DECISION_INSTANT,
    analysis_instant: PointInTime = _DECISION_INSTANT,
    analysis_listing: ListingReference | None = None,
    context_observed_at: PointInTime = _DECISION_INSTANT,
) -> AnalyzeAssetResult:
    """Hand-build a result whose parts disagree, which the pipeline never produces."""
    analysis = AssetAnalysis(
        analysis_listing or ListingReference(_SYMBOL, _EXCHANGE),
        analysis_instant,
        ("strong bullish",),
    )
    recommendation = Recommendation(
        action=RecommendationAction("BUY"),
        asset_analysis=analysis,
        strategy_identity=StrategyIdentity("mvp"),
        point_in_time=recommendation_instant,
    )
    return AnalyzeAssetResult(
        recommendation=recommendation,
        explanation=RecommendationExplanation(
            recommendation=recommendation,
            reasons=(ExplanationReason(rationale="Incoherent fixture."),),
        ),
        market_observation_context=_context(observed_at=context_observed_at),
    )


_HONEST_COUNT = object()


class RecordingStore(ForwardResearchRecordStore):
    """Accepts batches and records exactly what it was asked to freeze."""

    def __init__(self, return_value: object = _HONEST_COUNT) -> None:
        self.batches: list[tuple[ForwardResearchRecord, ...]] = []
        self._return_value = return_value

    def store(self, records: tuple[ForwardResearchRecord, ...]) -> int:
        self.batches.append(records)
        if self._return_value is _HONEST_COUNT:
            return len(records)
        return self._return_value


class FreezingStore(ForwardResearchRecordStore):
    """Reference implementation of the documented freeze semantics."""

    def __init__(self) -> None:
        self.frozen: dict[tuple[object, ...], ForwardResearchRecord] = {}

    def store(self, records: tuple[ForwardResearchRecord, ...]) -> int:
        seen: set[tuple[object, ...]] = set()
        for record in records:
            key = record.natural_key
            if key in seen:
                raise ForwardResearchRecordConflictError(
                    "Batch contains two records sharing one natural key."
                )
            seen.add(key)
            existing = self.frozen.get(key)
            if existing is not None and existing != record:
                raise ForwardResearchRecordConflictError(
                    "A different record is already frozen under this natural key."
                )
        for record in records:
            self.frozen.setdefault(record.natural_key, record)
        return len(records)


def _record(**kwargs: object) -> ForwardResearchRecord:
    timeframe = kwargs.pop("timeframe", _DAILY)
    return ForwardResearchRecord(result=_result(**kwargs), timeframe=timeframe)


# ---------------------------------------------------------------------------
# Valid record and derived identity
# ---------------------------------------------------------------------------


def test_creates_a_valid_record_from_a_produced_decision() -> None:
    result = _result()

    record = ForwardResearchRecord(result=result, timeframe=_DAILY)

    assert record.result is result
    assert record.timeframe == _DAILY


def test_derived_identity_fields_are_exposed() -> None:
    record = _record()

    assert record.listing_reference == ListingReference(_SYMBOL, _EXCHANGE)
    assert record.strategy_identity == StrategyIdentity("mvp")
    assert record.decision_instant == _DECISION_INSTANT


def test_derived_fields_are_not_duplicated_as_stored_state() -> None:
    record = _record()

    assert ForwardResearchRecord.__slots__ == ("result", "timeframe")
    assert not hasattr(record, "__dict__")


def test_decision_instant_matches_the_recorded_recommendation() -> None:
    record = _record()

    assert record.decision_instant == record.result.recommendation.point_in_time
    assert record.decision_instant == record.result.market_observation_context.observed_at


def test_record_stores_no_measurement_status() -> None:
    record = _record()

    for forbidden in ("status", "pending", "measured", "unavailable", "outcome", "recorded_at"):
        assert not hasattr(record, forbidden)


# ---------------------------------------------------------------------------
# Type validation
# ---------------------------------------------------------------------------


def test_rejects_none_result() -> None:
    with pytest.raises(TypeError, match="result cannot be None"):
        ForwardResearchRecord(result=None, timeframe=_DAILY)


def test_rejects_wrong_result_type() -> None:
    with pytest.raises(TypeError, match="must be an AnalyzeAssetResult"):
        ForwardResearchRecord(result="result", timeframe=_DAILY)


def test_rejects_none_timeframe() -> None:
    with pytest.raises(TypeError, match="timeframe cannot be None"):
        ForwardResearchRecord(result=_result(), timeframe=None)


def test_rejects_wrong_timeframe_type() -> None:
    with pytest.raises(TypeError, match="timeframe must be a Timeframe"):
        ForwardResearchRecord(result=_result(), timeframe="1d")


# ---------------------------------------------------------------------------
# Record coherence
# ---------------------------------------------------------------------------


def test_rejects_recommendation_instant_that_differs_from_observed_at() -> None:
    result = _incoherent_result(recommendation_instant=PointInTime("2026-01-21T16:00:00Z"))

    with pytest.raises(ValueError, match="recommendation instant must match"):
        ForwardResearchRecord(result=result, timeframe=_DAILY)


def test_rejects_asset_analysis_instant_that_differs_from_observed_at() -> None:
    result = _incoherent_result(analysis_instant=PointInTime("2026-01-19T16:00:00Z"))

    with pytest.raises(ValueError, match="asset analysis instant must match"):
        ForwardResearchRecord(result=result, timeframe=_DAILY)


def test_rejects_asset_analysis_listing_that_differs_from_the_context() -> None:
    result = _incoherent_result(
        analysis_listing=ListingReference(Symbol("MSFT"), _EXCHANGE),
    )

    with pytest.raises(ValueError, match="asset analysis listing must match"):
        ForwardResearchRecord(result=result, timeframe=_DAILY)


def test_accepts_equivalent_instants_expressed_in_different_offsets() -> None:
    result = _result(observed_at=PointInTime("2026-01-20T21:30:00+05:30"))

    record = ForwardResearchRecord(result=result, timeframe=_DAILY)

    assert record.decision_instant == _DECISION_INSTANT


# ---------------------------------------------------------------------------
# Natural key
# ---------------------------------------------------------------------------


def test_natural_key_is_the_documented_five_part_identity() -> None:
    record = _record()

    assert record.natural_key == (
        _SYMBOL,
        _EXCHANGE,
        _DAILY,
        _DECISION_INSTANT,
        StrategyIdentity("mvp"),
    )


def test_natural_key_distinguishes_by_strategy() -> None:
    assert _record().natural_key != _record(strategy="momentum").natural_key


def test_natural_key_distinguishes_by_timeframe() -> None:
    assert _record().natural_key != _record(timeframe=Timeframe("1h")).natural_key


def test_natural_key_distinguishes_by_symbol() -> None:
    assert _record().natural_key != _record(symbol=Symbol("MSFT")).natural_key


def test_natural_key_distinguishes_by_exchange() -> None:
    assert _record().natural_key != _record(exchange_code=ExchangeCode("NYSE")).natural_key


def test_natural_key_distinguishes_by_decision_instant() -> None:
    later = _record(observed_at=PointInTime("2026-01-21T16:00:00Z"))

    assert _record().natural_key != later.natural_key


def test_equivalent_records_share_one_natural_key() -> None:
    assert _record().natural_key == _record().natural_key


# ---------------------------------------------------------------------------
# Use case
# ---------------------------------------------------------------------------


def test_use_case_requires_a_store() -> None:
    with pytest.raises(TypeError, match="store cannot be None"):
        RecordForwardResearchDecisionUseCase(None)

    with pytest.raises(TypeError, match="must be a ForwardResearchRecordStore"):
        RecordForwardResearchDecisionUseCase("store")


def test_use_case_freezes_the_exact_record_it_returns() -> None:
    store = RecordingStore()
    result = _result()

    record = RecordForwardResearchDecisionUseCase(store).execute(result, _DAILY)

    assert store.batches == [(record,)]
    assert record.result is result
    assert record.timeframe == _DAILY


def test_use_case_stores_exactly_one_record_per_decision() -> None:
    store = RecordingStore()

    RecordForwardResearchDecisionUseCase(store).execute(_result(), _DAILY)

    assert len(store.batches) == 1
    assert len(store.batches[0]) == 1


def test_use_case_propagates_record_validation_failures() -> None:
    store = RecordingStore()
    incoherent = _incoherent_result(analysis_instant=PointInTime("2026-01-19T16:00:00Z"))

    with pytest.raises(ValueError, match="asset analysis instant must match"):
        RecordForwardResearchDecisionUseCase(store).execute(incoherent, _DAILY)

    assert store.batches == []


@pytest.mark.parametrize("returned", [0, 2, -1])
def test_use_case_rejects_a_wrong_store_count(returned: int) -> None:
    store = RecordingStore(return_value=returned)

    with pytest.raises(ForwardResearchContractViolationError, match="expected 1"):
        RecordForwardResearchDecisionUseCase(store).execute(_result(), _DAILY)


@pytest.mark.parametrize("returned", ["1", None, True])
def test_use_case_rejects_a_non_integer_store_count(returned: object) -> None:
    store = RecordingStore(return_value=returned)

    with pytest.raises(ForwardResearchContractViolationError, match="must return an integer"):
        RecordForwardResearchDecisionUseCase(store).execute(_result(), _DAILY)


# ---------------------------------------------------------------------------
# Freeze semantics, exercised through a reference store implementation
# ---------------------------------------------------------------------------


def test_new_natural_key_is_persisted() -> None:
    store = FreezingStore()

    RecordForwardResearchDecisionUseCase(store).execute(_result(), _DAILY)

    assert len(store.frozen) == 1


def test_identical_retry_is_idempotent() -> None:
    store = FreezingStore()
    use_case = RecordForwardResearchDecisionUseCase(store)
    result = _result()

    first = use_case.execute(result, _DAILY)
    second = use_case.execute(result, _DAILY)

    assert first == second
    assert len(store.frozen) == 1


def test_conflicting_same_key_record_is_rejected_and_not_overwritten() -> None:
    store = FreezingStore()
    use_case = RecordForwardResearchDecisionUseCase(store)
    original = _result()
    use_case.execute(original, _DAILY)

    divergent = AnalyzeAssetResult(
        recommendation=original.recommendation,
        explanation=RecommendationExplanation(
            recommendation=original.recommendation,
            reasons=(ExplanationReason(rationale="A different explanation."),),
        ),
        market_observation_context=original.market_observation_context,
    )

    with pytest.raises(ForwardResearchRecordConflictError, match="already frozen"):
        use_case.execute(divergent, _DAILY)

    assert store.frozen[_record().natural_key].result == original


def test_different_strategies_freeze_independently_at_one_instant() -> None:
    store = FreezingStore()
    use_case = RecordForwardResearchDecisionUseCase(store)

    use_case.execute(_result(strategy="mvp"), _DAILY)
    use_case.execute(_result(strategy="momentum"), _DAILY)

    assert len(store.frozen) == 2


def test_batch_with_duplicate_natural_keys_is_rejected() -> None:
    store = FreezingStore()
    record = _record()

    with pytest.raises(ForwardResearchRecordConflictError, match="Batch contains"):
        store.store((record, record))


def test_empty_batch_is_a_safe_no_op() -> None:
    store = FreezingStore()

    assert store.store(()) == 0
    assert store.frozen == {}


# ---------------------------------------------------------------------------
# Query contract
# ---------------------------------------------------------------------------


def test_query_carries_only_the_series_identity() -> None:
    query = ForwardResearchRecordQuery(_SYMBOL, _EXCHANGE, _DAILY)

    assert ForwardResearchRecordQuery.__slots__ == ("symbol", "exchange_code", "timeframe")
    assert query.symbol == _SYMBOL
    assert query.exchange_code == _EXCHANGE
    assert query.timeframe == _DAILY


def test_query_carries_no_status_or_range_filtering() -> None:
    query = ForwardResearchRecordQuery(_SYMBOL, _EXCHANGE, _DAILY)

    for forbidden in ("start", "end", "status", "pending", "measured", "strategy_identity"):
        assert not hasattr(query, forbidden)


@pytest.mark.parametrize(
    "symbol, exchange_code, timeframe, expected",
    [
        ("AAPL", _EXCHANGE, _DAILY, "symbol must be a Symbol"),
        (_SYMBOL, "NASDAQ", _DAILY, "exchange code must be an ExchangeCode"),
        (_SYMBOL, _EXCHANGE, "1d", "timeframe must be a Timeframe"),
    ],
)
def test_query_rejects_invalid_identity_values(
    symbol: object, exchange_code: object, timeframe: object, expected: str
) -> None:
    with pytest.raises(InvalidForwardResearchRecordQueryError, match=expected):
        ForwardResearchRecordQuery(symbol, exchange_code, timeframe)


def test_query_is_immutable_and_compares_by_value() -> None:
    left = ForwardResearchRecordQuery(_SYMBOL, _EXCHANGE, _DAILY)
    right = ForwardResearchRecordQuery(_SYMBOL, _EXCHANGE, _DAILY)

    assert left == right
    assert hash(left) == hash(right)
    with pytest.raises(AttributeError):
        left.timeframe = Timeframe("1h")


def test_repository_contract_preserves_decision_order_and_empty_results() -> None:
    class InMemoryRepository(ForwardResearchRecordRepository):
        def __init__(self, records: tuple[ForwardResearchRecord, ...]) -> None:
            self._records = records

        def get_records(
            self, query: ForwardResearchRecordQuery
        ) -> tuple[ForwardResearchRecord, ...]:
            matching = [
                record
                for record in self._records
                if record.listing_reference.symbol == query.symbol
                and record.listing_reference.exchange_code == query.exchange_code
                and record.timeframe == query.timeframe
            ]
            matching.sort(key=lambda record: record.decision_instant.value)
            return tuple(matching)

    later = _record(observed_at=PointInTime("2026-01-22T16:00:00Z"))
    earlier = _record()
    repository = InMemoryRepository((later, earlier))

    ordered = repository.get_records(ForwardResearchRecordQuery(_SYMBOL, _EXCHANGE, _DAILY))

    assert [record.decision_instant for record in ordered] == [
        earlier.decision_instant,
        later.decision_instant,
    ]
    assert (
        repository.get_records(ForwardResearchRecordQuery(Symbol("MSFT"), _EXCHANGE, _DAILY)) == ()
    )


# ---------------------------------------------------------------------------
# Value semantics
# ---------------------------------------------------------------------------


def test_record_is_immutable() -> None:
    record = _record()

    with pytest.raises(AttributeError):
        record.timeframe = Timeframe("1h")


def test_equivalent_records_compare_equal_and_hash_equal() -> None:
    left = _record()
    right = _record()

    assert left == right
    assert hash(left) == hash(right)


def test_records_differing_only_by_timeframe_are_not_equal() -> None:
    assert _record() != _record(timeframe=Timeframe("1h"))


def test_records_differing_by_strategy_are_not_equal() -> None:
    assert _record() != _record(strategy="momentum")


def test_record_is_usable_as_a_dictionary_key() -> None:
    index = {_record(): "frozen"}

    assert index[_record()] == "frozen"
