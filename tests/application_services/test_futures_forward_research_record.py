"""Tests for the futures forward research record, query and persistence ports.

The ports are abstract, so their freeze semantics are pinned through a minimal
reference store that implements exactly what the store contract documents.
That exercises the real record's natural key and value equality -- the two
facts the SQLite adapter in 9.8b will rely on -- without building persistence.
"""

from __future__ import annotations

import ast
import dataclasses
import importlib
import inspect
from decimal import Decimal
from functools import cmp_to_key
from pathlib import Path

import pytest
from northstar_core.derivatives import ExpirationDate, QuoteValue
from northstar_core.foundation.value_objects import (
    ExchangeCode,
    PointInTime,
    Quantity,
    Symbol,
    Timeframe,
)
from northstar_core.futures import FuturesContract, FuturesProductReference
from northstar_core.strategy import (
    FuturesAssetAnalysis,
    FuturesMarketObservationContext,
    Strategy,
    StrategyIdentity,
)

from northstar_application.application_services import (
    FuturesAnalysisResult,
    FuturesForwardResearchRecord,
    UnsupportedFuturesReplayTimeframeError,
)
from northstar_application.ports import (
    ForwardResearchRecordConflictError,
    FuturesForwardResearchRecordConflictError,
    FuturesForwardResearchRecordQuery,
    FuturesForwardResearchRecordRepository,
    FuturesForwardResearchRecordStore,
    InvalidFuturesForwardResearchRecordQueryError,
)


def _contract(product: str = "ES", exchange: str = "CME", expiry: str = "2026-12-18"):
    return FuturesContract(
        FuturesProductReference(Symbol(product), ExchangeCode(exchange)), ExpirationDate(expiry)
    )


_ES_DEC = _contract()
_DAILY = Timeframe("1d")
_INSTANT = "2026-09-15T21:00:00Z"
_STRATEGY = "futures-forward"


def _quote(value: str) -> QuoteValue:
    return QuoteValue(Decimal(value))


def _result(
    *,
    contract: FuturesContract = _ES_DEC,
    observed_at: str = _INSTANT,
    strategy: str = _STRATEGY,
    signal: str = "strong bullish",
    timeframe: Timeframe = _DAILY,
    latest: str = "7663.25",
    previous: str = "7650",
    closes: tuple[str, ...] | None = None,
    volumes: tuple[str, ...] | None = None,
) -> FuturesAnalysisResult:
    instant = PointInTime(observed_at)
    closes = closes if closes is not None else tuple(str(7600 + index) for index in range(20))
    volumes = volumes if volumes is not None else tuple(str(1000 + index) for index in range(20))
    context = FuturesMarketObservationContext(
        contract=contract,
        timeframe=timeframe,
        observed_at=instant,
        latest_quote=_quote(latest),
        previous_close=_quote(previous),
        latest_volume=Quantity(Decimal("1250")),
        session_high=_quote("7700"),
        session_low=_quote("7500"),
        recent_closes=tuple(_quote(close) for close in closes),
        recent_volumes=tuple(Quantity(Decimal(volume)) for volume in volumes),
    )
    recommendation = Strategy(StrategyIdentity(strategy)).evaluate_futures(
        FuturesAssetAnalysis(contract, instant, (signal,))
    )
    return FuturesAnalysisResult(recommendation, context)


def _record(**overrides: object) -> FuturesForwardResearchRecord:
    return FuturesForwardResearchRecord(_result(**overrides))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Record
# ---------------------------------------------------------------------------


def test_a_record_holds_exactly_one_field_the_result() -> None:
    result = _result()

    record = FuturesForwardResearchRecord(result)

    assert record.result is result
    assert [field.name for field in dataclasses.fields(FuturesForwardResearchRecord)] == ["result"]
    assert FuturesForwardResearchRecord.__slots__ == ("result",)


def test_a_record_is_frozen() -> None:
    record = _record()

    with pytest.raises(AttributeError):
        record.result = _result()  # type: ignore[misc]
    with pytest.raises(AttributeError):
        record.contract = _contract("MES")  # type: ignore[misc]


@pytest.mark.parametrize("value", [None, "result", object()])
def test_a_foreign_result_is_rejected(value: object) -> None:
    with pytest.raises(TypeError, match="FuturesAnalysisResult"):
        FuturesForwardResearchRecord(value)  # type: ignore[arg-type]


def test_a_daily_record_is_accepted() -> None:
    assert _record(timeframe=Timeframe("1d")).timeframe == _DAILY


@pytest.mark.parametrize("timeframe", [Timeframe("1m"), Timeframe("1h")], ids=["1m", "1h"])
def test_a_non_daily_record_is_rejected(timeframe: Timeframe) -> None:
    result = _result(timeframe=timeframe)

    with pytest.raises(UnsupportedFuturesReplayTimeframeError, match="session-daily") as raised:
        FuturesForwardResearchRecord(result)

    assert str(timeframe) in str(raised.value)


def test_the_record_reuses_the_existing_timeframe_error() -> None:
    import northstar_application.application_services.futures_forward_research_record as module
    import northstar_application.application_services.replay_futures_historical_market_data as rp

    assert (
        module.UnsupportedFuturesReplayTimeframeError is rp.UnsupportedFuturesReplayTimeframeError
    )
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    assert not [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name.endswith("Error")
    ]


def test_every_derived_property_reads_the_result() -> None:
    contract = _contract("ES", "CME", "2027-03-19")
    record = _record(contract=contract, observed_at="2026-09-16T21:00:00Z", strategy="alpha")

    assert record.contract == contract
    assert record.contract is record.result.recommendation.contract
    assert record.timeframe == _DAILY
    assert record.timeframe is record.result.market_observation_context.timeframe
    assert record.decision_instant == PointInTime("2026-09-16T21:00:00Z")
    assert record.decision_instant is record.result.market_observation_context.observed_at
    assert record.strategy_identity == StrategyIdentity("alpha")
    assert record.strategy_identity is record.result.recommendation.strategy_identity


def test_the_decision_instant_is_the_recommendation_instant() -> None:
    record = _record()

    assert record.decision_instant.compare(record.result.recommendation.point_in_time) == 0


def test_the_record_stores_no_measurement_or_execution_concept() -> None:
    record = _record()

    for absent in (
        "status",
        "state",
        "outcome",
        "measurement",
        "horizon",
        "position",
        "quantity",
        "multiplier",
        "margin",
        "pnl",
        "record_id",
        "raw_symbol",
        "instrument_id",
    ):
        assert not hasattr(record, absent)


# ---------------------------------------------------------------------------
# Natural key
# ---------------------------------------------------------------------------


def test_the_natural_key_is_the_documented_four_part_identity() -> None:
    record = _record()

    assert record.natural_key == (
        _ES_DEC,
        _DAILY,
        PointInTime(_INSTANT),
        StrategyIdentity(_STRATEGY),
    )
    hash(record.natural_key)  # usable as a mapping key by a store


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"contract": _contract("MES")}, id="es-vs-mes"),
        pytest.param({"contract": _contract(exchange="CBOT")}, id="cme-vs-cbot"),
        pytest.param({"contract": _contract(expiry="2027-03-19")}, id="dec-vs-mar"),
        pytest.param({"observed_at": "2026-09-16T21:00:00Z"}, id="another-instant"),
        pytest.param({"observed_at": "2026-09-15T21:00:00.5Z"}, id="half-a-second-later"),
        pytest.param({"strategy": "another-strategy"}, id="another-strategy"),
    ],
)
def test_the_natural_key_distinguishes_every_identity_part(overrides: dict[str, object]) -> None:
    assert _record(**overrides).natural_key != _record().natural_key


def test_the_natural_key_distinguishes_timeframe() -> None:
    """A non-daily record cannot exist, so the timeframe part is checked directly."""
    record = _record()
    other = (record.contract, Timeframe("1m"), record.decision_instant, record.strategy_identity)

    assert other != record.natural_key


@pytest.mark.parametrize(
    "spelling",
    [
        "2026-09-15T16:00:00-05:00",
        "2026-09-16T02:30:00+05:30",
        "2026-09-15T21:00:00.000Z",
        "2026-09-15T21:00:00+00:00",
    ],
)
def test_offset_equivalent_instants_give_one_natural_key(spelling: str) -> None:
    z = _record(observed_at=_INSTANT)
    other = _record(observed_at=spelling)

    assert other.natural_key == z.natural_key
    assert hash(other.natural_key) == hash(z.natural_key)
    assert other.decision_instant.compare(z.decision_instant) == 0
    assert other == z


# ---------------------------------------------------------------------------
# Full value equality
# ---------------------------------------------------------------------------


def test_separately_rebuilt_equal_records_are_equal() -> None:
    first = _record()
    second = _record()

    assert first is not second
    assert first.result is not second.result
    assert first == second
    assert hash(first) == hash(second)


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"latest": "7663.50"}, id="latest-quote"),
        pytest.param({"previous": "7649.75"}, id="previous-close"),
        pytest.param(
            {"closes": tuple(str(7600 + index) for index in range(19)) + ("9999",)},
            id="recent-closes",
        ),
        pytest.param(
            {"volumes": tuple(str(1000 + index) for index in range(19)) + ("1",)},
            id="recent-volumes",
        ),
        pytest.param({"signal": "neutral trend"}, id="analysis-signal-and-action"),
        pytest.param({"signal": "strong bearish"}, id="recommendation-action"),
    ],
)
def test_changed_evidence_under_the_same_key_is_a_different_record(
    overrides: dict[str, object],
) -> None:
    original = _record()
    changed = _record(**overrides)

    assert changed.natural_key == original.natural_key
    assert changed != original


def test_a_different_action_under_the_same_key_is_a_different_record() -> None:
    buy = _record(signal="strong bullish")
    sell = _record(signal="strong bearish")

    assert buy.result.recommendation.action.value == "BUY"
    assert sell.result.recommendation.action.value == "SELL"
    assert buy.natural_key == sell.natural_key
    assert buy != sell


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------


def test_a_query_names_one_contract_at_the_daily_timeframe() -> None:
    query = FuturesForwardResearchRecordQuery(_ES_DEC, _DAILY)

    assert query.contract == _ES_DEC
    assert query.timeframe == _DAILY
    assert [field.name for field in dataclasses.fields(query)] == ["contract", "timeframe"]


@pytest.mark.parametrize(
    ("contract", "timeframe", "message"),
    [
        pytest.param("ES", _DAILY, "contract must be a FuturesContract", id="text-contract"),
        pytest.param(None, _DAILY, "contract must be a FuturesContract", id="no-contract"),
        pytest.param(_ES_DEC, "1d", "timeframe must be a Timeframe", id="text-timeframe"),
        pytest.param(_ES_DEC, Timeframe("1m"), "session-daily", id="minute"),
        pytest.param(_ES_DEC, Timeframe("1h"), "session-daily", id="hour"),
    ],
)
def test_an_invalid_query_is_rejected(contract: object, timeframe: object, message: str) -> None:
    with pytest.raises(InvalidFuturesForwardResearchRecordQueryError, match=message):
        FuturesForwardResearchRecordQuery(contract, timeframe)  # type: ignore[arg-type]


def test_the_query_error_is_a_value_error() -> None:
    assert issubclass(InvalidFuturesForwardResearchRecordQueryError, ValueError)


def test_queries_are_immutable_and_compare_by_value() -> None:
    first = FuturesForwardResearchRecordQuery(_ES_DEC, _DAILY)
    second = FuturesForwardResearchRecordQuery(_contract(), Timeframe("1d"))

    assert first == second
    assert hash(first) == hash(second)
    assert first != FuturesForwardResearchRecordQuery(_contract("MES"), _DAILY)
    with pytest.raises(AttributeError):
        first.contract = _contract("MES")  # type: ignore[misc]


def test_the_query_carries_no_strategy_range_as_of_or_horizon() -> None:
    query = FuturesForwardResearchRecordQuery(_ES_DEC, _DAILY)

    for absent in ("strategy_identity", "start", "end", "as_of", "available_through", "horizon"):
        assert not hasattr(query, absent)


# ---------------------------------------------------------------------------
# Ports
# ---------------------------------------------------------------------------


def test_the_store_port_is_abstract_with_one_store_method() -> None:
    assert FuturesForwardResearchRecordStore.__abstractmethods__ == frozenset({"store"})
    with pytest.raises(TypeError):
        FuturesForwardResearchRecordStore()  # type: ignore[abstract]
    parameters = list(inspect.signature(FuturesForwardResearchRecordStore.store).parameters)
    assert parameters == ["self", "records"]


def test_the_repository_port_is_abstract_with_one_get_records_method() -> None:
    assert FuturesForwardResearchRecordRepository.__abstractmethods__ == frozenset({"get_records"})
    with pytest.raises(TypeError):
        FuturesForwardResearchRecordRepository()  # type: ignore[abstract]
    parameters = list(
        inspect.signature(FuturesForwardResearchRecordRepository.get_records).parameters
    )
    assert parameters == ["self", "query"]


def test_the_conflict_error_is_its_own_value_error() -> None:
    assert issubclass(FuturesForwardResearchRecordConflictError, ValueError)
    assert not issubclass(
        FuturesForwardResearchRecordConflictError, ForwardResearchRecordConflictError
    )
    assert not issubclass(
        ForwardResearchRecordConflictError, FuturesForwardResearchRecordConflictError
    )


class ReferenceStore(FuturesForwardResearchRecordStore):
    """Exactly the documented freeze semantics, keyed by the record's natural key."""

    def __init__(self) -> None:
        self.records: dict[tuple, FuturesForwardResearchRecord] = {}

    def store(self, records):
        keys = [record.natural_key for record in records]
        if len(set(keys)) != len(keys):
            raise FuturesForwardResearchRecordConflictError("batch shares a natural key")
        for record in records:
            existing = self.records.get(record.natural_key)
            if existing is not None and existing != record:
                raise FuturesForwardResearchRecordConflictError("different record under key")
        for record in records:
            self.records.setdefault(record.natural_key, record)
        return len(records)


def test_a_new_natural_key_is_persisted() -> None:
    store = ReferenceStore()

    assert store.store((_record(),)) == 1
    assert list(store.records.values()) == [_record()]


def test_an_equal_retry_is_idempotent_and_counted() -> None:
    store = ReferenceStore()
    store.store((_record(),))

    assert store.store((_record(),)) == 1
    assert len(store.records) == 1


def test_an_offset_spelled_retry_is_the_same_decision() -> None:
    store = ReferenceStore()
    store.store((_record(),))

    assert store.store((_record(observed_at="2026-09-15T16:00:00-05:00"),)) == 1
    assert len(store.records) == 1


def test_different_evidence_under_one_key_conflicts_and_is_not_overwritten() -> None:
    store = ReferenceStore()
    original = _record()
    store.store((original,))

    with pytest.raises(FuturesForwardResearchRecordConflictError):
        store.store((_record(latest="7501"),))

    assert store.records[original.natural_key] == original


def test_a_batch_repeating_one_key_is_rejected_even_when_equal() -> None:
    with pytest.raises(FuturesForwardResearchRecordConflictError):
        ReferenceStore().store((_record(), _record()))


def test_different_strategies_freeze_independently_at_one_instant() -> None:
    store = ReferenceStore()

    assert store.store((_record(strategy="alpha"), _record(strategy="beta"))) == 2
    assert len(store.records) == 2


def test_an_empty_batch_is_a_no_op_returning_zero() -> None:
    store = ReferenceStore()

    assert store.store(()) == 0
    assert store.records == {}


def _documented_order(
    left: FuturesForwardResearchRecord, right: FuturesForwardResearchRecord
) -> int:
    instant = left.decision_instant.compare(right.decision_instant)
    if instant:
        return instant
    left_id, right_id = left.strategy_identity.identity, right.strategy_identity.identity
    return (left_id > right_id) - (left_id < right_id)


def test_a_repository_returns_one_series_in_semantic_decision_order() -> None:
    """The documented order: instant by compare(), then strategy identity."""

    class ReferenceRepository(FuturesForwardResearchRecordRepository):
        def __init__(self, records: tuple[FuturesForwardResearchRecord, ...]) -> None:
            self._records = records

        def get_records(self, query):
            matching = [
                record
                for record in self._records
                if record.contract == query.contract and record.timeframe == query.timeframe
            ]
            return tuple(sorted(matching, key=cmp_to_key(_documented_order)))

    # As text "...00.5Z" sorts before "...00Z": only compare() orders these correctly.
    whole = _record(observed_at="2026-09-15T21:00:00Z", strategy="beta")
    half = _record(observed_at="2026-09-15T21:00:00.5Z", strategy="alpha")
    same_instant = _record(observed_at="2026-09-15T21:00:00Z", strategy="alpha")
    other_contract = _record(contract=_contract("MES"))
    repository = ReferenceRepository((half, whole, other_contract, same_instant))

    records = repository.get_records(FuturesForwardResearchRecordQuery(_ES_DEC, _DAILY))

    assert records == (same_instant, whole, half)
    for earlier, later in zip(records, records[1:], strict=False):
        order = earlier.decision_instant.compare(later.decision_instant)
        assert order < 0 or (
            order == 0 and earlier.strategy_identity.identity < later.strategy_identity.identity
        )
    assert (
        repository.get_records(
            FuturesForwardResearchRecordQuery(_contract(expiry="2027-03-19"), _DAILY)
        )
        == ()
    )


# ---------------------------------------------------------------------------
# Boundaries and exports
# ---------------------------------------------------------------------------

_MODULES = (
    "northstar_application.application_services.futures_forward_research_record",
    "northstar_application.ports.futures_forward_research_record_store",
    "northstar_application.ports.futures_forward_research_record_repository",
)


@pytest.mark.parametrize("module_name", _MODULES)
def test_the_contracts_depend_on_no_storage_provider_or_execution_concept(
    module_name: str,
) -> None:
    module = importlib.import_module(module_name)
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
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

    for module_path in modules:
        assert module_path.split(".")[0] not in {
            "northstar_infrastructure",
            "sqlite3",
            "databento",
            "exchange_calendars",
            "requests",
            "urllib",
            "socket",
            "time",
            "datetime",
            "random",
            "uuid",
        }
        assert "paper_trading" not in module_path
        assert "execution" not in module_path
    for forbidden in (
        "FuturesHistoricalMarketDataSource",
        "AcquireFuturesDailyHistoryUseCase",
        "FuturesTradingSessionResolver",
        "ReplayFuturesHistoricalMarketDataUseCase",
        "MeasureFuturesRecommendationOutcomeUseCase",
    ):
        assert forbidden not in names
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            assert node.attr not in {"now", "today", "utcnow", "action"}


def test_the_ports_import_the_record_for_typing_only() -> None:
    """A runtime import would cycle: application_services already imports ports."""
    for module_name in _MODULES[1:]:
        module = importlib.import_module(module_name)
        tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
        top_level_imports = {
            node.module
            for node in tree.body
            if isinstance(node, ast.ImportFrom) and node.module is not None
        }
        assert not any(name.startswith("northstar_application") for name in top_level_imports)


def test_the_contracts_are_exported() -> None:
    import northstar_application.application_services as services
    import northstar_application.ports as ports

    assert "FuturesForwardResearchRecord" in services.__all__
    for name in (
        "FuturesForwardResearchRecordQuery",
        "FuturesForwardResearchRecordStore",
        "FuturesForwardResearchRecordRepository",
        "FuturesForwardResearchRecordConflictError",
        "InvalidFuturesForwardResearchRecordQueryError",
    ):
        assert name in ports.__all__
    for private in ("_SUPPORTED_TIMEFRAME",):
        assert not hasattr(services, private)
        assert not hasattr(ports, private)
