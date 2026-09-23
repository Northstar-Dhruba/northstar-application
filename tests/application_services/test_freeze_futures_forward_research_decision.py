"""Tests for creating and freezing one futures forward research decision.

The repository here answers each query exactly as a conforming repository must
-- the queried contract and timeframe, inside the inclusive window, in
semantic order -- and its bars can be changed between runs to model appended
or back-filled history. The store implements exactly the documented freeze
semantics, so idempotency and conflict are exercised as the real SQLite store
will behave, not special-cased.
"""

from __future__ import annotations

import ast
from datetime import date, timedelta
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
from northstar_core.futures import FuturesContract, FuturesOHLCVBar, FuturesProductReference
from northstar_core.strategy import (
    FuturesAssetAnalysisGenerator,
    RecommendationAction,
    Strategy,
    StrategyIdentity,
)

from northstar_application.application_services import (
    ForwardResearchContractViolationError,
    FreezeFuturesForwardResearchDecisionUseCase,
    FuturesForwardResearchRecord,
    UnsupportedFuturesReplayTimeframeError,
)
from northstar_application.ports import (
    FuturesForwardResearchRecordConflictError,
    FuturesForwardResearchRecordStore,
    FuturesHistoricalMarketDataQuery,
    FuturesHistoricalMarketDataRepository,
)

_ES_DEC = FuturesContract(
    FuturesProductReference(Symbol("ES"), ExchangeCode("CME")), ExpirationDate("2026-12-18")
)
_MES_DEC = FuturesContract(
    FuturesProductReference(Symbol("MES"), ExchangeCode("CME")), ExpirationDate("2026-12-18")
)
_DAILY = Timeframe("1d")
_STRATEGY = Strategy(StrategyIdentity("futures-forward"))
_BUY = RecommendationAction("BUY")
_SELL = RecommendationAction("SELL")


def _instant(session: int) -> PointInTime:
    return PointInTime(f"{date(2026, 8, 1) + timedelta(days=session - 1)}T21:00:00Z")


def _bar(
    session: int,
    close: str,
    volume: str = "1000",
    *,
    instant: PointInTime | None = None,
    contract: FuturesContract = _ES_DEC,
) -> FuturesOHLCVBar:
    quote = Decimal(close)
    return FuturesOHLCVBar(
        contract=contract,
        point_in_time=instant if instant is not None else _instant(session),
        timeframe=_DAILY,
        open=QuoteValue(quote),
        high=QuoteValue(quote + 2),
        low=QuoteValue(quote - 2),
        close=QuoteValue(quote),
        volume=Quantity(Decimal(volume)),
    )


def _rising(count: int) -> tuple[FuturesOHLCVBar, ...]:
    """Fifteen flat sessions then a steady rise: strongly bullish from session 20."""
    closes = ["7600"] * 15 + [str(7601 + index) for index in range(max(count - 15, 0))]
    return tuple(_bar(session, close) for session, close in enumerate(closes[:count], start=1))


class ConformingRepository(FuturesHistoricalMarketDataRepository):
    def __init__(self, bars: tuple[FuturesOHLCVBar, ...] = ()) -> None:
        self.bars = bars
        self.queries: list[FuturesHistoricalMarketDataQuery] = []

    def get_bars(self, query):
        self.queries.append(query)
        matching = [
            bar
            for bar in self.bars
            if bar.contract == query.contract
            and bar.timeframe == query.timeframe
            and query.covers(bar.point_in_time)
        ]
        return tuple(
            sorted(matching, key=cmp_to_key(lambda a, b: a.point_in_time.compare(b.point_in_time)))
        )


class ReferenceStore(FuturesForwardResearchRecordStore):
    """Exactly the documented freeze semantics, recording every call."""

    def __init__(self) -> None:
        self.records: dict[tuple, FuturesForwardResearchRecord] = {}
        self.calls: list[tuple[FuturesForwardResearchRecord, ...]] = []

    def store(self, records):
        self.calls.append(records)
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


class CountingStore(FuturesForwardResearchRecordStore):
    def __init__(self, count: object) -> None:
        self.count = count

    def store(self, records):
        return self.count


def _use_case(
    repository: FuturesHistoricalMarketDataRepository,
    store: FuturesForwardResearchRecordStore,
) -> FreezeFuturesForwardResearchDecisionUseCase:
    return FreezeFuturesForwardResearchDecisionUseCase(
        repository, FuturesAssetAnalysisGenerator(), store
    )


def _freeze(
    repository: ConformingRepository,
    store: ReferenceStore,
    as_of: PointInTime,
    *,
    strategy: Strategy = _STRATEGY,
    contract: FuturesContract = _ES_DEC,
) -> FuturesForwardResearchRecord | None:
    return _use_case(repository, store).execute(contract, _DAILY, strategy, as_of)


# ---------------------------------------------------------------------------
# No data, warm-up and the first decision
# ---------------------------------------------------------------------------


def test_no_stored_history_freezes_nothing() -> None:
    repository, store = ConformingRepository(), ReferenceStore()

    assert _freeze(repository, store, _instant(30)) is None
    assert store.calls == []
    assert len(repository.queries) == 1


def test_nineteen_bars_are_warm_up_and_freeze_nothing() -> None:
    store = ReferenceStore()

    assert _freeze(ConformingRepository(_rising(19)), store, _instant(19)) is None
    assert store.calls == []


def test_the_twentieth_bar_freezes_the_first_decision() -> None:
    store = ReferenceStore()

    record = _freeze(ConformingRepository(_rising(20)), store, _instant(20))

    assert record is not None
    assert record.decision_instant == _instant(20)
    assert record.result.recommendation.action == _BUY
    assert store.calls == [(record,)]


def test_a_thirty_bar_history_freezes_only_the_latest_decision() -> None:
    """Forward testing takes the as-of decision, never a back-fill of old ones."""
    store = ReferenceStore()

    record = _freeze(ConformingRepository(_rising(30)), store, _instant(30))

    assert record is not None
    assert record.decision_instant == _instant(30)
    assert store.calls == [(record,)]
    assert list(store.records) == [record.natural_key]
    context = record.result.market_observation_context
    assert context.recent_closes == tuple(bar.close for bar in _rising(30)[-20:])


def test_the_replay_query_is_the_contract_up_to_as_of() -> None:
    repository = ConformingRepository(_rising(25))

    _freeze(repository, ReferenceStore(), _instant(25))

    (query,) = repository.queries
    assert query == FuturesHistoricalMarketDataQuery(_ES_DEC, _DAILY, None, _instant(25))


# ---------------------------------------------------------------------------
# The as-of boundary
# ---------------------------------------------------------------------------


def test_as_of_before_any_history_freezes_nothing() -> None:
    store = ReferenceStore()

    assert (
        _freeze(ConformingRepository(_rising(25)), store, PointInTime("2026-07-01T00:00:00Z"))
        is None
    )
    assert store.calls == []


def test_a_bar_completing_exactly_at_as_of_is_visible() -> None:
    record = _freeze(ConformingRepository(_rising(25)), ReferenceStore(), _instant(22))

    assert record is not None
    assert record.decision_instant == _instant(22)


def test_as_of_after_the_latest_bar_decides_on_that_bar() -> None:
    record = _freeze(
        ConformingRepository(_rising(25)), ReferenceStore(), PointInTime("2026-08-28T12:00:00Z")
    )

    assert record is not None
    assert record.decision_instant == _instant(25)  # 2026-08-25, not 2026-08-28


def test_as_of_later_than_the_latest_bar_is_the_same_decision() -> None:
    repository, store = ConformingRepository(_rising(25)), ReferenceStore()

    at_bar = _freeze(repository, store, _instant(25))
    days_later = _freeze(repository, store, PointInTime("2026-08-28T12:00:00Z"))

    assert days_later == at_bar
    assert len(store.records) == 1


def test_an_offset_spelled_as_of_is_the_same_decision() -> None:
    repository, store = ConformingRepository(_rising(25)), ReferenceStore()

    z = _freeze(repository, store, _instant(25))
    offset = _freeze(repository, store, PointInTime("2026-08-25T16:00:00-05:00"))

    assert offset == z
    assert len(store.calls) == 2
    assert len(store.records) == 1


def test_the_latest_bar_is_chosen_semantically_despite_the_text_order_trap() -> None:
    """Session 21 completes half a second after session 20 on the same day."""
    whole = _instant(20)
    fractional = PointInTime("2026-08-20T21:00:00.5Z")
    assert fractional.value < whole.value
    bars = (*_rising(20), _bar(21, "7606", instant=fractional))
    repository = ConformingRepository(tuple(reversed(bars)))

    at_fraction = _freeze(repository, ReferenceStore(), fractional)
    before_fraction = _freeze(repository, ReferenceStore(), PointInTime("2026-08-20T21:00:00.25Z"))

    assert at_fraction is not None and at_fraction.decision_instant == fractional
    assert at_fraction.result.market_observation_context.latest_quote == QuoteValue(Decimal("7606"))
    assert before_fraction is not None and before_fraction.decision_instant == whole


# ---------------------------------------------------------------------------
# Retries, later data and later boundaries
# ---------------------------------------------------------------------------


def test_the_same_inputs_retry_to_an_equal_record_idempotently() -> None:
    repository, store = ConformingRepository(_rising(25)), ReferenceStore()

    first = _freeze(repository, store, _instant(25))
    second = _freeze(repository, store, _instant(25))
    fresh = _freeze(ConformingRepository(_rising(25)), ReferenceStore(), _instant(25))

    assert first == second == fresh
    assert len(store.calls) == 2
    assert len(store.records) == 1


def test_bars_appended_after_as_of_cannot_change_the_decision() -> None:
    """Load-bearing: the appended collapse would flip the signal if it were read."""
    repository, store = ConformingRepository(_rising(25)), ReferenceStore()
    before = _freeze(repository, store, _instant(25))

    repository.bars = (
        *repository.bars,
        *(
            _bar(session, close, "250000")
            for session, close in zip(
                range(26, 31), ["3000", "-500", "-4000", "-7500", "-11000"], strict=True
            )
        ),
    )
    after = _freeze(repository, store, _instant(25))

    assert after == before
    assert after.result.recommendation.action == _BUY
    assert len(store.records) == 1
    assert repository.queries[-1].end == _instant(25)


def test_a_later_as_of_freezes_a_new_decision_beside_the_earlier_one() -> None:
    repository, store = ConformingRepository(_rising(25)), ReferenceStore()
    earlier = _freeze(repository, store, _instant(25))
    repository.bars = (
        *repository.bars,
        *(
            _bar(session, close, "250000")
            for session, close in zip(
                range(26, 31), ["3000", "-500", "-4000", "-7500", "-11000"], strict=True
            )
        ),
    )

    later = _freeze(repository, store, _instant(30))

    assert later is not None and earlier is not None
    assert later.decision_instant == _instant(30)
    assert later.natural_key != earlier.natural_key
    assert later.result.recommendation.action == _SELL
    assert store.records == {earlier.natural_key: earlier, later.natural_key: later}


def test_two_strategies_at_one_instant_freeze_side_by_side() -> None:
    repository, store = ConformingRepository(_rising(25)), ReferenceStore()

    alpha = _freeze(repository, store, _instant(25), strategy=Strategy(StrategyIdentity("alpha")))
    beta = _freeze(repository, store, _instant(25), strategy=Strategy(StrategyIdentity("beta")))

    assert alpha.decision_instant == beta.decision_instant
    assert alpha.natural_key != beta.natural_key
    assert len(store.records) == 2
    assert (
        _freeze(repository, store, _instant(25), strategy=Strategy(StrategyIdentity("alpha")))
        == alpha
    )
    assert len(store.records) == 2


def test_other_contracts_in_the_repository_do_not_enter_the_decision() -> None:
    noise = tuple(_bar(session, "-9999", contract=_MES_DEC) for session in range(1, 26))
    alone = _freeze(ConformingRepository(_rising(25)), ReferenceStore(), _instant(25))

    mixed = _freeze(ConformingRepository(_rising(25) + noise), ReferenceStore(), _instant(25))

    assert mixed == alone


# ---------------------------------------------------------------------------
# Frozen means frozen: changed evidence under the same key conflicts
# ---------------------------------------------------------------------------


def test_a_session_back_filled_inside_the_window_conflicts_with_the_frozen_decision() -> None:
    """Session 18 was missing when the decision was frozen, then arrived later.

    Its close (7603) differs from its neighbours, so the back-filled window is
    genuinely different evidence. A back-filled session identical to what it
    displaces would recompute an equal record and be an idempotent retry.
    """
    full = _rising(22)
    repository, store = ConformingRepository(full[:17] + full[18:]), ReferenceStore()
    frozen = _freeze(repository, store, _instant(22))
    assert frozen is not None and frozen.decision_instant == _instant(22)

    repository.bars = full  # the back-fill
    with pytest.raises(FuturesForwardResearchRecordConflictError):
        _freeze(repository, store, _instant(22))

    assert store.records == {frozen.natural_key: frozen}


def test_changed_evidence_at_the_same_instant_conflicts_rather_than_replacing() -> None:
    """Same contract, timeframe, instant and strategy; one earlier close differs."""
    repository, store = ConformingRepository(_rising(25)), ReferenceStore()
    frozen = _freeze(repository, store, _instant(25))

    corrected = list(_rising(25))
    corrected[14] = _bar(15, "7599.75")
    repository.bars = tuple(corrected)
    with pytest.raises(FuturesForwardResearchRecordConflictError):
        _freeze(repository, store, _instant(25))

    assert store.records == {frozen.natural_key: frozen}


def test_the_recomputed_record_really_differs_under_the_same_key() -> None:
    """Guard: the conflict tests are load-bearing only if this holds."""
    original = _freeze(ConformingRepository(_rising(25)), ReferenceStore(), _instant(25))
    corrected = list(_rising(25))
    corrected[14] = _bar(15, "7599.75")

    recomputed = _freeze(ConformingRepository(tuple(corrected)), ReferenceStore(), _instant(25))

    assert recomputed.natural_key == original.natural_key
    assert recomputed != original


# ---------------------------------------------------------------------------
# Store interaction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("count", [0, 2, True, "1", None])
def test_an_unexpected_store_count_is_a_contract_violation(count: object) -> None:
    with pytest.raises(ForwardResearchContractViolationError):
        _use_case(ConformingRepository(_rising(25)), CountingStore(count)).execute(
            _ES_DEC, _DAILY, _STRATEGY, _instant(25)
        )


def test_a_store_conflict_propagates_unchanged() -> None:
    class ConflictingStore(FuturesForwardResearchRecordStore):
        def store(self, records):
            raise FuturesForwardResearchRecordConflictError("frozen elsewhere")

    with pytest.raises(FuturesForwardResearchRecordConflictError, match="frozen elsewhere"):
        _use_case(ConformingRepository(_rising(25)), ConflictingStore()).execute(
            _ES_DEC, _DAILY, _STRATEGY, _instant(25)
        )


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("timeframe", [Timeframe("1m"), Timeframe("1h")], ids=["1m", "1h"])
def test_a_non_daily_timeframe_is_rejected_before_any_read(timeframe: Timeframe) -> None:
    repository, store = ConformingRepository(_rising(25)), ReferenceStore()

    with pytest.raises(UnsupportedFuturesReplayTimeframeError, match="session-daily"):
        _use_case(repository, store).execute(_ES_DEC, timeframe, _STRATEGY, _instant(25))

    assert repository.queries == []
    assert store.calls == []


@pytest.mark.parametrize("position", [0, 1, 2, 3])
def test_every_input_is_type_checked_before_any_read(position: int) -> None:
    arguments: list[object] = [_ES_DEC, _DAILY, _STRATEGY, _instant(25)]
    arguments[position] = "not an input"
    repository, store = ConformingRepository(_rising(25)), ReferenceStore()

    with pytest.raises(TypeError):
        _use_case(repository, store).execute(*arguments)  # type: ignore[arg-type]

    assert repository.queries == []
    assert store.calls == []


@pytest.mark.parametrize("position", [0, 1, 2])
def test_every_dependency_is_type_checked(position: int) -> None:
    dependencies: list[object] = [
        ConformingRepository(),
        FuturesAssetAnalysisGenerator(),
        ReferenceStore(),
    ]
    dependencies[position] = "not a dependency"

    with pytest.raises(TypeError):
        FreezeFuturesForwardResearchDecisionUseCase(*dependencies)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Boundaries and exports
# ---------------------------------------------------------------------------


def _module_tree() -> ast.Module:
    import northstar_application.application_services.freeze_futures_forward_research_decision as m

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
        "AcquireFuturesDailyHistoryUseCase",
        "FuturesTradingSessionResolver",
        "MeasureFuturesRecommendationOutcomeUseCase",
    ):
        assert forbidden not in names
    for module_path in modules:
        assert module_path.split(".")[0] not in {
            "databento",
            "exchange_calendars",
            "northstar_infrastructure",
            "sqlite3",
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


def test_the_use_case_reads_no_clock_no_action_and_computes_nothing() -> None:
    for node in ast.walk(_module_tree()):
        if isinstance(node, ast.Attribute):
            assert node.attr not in {"now", "today", "utcnow", "action", "value"}
        if isinstance(node, ast.BinOp) and not isinstance(node.op, ast.BitOr):
            raise AssertionError(f"arithmetic at line {node.lineno}")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in {"sorted", "Decimal", "sum"}


def test_the_use_case_is_exported() -> None:
    import northstar_application.application_services as services

    assert "FreezeFuturesForwardResearchDecisionUseCase" in services.__all__
    assert not hasattr(services, "_SUPPORTED_TIMEFRAME")
