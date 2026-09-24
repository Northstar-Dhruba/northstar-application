"""Tests for the operational futures paper-trading session."""

from __future__ import annotations

import ast
import importlib
import inspect
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
from northstar_core.futures import (
    FuturesContract,
    FuturesOHLCVBar,
    FuturesProductEconomics,
    FuturesProductReference,
)
from northstar_core.paper_trading import (
    FuturesContractCount,
    FuturesPosition,
    OrderSide,
    PaperPortfolioIdentity,
)
from northstar_core.strategy import (
    FuturesAssetAnalysis,
    FuturesAssetAnalysisGenerator,
    FuturesMarketObservationContext,
    FuturesRecommendation,
    RecommendationAction,
    Strategy,
    StrategyIdentity,
)

from northstar_application.application_services import (
    BuildFuturesPaperTradingReportUseCase,
    FreezeFuturesForwardResearchDecisionUseCase,
    FuturesAnalysisResult,
    FuturesExecutionIntentNoIntentReason,
    FuturesForwardResearchRecord,
    FuturesPaperPortfolioStrategyConflictError,
    FuturesPaperTradingSessionResult,
    RunFuturesPaperTradingSessionUseCase,
)
from northstar_application.ports import (
    FuturesForwardResearchRecordConflictError,
    FuturesForwardResearchRecordQuery,
    FuturesForwardResearchRecordRepository,
    FuturesForwardResearchRecordStore,
    FuturesHistoricalMarketDataQuery,
    FuturesHistoricalMarketDataRepository,
    FuturesPaperFillConflictError,
    FuturesPaperFillQuery,
    FuturesPaperFillRepository,
    FuturesPaperFillStore,
    FuturesPaperOrderConflictError,
    FuturesPaperOrderQuery,
    FuturesPaperOrderRepository,
    FuturesPaperOrderStore,
    FuturesProductEconomicsConflictError,
    FuturesProductEconomicsStore,
)

_ES_DEC = FuturesContract(
    FuturesProductReference(Symbol("ES"), ExchangeCode("CME")), ExpirationDate("2026-12-18")
)
_ES_MAR = FuturesContract(_ES_DEC.product, ExpirationDate("2027-03-19"))
_DAILY = Timeframe("1d")
_ALPHA = StrategyIdentity("alpha")
_BETA = StrategyIdentity("beta")
_PORTFOLIO = PaperPortfolioIdentity("futures-paper-alpha")
_ONE = FuturesContractCount(1)

BUY = OrderSide.BUY
SELL = OrderSide.SELL
_HOLD = FuturesExecutionIntentNoIntentReason.HOLD
_MET = FuturesExecutionIntentNoIntentReason.TARGET_ALREADY_MET


def _session_dates(count: int) -> list[date]:
    dates: list[date] = []
    day = date(2026, 6, 1)
    while len(dates) < count:
        if day.weekday() < 5 and day != date(2026, 6, 19):
            dates.append(day)
        day += timedelta(days=1)
    return dates


_DATES = _session_dates(35)


def _at(session: int) -> PointInTime:
    return PointInTime(f"{_DATES[session - 1].isoformat()}T21:00:00Z")


def _bar(
    session: int,
    close: str,
    *,
    open_: str | None = None,
    high: str | None = None,
    low: str | None = None,
    volume: str = "1000",
    contract: FuturesContract = _ES_DEC,
) -> FuturesOHLCVBar:
    closing = Decimal(close)
    opening = Decimal(open_) if open_ is not None else closing
    return FuturesOHLCVBar(
        contract=contract,
        point_in_time=_at(session),
        timeframe=_DAILY,
        open=QuoteValue(opening),
        high=QuoteValue(Decimal(high) if high is not None else max(opening, closing) + 2),
        low=QuoteValue(Decimal(low) if low is not None else min(opening, closing) - 2),
        close=QuoteValue(closing),
        volume=Quantity(Decimal(volume)),
    )


# Fifteen flat closes, then a steady rise: session 25 decides BUY.
_RISE = ["7600"] * 15 + [str(7601 + index) for index in range(10)]


def _phase_a(sessions: int = 25, contract: FuturesContract = _ES_DEC) -> tuple:
    return tuple(
        _bar(session, close, contract=contract)
        for session, close in enumerate(_RISE[:sessions], start=1)
    )


_S26 = _bar(26, "7611", open_="7650", high="7700", low="7600")
_S27 = _bar(27, "3000", open_="7600", high="7610", low="2990", volume="250000")
_S28 = _bar(28, "2900", open_="2950", high="3050", low="2800", volume="250000")
_S29 = _bar(29, "2900", open_="2900", high="2950", low="2850")


def _compare_text(left: str, right: str) -> int:
    return (left > right) - (left < right)


# ---------------------------------------------------------------------------
# Reference ports over one shared in-memory world
# ---------------------------------------------------------------------------


class World:
    def __init__(self, *bars: FuturesOHLCVBar) -> None:
        self.bars: list[FuturesOHLCVBar] = list(bars)
        self.records: dict[tuple, FuturesForwardResearchRecord] = {}
        self.orders: dict[str, object] = {}
        self.fills: dict[str, object] = {}

    def counts(self) -> tuple[int, int, int]:
        return (len(self.records), len(self.orders), len(self.fills))


def _immutable_put(target: dict, key, value, error: type[Exception]) -> None:
    existing = target.get(key)
    if existing is not None and existing != value:
        raise error("a different value is already stored under this key")
    target[key] = value


class MarketRepository(FuturesHistoricalMarketDataRepository):
    def __init__(self, world: World) -> None:
        self.world = world

    def get_bars(self, query: FuturesHistoricalMarketDataQuery):
        matching = [
            bar
            for bar in self.world.bars
            if bar.contract == query.contract
            and bar.timeframe == query.timeframe
            and query.covers(bar.point_in_time)
        ]
        return tuple(
            sorted(matching, key=cmp_to_key(lambda a, b: a.point_in_time.compare(b.point_in_time)))
        )


class ForwardStore(FuturesForwardResearchRecordStore):
    def __init__(self, world: World) -> None:
        self.world = world

    def store(self, records):
        for record in records:
            _immutable_put(
                self.world.records,
                record.natural_key,
                record,
                FuturesForwardResearchRecordConflictError,
            )
        return len(records)


class ForwardRepository(FuturesForwardResearchRecordRepository):
    def __init__(self, world: World) -> None:
        self.world = world

    def get_records(self, query: FuturesForwardResearchRecordQuery):
        def compare(left, right) -> int:
            instant = left.decision_instant.compare(right.decision_instant)
            return instant or _compare_text(
                left.strategy_identity.identity, right.strategy_identity.identity
            )

        matching = [
            r
            for r in self.world.records.values()
            if r.contract == query.contract and r.timeframe == query.timeframe
        ]
        return tuple(sorted(matching, key=cmp_to_key(compare)))


class OrderStore(FuturesPaperOrderStore):
    def __init__(self, world: World) -> None:
        self.world = world

    def store(self, orders):
        for order in orders:
            _immutable_put(
                self.world.orders, order.identity.identity, order, FuturesPaperOrderConflictError
            )
        return len(orders)


class OrderRepository(FuturesPaperOrderRepository):
    def __init__(self, world: World) -> None:
        self.world = world

    def get_orders(self, query: FuturesPaperOrderQuery):
        def compare(left, right) -> int:
            instant = left.intent.decided_at.compare(right.intent.decided_at)
            return instant or _compare_text(left.identity.identity, right.identity.identity)

        owned = [
            o
            for o in self.world.orders.values()
            if o.intent.portfolio_identity == query.portfolio_identity
        ]
        return tuple(sorted(owned, key=cmp_to_key(compare)))


class FillStore(FuturesPaperFillStore):
    def __init__(self, world: World) -> None:
        self.world = world

    def store(self, fills):
        for fill in fills:
            _immutable_put(
                self.world.fills, fill.identity.identity, fill, FuturesPaperFillConflictError
            )
        return len(fills)


class FillRepository(FuturesPaperFillRepository):
    def __init__(self, world: World) -> None:
        self.world = world

    def get_fills(self, query: FuturesPaperFillQuery):
        def compare(left, right) -> int:
            instant = left.filled_at.compare(right.filled_at)
            return instant or _compare_text(
                left.order_identity.identity, right.order_identity.identity
            )

        owned = [
            f for f in self.world.fills.values() if f.portfolio_identity == query.portfolio_identity
        ]
        return tuple(sorted(owned, key=cmp_to_key(compare)))


def _ports(world: World) -> dict:
    return {
        "market_repository": MarketRepository(world),
        "forward_store": ForwardStore(world),
        "forward_repository": ForwardRepository(world),
        "order_store": OrderStore(world),
        "order_repository": OrderRepository(world),
        "fill_store": FillStore(world),
        "fill_repository": FillRepository(world),
    }


def _session(
    world: World,
    through: PointInTime,
    *,
    strategy: StrategyIdentity = _ALPHA,
    target: FuturesContractCount = _ONE,
    contract: FuturesContract = _ES_DEC,
) -> FuturesPaperTradingSessionResult:
    return RunFuturesPaperTradingSessionUseCase(**_ports(world)).execute(
        contract, strategy, _PORTFOLIO, target, through
    )


def _freeze_directly(world: World, through: PointInTime) -> FuturesForwardResearchRecord:
    record = FreezeFuturesForwardResearchDecisionUseCase(
        MarketRepository(world), FuturesAssetAnalysisGenerator(), ForwardStore(world)
    ).execute(_ES_DEC, _DAILY, Strategy(_ALPHA), through)
    assert record is not None
    return record


def _action(record: FuturesForwardResearchRecord) -> str:
    return record.result.recommendation.action.value


# ---------------------------------------------------------------------------
# Freeze, run and report through one cutoff
# ---------------------------------------------------------------------------


def test_a_buy_decision_is_frozen_and_left_pending_without_a_next_bar() -> None:
    world = World(*_phase_a())

    session = _session(world, _at(25))

    record = session.frozen_record
    assert _action(record) == "BUY"
    assert record.decision_instant == _at(25)
    assert world.records == {record.natural_key: record}
    (result,) = session.run.results
    assert result.record == record
    assert (result.order.intent.side, result.order.intent.contracts) == (BUY, _ONE)
    assert result.fill is None
    assert (session.report.order_count, session.report.pending_order_count) == (1, 1)
    assert session.run.available_through == _at(25)
    assert session.run.query == FuturesForwardResearchRecordQuery(_ES_DEC, Timeframe("1d"))
    assert world.counts() == (1, 1, 0)


def test_the_next_session_settles_the_pending_order_at_its_open() -> None:
    world = World(*_phase_a())
    pending = _session(world, _at(25)).run.results[0].order

    world.bars.append(_S26)
    session = _session(world, _at(26))

    first = session.run.results[0]
    assert first.order == pending
    assert first.fill.fill_quote == _S26.open
    assert first.fill.filled_at == _at(26)
    assert session.frozen_record.decision_instant == _at(26)
    assert session.report.fill_count == 1
    assert session.report.positions == (FuturesPosition(_ES_DEC, 1, _S26.open),)
    assert (len(world.orders), len(world.fills)) == (1, 1)


def test_a_pending_order_settles_when_the_freeze_reuses_an_existing_decision() -> None:
    world = World(*_phase_a())
    _session(world, _at(25))
    world.bars.append(_S26)
    already_frozen = _freeze_directly(world, _at(26))
    records_before = dict(world.records)

    session = _session(world, _at(26))

    assert session.frozen_record == already_frozen
    assert world.records == records_before
    assert session.run.results[0].fill.fill_quote == _S26.open
    assert len(world.fills) == 1


def test_a_repeated_session_at_one_cutoff_is_equal_and_writes_nothing_new() -> None:
    world = World(*_phase_a(), _S26)
    first = _session(world, _at(26))
    counts = world.counts()

    second = _session(world, _at(26))

    assert second == first
    assert world.counts() == counts


def test_a_later_cutoff_without_a_new_bar_reproduces_the_same_decision() -> None:
    world = World(*_phase_a())
    first = _session(world, _at(25))

    later = _session(world, _at(28))

    assert later.frozen_record == first.frozen_record
    assert later.run.available_through == _at(28)
    assert later.run.results[0].order == first.run.results[0].order
    assert later.report.pending_order_count == 1
    assert world.counts() == (1, 1, 0)


def test_warm_up_freezes_nothing_and_still_reports() -> None:
    world = World(*_phase_a(19))

    session = _session(world, _at(19))

    assert session.frozen_record is None
    assert session.run.results == ()
    assert session.report.decision_count == 0
    assert session.report.positions == ()
    assert world.counts() == (0, 0, 0)


def test_daily_sessions_keep_hold_and_target_already_met_without_orders() -> None:
    world = World(*_phase_a())
    for session_number, bar in ((25, None), (26, _S26), (27, _S27), (28, _S28), (29, _S29)):
        if bar is not None:
            world.bars.append(bar)
        session = _session(world, _at(session_number))

    by_instant = {r.record.decision_instant: r for r in session.run.results}
    assert by_instant[_at(28)].decision.no_intent_reason is _MET
    assert by_instant[_at(28)].order is None
    assert by_instant[_at(29)].decision.no_intent_reason is _HOLD
    assert by_instant[_at(29)].order is None
    assert _action(by_instant[_at(27)].record) == "SELL"
    assert by_instant[_at(27)].order.intent.contracts == FuturesContractCount(2)
    assert session.report.positions == (FuturesPosition(_ES_DEC, -1, _S28.open),)
    assert session.frozen_record == by_instant[_at(29)].record


def test_the_report_is_exactly_the_report_builders() -> None:
    world = World(*_phase_a(), _S26, _S27)
    session = _session(world, _at(27))

    assert BuildFuturesPaperTradingReportUseCase().execute(session.run) == session.report
    assert session.report.run == session.run


# ---------------------------------------------------------------------------
# Conflicts
# ---------------------------------------------------------------------------


def test_another_strategy_on_the_portfolio_is_a_stable_conflict() -> None:
    world = World(*_phase_a())
    _session(world, _at(25))
    orders, fills = dict(world.orders), dict(world.fills)

    with pytest.raises(FuturesPaperPortfolioStrategyConflictError) as caught:
        _session(world, _at(25), strategy=_BETA)

    assert isinstance(caught.value, ValueError)
    assert "one paper portfolio belongs to one strategy" in str(caught.value)
    assert (world.orders, world.fills) == (orders, fills)


def test_a_changed_target_for_an_executed_decision_conflicts() -> None:
    world = World(*_phase_a())
    _session(world, _at(25))
    orders = dict(world.orders)

    with pytest.raises(FuturesPaperOrderConflictError):
        _session(world, _at(25), target=FuturesContractCount(2))

    assert world.orders == orders


def test_inputs_and_ports_are_type_checked() -> None:
    world = World()
    for name in _ports(world):
        with pytest.raises(TypeError, match=name):
            RunFuturesPaperTradingSessionUseCase(**{**_ports(world), name: object()})

    use_case = RunFuturesPaperTradingSessionUseCase(**_ports(world))
    arguments = (_ES_DEC, _ALPHA, _PORTFOLIO, _ONE, _at(25))
    for index, label in enumerate(
        ("contract", "strategy identity", "portfolio identity", "target contracts", "available")
    ):
        bad = list(arguments)
        bad[index] = "wrong"
        with pytest.raises(TypeError, match=label):
            use_case.execute(*bad)
    assert world.counts() == (0, 0, 0)


# ---------------------------------------------------------------------------
# FuturesPaperTradingSessionResult
# ---------------------------------------------------------------------------


def _record(
    instant: PointInTime,
    *,
    contract: FuturesContract = _ES_DEC,
    strategy: StrategyIdentity = _ALPHA,
) -> FuturesForwardResearchRecord:
    context = FuturesMarketObservationContext(
        contract=contract,
        timeframe=_DAILY,
        observed_at=instant,
        latest_quote=QuoteValue(Decimal("7610")),
        previous_close=QuoteValue(Decimal("7609")),
        latest_volume=Quantity(Decimal("1000")),
        session_high=QuoteValue(Decimal("7700")),
        session_low=QuoteValue(Decimal("7500")),
        recent_closes=tuple(QuoteValue(Decimal(7600 + index)) for index in range(20)),
        recent_volumes=tuple(Quantity(Decimal(1000)) for _ in range(20)),
    )
    recommendation = FuturesRecommendation(
        action=RecommendationAction("BUY"),
        asset_analysis=FuturesAssetAnalysis(contract, instant, ("signal",)),
        strategy_identity=strategy,
        point_in_time=instant,
    )
    return FuturesForwardResearchRecord(FuturesAnalysisResult(recommendation, context))


def _valid() -> FuturesPaperTradingSessionResult:
    return _session(World(*_phase_a(), _S26), _at(26))


def test_a_warm_up_result_needs_no_record() -> None:
    session = _valid()

    result = FuturesPaperTradingSessionResult(None, session.run, session.report)

    assert result.frozen_record is None


@pytest.mark.parametrize(
    ("record", "message"),
    [
        (_record(_at(26), contract=_ES_MAR), "contract and timeframe"),
        (_record(_at(26), strategy=_BETA), "run's strategy"),
        (_record(_at(27)), "after the run cutoff"),
        (_record(_at(26)), "one of the run's decisions"),
    ],
    ids=["contract", "strategy", "after-cutoff", "not-executed"],
)
def test_an_incoherent_frozen_record_is_rejected(record, message: str) -> None:
    session = _valid()

    with pytest.raises(ValueError, match=message):
        FuturesPaperTradingSessionResult(record, session.run, session.report)


def test_result_types_and_report_pairing_are_checked() -> None:
    session = _valid()
    other = _session(World(*_phase_a()), _at(25))

    with pytest.raises(TypeError, match="frozen record"):
        FuturesPaperTradingSessionResult("record", session.run, session.report)
    with pytest.raises(TypeError, match="run must be"):
        FuturesPaperTradingSessionResult(session.frozen_record, session.report, session.report)
    with pytest.raises(TypeError, match="report must be"):
        FuturesPaperTradingSessionResult(session.frozen_record, session.run, session.run)
    with pytest.raises(ValueError, match="report must describe"):
        FuturesPaperTradingSessionResult(session.frozen_record, session.run, other.report)


# ---------------------------------------------------------------------------
# Economics store port
# ---------------------------------------------------------------------------


def test_the_economics_store_port_is_an_abstract_product_level_batch_store() -> None:
    with pytest.raises(TypeError):
        FuturesProductEconomicsStore()
    public = [name for name in vars(FuturesProductEconomicsStore) if not name.startswith("_")]
    signature = inspect.signature(FuturesProductEconomicsStore.store)

    assert public == ["store"]
    assert list(signature.parameters) == ["self", "economics"]
    assert "FuturesProductEconomics" in str(signature.parameters["economics"].annotation)
    assert "Contract" not in str(signature.parameters["economics"].annotation)
    assert signature.return_annotation in (int, "int")


def test_a_conforming_economics_store_can_be_implemented() -> None:
    class Store(FuturesProductEconomicsStore):
        def store(self, economics: tuple[FuturesProductEconomics, ...]) -> int:
            return len(economics)

    assert Store().store(()) == 0


def test_the_economics_conflict_error_lives_with_the_port() -> None:
    import northstar_application.ports.futures_product_economics_store as port_module

    assert issubclass(FuturesProductEconomicsConflictError, ValueError)
    assert FuturesProductEconomicsConflictError.__module__ == port_module.__name__


# ---------------------------------------------------------------------------
# Structure and exports
# ---------------------------------------------------------------------------

_MODULE = "northstar_application.application_services.run_futures_paper_trading_session"


def _imports() -> tuple[set[str], set[str]]:
    tree = ast.parse(Path(importlib.import_module(_MODULE).__file__).read_text(encoding="utf-8"))
    modules: set[str] = set()
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
    return modules, names


def test_the_session_neither_acquires_nor_values() -> None:
    modules, names = _imports()

    for module in modules:
        assert module.split(".")[0] not in {
            "northstar_infrastructure",
            "sqlite3",
            "databento",
            "exchange_calendars",
            "os",
            "argparse",
            "fastapi",
            "time",
            "datetime",
        }
        for forbidden in ("acquire", "valuation", "unrealized", "realized", "economics"):
            assert forbidden not in module
    for forbidden in (
        "AcquireFuturesDailyHistoryUseCase",
        "FuturesHistoricalMarketDataSource",
        "BuildFuturesPaperTradingValuationUseCase",
        "FuturesProductEconomicsRepository",
        "CalculateFuturesPaperTradingMetricsUseCase",
    ):
        assert forbidden not in names


def test_the_public_surface_is_exported() -> None:
    import northstar_application.application_services as services
    import northstar_application.ports as ports

    for name in (
        "RunFuturesPaperTradingSessionUseCase",
        "FuturesPaperTradingSessionResult",
        "FuturesPaperPortfolioStrategyConflictError",
    ):
        assert name in services.__all__
    for name in ("FuturesProductEconomicsStore", "FuturesProductEconomicsConflictError"):
        assert name in ports.__all__
