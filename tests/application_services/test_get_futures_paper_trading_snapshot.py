"""Tests for the canonical futures paper-trading snapshot read."""

from __future__ import annotations

import ast
import importlib
from datetime import date, timedelta
from decimal import Decimal
from functools import cmp_to_key
from pathlib import Path

import pytest
from northstar_core.derivatives import ExpirationDate, QuoteValue
from northstar_core.foundation.value_objects import (
    Currency,
    ExchangeCode,
    Money,
    PointInTime,
    Quantity,
    Symbol,
    Timeframe,
)
from northstar_core.futures import (
    FuturesContract,
    FuturesOHLCVBar,
    FuturesPointValue,
    FuturesProductEconomics,
    FuturesProductReference,
)
from northstar_core.paper_trading import (
    FuturesContractCount,
    FuturesExecutionIntent,
    FuturesPaperFill,
    FuturesPaperOrder,
    FuturesPaperPortfolio,
    FuturesPosition,
    OrderSide,
    PaperFillIdentity,
    PaperOrderIdentity,
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
    FreezeFuturesForwardResearchDecisionUseCase,
    FuturesAnalysisResult,
    FuturesForwardResearchRecord,
    FuturesPaperDecisionSnapshot,
    FuturesPaperPortfolioStrategyConflictError,
    FuturesPaperTradingSnapshot,
    FuturesProductEconomicsContractViolationError,
    FuturesProductEconomicsNotFoundError,
    GetFuturesPaperTradingSnapshotUseCase,
    RunFuturesPaperTradingSessionUseCase,
)
from northstar_application.ports import (
    FuturesForwardResearchRecordConflictError,
    FuturesForwardResearchRecordRepository,
    FuturesForwardResearchRecordStore,
    FuturesHistoricalMarketDataRepository,
    FuturesPaperFillConflictError,
    FuturesPaperFillRepository,
    FuturesPaperFillStore,
    FuturesPaperOrderConflictError,
    FuturesPaperOrderRepository,
    FuturesPaperOrderStore,
    FuturesProductEconomicsRepository,
)

_ES = FuturesProductReference(Symbol("ES"), ExchangeCode("CME"))
_FESX = FuturesProductReference(Symbol("FESX"), ExchangeCode("EUREX"))
_ES_DEC = FuturesContract(_ES, ExpirationDate("2026-12-18"))
_ES_MAR = FuturesContract(_ES, ExpirationDate("2027-03-19"))
_FESX_DEC = FuturesContract(_FESX, ExpirationDate("2026-12-18"))
_DAILY = Timeframe("1d")
_ALPHA = StrategyIdentity("alpha")
_BETA = StrategyIdentity("beta")
_PORTFOLIO = PaperPortfolioIdentity("futures-paper-alpha")
_USD, _EUR = Currency("USD"), Currency("EUR")
_ES_ECONOMICS = FuturesProductEconomics(_ES, FuturesPointValue(Decimal("50"), _USD))
_FESX_ECONOMICS = FuturesProductEconomics(_FESX, FuturesPointValue(Decimal("10"), _EUR))


def _session_dates(count: int) -> list[date]:
    dates: list[date] = []
    day = date(2026, 6, 1)
    while len(dates) < count:
        if day.weekday() < 5 and day != date(2026, 6, 19):
            dates.append(day)
        day += timedelta(days=1)
    return dates


_DATES = _session_dates(35)


def _at(session: int, clock: str = "21:00:00") -> PointInTime:
    return PointInTime(f"{_DATES[session - 1].isoformat()}T{clock}Z")


def _bar(
    session: int,
    close: str,
    *,
    open_: str | None = None,
    high: str | None = None,
    low: str | None = None,
    volume: str = "1000",
    contract: FuturesContract = _ES_DEC,
    instant: PointInTime | None = None,
) -> FuturesOHLCVBar:
    closing = Decimal(close)
    opening = Decimal(open_) if open_ is not None else closing
    return FuturesOHLCVBar(
        contract=contract,
        point_in_time=instant or _at(session),
        timeframe=_DAILY,
        open=QuoteValue(opening),
        high=QuoteValue(Decimal(high) if high is not None else max(opening, closing) + 2),
        low=QuoteValue(Decimal(low) if low is not None else min(opening, closing) - 2),
        close=QuoteValue(closing),
        volume=Quantity(Decimal(volume)),
    )


_RISE = ["7600"] * 15 + [str(7601 + index) for index in range(10)]


def _phase_a(sessions: int = 25, contract: FuturesContract = _ES_DEC) -> list[FuturesOHLCVBar]:
    return [_bar(n, close, contract=contract) for n, close in enumerate(_RISE[:sessions], start=1)]


_LATER = {
    26: _bar(26, "7611", open_="7650", high="7700", low="7600"),
    27: _bar(27, "3000", open_="7600", high="7610", low="2990", volume="250000"),
    28: _bar(28, "2900", open_="2950", high="3050", low="2800", volume="250000"),
    29: _bar(29, "2900", open_="2900", high="2950", low="2850"),
}


def _text(left: str, right: str) -> int:
    return (left > right) - (left < right)


# ---------------------------------------------------------------------------
# Reference ports over one in-memory world; writes are counted
# ---------------------------------------------------------------------------


class World:
    def __init__(self, *bars: FuturesOHLCVBar, economics=(_ES_ECONOMICS, _FESX_ECONOMICS)):
        self.bars = list(bars)
        self.records: dict = {}
        self.orders: dict = {}
        self.fills: dict = {}
        self.economics = {entry.reference: entry for entry in economics}
        self.writes = 0

    def facts(self) -> tuple:
        return (dict(self.records), dict(self.orders), dict(self.fills), list(self.bars))


def _put(world: World, target: dict, key, value, error) -> None:
    existing = target.get(key)
    if existing is not None and existing != value:
        raise error("different value under this key")
    target[key] = value
    world.writes += 1


class Market(FuturesHistoricalMarketDataRepository):
    def __init__(self, world: World) -> None:
        self.world = world

    def get_bars(self, query):
        matching = [
            b
            for b in self.world.bars
            if b.contract == query.contract
            and b.timeframe == query.timeframe
            and query.covers(b.point_in_time)
        ]
        return tuple(
            sorted(matching, key=cmp_to_key(lambda a, b: a.point_in_time.compare(b.point_in_time)))
        )


class ForwardStore(FuturesForwardResearchRecordStore):
    def __init__(self, world: World) -> None:
        self.world = world

    def store(self, records):
        for record in records:
            _put(
                self.world,
                self.world.records,
                record.natural_key,
                record,
                FuturesForwardResearchRecordConflictError,
            )
        return len(records)


class Forward(FuturesForwardResearchRecordRepository):
    def __init__(self, world: World) -> None:
        self.world = world

    def get_records(self, query):
        def compare(left, right) -> int:
            instant = left.decision_instant.compare(right.decision_instant)
            return instant or _text(
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
            _put(
                self.world,
                self.world.orders,
                order.identity.identity,
                order,
                FuturesPaperOrderConflictError,
            )
        return len(orders)


class Orders(FuturesPaperOrderRepository):
    def __init__(self, world: World) -> None:
        self.world = world

    def get_orders(self, query):
        def compare(left, right) -> int:
            instant = left.intent.decided_at.compare(right.intent.decided_at)
            return instant or _text(left.identity.identity, right.identity.identity)

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
            _put(
                self.world,
                self.world.fills,
                fill.identity.identity,
                fill,
                FuturesPaperFillConflictError,
            )
        return len(fills)


class Fills(FuturesPaperFillRepository):
    def __init__(self, world: World) -> None:
        self.world = world

    def get_fills(self, query):
        def compare(left, right) -> int:
            instant = left.filled_at.compare(right.filled_at)
            return instant or _text(left.order_identity.identity, right.order_identity.identity)

        owned = [
            f for f in self.world.fills.values() if f.portfolio_identity == query.portfolio_identity
        ]
        return tuple(sorted(owned, key=cmp_to_key(compare)))


class Economics(FuturesProductEconomicsRepository):
    def __init__(self, world: World) -> None:
        self.world = world

    def get_economics(self, reference):
        return self.world.economics.get(reference)


def _session(world: World, session: int, *, contract: FuturesContract = _ES_DEC) -> None:
    RunFuturesPaperTradingSessionUseCase(
        market_repository=Market(world),
        forward_store=ForwardStore(world),
        forward_repository=Forward(world),
        order_store=OrderStore(world),
        order_repository=Orders(world),
        fill_store=FillStore(world),
        fill_repository=Fills(world),
    ).execute(contract, _ALPHA, _PORTFOLIO, FuturesContractCount(1), _at(session))


def _daily(world: World, last: int) -> World:
    """Operate daily from session 25 through ``last``, adding each new bar first."""
    for session in range(25, last + 1):
        if session in _LATER:
            world.bars.append(_LATER[session])
        _session(world, session)
    return world


def _snapshot(
    world: World,
    through: PointInTime,
    *,
    contract: FuturesContract | None = _ES_DEC,
    strategy: StrategyIdentity = _ALPHA,
    economics: FuturesProductEconomicsRepository | None = None,
) -> FuturesPaperTradingSnapshot:
    return GetFuturesPaperTradingSnapshotUseCase(
        market_repository=Market(world),
        forward_repository=Forward(world),
        order_repository=Orders(world),
        fill_repository=Fills(world),
        economics_repository=economics or Economics(world),
    ).execute(contract, strategy, _PORTFOLIO, through)


def _action(snapshot: FuturesPaperTradingSnapshot) -> str:
    return snapshot.latest_forward_record.result.recommendation.action.value


def _record(
    instant: PointInTime,
    action: str = "BUY",
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
        recent_closes=tuple(QuoteValue(Decimal(7600 + i)) for i in range(20)),
        recent_volumes=tuple(Quantity(Decimal(1000)) for _ in range(20)),
    )
    recommendation = FuturesRecommendation(
        action=RecommendationAction(action),
        asset_analysis=FuturesAssetAnalysis(contract, instant, ("signal",)),
        strategy_identity=strategy,
        point_in_time=instant,
    )
    return FuturesForwardResearchRecord(FuturesAnalysisResult(recommendation, context))


def _plant(record: FuturesForwardResearchRecord, world: World) -> None:
    world.records[record.natural_key] = record


def _order(
    order_id: str,
    contract: FuturesContract,
    side: OrderSide,
    contracts: int,
    decided: PointInTime,
    strategy: StrategyIdentity = _ALPHA,
) -> FuturesPaperOrder:
    return FuturesPaperOrder(
        PaperOrderIdentity(order_id),
        FuturesExecutionIntent(
            _PORTFOLIO, contract, side, FuturesContractCount(contracts), strategy, decided
        ),
    )


def _fill(order: FuturesPaperOrder, quote: str, at: PointInTime) -> FuturesPaperFill:
    return FuturesPaperFill(
        PaperFillIdentity(f"fill-{order.identity.identity}"),
        order.identity,
        order.intent,
        order.intent.contracts,
        QuoteValue(Decimal(quote)),
        at,
    )


def _trade(world: World, order: FuturesPaperOrder, quote: str | None, at=None) -> None:
    world.orders[order.identity.identity] = order
    if quote is not None:
        fill = _fill(order, quote, at)
        world.fills[fill.identity.identity] = fill


# ---------------------------------------------------------------------------
# Selected contract: market and research
# ---------------------------------------------------------------------------


def test_no_market_data_and_no_record_is_a_valid_empty_snapshot() -> None:
    snapshot = _snapshot(World(), _at(25))

    assert snapshot.latest_market_bar is None
    assert snapshot.latest_forward_record is None
    assert snapshot.recent_decisions == ()
    assert snapshot.portfolio == FuturesPaperPortfolio(_PORTFOLIO, _ALPHA, (), _at(25))
    assert (snapshot.orders, snapshot.fills, snapshot.pending_orders) == ((), (), ())
    assert snapshot.valuation.contracts == ()
    assert snapshot.missing_economics is None
    assert (snapshot.portfolio_identity, snapshot.strategy_identity) == (_PORTFOLIO, _ALPHA)
    assert snapshot.available_through == _at(25)


def test_warm_up_shows_the_market_bar_and_no_research() -> None:
    snapshot = _snapshot(World(*_phase_a(19)), _at(19))

    assert snapshot.latest_market_bar == _phase_a(19)[-1]
    assert snapshot.latest_forward_record is None
    assert snapshot.recent_decisions == ()


def test_the_latest_frozen_buy_and_its_pending_order() -> None:
    world = _daily(World(*_phase_a()), 25)

    snapshot = _snapshot(world, _at(25))

    assert _action(snapshot) == "BUY"
    assert snapshot.latest_forward_record.decision_instant == _at(25)
    assert snapshot.latest_market_bar.point_in_time == _at(25)
    (decision,) = snapshot.recent_decisions
    assert decision.order.intent.side is OrderSide.BUY
    assert decision.fill is None
    assert snapshot.pending_orders == (decision.order,)
    assert snapshot.portfolio.positions == ()


@pytest.mark.parametrize(("session", "action"), [(27, "SELL"), (29, "HOLD")])
def test_the_latest_frozen_sell_and_hold(session: int, action: str) -> None:
    world = _daily(World(*_phase_a()), 29)

    snapshot = _snapshot(world, _at(session))

    assert _action(snapshot) == action
    assert snapshot.latest_forward_record.decision_instant == _at(session)


def test_decisions_are_newest_first_with_their_paper_state() -> None:
    world = _daily(World(*_phase_a()), 29)

    snapshot = _snapshot(world, _at(29))

    instants = [d.record.decision_instant for d in snapshot.recent_decisions]
    assert instants == [_at(s) for s in (29, 28, 27, 26, 25)]
    by_instant = {d.record.decision_instant: d for d in snapshot.recent_decisions}
    assert by_instant[_at(29)].order is None  # HOLD: no order persisted, no reason inferred
    assert by_instant[_at(27)].order.intent.side is OrderSide.SELL
    assert by_instant[_at(27)].fill.fill_quote == _LATER[28].open
    assert by_instant[_at(25)].fill.fill_quote == _LATER[26].open
    assert not hasattr(by_instant[_at(29)], "no_intent_reason")


def test_a_later_decision_and_fill_are_invisible_at_an_earlier_cutoff() -> None:
    world = _daily(World(*_phase_a()), 26)

    snapshot = _snapshot(world, _at(25))

    assert [d.record.decision_instant for d in snapshot.recent_decisions] == [_at(25)]
    assert snapshot.latest_market_bar.point_in_time == _at(25)
    assert snapshot.fills == ()
    assert len(snapshot.pending_orders) == 1
    assert snapshot.recent_decisions[0].fill is None


def test_other_strategies_and_contracts_are_excluded() -> None:
    world = _daily(World(*_phase_a(), *_phase_a(contract=_ES_MAR)), 25)
    for contract, strategy in ((_ES_DEC, _BETA), (_ES_MAR, _ALPHA)):
        FreezeFuturesForwardResearchDecisionUseCase(
            Market(world), FuturesAssetAnalysisGenerator(), ForwardStore(world)
        ).execute(contract, _DAILY, Strategy(strategy), _at(25))

    snapshot = _snapshot(world, _at(25))

    assert len(world.records) == 3
    assert [(d.record.contract, d.record.strategy_identity) for d in snapshot.recent_decisions] == [
        (_ES_DEC, _ALPHA)
    ]


def test_recent_decisions_are_bounded_to_the_newest_ten() -> None:
    world = World()
    for session in range(1, 13):
        _plant(_record(_at(session)), world)

    snapshot = _snapshot(world, _at(30))

    assert [d.record.decision_instant for d in snapshot.recent_decisions] == [
        _at(s) for s in range(12, 2, -1)
    ]


def test_latest_selection_is_semantic_not_textual() -> None:
    whole, half = _at(5), _at(5, "21:00:00.5")
    world = World(_bar(5, "100", instant=whole), _bar(5, "101", instant=half))
    _plant(_record(whole, "SELL"), world)
    _plant(_record(half, "BUY"), world)
    offset_half = PointInTime(f"{(_DATES[4] + timedelta(days=1)).isoformat()}T02:30:00.5+05:30")

    snapshot = _snapshot(world, offset_half)

    assert half.value < whole.value  # the text trap
    assert offset_half == half
    assert snapshot.latest_market_bar.point_in_time == half
    assert snapshot.latest_forward_record.decision_instant == half
    assert _action(snapshot) == "BUY"


def test_a_whole_portfolio_request_has_no_research_scope() -> None:
    world = _daily(World(*_phase_a()), 26)

    snapshot = _snapshot(world, _at(26), contract=None)

    assert (snapshot.contract, snapshot.latest_market_bar, snapshot.recent_decisions) == (
        None,
        None,
        (),
    )
    assert snapshot.portfolio.positions == (FuturesPosition(_ES_DEC, 1, _LATER[26].open),)


# ---------------------------------------------------------------------------
# Whole portfolio: orders, fills, positions
# ---------------------------------------------------------------------------


def test_an_open_long_and_an_open_short_with_exact_pnl() -> None:
    world = _daily(World(*_phase_a()), 28)

    long = _snapshot(world, _at(26))
    short = _snapshot(world, _at(28))

    assert long.portfolio.positions == (FuturesPosition(_ES_DEC, 1, QuoteValue(Decimal("7650"))),)
    (row,) = long.valuation.contracts
    assert (row.realized_pnl, row.unrealized_pnl) == (
        Money(Decimal("0"), _USD),
        Money(Decimal("-1950"), _USD),
    )
    assert short.portfolio.positions == (FuturesPosition(_ES_DEC, -1, QuoteValue(Decimal("2950"))),)
    (row,) = short.valuation.contracts
    assert (row.realized_pnl, row.mark_quote, row.unrealized_pnl) == (
        Money(Decimal("-235000"), _USD),
        QuoteValue(Decimal("2900")),
        Money(Decimal("2500"), _USD),
    )
    assert short.pending_orders == ()


def test_a_multi_contract_multi_currency_portfolio_is_whole() -> None:
    world = World(_bar(3, "5010", contract=_FESX_DEC), _bar(3, "105"))
    es = _order("o-es", _ES_DEC, OrderSide.BUY, 1, _at(1))
    fesx = _order("o-fesx", _FESX_DEC, OrderSide.SELL, 2, _at(1))
    pending = _order("o-mar", _ES_MAR, OrderSide.BUY, 1, _at(2))
    _trade(world, es, "100", _at(2))
    _trade(world, fesx, "5000", _at(2))
    _trade(world, pending, None)

    snapshot = _snapshot(world, _at(3))

    assert snapshot.recent_decisions == ()
    assert [p.contract for p in snapshot.portfolio.positions] == [_ES_DEC, _FESX_DEC]
    assert snapshot.pending_orders == (pending,)
    rows = {row.contract: row for row in snapshot.valuation.contracts}
    assert rows[_ES_DEC].unrealized_pnl == Money(Decimal("250"), _USD)
    assert rows[_FESX_DEC].unrealized_pnl == Money(Decimal("-200"), _EUR)
    for name in ("total_pnl", "total", "total_realized", "total_unrealized"):
        assert not hasattr(snapshot, name)
        assert not hasattr(snapshot.valuation, name)


def test_a_mixed_strategy_portfolio_is_rejected() -> None:
    world = World()
    _trade(world, _order("o-beta", _ES_DEC, OrderSide.BUY, 1, _at(1), _BETA), None)

    with pytest.raises(FuturesPaperPortfolioStrategyConflictError):
        _snapshot(world, _at(3))


# ---------------------------------------------------------------------------
# Valuation and missing economics
# ---------------------------------------------------------------------------


def test_missing_economics_leaves_every_other_section_available() -> None:
    world = _daily(World(*_phase_a(), economics=()), 26)

    snapshot = _snapshot(world, _at(26))

    assert snapshot.valuation is None
    assert snapshot.missing_economics == _ES
    assert _action(snapshot) == "BUY"
    assert snapshot.latest_market_bar == _LATER[26]
    assert snapshot.portfolio.positions == (FuturesPosition(_ES_DEC, 1, _LATER[26].open),)
    assert len(snapshot.recent_decisions) == 2


def test_the_first_missing_product_is_named_when_several_are_missing() -> None:
    world = World(economics=())
    _trade(world, _order("o-fesx", _FESX_DEC, OrderSide.BUY, 1, _at(1)), "5000", _at(2))
    _trade(world, _order("o-es", _ES_DEC, OrderSide.BUY, 1, _at(1)), "100", _at(3))

    snapshot = _snapshot(world, _at(4))

    assert snapshot.missing_economics == _FESX


def test_the_not_found_error_names_its_product() -> None:
    error = FuturesProductEconomicsNotFoundError("message", _ES)

    assert (str(error), error.reference) == ("message", _ES)
    assert FuturesProductEconomicsNotFoundError("message").reference is None


def test_a_misbehaving_economics_repository_still_raises() -> None:
    class Corrupt(FuturesProductEconomicsRepository):
        def get_economics(self, reference):
            return "economics"

    world = _daily(World(*_phase_a()), 26)

    with pytest.raises(FuturesProductEconomicsContractViolationError):
        _snapshot(world, _at(26), economics=Corrupt())


# ---------------------------------------------------------------------------
# Read only
# ---------------------------------------------------------------------------


def test_the_snapshot_writes_nothing_and_never_recomputes(monkeypatch) -> None:
    world = _daily(World(*_phase_a()), 28)
    facts, writes = world.facts(), world.writes

    def forbidden(*args, **kwargs):
        raise AssertionError("a read must never analyse or evaluate")

    monkeypatch.setattr(FuturesAssetAnalysisGenerator, "generate", forbidden)
    monkeypatch.setattr(Strategy, "evaluate_futures", forbidden)
    first = _snapshot(world, _at(28))
    second = _snapshot(world, _at(28))

    assert first == second
    assert world.facts() == facts
    assert world.writes == writes


_MODULE = "northstar_application.application_services.get_futures_paper_trading_snapshot"


def test_the_use_case_depends_on_read_ports_only() -> None:
    tree = ast.parse(Path(importlib.import_module(_MODULE).__file__).read_text(encoding="utf-8"))
    names = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    modules = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}

    assert not [name for name in names if name.endswith("Store")]
    for forbidden in (
        "FuturesAssetAnalysisGenerator",
        "Strategy",
        "FreezeFuturesForwardResearchDecisionUseCase",
        "RunFuturesPaperTradingSessionUseCase",
        "AcquireFuturesDailyHistoryUseCase",
        "FuturesHistoricalMarketDataSource",
    ):
        assert forbidden not in names
    for module in modules:
        assert module.split(".")[0] not in {"northstar_infrastructure", "datetime", "time"}


def test_ports_are_type_checked() -> None:
    world = World()
    ports = {
        "market_repository": Market(world),
        "forward_repository": Forward(world),
        "order_repository": Orders(world),
        "fill_repository": Fills(world),
        "economics_repository": Economics(world),
    }
    for name in ports:
        with pytest.raises(TypeError, match=name):
            GetFuturesPaperTradingSnapshotUseCase(**{**ports, name: object()})
    use_case = GetFuturesPaperTradingSnapshotUseCase(**ports)
    with pytest.raises(TypeError, match="contract"):
        use_case.execute("ES", _ALPHA, _PORTFOLIO, _at(1))
    with pytest.raises(TypeError, match="available-through"):
        use_case.execute(_ES_DEC, _ALPHA, _PORTFOLIO, "2026-06-01T21:00:00Z")


# ---------------------------------------------------------------------------
# Result coherence
# ---------------------------------------------------------------------------


def _valid() -> FuturesPaperTradingSnapshot:
    return _snapshot(_daily(World(*_phase_a()), 26), _at(26))


def _replace(snapshot: FuturesPaperTradingSnapshot, **changes) -> FuturesPaperTradingSnapshot:
    fields = {
        name: getattr(snapshot, name)
        for name in (
            "contract",
            "portfolio",
            "latest_market_bar",
            "recent_decisions",
            "orders",
            "fills",
            "valuation",
            "missing_economics",
        )
    }
    return FuturesPaperTradingSnapshot(**{**fields, **changes})


_FUTURE_DECISION = (FuturesPaperDecisionSnapshot(_record(_at(27)), None, None),)
_BETA_DECISION = (FuturesPaperDecisionSnapshot(_record(_at(26), strategy=_BETA), None, None),)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"latest_market_bar": _bar(26, "1", contract=_ES_MAR)}, "market bar"),
        ({"latest_market_bar": _bar(27, "1")}, "market bar"),
        ({"recent_decisions": _FUTURE_DECISION}, "visible daily"),
        ({"recent_decisions": _BETA_DECISION}, "visible daily"),
        ({"missing_economics": _ES}, "not both"),
        ({"valuation": None}, "not both"),
        ({"fills": ()}, "decision fills"),
    ],
)
def test_incoherent_snapshots_are_rejected(changes: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _replace(_valid(), **changes)


def test_decisions_must_be_newest_first() -> None:
    snapshot = _valid()

    with pytest.raises(ValueError, match="newest first"):
        _replace(snapshot, recent_decisions=tuple(reversed(snapshot.recent_decisions)))


def test_a_decision_snapshot_must_describe_its_own_order() -> None:
    order = _order("o-1", _ES_DEC, OrderSide.BUY, 1, _at(1))

    with pytest.raises(ValueError, match="its own decision"):
        FuturesPaperDecisionSnapshot(_record(_at(2)), order, None)
    with pytest.raises(ValueError, match="without its order"):
        FuturesPaperDecisionSnapshot(_record(_at(1)), None, _fill(order, "1", _at(2)))
