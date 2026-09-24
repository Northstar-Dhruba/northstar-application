"""Tests for running one frozen futures decision through paper trading.

Every port is an in-memory reference implementation sharing one ``World``, so
a store write is visible to the matching repository exactly as it would be in
persistence. Tests that need a port to misbehave subclass the reference.
"""

from __future__ import annotations

import ast
import importlib
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
    FuturesMarketObservationContext,
    FuturesRecommendation,
    RecommendationAction,
    StrategyIdentity,
)

from northstar_application.application_services import (
    ForwardResearchContractViolationError,
    FuturesAnalysisResult,
    FuturesExecutionIntentDecision,
    FuturesExecutionIntentNoIntentReason,
    FuturesForwardResearchRecord,
    FuturesPaperExecutionIdentityService,
    FuturesPaperTradingContractViolationError,
    FuturesPaperTradingDecisionResult,
    RunFuturesPaperTradingDecisionUseCase,
)
from northstar_application.ports import (
    FuturesForwardResearchRecordQuery,
    FuturesForwardResearchRecordRepository,
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
)


def _contract(product: str = "ES", expiry: str = "2026-12-18") -> FuturesContract:
    return FuturesContract(
        FuturesProductReference(Symbol(product), ExchangeCode("CME")), ExpirationDate(expiry)
    )


_ES_DEC = _contract()
_ES_MAR = _contract(expiry="2027-03-19")
_MES_DEC = _contract("MES")
_DAILY = Timeframe("1d")
_STRATEGY = "futures-forward"
_PORTFOLIO = "futures-paper-1"

_D1 = "2026-09-14T21:00:00Z"
_D2 = "2026-09-15T21:00:00Z"
_D3 = "2026-09-16T21:00:00Z"
_D4 = "2026-09-17T21:00:00Z"
_D5 = "2026-09-18T21:00:00Z"

_HOLD = FuturesExecutionIntentNoIntentReason.HOLD
_MET = FuturesExecutionIntentNoIntentReason.TARGET_ALREADY_MET


def _quote(value: str) -> QuoteValue:
    return QuoteValue(Decimal(value))


def _record(
    action: str = "BUY",
    *,
    contract: FuturesContract = _ES_DEC,
    instant: str = _D1,
    strategy: str = _STRATEGY,
    latest: str = "7663.25",
) -> FuturesForwardResearchRecord:
    observed_at = PointInTime(instant)
    context = FuturesMarketObservationContext(
        contract=contract,
        timeframe=_DAILY,
        observed_at=observed_at,
        latest_quote=_quote(latest),
        previous_close=_quote("7650"),
        latest_volume=Quantity(Decimal("1250")),
        session_high=_quote("7700"),
        session_low=_quote("7500"),
        recent_closes=tuple(_quote(str(7600 + index)) for index in range(20)),
        recent_volumes=tuple(Quantity(Decimal(1000 + index)) for index in range(20)),
    )
    recommendation = FuturesRecommendation(
        action=RecommendationAction(action),
        asset_analysis=FuturesAssetAnalysis(contract, observed_at, ("signal",)),
        strategy_identity=StrategyIdentity(strategy),
        point_in_time=observed_at,
    )
    return FuturesForwardResearchRecord(FuturesAnalysisResult(recommendation, context))


def _bar(instant: str, open_: str, contract: FuturesContract = _ES_DEC) -> FuturesOHLCVBar:
    base = Decimal(open_)
    return FuturesOHLCVBar(
        contract=contract,
        point_in_time=PointInTime(instant),
        timeframe=_DAILY,
        open=QuoteValue(base),
        high=QuoteValue(base + 10),
        low=QuoteValue(base - 10),
        close=QuoteValue(base + 5),
        volume=Quantity(Decimal("1000")),
    )


def _compare_text(left: str, right: str) -> int:
    return (left > right) - (left < right)


# ---------------------------------------------------------------------------
# Reference ports over one shared world
# ---------------------------------------------------------------------------


class World:
    def __init__(self) -> None:
        self.records: list[FuturesForwardResearchRecord] = []
        self.bars: list[FuturesOHLCVBar] = []
        self.orders: dict[str, FuturesPaperOrder] = {}
        self.fills: dict[str, FuturesPaperFill] = {}
        self.writes: list[tuple[str, int]] = []
        self.market_queries: list[FuturesHistoricalMarketDataQuery] = []

    def freeze(self, *records: FuturesForwardResearchRecord) -> None:
        self.records.extend(records)

    def add_bars(self, *bars: FuturesOHLCVBar) -> None:
        self.bars.extend(bars)

    def fill_for(self, order: FuturesPaperOrder) -> FuturesPaperFill | None:
        return next((f for f in self.fills.values() if f.order_identity == order.identity), None)


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
            for r in self.world.records
            if r.contract == query.contract and r.timeframe == query.timeframe
        ]
        return tuple(sorted(matching, key=cmp_to_key(compare)))


class MarketRepository(FuturesHistoricalMarketDataRepository):
    def __init__(self, world: World) -> None:
        self.world = world

    def get_bars(self, query: FuturesHistoricalMarketDataQuery):
        self.world.market_queries.append(query)
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


class OrderStore(FuturesPaperOrderStore):
    def __init__(self, world: World) -> None:
        self.world = world

    def store(self, orders):
        self.world.writes.append(("orders", len(orders)))
        for order in orders:
            existing = self.world.orders.get(order.identity.identity)
            if existing is not None and existing != order:
                raise FuturesPaperOrderConflictError("different order under this identity")
        for order in orders:
            self.world.orders[order.identity.identity] = order
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
        self.world.writes.append(("fills", len(fills)))
        for fill in fills:
            existing = self.world.fills.get(fill.identity.identity)
            if existing is not None and existing != fill:
                raise FuturesPaperFillConflictError("different fill under this identity")
            attached = next(
                (
                    f
                    for f in self.world.fills.values()
                    if f.order_identity == fill.order_identity and f.identity != fill.identity
                ),
                None,
            )
            if attached is not None:
                raise FuturesPaperFillConflictError("order already filled")
        for fill in fills:
            self.world.fills[fill.identity.identity] = fill
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


def _use_case(world: World, **overrides: object) -> RunFuturesPaperTradingDecisionUseCase:
    ports: dict[str, object] = {
        "forward_repository": ForwardRepository(world),
        "market_repository": MarketRepository(world),
        "order_store": OrderStore(world),
        "order_repository": OrderRepository(world),
        "fill_store": FillStore(world),
        "fill_repository": FillRepository(world),
    }
    ports.update(overrides)
    return RunFuturesPaperTradingDecisionUseCase(**ports)


def _run(
    world: World,
    record: FuturesForwardResearchRecord,
    *,
    portfolio: str = _PORTFOLIO,
    target: int = 2,
    through: str | None = None,
    **overrides: object,
) -> FuturesPaperTradingDecisionResult:
    return _use_case(world, **overrides).execute(
        record,
        PaperPortfolioIdentity(portfolio),
        FuturesContractCount(target),
        PointInTime(through) if through is not None else record.decision_instant,
    )


def _order_id(record: FuturesForwardResearchRecord, portfolio: str = _PORTFOLIO):
    return FuturesPaperExecutionIdentityService().order_identity(
        record, PaperPortfolioIdentity(portfolio)
    )


def _world(*records: FuturesForwardResearchRecord) -> World:
    world = World()
    world.freeze(*records)
    return world


# ---------------------------------------------------------------------------
# Forward-record verification
# ---------------------------------------------------------------------------


def test_a_separately_rebuilt_persisted_record_is_executable() -> None:
    world = _world(_record())

    result = _run(world, _record())

    assert result.order is not None


def test_a_record_that_was_never_frozen_is_rejected_before_any_write() -> None:
    world = _world(_record(instant=_D2))

    with pytest.raises(ValueError, match="not persisted"):
        _run(world, _record())

    assert world.writes == []


@pytest.mark.parametrize(
    "impostor",
    [_record("SELL"), _record(latest="7601.5")],
    ids=["different-action", "different-evidence"],
)
def test_a_payload_differing_from_the_frozen_record_is_rejected(impostor) -> None:
    world = _world(_record())

    with pytest.raises(ValueError, match="differs from the frozen"):
        _run(world, impostor)

    assert world.writes == []


class _RawForward(FuturesForwardResearchRecordRepository):
    def __init__(self, output: object) -> None:
        self.output = output

    def get_records(self, query):
        return self.output


@pytest.mark.parametrize(
    "output",
    [
        [_record()],
        (_record(), "record"),
        (_record(contract=_MES_DEC),),
        (_record(instant=_D2), _record()),
        (_record(), _record()),
    ],
    ids=["list", "foreign", "wrong-contract", "unordered", "duplicate-key"],
)
def test_malformed_forward_repository_output_is_a_contract_violation(output) -> None:
    world = _world(_record())

    with pytest.raises(ForwardResearchContractViolationError):
        _run(world, _record(), forward_repository=_RawForward(output))

    assert world.writes == []


# ---------------------------------------------------------------------------
# Basic decisions
# ---------------------------------------------------------------------------


def test_hold_creates_no_order() -> None:
    world = _world(_record("HOLD"))
    world.add_bars(_bar(_D2, "100"))

    result = _run(world, _record("HOLD"), through=_D2)

    assert result.decision.no_intent_reason is _HOLD
    assert (result.order, result.fill) == (None, None)
    assert world.orders == {}


def test_hold_keeps_existing_exposure() -> None:
    world = _world(_record(instant=_D1), _record("HOLD", instant=_D3))
    world.add_bars(_bar(_D2, "100"))
    _run(world, _record(instant=_D1), through=_D2)

    result = _run(world, _record("HOLD", instant=_D3))

    assert result.portfolio_before.get_position(_ES_DEC).net_contracts == 2
    assert len(world.orders) == 1


def test_target_already_met_creates_no_order_and_is_not_a_hold() -> None:
    world = _world(_record(instant=_D1), _record(instant=_D3))
    world.add_bars(_bar(_D2, "100"))
    _run(world, _record(instant=_D1), through=_D2)

    result = _run(world, _record(instant=_D3))

    assert result.decision.no_intent_reason is _MET
    assert result.order is None
    assert len(world.orders) == 1


def test_buy_from_flat_creates_a_pending_buy_order() -> None:
    world = _world(_record())

    result = _run(world, _record())

    assert result.order == FuturesPaperOrder(_order_id(_record()), result.decision.intent)
    assert result.order.intent.side is OrderSide.BUY
    assert result.order.intent.contracts == FuturesContractCount(2)
    assert result.fill is None
    assert world.orders == {result.order.identity.identity: result.order}


def test_sell_from_flat_targets_a_short() -> None:
    world = _world(_record("SELL"))

    result = _run(world, _record("SELL"))

    assert result.order.intent.side is OrderSide.SELL
    assert result.order.intent.contracts.value == 2


def test_a_reversal_trades_through_zero_in_one_order() -> None:
    world = _world(_record(instant=_D1), _record("SELL", instant=_D3))
    world.add_bars(_bar(_D2, "100"))
    _run(world, _record(instant=_D1), through=_D2)

    result = _run(world, _record("SELL", instant=_D3))

    assert result.order.intent.side is OrderSide.SELL
    assert result.order.intent.contracts.value == 4


def test_a_first_run_at_the_decision_is_pending_even_if_later_bars_are_stored() -> None:
    world = _world(_record())
    world.add_bars(_bar(_D1, "90"), _bar(_D2, "100"))

    result = _run(world, _record(), through=_D1)

    assert result.order is not None
    assert result.fill is None
    assert world.fills == {}


def test_a_first_run_with_the_next_bar_visible_fills_immediately() -> None:
    world = _world(_record())
    world.add_bars(_bar(_D1, "90"), _bar(_D3, "104"))

    result = _run(world, _record(), through=_D4)

    assert result.fill is not None
    assert result.fill.fill_quote == _quote("104")
    assert result.fill.filled_at == PointInTime(_D3)
    assert result.fill.identity == FuturesPaperExecutionIdentityService().fill_identity(
        result.order.identity
    )
    assert world.writes == [("orders", 1), ("fills", 1)]
    assert list(world.fills.values()) == [result.fill]


@pytest.mark.parametrize("open_", ["0", "-37.63"])
def test_zero_and_negative_next_bar_opens_fill_normally(open_: str) -> None:
    world = _world(_record())
    world.add_bars(_bar(_D2, open_))

    result = _run(world, _record(), through=_D2)

    assert result.fill.fill_quote == _quote(open_)
    assert list(world.fills.values()) == [result.fill]


# ---------------------------------------------------------------------------
# Pending -> filled retry
# ---------------------------------------------------------------------------


def test_a_pending_order_settles_on_retry_without_duplicates() -> None:
    world = _world(_record())
    first = _run(world, _record(), through=_D1)
    assert first.fill is None

    world.add_bars(_bar(_D2, "101"))
    second = _run(world, _record(), through=_D2)

    # The pending order settled first; the retry reused that fill.
    assert world.writes == [("orders", 1), ("fills", 1), ("orders", 1)]
    assert second.order == first.order
    assert second.fill.filled_at == PointInTime(_D2)
    assert second.fill.fill_quote == _quote("101")
    assert len(world.orders) == 1
    assert len(world.fills) == 1
    # The decision's own fill is later than the decision, so it never counts.
    assert second.portfolio_before.positions == ()
    assert second.decision == first.decision


def test_a_filled_retry_is_idempotent_and_still_ignores_its_own_fill() -> None:
    world = _world(_record())
    world.add_bars(_bar(_D2, "101"))
    first = _run(world, _record(), through=_D2)

    second = _run(world, _record(), through=_D3)

    assert second == FuturesPaperTradingDecisionResult(
        record=first.record,
        portfolio_before=first.portfolio_before,
        decision=first.decision,
        order=first.order,
        fill=first.fill,
        available_through=PointInTime(_D3),
    )
    assert second.decision.intent.contracts.value == 2  # not TARGET_ALREADY_MET
    assert len(world.orders) == len(world.fills) == 1


def test_a_filled_order_retried_at_an_earlier_cutoff_reports_pending() -> None:
    world = _world(_record())
    world.add_bars(_bar(_D2, "101"))
    _run(world, _record(), through=_D2)

    result = _run(world, _record(), through=_D1)

    assert result.fill is None
    assert len(world.fills) == 1


def test_an_order_whose_fill_write_failed_settles_on_the_next_run() -> None:
    world = _world(_record())
    world.add_bars(_bar(_D2, "101"))

    class FailingFillStore(FillStore):
        def store(self, fills):
            raise RuntimeError("fill storage down")

    with pytest.raises(RuntimeError):
        _run(world, _record(), through=_D2, fill_store=FailingFillStore(world))

    assert len(world.orders) == 1
    assert world.fills == {}

    result = _run(world, _record(), through=_D2)

    assert result.fill is not None
    assert len(world.fills) == 1


# ---------------------------------------------------------------------------
# Settling older pending orders
# ---------------------------------------------------------------------------


def test_older_pending_orders_settle_before_the_decision_across_contracts() -> None:
    es, mes, current = (
        _record(instant=_D1),
        _record("SELL", contract=_MES_DEC, instant=_D1),
        _record(contract=_ES_MAR, instant=_D3),
    )
    world = _world(es, mes, current)
    _run(world, es)
    _run(world, mes, target=3)
    world.add_bars(_bar(_D2, "100"), _bar(_D2, "20", _MES_DEC))

    result = _run(world, current)

    assert len(world.fills) == 2
    assert result.portfolio_before.positions == (
        FuturesPosition(_ES_DEC, 2, _quote("100")),
        FuturesPosition(_MES_DEC, -3, _quote("20")),
    )
    assert result.order.intent.contracts.value == 2  # ES Mar is flat
    assert world.writes[-2:] == [("fills", 2), ("orders", 1)]


def test_already_filled_orders_are_not_simulated_again() -> None:
    older, current = _record(contract=_ES_MAR, instant=_D1), _record(instant=_D3)
    world = _world(older, current)
    world.add_bars(_bar(_D2, "100", _ES_MAR))
    _run(world, older, through=_D2)
    world.market_queries.clear()

    _run(world, current)

    assert [query.contract for query in world.market_queries] == [_ES_DEC]


def test_orders_decided_after_the_cutoff_are_not_settled_or_counted() -> None:
    later, earlier = _record(instant=_D4), _record(instant=_D2)
    world = _world(later, earlier)
    _run(world, later)
    world.add_bars(_bar(_D3, "100"), _bar(_D5, "105"))

    result = _run(world, earlier, through=_D3)

    later_order = world.orders[_order_id(later).identity]
    assert world.fill_for(later_order) is None
    assert result.fill.filled_at == PointInTime(_D3)
    assert result.portfolio_before.positions == ()


def test_a_later_fill_never_enters_an_earlier_decisions_portfolio() -> None:
    later, earlier = _record(instant=_D3), _record("SELL", instant=_D1)
    world = _world(later, earlier)
    world.add_bars(_bar(_D4, "100"))
    _run(world, later, through=_D4)  # fills at D4

    result = _run(world, earlier)

    assert result.portfolio_before.positions == ()
    assert result.order.intent.contracts.value == 2


# ---------------------------------------------------------------------------
# Conflicts
# ---------------------------------------------------------------------------


def _plant_long(world: World, net: int) -> None:
    """Present an earlier filled BUY as if it had always been in history."""
    order = FuturesPaperOrder(
        PaperOrderIdentity("planted-order"),
        FuturesExecutionIntent(
            PaperPortfolioIdentity(_PORTFOLIO),
            _ES_DEC,
            OrderSide.BUY,
            FuturesContractCount(net),
            StrategyIdentity(_STRATEGY),
            PointInTime("2026-09-10T21:00:00Z"),
        ),
    )
    world.orders[order.identity.identity] = order
    fill = FuturesPaperFill(
        PaperFillIdentity("planted-fill"),
        order.identity,
        order.intent,
        order.intent.contracts,
        _quote("100"),
        PointInTime("2026-09-11T21:00:00Z"),
    )
    world.fills[fill.identity.identity] = fill


def test_changed_pre_decision_exposure_conflicts_with_the_frozen_order() -> None:
    record = _record(instant=_D3)
    world = _world(record)
    original = _run(world, record).order
    _plant_long(world, 1)

    with pytest.raises(FuturesPaperOrderConflictError):
        _run(world, record)

    assert world.orders[original.identity.identity] == original


def test_changed_exposure_that_now_needs_no_trade_still_conflicts() -> None:
    record = _record(instant=_D3)
    world = _world(record)
    original = _run(world, record).order
    _plant_long(world, 2)

    with pytest.raises(FuturesPaperOrderConflictError, match="now intends no execution"):
        _run(world, record)

    assert world.orders[original.identity.identity] == original


def test_a_back_filled_earlier_next_bar_conflicts_with_the_persisted_fill() -> None:
    world = _world(_record())
    world.add_bars(_bar(_D3, "104"))
    original = _run(world, _record(), through=_D3).fill

    world.add_bars(_bar(_D2, "101"))
    with pytest.raises(FuturesPaperFillConflictError):
        _run(world, _record(), through=_D3)

    assert list(world.fills.values()) == [original]


def test_an_order_store_conflict_is_not_swallowed() -> None:
    world = _world(_record())

    class ConflictingOrderStore(OrderStore):
        def store(self, orders):
            raise FuturesPaperOrderConflictError("stored elsewhere")

    with pytest.raises(FuturesPaperOrderConflictError, match="stored elsewhere"):
        _run(world, _record(), order_store=ConflictingOrderStore(world))


def test_a_fill_store_conflict_is_not_swallowed() -> None:
    world = _world(_record())
    world.add_bars(_bar(_D2, "101"))

    class ConflictingFillStore(FillStore):
        def store(self, fills):
            raise FuturesPaperFillConflictError("stored elsewhere")

    with pytest.raises(FuturesPaperFillConflictError, match="stored elsewhere"):
        _run(world, _record(), through=_D2, fill_store=ConflictingFillStore(world))


# ---------------------------------------------------------------------------
# Portfolio and strategy isolation
# ---------------------------------------------------------------------------


def test_the_same_decision_in_another_portfolio_is_a_separate_simulation() -> None:
    world = _world(_record(instant=_D1), _record(instant=_D3))
    world.add_bars(_bar(_D2, "100"))
    _run(world, _record(instant=_D1), through=_D2)

    mine = _run(world, _record(instant=_D3))
    theirs = _run(world, _record(instant=_D3), portfolio="futures-paper-2")

    assert mine.decision.no_intent_reason is _MET
    assert theirs.order.intent.contracts.value == 2
    assert theirs.order.identity == _order_id(_record(instant=_D3), "futures-paper-2")
    assert theirs.order.identity != _order_id(_record(instant=_D3))
    assert theirs.portfolio_before.positions == ()


def test_a_portfolio_holding_another_strategy_is_rejected_before_any_write() -> None:
    foreign = _record(contract=_ES_MAR, instant=_D1, strategy="other")
    world = _world(foreign, _record(instant=_D3))
    _run(world, foreign)
    world.add_bars(_bar(_D2, "100", _ES_MAR))
    writes = list(world.writes)

    with pytest.raises(ValueError, match="one paper portfolio belongs to one strategy"):
        _run(world, _record(instant=_D3))

    assert world.writes == writes
    assert world.fills == {}


def test_one_strategy_may_run_in_several_portfolios() -> None:
    world = _world(_record())

    first = _run(world, _record(), portfolio="a")
    second = _run(world, _record(), portfolio="b")

    assert first.order.identity != second.order.identity
    assert len(world.orders) == 2


def test_other_contract_positions_are_kept_but_do_not_drive_the_target() -> None:
    world = _world(_record(contract=_ES_MAR, instant=_D1), _record(instant=_D3))
    world.add_bars(_bar(_D2, "200", _ES_MAR))
    _run(world, _record(contract=_ES_MAR, instant=_D1), target=3, through=_D2)

    result = _run(world, _record(instant=_D3))

    assert result.portfolio_before.positions == (FuturesPosition(_ES_MAR, 3, _quote("200")),)
    assert result.order.intent.side is OrderSide.BUY
    assert result.order.intent.contracts.value == 2


# ---------------------------------------------------------------------------
# Cutoff
# ---------------------------------------------------------------------------


def test_a_cutoff_before_the_decision_is_rejected_before_reading_anything() -> None:
    world = _world(_record(instant=_D2))

    class Unreadable(ForwardRepository):
        def get_records(self, query):
            raise AssertionError("must not be read")

    with pytest.raises(ValueError, match="precedes the decision instant"):
        _run(world, _record(instant=_D2), through=_D1, forward_repository=Unreadable(world))


def test_an_offset_equivalent_cutoff_at_the_decision_is_pending() -> None:
    world = _world(_record())
    world.add_bars(_bar(_D2, "100"))

    result = _run(world, _record(), through="2026-09-15T02:30:00+05:30")

    assert result.order is not None
    assert result.fill is None


def test_a_sub_second_next_bar_fills_by_semantic_time() -> None:
    world = _world(_record())
    half = "2026-09-14T21:00:00.5Z"
    world.add_bars(_bar(_D1, "90"), _bar(half, "91"))

    result = _run(world, _record(), through="2026-09-14T21:00:01Z")

    assert half < _D1  # the text trap
    assert result.fill.filled_at == PointInTime(half)
    assert result.fill.fill_quote == _quote("91")


# ---------------------------------------------------------------------------
# Store counts and repository contracts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("count", [0, 2, True, "1", None])
def test_a_wrong_order_store_count_is_a_contract_violation(count: object) -> None:
    world = _world(_record())

    class Miscounting(OrderStore):
        def store(self, orders):
            super().store(orders)
            return count

    with pytest.raises(FuturesPaperTradingContractViolationError, match="FuturesPaperOrderStore"):
        _run(world, _record(), order_store=Miscounting(world))


@pytest.mark.parametrize("count", [0, 2, True, "1", None])
def test_a_wrong_current_fill_store_count_is_a_contract_violation(count: object) -> None:
    world = _world(_record())
    world.add_bars(_bar(_D2, "100"))

    class Miscounting(FillStore):
        def store(self, fills):
            super().store(fills)
            return count

    with pytest.raises(FuturesPaperTradingContractViolationError, match="FuturesPaperFillStore"):
        _run(world, _record(), through=_D2, fill_store=Miscounting(world))


def test_a_wrong_settlement_fill_store_count_is_a_contract_violation() -> None:
    world = _world(_record(instant=_D1), _record(contract=_MES_DEC, instant=_D1))
    _run(world, _record(instant=_D1))
    _run(world, _record(contract=_MES_DEC, instant=_D1))
    world.add_bars(_bar(_D2, "100"), _bar(_D2, "10", _MES_DEC))

    class Miscounting(FillStore):
        def store(self, fills):
            super().store(fills)
            return 1

    with pytest.raises(FuturesPaperTradingContractViolationError, match="exactly 2"):
        _run(world, _record(instant=_D1), through=_D2, fill_store=Miscounting(world))


def _order(
    order_id: str, *, decided_at: str = _D1, portfolio: str = _PORTFOLIO, side=OrderSide.BUY
):
    return FuturesPaperOrder(
        PaperOrderIdentity(order_id),
        FuturesExecutionIntent(
            PaperPortfolioIdentity(portfolio),
            _ES_MAR,
            side,
            FuturesContractCount(1),
            StrategyIdentity(_STRATEGY),
            PointInTime(decided_at),
        ),
    )


def _fill_of(order: FuturesPaperOrder, filled_at: str = _D2, *, intent=None) -> FuturesPaperFill:
    intent = intent or order.intent
    return FuturesPaperFill(
        PaperFillIdentity(f"fill-{order.identity.identity}"),
        order.identity,
        intent,
        intent.contracts,
        _quote("100"),
        PointInTime(filled_at),
    )


class _RawOrders(FuturesPaperOrderRepository):
    def __init__(self, output: object) -> None:
        self.output = output

    def get_orders(self, query):
        return self.output


class _RawFills(FuturesPaperFillRepository):
    def __init__(self, output: object) -> None:
        self.output = output

    def get_fills(self, query):
        return self.output


_A, _B = _order("a"), _order("b")


@pytest.mark.parametrize(
    "orders",
    [
        [_A],
        (_A, "order"),
        (_B, _A),
        (_A, _A),
        (_order("x", portfolio="other"),),
        (_order("z", decided_at=_D2), _order("y", decided_at=_D1)),
    ],
    ids=["list", "foreign", "unordered-ids", "duplicate", "other-portfolio", "unordered-time"],
)
def test_malformed_order_repository_output_is_rejected_before_writes(orders) -> None:
    world = _world(_record())

    with pytest.raises(FuturesPaperTradingContractViolationError):
        _run(world, _record(), order_repository=_RawOrders(orders))

    assert world.writes == []


@pytest.mark.parametrize(
    ("orders", "fills"),
    [
        ((_A, _B), [_fill_of(_A)]),
        ((_A, _B), (_fill_of(_A), "fill")),
        ((_A, _B), (_fill_of(_B), _fill_of(_A))),
        ((_A, _B), (_fill_of(_A, _D3), _fill_of(_B, _D2))),
        ((_A,), (_fill_of(_A), _fill_of(_A))),
        ((_A,), (_fill_of(_B),)),
        ((_A,), (_fill_of(_A, intent=_order("a", side=OrderSide.SELL).intent),)),
    ],
    ids=[
        "list",
        "foreign",
        "unordered-order-ids",
        "unordered-time",
        "duplicate",
        "fill-without-order",
        "fill-intent-mismatch",
    ],
)
def test_malformed_fill_history_is_rejected_before_writes(orders, fills) -> None:
    world = _world(_record())

    with pytest.raises(FuturesPaperTradingContractViolationError):
        _run(
            world,
            _record(),
            order_repository=_RawOrders(orders),
            fill_repository=_RawFills(fills),
        )

    assert world.writes == []


# ---------------------------------------------------------------------------
# Construction and inputs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "forward_repository",
        "market_repository",
        "order_store",
        "order_repository",
        "fill_store",
        "fill_repository",
    ],
)
def test_every_port_is_type_checked(name: str) -> None:
    with pytest.raises(TypeError, match=name):
        _use_case(World(), **{name: object()})


@pytest.mark.parametrize(
    ("argument", "value", "message"),
    [
        (0, _record().result, "record must be a FuturesForwardResearchRecord"),
        (1, _PORTFOLIO, "portfolio identity must be a PaperPortfolioIdentity"),
        (2, 2, "target contracts must be a FuturesContractCount"),
        (3, _D1, "available-through must be a PointInTime"),
    ],
)
def test_inputs_are_type_checked_before_any_read(argument: int, value: object, message: str):
    world = _world(_record())
    arguments: list[object] = [
        _record(),
        PaperPortfolioIdentity(_PORTFOLIO),
        FuturesContractCount(2),
        PointInTime(_D1),
    ]
    arguments[argument] = value

    with pytest.raises(TypeError, match=message):
        _use_case(world).execute(*arguments)

    assert world.writes == []


# ---------------------------------------------------------------------------
# Result value
# ---------------------------------------------------------------------------


def _result_kwargs(**overrides: object) -> dict:
    world = _world(_record())
    world.add_bars(_bar(_D2, "100"))
    result = _run(world, _record(), through=_D2)
    values = {
        "record": result.record,
        "portfolio_before": result.portfolio_before,
        "decision": result.decision,
        "order": result.order,
        "fill": result.fill,
        "available_through": result.available_through,
    }
    values.update(overrides)
    return values


def test_a_result_is_immutable_and_comparable() -> None:
    result = FuturesPaperTradingDecisionResult(**_result_kwargs())

    assert result == FuturesPaperTradingDecisionResult(**_result_kwargs())
    with pytest.raises(AttributeError):
        result.fill = None  # type: ignore[misc]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"order": None}, "must carry an order"),
        ({"available_through": PointInTime("2026-09-13T21:00:00Z")}, "cannot precede"),
        ({"available_through": PointInTime(_D1)}, "visible at available-through"),
        (
            {"decision": FuturesExecutionIntentDecision(no_intent_reason=_MET)},
            "cannot carry an order or fill",
        ),
        (
            {
                "decision": FuturesExecutionIntentDecision(no_intent_reason=_HOLD),
                "order": None,
                "fill": None,
            },
            "cannot decline a directional decision as a hold",
        ),
        (
            {
                "portfolio_before": FuturesPaperPortfolio(
                    PaperPortfolioIdentity(_PORTFOLIO),
                    StrategyIdentity("other"),
                    (),
                    PointInTime(_D1),
                )
            },
            "decision strategy",
        ),
        (
            {
                "portfolio_before": FuturesPaperPortfolio(
                    PaperPortfolioIdentity(_PORTFOLIO),
                    StrategyIdentity(_STRATEGY),
                    (),
                    PointInTime(_D2),
                )
            },
            "as of the decision instant",
        ),
        (
            {
                "portfolio_before": FuturesPaperPortfolio(
                    PaperPortfolioIdentity("other"),
                    StrategyIdentity(_STRATEGY),
                    (),
                    PointInTime(_D1),
                )
            },
            "belong to the portfolio",
        ),
    ],
)
def test_an_incoherent_result_is_rejected(overrides: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        FuturesPaperTradingDecisionResult(**_result_kwargs(**overrides))


def test_a_hold_result_must_carry_the_hold_reason() -> None:
    world = _world(_record("HOLD"))
    result = _run(world, _record("HOLD"))

    with pytest.raises(ValueError, match="decline a HOLD decision as a hold"):
        FuturesPaperTradingDecisionResult(
            record=result.record,
            portfolio_before=result.portfolio_before,
            decision=FuturesExecutionIntentDecision(no_intent_reason=_MET),
            order=None,
            fill=None,
            available_through=result.available_through,
        )


# ---------------------------------------------------------------------------
# Boundaries and exports
# ---------------------------------------------------------------------------

_MODULE_NAME = "northstar_application.application_services.run_futures_paper_trading_decision"


def _tree() -> ast.Module:
    module = importlib.import_module(_MODULE_NAME)
    return ast.parse(Path(module.__file__).read_text(encoding="utf-8"))


def test_the_orchestration_depends_only_on_application_ports_and_core() -> None:
    modules: set[str] = set()
    names: set[str] = set()
    for node in ast.walk(_tree()):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.add(node.module)
            names.update(alias.name for alias in node.names)

    for module in modules:
        assert module.split(".")[0] not in {
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
        assert "acquire" not in module
    assert not names & {
        "FuturesHistoricalMarketDataSource",
        "FuturesTradingSessionResolver",
        "FuturesProductSpecification",
        "Money",
        "Price",
        "Currency",
        "PaperOrderStatus",
    }
    for node in ast.walk(_tree()):
        if isinstance(node, ast.Attribute):
            assert node.attr not in {"now", "today", "utcnow"}


def test_the_public_surface_is_exported_without_helpers() -> None:
    import northstar_application.application_services as services

    for name in (
        "RunFuturesPaperTradingDecisionUseCase",
        "FuturesPaperTradingDecisionResult",
        "FuturesPaperTradingContractViolationError",
    ):
        assert name in services.__all__
    for private in ("_compare_orders", "_compare_fills", "_HOLD_ACTION"):
        assert not hasattr(services, private)
    assert "FuturesPaperOrderState" not in services.__all__
