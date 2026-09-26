"""Tests for the futures paper-trading run, its metrics and its report.

Every port is an in-memory reference implementation sharing one ``World``, so
writes made by one run are visible to the next exactly as with persistence.
"""

from __future__ import annotations

import ast
import dataclasses
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
    BuildFuturesPaperTradingReportUseCase,
    CalculateFuturesPaperTradingMetricsUseCase,
    ForwardResearchContractViolationError,
    FuturesAnalysisResult,
    FuturesExecutionIntentNoIntentReason,
    FuturesForwardResearchRecord,
    FuturesPaperTradingReport,
    FuturesPaperTradingRun,
    FuturesPaperTradingStrategyContractMetrics,
    RunFuturesPaperTradingUseCase,
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
_DAILY = Timeframe("1d")
_STRATEGY = "futures-forward"
_PORTFOLIO = "futures-paper-1"

_D1 = "2026-09-14T21:00:00Z"
_D2 = "2026-09-15T21:00:00Z"
_D3 = "2026-09-16T21:00:00Z"
_D4 = "2026-09-17T21:00:00Z"
_D5 = "2026-09-18T21:00:00Z"
_D6 = "2026-09-21T21:00:00Z"
_D7 = "2026-09-22T21:00:00Z"
_D8 = "2026-09-23T21:00:00Z"
_LATE = "2026-09-30T00:00:00Z"

BUY = OrderSide.BUY
SELL = OrderSide.SELL
_HOLD = FuturesExecutionIntentNoIntentReason.HOLD
_MET = FuturesExecutionIntentNoIntentReason.TARGET_ALREADY_MET


def _quote(value: str) -> QuoteValue:
    return QuoteValue(Decimal(value))


def _record(
    action: str = "BUY",
    instant: str = _D1,
    *,
    contract: FuturesContract = _ES_DEC,
    strategy: str = _STRATEGY,
) -> FuturesForwardResearchRecord:
    observed_at = PointInTime(instant)
    context = FuturesMarketObservationContext(
        contract=contract,
        timeframe=_DAILY,
        observed_at=observed_at,
        latest_quote=_quote("7663.25"),
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

    def freeze(self, *records: FuturesForwardResearchRecord) -> World:
        self.records.extend(records)
        return self

    def add_bars(self, *bars: FuturesOHLCVBar) -> World:
        self.bars.extend(bars)
        return self


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


def _use_case(world: World, **overrides: object) -> RunFuturesPaperTradingUseCase:
    ports: dict[str, object] = {
        "forward_repository": ForwardRepository(world),
        "market_repository": MarketRepository(world),
        "order_store": OrderStore(world),
        "order_repository": OrderRepository(world),
        "fill_store": FillStore(world),
        "fill_repository": FillRepository(world),
    }
    ports.update(overrides)
    return RunFuturesPaperTradingUseCase(**ports)


def _run(
    world: World,
    *,
    through: str = _D8,
    strategy: str = _STRATEGY,
    portfolio: str = _PORTFOLIO,
    target: int = 2,
    contract: FuturesContract = _ES_DEC,
    **overrides: object,
) -> FuturesPaperTradingRun:
    return _use_case(world, **overrides).execute(
        FuturesForwardResearchRecordQuery(contract, _DAILY),
        PaperPortfolioIdentity(portfolio),
        StrategyIdentity(strategy),
        FuturesContractCount(target),
        PointInTime(through),
    )


def _plant(
    world: World,
    net: int,
    *,
    contract: FuturesContract = _ES_DEC,
    strategy: str = _STRATEGY,
    quote: str = "100",
    name: str = "planted",
) -> None:
    """Put an earlier filled order in history that no selected decision produced."""
    order = FuturesPaperOrder(
        PaperOrderIdentity(f"{name}-order"),
        FuturesExecutionIntent(
            PaperPortfolioIdentity(_PORTFOLIO),
            contract,
            BUY if net > 0 else SELL,
            FuturesContractCount(abs(net)),
            StrategyIdentity(strategy),
            PointInTime("2026-09-10T21:00:00Z"),
        ),
    )
    world.orders[order.identity.identity] = order
    fill = FuturesPaperFill(
        PaperFillIdentity(f"{name}-fill"),
        order.identity,
        order.intent,
        order.intent.contracts,
        _quote(quote),
        PointInTime("2026-09-11T21:00:00Z"),
    )
    world.fills[fill.identity.identity] = fill


def _sequence_world(opens: tuple[str, str, str, str] = ("100", "110", "120", "130")) -> World:
    """BUY, BUY again at target, SELL reversal, HOLD -- each with a next bar."""
    return (
        World()
        .freeze(
            _record("BUY", _D1),
            _record("BUY", _D3),
            _record("SELL", _D5),
            _record("HOLD", _D7),
        )
        .add_bars(
            _bar(_D2, opens[0]), _bar(_D4, opens[1]), _bar(_D6, opens[2]), _bar(_D8, opens[3])
        )
    )


def _metrics(run: FuturesPaperTradingRun) -> tuple[FuturesPaperTradingStrategyContractMetrics, ...]:
    return CalculateFuturesPaperTradingMetricsUseCase().execute(run)


def _only_metrics(run: FuturesPaperTradingRun) -> FuturesPaperTradingStrategyContractMetrics:
    (entry,) = _metrics(run)
    return entry


# ---------------------------------------------------------------------------
# Run: basic decisions
# ---------------------------------------------------------------------------


def test_an_empty_run_still_reports_the_existing_portfolio() -> None:
    world = World()
    _plant(world, 3, contract=_ES_MAR)

    run = _run(world)

    assert run.results == ()
    assert world.writes == []
    assert run.portfolio == FuturesPaperPortfolio(
        PaperPortfolioIdentity(_PORTFOLIO),
        StrategyIdentity(_STRATEGY),
        (FuturesPosition(_ES_MAR, 3, _quote("100")),),
        PointInTime(_D8),
    )
    assert (run.first_decision_instant, run.last_decision_instant) == (None, None)


def test_one_buy_at_its_own_instant_is_pending() -> None:
    world = World().freeze(_record()).add_bars(_bar(_D2, "100"))

    run = _run(world, through=_D1)

    (result,) = run.results
    assert result.order.intent.side is BUY
    assert result.fill is None
    assert run.portfolio.positions == ()


def test_one_buy_with_its_next_bar_visible_fills() -> None:
    world = World().freeze(_record()).add_bars(_bar(_D2, "100"))

    run = _run(world, through=_D2)

    assert run.results[0].fill.fill_quote == _quote("100")
    assert run.portfolio.positions == (FuturesPosition(_ES_DEC, 2, _quote("100")),)


def test_one_sell_opens_a_short() -> None:
    world = World().freeze(_record("SELL")).add_bars(_bar(_D2, "100"))

    run = _run(world)

    assert run.portfolio.positions == (FuturesPosition(_ES_DEC, -2, _quote("100")),)


def test_hold_places_no_order() -> None:
    world = World().freeze(_record("HOLD")).add_bars(_bar(_D2, "100"))

    run = _run(world)

    assert run.results[0].decision.no_intent_reason is _HOLD
    assert run.orders == ()
    assert world.orders == {}


def test_a_met_target_places_no_second_order() -> None:
    world = World().freeze(_record("BUY", _D1), _record("BUY", _D3)).add_bars(_bar(_D2, "100"))

    run = _run(world, through=_D3)

    assert run.results[1].decision.no_intent_reason is _MET
    assert len(run.orders) == 1


def test_decisions_execute_chronologically_into_one_history() -> None:
    world = _sequence_world()

    run = _run(world)

    assert [r.record.decision_instant for r in run.results] == [
        PointInTime(instant) for instant in (_D1, _D3, _D5, _D7)
    ]
    first, met, reversal, hold = run.results
    assert (first.order.intent.side, first.order.intent.contracts.value) == (BUY, 2)
    assert met.decision.no_intent_reason is _MET
    assert (reversal.order.intent.side, reversal.order.intent.contracts.value) == (SELL, 4)
    assert reversal.portfolio_before.get_position(_ES_DEC).net_contracts == 2
    assert hold.decision.no_intent_reason is _HOLD
    assert run.portfolio.positions == (FuturesPosition(_ES_DEC, -2, _quote("120")),)


# ---------------------------------------------------------------------------
# Run: selection
# ---------------------------------------------------------------------------


def test_only_the_requested_strategy_executes() -> None:
    world = World().freeze(
        _record("BUY", _D1),
        _record("SELL", _D1, strategy="other"),
        _record("BUY", _D3, strategy="other"),
    )

    run = _run(world)

    assert [r.record.strategy_identity for r in run.results] == [StrategyIdentity(_STRATEGY)]
    assert {o.intent.strategy_identity for o in world.orders.values()} == {
        StrategyIdentity(_STRATEGY)
    }


def test_decisions_after_the_cutoff_do_not_execute() -> None:
    world = World().freeze(_record("BUY", _D1), _record("SELL", _D5))

    run = _run(world, through=_D3)

    assert len(run.results) == 1
    assert len(world.orders) == 1


def test_every_decision_executes_at_the_run_cutoff() -> None:
    """With per-decision cutoffs the reversal would still be pending at D3."""
    world = (
        World()
        .freeze(_record("BUY", _D1), _record("SELL", _D3))
        .add_bars(_bar(_D2, "100"), _bar(_D4, "90"))
    )

    run = _run(world, through=_D5)

    assert all(r.available_through == PointInTime(_D5) for r in run.results)
    assert run.results[1].fill.filled_at == PointInTime(_D4)
    assert run.portfolio.positions == (FuturesPosition(_ES_DEC, -2, _quote("90")),)


def test_the_final_portfolio_is_as_of_the_cutoff_not_the_last_decision() -> None:
    world = World().freeze(_record()).add_bars(_bar(_D2, "100"))

    run = _run(world, through=_D3)

    assert run.results[-1].portfolio_before.positions == ()
    assert run.portfolio.as_of == PointInTime(_D3)
    assert run.portfolio.positions == (FuturesPosition(_ES_DEC, 2, _quote("100")),)


def test_other_contract_history_is_kept_in_the_final_portfolio() -> None:
    world = World().freeze(_record()).add_bars(_bar(_D2, "100"))
    _plant(world, 3, contract=_ES_MAR, quote="200")

    run = _run(world)

    assert run.portfolio.positions == (
        FuturesPosition(_ES_DEC, 2, _quote("100")),
        FuturesPosition(_ES_MAR, 3, _quote("200")),
    )
    assert run.results[0].order.intent.contracts.value == 2


# ---------------------------------------------------------------------------
# Run: repeats and cutoffs
# ---------------------------------------------------------------------------


def test_a_repeated_run_is_equal_and_writes_nothing_new() -> None:
    world = _sequence_world()
    first = _run(world)
    orders, fills = dict(world.orders), dict(world.fills)

    second = _run(world)

    assert second == first
    assert world.orders == orders
    assert world.fills == fills


def test_later_market_data_does_not_leak_into_the_same_cutoff() -> None:
    world = World().freeze(_record("BUY", _D1), _record("SELL", _D3)).add_bars(_bar(_D2, "100"))
    first = _run(world, through=_D3)
    fills = dict(world.fills)

    world.add_bars(_bar(_D4, "90"), _bar(_D5, "95"))
    second = _run(world, through=_D3)

    assert second == first
    assert second.results[1].fill is None
    assert world.fills == fills


def test_a_later_cutoff_settles_pending_orders_without_rewriting_history() -> None:
    world = World().freeze(_record("BUY", _D1), _record("SELL", _D3)).add_bars(_bar(_D2, "100"))
    first = _run(world, through=_D3)
    world.add_bars(_bar(_D4, "90"))

    later = _run(world, through=_D5)

    assert first.results[1].fill is None
    assert later.results[1].fill.filled_at == PointInTime(_D4)
    assert later.orders == first.orders
    assert later.results[0].fill == first.results[0].fill
    assert later.portfolio.get_position(_ES_DEC).net_contracts == -2


def test_a_later_frozen_decision_joins_only_later_cutoffs() -> None:
    world = World().freeze(_record("BUY", _D1)).add_bars(_bar(_D2, "100"))
    first = _run(world, through=_D3)
    world.freeze(_record("SELL", _D5))

    assert _run(world, through=_D3) == first
    assert len(_run(world, through=_D6).results) == 2


def test_a_back_filled_frozen_decision_joins_a_rerun_at_the_same_cutoff() -> None:
    """Accepted: runs are re-derived views, not persisted snapshots."""
    world = World().freeze(_record("BUY", _D1)).add_bars(_bar(_D2, "100"))
    first = _run(world, through=_D5)
    world.freeze(_record("SELL", _D3))

    rerun = _run(world, through=_D5)

    assert len(first.results) == 1
    assert len(rerun.results) == 2


def test_a_back_filled_market_bar_still_conflicts() -> None:
    world = World().freeze(_record()).add_bars(_bar(_D3, "104"))
    original = _run(world, through=_D3).results[0].fill

    world.add_bars(_bar(_D2, "101"))
    with pytest.raises(FuturesPaperFillConflictError):
        _run(world, through=_D3)

    assert list(world.fills.values()) == [original]


# ---------------------------------------------------------------------------
# Run: isolation and contracts
# ---------------------------------------------------------------------------


def test_a_mixed_strategy_portfolio_is_rejected_before_any_write() -> None:
    world = World().freeze(_record())
    _plant(world, 1, contract=_ES_MAR, strategy="other")

    with pytest.raises(ValueError, match="one paper portfolio belongs to one strategy"):
        _run(world)
    with pytest.raises(ValueError, match="one paper portfolio belongs to one strategy"):
        _run(world, contract=_ES_MAR)  # selects nothing, still rejected

    assert world.writes == []


def test_portfolios_are_isolated() -> None:
    world = _sequence_world()

    mine = _run(world)
    theirs = _run(world, portfolio="futures-paper-2")

    assert mine.portfolio.positions == theirs.portfolio.positions
    assert {o.identity for o in mine.orders}.isdisjoint({o.identity for o in theirs.orders})
    assert len(world.orders) == 4


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
        (_record(contract=_ES_MAR),),
        (_record("BUY", _D3), _record("BUY", _D1)),
        (_record(), _record()),
    ],
    ids=["list", "foreign", "wrong-contract", "unordered", "duplicate"],
)
def test_malformed_forward_output_is_rejected_before_writes(output) -> None:
    world = World().freeze(_record())

    with pytest.raises(ForwardResearchContractViolationError):
        _run(world, forward_repository=_RawForward(output))

    assert world.writes == []


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
    with pytest.raises(TypeError, match=f"RunFuturesPaperTradingUseCase {name}"):
        _use_case(World(), **{name: object()})


@pytest.mark.parametrize("argument", range(5))
def test_execute_inputs_are_type_checked(argument: int) -> None:
    arguments: list[object] = [
        FuturesForwardResearchRecordQuery(_ES_DEC, _DAILY),
        PaperPortfolioIdentity(_PORTFOLIO),
        StrategyIdentity(_STRATEGY),
        FuturesContractCount(2),
        PointInTime(_D8),
    ]
    arguments[argument] = "wrong"

    with pytest.raises(TypeError, match="RunFuturesPaperTradingUseCase"):
        _use_case(World()).execute(*arguments)


# ---------------------------------------------------------------------------
# Run value coherence
# ---------------------------------------------------------------------------


def test_an_incoherent_run_is_rejected() -> None:
    run = _run(_sequence_world())
    other_portfolio = FuturesPaperPortfolio(
        PaperPortfolioIdentity(_PORTFOLIO), StrategyIdentity(_STRATEGY), (), PointInTime(_D7)
    )

    cases = [
        ({"results": tuple(reversed(run.results))}, "strictly ordered"),
        ({"results": (run.results[0], run.results[0])}, "strictly ordered"),
        ({"strategy_identity": StrategyIdentity("other")}, "run strategy"),
        ({"available_through": PointInTime(_LATE)}, "run cutoff"),
        ({"portfolio_identity": PaperPortfolioIdentity("x")}, "run's portfolio"),
        ({"query": FuturesForwardResearchRecordQuery(_ES_MAR, _DAILY)}, "query contract"),
        ({"portfolio": other_portfolio}, "as of the run cutoff"),
    ]
    for overrides, message in cases:
        with pytest.raises(ValueError, match=message):
            dataclasses.replace(run, **overrides)
    with pytest.raises(TypeError, match="results must be a tuple"):
        dataclasses.replace(run, results=list(run.results))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def test_an_empty_run_has_no_metrics() -> None:
    world = World()
    _plant(world, 3)

    assert _metrics(_run(world)) == ()


def test_the_sequence_metrics_are_exact() -> None:
    entry = _only_metrics(_run(_sequence_world()))

    assert entry == FuturesPaperTradingStrategyContractMetrics(
        contract=_ES_DEC,
        strategy_identity=StrategyIdentity(_STRATEGY),
        decision_count=4,
        hold_count=1,
        target_already_met_count=1,
        order_count=2,
        filled_order_count=2,
        pending_order_count=0,
        contracts_bought=2,
        contracts_sold=4,
        current_net_contracts=-2,
    )


def test_hold_only_metrics() -> None:
    entry = _only_metrics(_run(World().freeze(_record("HOLD"))))

    assert (entry.decision_count, entry.hold_count, entry.target_already_met_count) == (1, 1, 0)
    assert entry.order_count == 0


def test_target_met_only_metrics_keep_the_existing_position() -> None:
    world = World().freeze(_record())
    _plant(world, 2)

    entry = _only_metrics(_run(world))

    assert (entry.hold_count, entry.target_already_met_count, entry.order_count) == (0, 1, 0)
    assert entry.current_net_contracts == 2
    assert (entry.contracts_bought, entry.contracts_sold) == (0, 0)


def test_a_pending_order_executes_no_contracts() -> None:
    entry = _only_metrics(_run(World().freeze(_record()), through=_D1))

    assert (entry.order_count, entry.filled_order_count, entry.pending_order_count) == (1, 0, 1)
    assert (entry.contracts_bought, entry.contracts_sold) == (0, 0)
    assert entry.current_net_contracts == 0


@pytest.mark.parametrize(
    ("action", "bought", "sold", "net"), [("BUY", 2, 0, 2), ("SELL", 0, 2, -2)]
)
def test_a_filled_order_counts_its_executed_side(action, bought, sold, net) -> None:
    world = World().freeze(_record(action)).add_bars(_bar(_D2, "100"))

    entry = _only_metrics(_run(world))

    assert (entry.contracts_bought, entry.contracts_sold) == (bought, sold)
    assert entry.current_net_contracts == net


def test_a_research_buy_over_target_counts_as_contracts_sold() -> None:
    world = World().freeze(_record("BUY")).add_bars(_bar(_D2, "100"))
    _plant(world, 3)

    entry = _only_metrics(_run(world))

    assert (entry.contracts_bought, entry.contracts_sold) == (0, 1)
    assert entry.current_net_contracts == 2


def test_a_reversal_counts_the_whole_traded_size() -> None:
    world = World().freeze(_record("SELL")).add_bars(_bar(_D2, "100"))
    _plant(world, 1)

    run = _run(world, target=1)
    entry = _only_metrics(run)

    assert run.results[0].order.intent.contracts.value == 2
    assert (entry.order_count, entry.filled_order_count) == (1, 1)
    assert entry.contracts_sold == 2
    assert entry.current_net_contracts == -1


def test_metrics_are_scoped_to_the_decided_contract() -> None:
    world = World().freeze(_record()).add_bars(_bar(_D2, "100"))
    _plant(world, 3, contract=_ES_MAR)

    run = _run(world)

    assert [entry.contract for entry in _metrics(run)] == [_ES_DEC]
    assert _only_metrics(run).current_net_contracts == 2


def test_quote_sign_never_changes_counts_or_exposure() -> None:
    positive = _only_metrics(_run(_sequence_world()))
    signed = _only_metrics(_run(_sequence_world(("0", "-37.63", "-10", "0"))))

    assert signed == positive


def test_equal_runs_give_equal_metrics() -> None:
    world = _sequence_world()

    assert _metrics(_run(world)) == _metrics(_run(world))


_VALID_METRICS = {
    "contract": _ES_DEC,
    "strategy_identity": StrategyIdentity(_STRATEGY),
    "decision_count": 3,
    "hold_count": 1,
    "target_already_met_count": 1,
    "order_count": 1,
    "filled_order_count": 1,
    "pending_order_count": 0,
    "contracts_bought": 2,
    "contracts_sold": 0,
    "current_net_contracts": 0,
}


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"decision_count": True}, TypeError),
        ({"contracts_sold": 1.0}, TypeError),
        ({"current_net_contracts": False}, TypeError),
        ({"contracts_bought": -1}, ValueError),
        ({"pending_order_count": 1}, ValueError),
        ({"decision_count": 4}, ValueError),
        ({"contract": "ES"}, TypeError),
    ],
)
def test_metrics_values_are_validated(overrides: dict, error: type) -> None:
    with pytest.raises(error):
        FuturesPaperTradingStrategyContractMetrics(**{**_VALID_METRICS, **overrides})


def test_net_contracts_may_be_negative() -> None:
    entry = FuturesPaperTradingStrategyContractMetrics(
        **{**_VALID_METRICS, "current_net_contracts": -5}
    )

    assert entry.current_net_contracts == -5


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _report(run: FuturesPaperTradingRun) -> FuturesPaperTradingReport:
    return BuildFuturesPaperTradingReportUseCase().execute(run)


def test_a_report_surfaces_the_run() -> None:
    world = _sequence_world()
    _plant(world, 3, contract=_ES_MAR, quote="200")
    run = _run(world, through=_LATE, target=2)

    report = _report(run)

    assert report.metrics == _metrics(run)
    assert report.portfolio_identity == PaperPortfolioIdentity(_PORTFOLIO)
    assert report.strategy_identity == StrategyIdentity(_STRATEGY)
    assert (report.contract, report.timeframe) == (_ES_DEC, _DAILY)
    assert report.available_through == PointInTime(_LATE)
    assert report.target_contracts == FuturesContractCount(2)
    assert (report.decision_count, report.order_count, report.fill_count) == (4, 2, 2)
    assert report.pending_order_count == 0
    assert report.first_decision_instant == PointInTime(_D1)
    assert report.last_decision_instant == PointInTime(_D7)
    assert report.positions == (
        FuturesPosition(_ES_DEC, -2, _quote("120")),
        FuturesPosition(_ES_MAR, 3, _quote("200")),
    )
    assert report.position_count == 2
    assert report.portfolio is run.portfolio


def test_an_empty_report_keeps_its_configuration_and_positions() -> None:
    world = World()
    _plant(world, 3, contract=_ES_MAR)

    report = _report(_run(world, through=_D2, target=5))

    assert report.metrics == ()
    assert report.decision_count == 0
    assert (report.first_decision_instant, report.last_decision_instant) == (None, None)
    assert report.available_through == PointInTime(_D2)
    assert report.target_contracts == FuturesContractCount(5)
    assert report.position_count == 1


def test_a_pending_report_counts_the_pending_order() -> None:
    report = _report(_run(World().freeze(_record()), through=_D1))

    assert (report.order_count, report.fill_count, report.pending_order_count) == (1, 0, 1)
    assert report.positions == ()


def test_repeated_builds_are_equal() -> None:
    run = _run(_sequence_world())

    assert _report(run) == _report(run)


def test_bad_metrics_are_rejected() -> None:
    run = _run(_sequence_world())
    (entry,) = _metrics(run)
    miscounted = dataclasses.replace(
        entry, decision_count=entry.decision_count + 1, hold_count=entry.hold_count + 1
    )

    cases = [
        ((), "exactly one metrics entry"),
        ((entry, entry), "exactly one metrics entry"),
        ((dataclasses.replace(entry, contract=_ES_MAR),), "canonical contract order"),
        ((dataclasses.replace(entry, strategy_identity=StrategyIdentity("x")),), "run strategy"),
        ((miscounted,), "do not describe the run"),
        ((dataclasses.replace(entry, current_net_contracts=0),), "do not describe the run"),
    ]
    for metrics, message in cases:
        with pytest.raises(ValueError, match=message):
            FuturesPaperTradingReport(run=run, metrics=metrics)
    with pytest.raises(TypeError):
        FuturesPaperTradingReport(run=run, metrics=[entry])


def test_the_report_builder_is_type_checked() -> None:
    with pytest.raises(TypeError):
        BuildFuturesPaperTradingReportUseCase(calculate_metrics=object())
    with pytest.raises(TypeError):
        BuildFuturesPaperTradingReportUseCase().execute("run")
    with pytest.raises(TypeError):
        CalculateFuturesPaperTradingMetricsUseCase().execute("run")


# ---------------------------------------------------------------------------
# Boundaries and exports
# ---------------------------------------------------------------------------

_SERVICES = "northstar_application.application_services"


def _imports(module_name: str) -> tuple[set[str], set[str]]:
    module = importlib.import_module(module_name)
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    modules: set[str] = set()
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.add(node.module)
            names.update(alias.name for alias in node.names)
    return modules, names


_FORBIDDEN_ROOTS = {
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


def test_the_run_uses_only_application_ports_and_services() -> None:
    modules, names = _imports(f"{_SERVICES}.run_futures_paper_trading")

    for module in modules:
        assert module.split(".")[0] not in _FORBIDDEN_ROOTS
        assert "acquire" not in module
    assert not names & {"FuturesHistoricalMarketDataSource", "FuturesTradingSessionResolver"}


@pytest.mark.parametrize(
    "module_name",
    [
        f"{_SERVICES}.calculate_futures_paper_trading_metrics",
        f"{_SERVICES}.build_futures_paper_trading_report",
    ],
)
def test_metrics_and_report_consume_only_the_frozen_run(module_name: str) -> None:
    modules, names = _imports(module_name)

    for module in modules:
        assert module.split(".")[0] not in _FORBIDDEN_ROOTS
        assert not module.startswith("northstar_application.ports")
        assert "measure" not in module and "market_data" not in module
    assert not [name for name in names if name.endswith(("Repository", "Store", "Source"))]
    assert "RunFuturesPaperTradingUseCase" not in names
    assert "RunFuturesPaperTradingDecisionUseCase" not in names
    assert not names & {"Money", "Price", "Currency", "FuturesProductSpecification"}


def test_the_public_surface_is_exported_without_helpers() -> None:
    import northstar_application.application_services as services

    for name in (
        "FuturesPaperTradingRun",
        "RunFuturesPaperTradingUseCase",
        "FuturesPaperTradingStrategyContractMetrics",
        "CalculateFuturesPaperTradingMetricsUseCase",
        "FuturesPaperTradingReport",
        "BuildFuturesPaperTradingReportUseCase",
    ):
        assert name in services.__all__
    for private in ("_COUNT_FIELDS", "_load_history", "_validate_forward_output"):
        assert not hasattr(services, private)
