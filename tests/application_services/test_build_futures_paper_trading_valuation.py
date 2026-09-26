"""Tests for the gross simulated futures paper-trading valuation at one cutoff."""

from __future__ import annotations

import ast
import importlib
from decimal import (
    ROUND_CEILING,
    ROUND_DOWN,
    ROUND_HALF_EVEN,
    ROUND_HALF_UP,
    Context,
    Decimal,
    localcontext,
)
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
from northstar_core.strategy import StrategyIdentity

from northstar_application.application_services import (
    BuildFuturesPaperTradingValuationUseCase,
    FuturesContractPnl,
    FuturesPaperTradingContractViolationError,
    FuturesPaperTradingValuation,
    FuturesProductEconomicsContractViolationError,
    FuturesProductEconomicsNotFoundError,
)
from northstar_application.ports import (
    FuturesHistoricalMarketDataQuery,
    FuturesHistoricalMarketDataRepository,
    FuturesPaperFillQuery,
    FuturesPaperFillRepository,
    FuturesPaperOrderQuery,
    FuturesPaperOrderRepository,
    FuturesProductEconomicsRepository,
)

_USD = Currency("USD")
_EUR = Currency("EUR")
_DAILY = Timeframe("1d")
_ES = FuturesProductReference(Symbol("ES"), ExchangeCode("CME"))
_FESX = FuturesProductReference(Symbol("FESX"), ExchangeCode("EUREX"))
_ES_DEC = FuturesContract(_ES, ExpirationDate("2026-12-18"))
_ES_MAR = FuturesContract(_ES, ExpirationDate("2027-03-19"))
_FESX_DEC = FuturesContract(_FESX, ExpirationDate("2026-12-18"))
_ES_ECONOMICS = FuturesProductEconomics(_ES, FuturesPointValue(Decimal("50"), _USD))
_FESX_ECONOMICS = FuturesProductEconomics(_FESX, FuturesPointValue(Decimal("10"), _EUR))
_PORTFOLIO = PaperPortfolioIdentity("futures-paper-1")
_STRATEGY = StrategyIdentity("futures-forward")
_OTHER_STRATEGY = StrategyIdentity("other-strategy")
_DECIDED = PointInTime("2026-09-01T21:00:00Z")
_CONTEXT = Context(prec=28, rounding=ROUND_HALF_EVEN)

BUY = OrderSide.BUY
SELL = OrderSide.SELL


def _at(day: int) -> PointInTime:
    return PointInTime(f"2026-09-{day:02d}T21:00:00Z")


_T = _at(10)


def _usd(amount: str) -> Money:
    return Money(Decimal(amount), _USD)


def _eur(amount: str) -> Money:
    return Money(Decimal(amount), _EUR)


def _quote(value: str) -> QuoteValue:
    return QuoteValue(Decimal(value))


def _sort(items, compare):
    return tuple(sorted(items, key=cmp_to_key(compare)))


class OrderRepository(FuturesPaperOrderRepository):
    def __init__(self, orders: list[FuturesPaperOrder]) -> None:
        self.orders = orders

    def get_orders(self, query: FuturesPaperOrderQuery):
        return _sort(
            (o for o in self.orders if o.intent.portfolio_identity == query.portfolio_identity),
            lambda a, b: (
                a.intent.decided_at.compare(b.intent.decided_at)
                or (a.identity.identity > b.identity.identity)
                - (a.identity.identity < b.identity.identity)
            ),
        )


class FillRepository(FuturesPaperFillRepository):
    def __init__(self, fills: list[FuturesPaperFill]) -> None:
        self.fills = fills

    def get_fills(self, query: FuturesPaperFillQuery):
        return _sort(
            (f for f in self.fills if f.portfolio_identity == query.portfolio_identity),
            lambda a, b: (
                a.filled_at.compare(b.filled_at)
                or (a.order_identity.identity > b.order_identity.identity)
                - (a.order_identity.identity < b.order_identity.identity)
            ),
        )


class MarketRepository(FuturesHistoricalMarketDataRepository):
    def __init__(self, bars: list[FuturesOHLCVBar]) -> None:
        self.bars = bars
        self.queries: list[FuturesHistoricalMarketDataQuery] = []

    def get_bars(self, query: FuturesHistoricalMarketDataQuery):
        self.queries.append(query)
        return _sort(
            (
                bar
                for bar in self.bars
                if bar.contract == query.contract
                and bar.timeframe == query.timeframe
                and query.covers(bar.point_in_time)
            ),
            lambda a, b: a.point_in_time.compare(b.point_in_time),
        )


class EconomicsRepository(FuturesProductEconomicsRepository):
    def __init__(self, *economics: FuturesProductEconomics) -> None:
        self.economics = {entry.reference: entry for entry in economics}
        self.calls: list[FuturesProductReference] = []

    def get_economics(self, reference: FuturesProductReference):
        self.calls.append(reference)
        return self.economics.get(reference)


class Raw:
    """A port double returning one fixed, possibly malformed, answer."""

    def __init__(self, output: object) -> None:
        self.output = output

    def get_orders(self, query):
        return self.output

    def get_fills(self, query):
        return self.output

    def get_economics(self, reference):
        return self.output


class RawOrders(Raw, FuturesPaperOrderRepository):
    pass


class RawFills(Raw, FuturesPaperFillRepository):
    pass


class RawEconomics(Raw, FuturesProductEconomicsRepository):
    pass


def _intent(
    side: OrderSide,
    contracts: int,
    contract: FuturesContract,
    strategy: StrategyIdentity = _STRATEGY,
) -> FuturesExecutionIntent:
    return FuturesExecutionIntent(
        portfolio_identity=_PORTFOLIO,
        contract=contract,
        side=side,
        contracts=FuturesContractCount(contracts),
        strategy_identity=strategy,
        decided_at=_DECIDED,
    )


class World:
    """Stored paper history, market bars and economics for one portfolio."""

    def __init__(self, *economics: FuturesProductEconomics) -> None:
        self.orders: list[FuturesPaperOrder] = []
        self.fills: list[FuturesPaperFill] = []
        self.bars: list[FuturesOHLCVBar] = []
        self.economics = EconomicsRepository(*(economics or (_ES_ECONOMICS, _FESX_ECONOMICS)))
        self.market = MarketRepository(self.bars)

    def trade(
        self,
        side: OrderSide,
        contracts: int,
        quote: str | None,
        day: int,
        *,
        contract: FuturesContract = _ES_DEC,
        strategy: StrategyIdentity = _STRATEGY,
    ) -> World:
        """Store one order, and its fill on ``day`` unless ``quote`` is None."""
        order = FuturesPaperOrder(
            PaperOrderIdentity(f"o-{len(self.orders):02d}"),
            _intent(side, contracts, contract, strategy),
        )
        self.orders.append(order)
        if quote is not None:
            self.fills.append(
                FuturesPaperFill(
                    identity=PaperFillIdentity(f"fill-{order.identity.identity}"),
                    order_identity=order.identity,
                    intent=order.intent,
                    contracts=order.intent.contracts,
                    fill_quote=_quote(quote),
                    filled_at=_at(day),
                )
            )
        return self

    def bar(self, day: int, close: str, contract: FuturesContract = _ES_DEC) -> World:
        closing = Decimal(close)
        self.bars.append(
            FuturesOHLCVBar(
                contract=contract,
                point_in_time=_at(day),
                timeframe=_DAILY,
                open=QuoteValue(closing + 3),
                high=QuoteValue(closing + 10),
                low=QuoteValue(closing - 10),
                close=QuoteValue(closing),
                volume=Quantity(Decimal("1000")),
            )
        )
        return self

    def use_case(self, **overrides) -> BuildFuturesPaperTradingValuationUseCase:
        ports = {
            "order_repository": OrderRepository(self.orders),
            "fill_repository": FillRepository(self.fills),
            "market_repository": self.market,
            "economics_repository": self.economics,
            **overrides,
        }
        return BuildFuturesPaperTradingValuationUseCase(**ports)

    def value(self, through: PointInTime = _T, **overrides) -> FuturesPaperTradingValuation:
        return self.use_case(**overrides).execute(_PORTFOLIO, _STRATEGY, through)


def _row(valuation: FuturesPaperTradingValuation, contract: FuturesContract = _ES_DEC):
    (row,) = [row for row in valuation.contracts if row.contract == contract]
    return row


# ---------------------------------------------------------------------------
# Empty and pending history
# ---------------------------------------------------------------------------


def test_an_empty_history_values_to_an_empty_portfolio_without_reads() -> None:
    world = World()

    valuation = world.value()

    assert valuation.contracts == ()
    assert valuation.portfolio == FuturesPaperPortfolio(_PORTFOLIO, _STRATEGY, (), _T)
    assert (world.economics.calls, world.market.queries) == ([], [])
    assert (valuation.contract_count, valuation.open_position_count) == (0, 0)


def test_a_pending_order_has_no_pnl() -> None:
    world = World().trade(BUY, 1, None, 3)

    valuation = world.value()

    assert valuation.contracts == ()
    assert (world.economics.calls, world.market.queries) == ([], [])


# ---------------------------------------------------------------------------
# One contract
# ---------------------------------------------------------------------------


def test_an_opening_fill_with_a_mark_is_realized_zero_and_unrealized_known() -> None:
    world = World().trade(BUY, 1, "100", 3).bar(9, "110")

    row = _row(world.value())

    assert row.realized_pnl == _usd("0")
    assert row.position == FuturesPosition(_ES_DEC, 1, _quote("100"))
    assert (row.mark_quote, row.mark_instant, row.unrealized_pnl) == (
        _quote("110"),
        _at(9),
        _usd("500"),
    )


def test_an_opening_fill_without_a_mark_is_unavailable_not_zero() -> None:
    world = World().trade(BUY, 1, "100", 3).bar(11, "110")

    valuation = world.value()
    row = _row(valuation)

    assert row.realized_pnl == _usd("0")
    assert row.is_open
    assert (row.mark_quote, row.mark_instant, row.unrealized_pnl) == (None, None, None)
    assert valuation.open_position_count == 1


def test_a_fully_closed_contract_keeps_realized_and_has_exact_zero_unrealized() -> None:
    world = World().trade(BUY, 1, "100", 3).trade(SELL, 1, "110", 4)

    valuation = world.value()
    row = _row(valuation)

    assert valuation.portfolio.positions == ()
    assert row.realized_pnl == _usd("500")
    assert row.position is None
    assert (row.mark_quote, row.mark_instant) == (None, None)
    assert row.unrealized_pnl == _usd("0")
    assert world.market.queries == []


def test_a_partial_reduction_marks_only_the_remaining_contracts() -> None:
    world = World().trade(BUY, 3, "100", 3).trade(SELL, 1, "110", 4).bar(9, "105")

    row = _row(world.value())

    assert row.realized_pnl == _usd("500")
    assert row.position == FuturesPosition(_ES_DEC, 2, _quote("100"))
    assert row.unrealized_pnl == _usd("500")


def test_a_reversal_exposes_the_closed_long_and_the_open_short_separately() -> None:
    world = World().trade(BUY, 1, "100", 3).trade(SELL, 2, "110", 4).bar(9, "90")

    row = _row(world.value())

    assert row.realized_pnl == _usd("500")
    assert row.position == FuturesPosition(_ES_DEC, -1, _quote("110"))
    assert row.unrealized_pnl == _usd("1000")


def test_a_same_direction_add_is_marked_against_the_portfolio_average() -> None:
    world = World().trade(BUY, 2, "100", 3).trade(BUY, 1, "102", 4).bar(9, "101")

    valuation = world.value()
    row = _row(valuation)
    average = valuation.portfolio.get_position(_ES_DEC).average_entry

    assert row.position.average_entry == average == _quote("100.6666666666666666666666667")
    assert row.realized_pnl == _usd("0")
    expected = _CONTEXT.multiply(
        _CONTEXT.multiply(_CONTEXT.subtract(Decimal("101"), average.value), 3), Decimal("50")
    )
    assert row.unrealized_pnl == Money(expected, _USD)
    assert row.unrealized_pnl != _usd("50")  # the exact 302/3 basis would give 50


@pytest.mark.parametrize(
    ("trades", "mark", "realized", "position", "unrealized"),
    [
        ([(BUY, 2, "-20"), (SELL, 1, "-10")], "-5", "500", (1, "-20"), "750"),
        ([(SELL, 2, "-20"), (BUY, 1, "-10")], "-5", "-500", (-1, "-20"), "-750"),
        ([(BUY, 2, "10"), (SELL, 1, "-5")], "0", "-750", (1, "10"), "-500"),
        ([(SELL, 2, "10"), (BUY, 1, "-5")], "0", "750", (-1, "10"), "500"),
        ([(BUY, 1, "-10"), (SELL, 2, "5")], "-5", "750", (-1, "5"), "500"),
    ],
    ids=["long-negative", "short-negative", "long-cross", "short-cross", "reverse-across-zero"],
)
def test_negative_quotes_and_zero_crossings_stay_linear(
    trades, mark, realized, position, unrealized
) -> None:
    world = World()
    for day, (side, contracts, quote) in enumerate(trades, start=3):
        world.trade(side, contracts, quote, day)
    world.bar(9, mark)

    row = _row(world.value())

    assert row.realized_pnl == _usd(realized)
    assert row.position == FuturesPosition(_ES_DEC, position[0], _quote(position[1]))
    assert row.unrealized_pnl == _usd(unrealized)


def test_an_available_zero_mark_is_not_unavailable() -> None:
    marked = _row(World().trade(BUY, 1, "100", 3).bar(9, "100").value())
    unmarked = _row(World().trade(BUY, 1, "100", 3).value())

    assert marked.unrealized_pnl == _usd("0")
    assert marked.mark_quote == _quote("100")
    assert unmarked.unrealized_pnl is None


# ---------------------------------------------------------------------------
# Several contracts
# ---------------------------------------------------------------------------


def _mixed_world() -> World:
    """ES Dec closed, ES Mar open without a mark, FESX Dec open and marked."""
    return (
        World()
        .trade(BUY, 1, "100", 2, contract=_ES_DEC)
        .trade(SELL, 1, "104", 3, contract=_ES_DEC)
        .trade(SELL, 2, "200", 4, contract=_ES_MAR)
        .trade(BUY, 3, "5000", 5, contract=_FESX_DEC)
        .bar(9, "5010", _FESX_DEC)
        .bar(11, "190", _ES_MAR)
    )


def test_closed_and_unavailable_contracts_stay_distinct() -> None:
    valuation = _mixed_world().value()
    closed, unmarked = _row(valuation, _ES_DEC), _row(valuation, _ES_MAR)

    assert (closed.position, closed.unrealized_pnl) == (None, _usd("0"))
    assert closed.realized_pnl == _usd("200")
    assert closed.position is None and not closed.is_open
    assert unmarked.position == FuturesPosition(_ES_MAR, -2, _quote("200"))
    assert unmarked.unrealized_pnl is None
    assert unmarked.realized_pnl == _usd("0")


def test_every_traded_contract_gets_one_row_in_contract_order() -> None:
    world = _mixed_world()

    valuation = world.value()

    assert [row.contract for row in valuation.contracts] == sorted(
        [_ES_DEC, _ES_MAR, _FESX_DEC], key=lambda contract: contract.natural_key
    )
    assert tuple(r.position for r in valuation.contracts if r.is_open) == (
        valuation.portfolio.positions
    )
    assert (valuation.contract_count, valuation.open_position_count) == (3, 2)
    assert [query.contract for query in world.market.queries] == [_ES_MAR, _FESX_DEC]


def test_currencies_are_kept_per_contract_and_never_summed() -> None:
    valuation = _mixed_world().value()
    fesx = _row(valuation, _FESX_DEC)

    assert (fesx.realized_pnl, fesx.unrealized_pnl) == (_eur("0"), _eur("300"))
    assert fesx.settlement_currency == _EUR
    assert _row(valuation, _ES_DEC).settlement_currency == _USD
    for name in ("total_pnl", "gross_pnl", "total_realized", "total_unrealized"):
        assert not hasattr(valuation, name)
        assert not hasattr(fesx, name)


def test_expiries_share_economics_but_not_pnl() -> None:
    world = (
        World()
        .trade(BUY, 1, "100", 3, contract=_ES_DEC)
        .trade(BUY, 1, "100", 4, contract=_ES_MAR)
        .bar(9, "110", _ES_DEC)
        .bar(9, "90", _ES_MAR)
    )

    valuation = world.value()

    assert [row.unrealized_pnl for row in valuation.contracts] == [_usd("500"), _usd("-500")]
    # One lookup for the realized side, one inside the unrealized service.
    assert world.economics.calls == [_ES, _ES]
    assert [query.contract for query in world.market.queries] == [_ES_DEC, _ES_MAR]


# ---------------------------------------------------------------------------
# Cutoff
# ---------------------------------------------------------------------------


def test_a_future_fill_changes_nothing_and_needs_no_economics() -> None:
    world = World(_ES_ECONOMICS).trade(BUY, 1, "100", 3).bar(9, "110")
    before = world.value()

    world.trade(SELL, 1, "200", 11)
    world.trade(BUY, 1, "5000", 12, contract=_FESX_DEC)
    after = world.value()

    assert after == before
    assert [row.contract for row in after.contracts] == [_ES_DEC]
    assert _FESX not in world.economics.calls


def test_a_future_bar_changes_nothing() -> None:
    world = World().trade(BUY, 1, "100", 3).bar(9, "110")
    before = world.value()

    world.bar(11, "110").bar(12, "999").bar(13, "-999")

    assert world.value() == before
    assert _row(before).mark_quote == _quote("110")


def test_a_later_cutoff_evolves_while_the_earlier_view_is_stable() -> None:
    world = World().trade(BUY, 1, "100", 3).trade(SELL, 1, "120", 12).bar(11, "115")

    early = world.value(_T)
    late = world.value(_at(13))

    assert _row(early).unrealized_pnl is None and _row(early).realized_pnl == _usd("0")
    assert _row(late).position is None
    assert (_row(late).realized_pnl, _row(late).unrealized_pnl) == (_usd("1000"), _usd("0"))
    assert world.value(_T) == early
    assert len(world.fills) == 2


# ---------------------------------------------------------------------------
# History validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("day", [3, 20], ids=["visible", "future"])
def test_a_mixed_strategy_history_is_rejected(day: int) -> None:
    world = World().trade(BUY, 1, "100", 3)
    world.trade(BUY, 1, "101", day, contract=_ES_MAR, strategy=_OTHER_STRATEGY)

    with pytest.raises(ValueError, match="one paper portfolio belongs to one strategy"):
        world.value()


def test_a_pending_order_of_another_strategy_is_rejected() -> None:
    world = World().trade(BUY, 1, "100", 3).trade(BUY, 1, None, 4, strategy=_OTHER_STRATEGY)

    with pytest.raises(ValueError, match="one paper portfolio belongs to one strategy"):
        world.value()


def test_malformed_order_output_is_rejected() -> None:
    world = World().trade(BUY, 1, "100", 3)

    with pytest.raises(FuturesPaperTradingContractViolationError, match="tuple"):
        world.value(order_repository=RawOrders(list(world.orders)))
    with pytest.raises(FuturesPaperTradingContractViolationError, match="out of order"):
        world.value(order_repository=RawOrders((world.orders[0], world.orders[0])))


def test_malformed_fill_output_is_rejected() -> None:
    world = World().trade(BUY, 1, "100", 3).trade(SELL, 1, "101", 4)

    with pytest.raises(FuturesPaperTradingContractViolationError, match="tuple"):
        world.value(fill_repository=RawFills(list(world.fills)))
    with pytest.raises(FuturesPaperTradingContractViolationError, match="out of order"):
        world.value(fill_repository=RawFills(tuple(reversed(world.fills))))


def test_a_fill_without_its_order_is_rejected() -> None:
    world = World().trade(BUY, 1, "100", 3)

    with pytest.raises(FuturesPaperTradingContractViolationError, match="no loaded order"):
        world.value(order_repository=RawOrders(()))


def test_a_fill_carrying_another_intent_is_rejected() -> None:
    world = World().trade(BUY, 1, "100", 3)
    stored = world.fills[0]
    impostor = FuturesPaperFill(
        identity=stored.identity,
        order_identity=stored.order_identity,
        intent=_intent(SELL, 1, _ES_DEC),
        contracts=stored.contracts,
        fill_quote=stored.fill_quote,
        filled_at=stored.filled_at,
    )

    with pytest.raises(FuturesPaperTradingContractViolationError, match="order's intent"):
        world.value(fill_repository=RawFills((impostor,)))


def test_missing_economics_for_a_visible_product_fails() -> None:
    world = World(_ES_ECONOMICS).trade(BUY, 1, "5000", 3, contract=_FESX_DEC)

    with pytest.raises(FuturesProductEconomicsNotFoundError, match="FESX@EUREX"):
        world.value()


@pytest.mark.parametrize(
    "output", ["economics", _FESX_ECONOMICS], ids=["wrong-type", "wrong-reference"]
)
def test_a_misbehaving_economics_repository_fails(output) -> None:
    world = World().trade(BUY, 1, "100", 3)

    with pytest.raises(FuturesProductEconomicsContractViolationError):
        world.value(economics_repository=RawEconomics(output))


def test_dependencies_and_inputs_are_type_checked() -> None:
    world = World()
    for name in (
        "order_repository",
        "fill_repository",
        "market_repository",
        "economics_repository",
    ):
        with pytest.raises(TypeError, match=name):
            world.use_case(**{name: object()})

    use_case = world.use_case()
    with pytest.raises(TypeError, match="portfolio identity"):
        use_case.execute("futures-paper-1", _STRATEGY, _T)
    with pytest.raises(TypeError, match="strategy identity"):
        use_case.execute(_PORTFOLIO, "futures-forward", _T)
    with pytest.raises(TypeError, match="available-through"):
        use_case.execute(_PORTFOLIO, _STRATEGY, "2026-09-10T21:00:00Z")


# ---------------------------------------------------------------------------
# Decimal determinism
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("precision", [6, 28, 50])
@pytest.mark.parametrize("rounding", [ROUND_DOWN, ROUND_CEILING, ROUND_HALF_UP])
def test_the_valuation_ignores_the_callers_decimal_context(precision: int, rounding) -> None:
    economics = FuturesProductEconomics(
        _ES, FuturesPointValue(Decimal("12.34567890123456789012345678"), _USD)
    )

    def build() -> FuturesPaperTradingValuation:
        return (
            World(economics)
            .trade(BUY, 3, "100.1234567890123456789012345", 3)
            .trade(BUY, 4, "101.9876543210987654321098765", 4)
            .trade(SELL, 2, "102.5555555555555555555555555", 5)
            .bar(9, "99.87654321098765432109876543")
            .value()
        )

    baseline = build()
    with localcontext() as ambient:
        ambient.prec = precision
        ambient.rounding = rounding
        assert build() == baseline
        assert (ambient.prec, ambient.rounding) == (precision, rounding)


# ---------------------------------------------------------------------------
# FuturesContractPnl
# ---------------------------------------------------------------------------

_OPEN = FuturesPosition(_ES_DEC, 2, _quote("100"))


def _pnl(**fields) -> FuturesContractPnl:
    values = {
        "contract": _ES_DEC,
        "realized_pnl": _usd("10"),
        "position": _OPEN,
        "mark_quote": _quote("101"),
        "mark_instant": _T,
        "unrealized_pnl": _usd("100"),
        **fields,
    }
    return FuturesContractPnl(**values)


_FLAT = {"position": None, "mark_quote": None, "mark_instant": None, "unrealized_pnl": _usd("0")}
_UNMARKED = {"mark_quote": None, "mark_instant": None, "unrealized_pnl": None}


def test_the_three_valid_contract_states() -> None:
    flat, marked, unmarked = _pnl(**_FLAT), _pnl(), _pnl(**_UNMARKED)

    assert (flat.is_open, flat.unrealized_pnl) == (False, _usd("0"))
    assert (marked.is_open, marked.unrealized_pnl) == (True, _usd("100"))
    assert (unmarked.is_open, unmarked.unrealized_pnl) == (True, None)
    assert flat.settlement_currency == _USD


@pytest.mark.parametrize(
    "fields",
    [
        {"contract": "ES"},
        {"realized_pnl": Decimal("10")},
        {"position": "long"},
        {"mark_quote": Decimal("101")},
        {"mark_instant": "2026-09-10T21:00:00Z"},
        {"unrealized_pnl": Decimal("100")},
    ],
)
def test_field_types_are_checked(fields: dict) -> None:
    with pytest.raises(TypeError):
        _pnl(**fields)


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"position": FuturesPosition(_ES_MAR, 2, _quote("100"))}, "for its contract"),
        ({**_FLAT, "unrealized_pnl": None}, "exactly zero"),
        ({**_FLAT, "unrealized_pnl": _usd("1")}, "exactly zero"),
        ({**_FLAT, "unrealized_pnl": _eur("0")}, "exactly zero"),
        ({**_FLAT, "mark_quote": _quote("101")}, "cannot carry a mark"),
        ({**_FLAT, "mark_instant": _T}, "cannot carry a mark"),
        ({"unrealized_pnl": None}, "together, or none"),
        ({"mark_quote": None}, "together, or none"),
        ({**_UNMARKED, "mark_instant": _T}, "together, or none"),
        ({**_UNMARKED, "unrealized_pnl": _usd("0")}, "together, or none"),
        ({"unrealized_pnl": _eur("100")}, "one settlement currency"),
    ],
    ids=[
        "position-other-contract",
        "flat-none",
        "flat-nonzero",
        "flat-other-currency",
        "flat-mark-quote",
        "flat-mark-instant",
        "open-no-pnl",
        "open-no-quote",
        "unmarked-with-instant",
        "unmarked-with-zero",
        "currency-mismatch",
    ],
)
def test_incoherent_contract_states_are_rejected(fields: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _pnl(**fields)


# ---------------------------------------------------------------------------
# FuturesPaperTradingValuation
# ---------------------------------------------------------------------------

_MAR_OPEN = FuturesPosition(_ES_MAR, -1, _quote("200"))
_PORTFOLIO_AT_T = FuturesPaperPortfolio(_PORTFOLIO, _STRATEGY, (_OPEN, _MAR_OPEN), _T)
_DEC_ROW = _pnl()
_MAR_ROW = _pnl(contract=_ES_MAR, position=_MAR_OPEN, **_UNMARKED)
_FESX_FLAT = _pnl(contract=_FESX_DEC, **{**_FLAT, "unrealized_pnl": _usd("0")})


def _valuation(**fields) -> FuturesPaperTradingValuation:
    values = {
        "portfolio_identity": _PORTFOLIO,
        "strategy_identity": _STRATEGY,
        "available_through": _T,
        "portfolio": _PORTFOLIO_AT_T,
        "contracts": tuple(
            sorted((_DEC_ROW, _MAR_ROW, _FESX_FLAT), key=lambda r: r.contract.natural_key)
        ),
        **fields,
    }
    return FuturesPaperTradingValuation(**values)


def test_flat_rows_beyond_the_portfolio_are_allowed() -> None:
    valuation = _valuation()

    assert (valuation.contract_count, valuation.open_position_count) == (3, 2)


@pytest.mark.parametrize(
    "fields",
    [
        {"portfolio_identity": "futures-paper-1"},
        {"strategy_identity": "futures-forward"},
        {"available_through": "2026-09-10T21:00:00Z"},
        {"portfolio": ()},
        {"contracts": [_DEC_ROW, _MAR_ROW]},
        {"contracts": (_DEC_ROW, "row")},
    ],
)
def test_valuation_field_types_are_checked(fields: dict) -> None:
    with pytest.raises(TypeError):
        _valuation(**fields)


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"portfolio_identity": PaperPortfolioIdentity("other")}, "valued portfolio"),
        ({"strategy_identity": _OTHER_STRATEGY}, "valued strategy"),
        ({"available_through": _at(11)}, "valuation cutoff"),
        ({"contracts": (_DEC_ROW, _DEC_ROW, _MAR_ROW)}, "unique and ordered"),
        ({"contracts": (_MAR_ROW, _DEC_ROW)}, "unique and ordered"),
        ({"contracts": (_DEC_ROW,)}, "exactly the portfolio's positions"),
        (
            {"contracts": (_pnl(position=FuturesPosition(_ES_DEC, 3, _quote("100"))), _MAR_ROW)},
            "exactly the portfolio's positions",
        ),
        (
            {"portfolio": FuturesPaperPortfolio(_PORTFOLIO, _STRATEGY, (_OPEN,), _T)},
            "exactly the portfolio's positions",
        ),
    ],
    ids=[
        "identity",
        "strategy",
        "cutoff",
        "duplicate",
        "order",
        "missing-open-row",
        "different-position",
        "extra-open-row",
    ],
)
def test_incoherent_valuations_are_rejected(fields: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _valuation(**fields)


# ---------------------------------------------------------------------------
# Structure and boundaries
# ---------------------------------------------------------------------------

_MODULE = "northstar_application.application_services.build_futures_paper_trading_valuation"


def _tree() -> ast.Module:
    return ast.parse(Path(importlib.import_module(_MODULE).__file__).read_text(encoding="utf-8"))


def test_the_orchestration_performs_no_arithmetic() -> None:
    tree = _tree()

    arithmetic = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Pow, ast.USub)

    assert not [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.AugAssign)
        or (isinstance(node, (ast.BinOp, ast.UnaryOp)) and isinstance(node.op, arithmetic))
    ]


def test_the_valuation_touches_no_outer_layer_or_store() -> None:
    modules: set[str] = set()
    names: set[str] = set()
    for node in ast.walk(_tree()):
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)

    for module in modules:
        assert module.split(".")[0] not in {
            "northstar_infrastructure",
            "sqlite3",
            "databento",
            "exchange_calendars",
            "requests",
            "urllib",
            "time",
            "datetime",
        }
        assert "report" not in module and "metrics" not in module
    for forbidden in (
        "FuturesPaperOrderStore",
        "FuturesPaperFillStore",
        "FuturesHistoricalMarketDataSource",
        "FuturesPaperTradingRun",
        "FuturesPaperTradingReport",
    ):
        assert forbidden not in names


def test_the_public_surface_is_exported_without_helpers() -> None:
    import northstar_application.application_services as services

    for name in (
        "FuturesContractPnl",
        "FuturesPaperTradingValuation",
        "BuildFuturesPaperTradingValuationUseCase",
    ):
        assert name in services.__all__
    for private in ("_combine", "_economics_for", "_load_history"):
        assert not hasattr(services, private)
