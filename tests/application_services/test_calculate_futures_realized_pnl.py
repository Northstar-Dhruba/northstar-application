"""Tests for reconstructing gross realized futures P&L from fills and economics."""

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
from pathlib import Path

import pytest
from northstar_core.derivatives import ExpirationDate, QuoteValue
from northstar_core.foundation.value_objects import (
    Currency,
    ExchangeCode,
    Money,
    PointInTime,
    Symbol,
)
from northstar_core.futures import (
    FuturesContract,
    FuturesPointValue,
    FuturesProductEconomics,
    FuturesProductReference,
)
from northstar_core.paper_trading import (
    FuturesContractCount,
    FuturesExecutionIntent,
    FuturesPaperFill,
    FuturesPosition,
    OrderSide,
    PaperFillIdentity,
    PaperOrderIdentity,
    PaperPortfolioIdentity,
)
from northstar_core.strategy import StrategyIdentity

from northstar_application.application_services import (
    BuildFuturesPaperPortfolioUseCase,
    CalculateFuturesRealizedPnlUseCase,
    FuturesContractRealizedPnl,
    InvalidFuturesPaperFillHistoryError,
    InvalidFuturesProductEconomicsInputError,
)

_USD = Currency("USD")
_EUR = Currency("EUR")
_ES = FuturesProductReference(Symbol("ES"), ExchangeCode("CME"))
_FESX = FuturesProductReference(Symbol("FESX"), ExchangeCode("EUREX"))
_ES_DEC = FuturesContract(_ES, ExpirationDate("2026-12-18"))
_ES_MAR = FuturesContract(_ES, ExpirationDate("2027-03-19"))
_FESX_DEC = FuturesContract(_FESX, ExpirationDate("2026-12-18"))
_ES_ECONOMICS = FuturesProductEconomics(_ES, FuturesPointValue(Decimal("50"), _USD))
_FESX_ECONOMICS = FuturesProductEconomics(_FESX, FuturesPointValue(Decimal("10"), _EUR))
_PORTFOLIO = PaperPortfolioIdentity("futures-paper-1")
_STRATEGY = StrategyIdentity("futures-forward")
_DECIDED = PointInTime("2026-09-01T21:00:00Z")
_AS_OF = PointInTime("2026-10-30T21:00:00Z")
_CONTEXT = Context(prec=28, rounding=ROUND_HALF_EVEN)

BUY = OrderSide.BUY
SELL = OrderSide.SELL


def _at(day: int, fraction: str = "") -> PointInTime:
    return PointInTime(f"2026-09-{day:02d}T21:00:00{fraction}Z")


def _fill(
    order_id: str,
    side: OrderSide,
    contracts: int,
    quote: str,
    filled_at: PointInTime,
    *,
    contract: FuturesContract = _ES_DEC,
    portfolio: PaperPortfolioIdentity = _PORTFOLIO,
    strategy: StrategyIdentity = _STRATEGY,
) -> FuturesPaperFill:
    count = FuturesContractCount(contracts)
    return FuturesPaperFill(
        identity=PaperFillIdentity(f"fill-{order_id}"),
        order_identity=PaperOrderIdentity(order_id),
        intent=FuturesExecutionIntent(
            portfolio_identity=portfolio,
            contract=contract,
            side=side,
            contracts=count,
            strategy_identity=strategy,
            decided_at=_DECIDED,
        ),
        contracts=count,
        fill_quote=QuoteValue(Decimal(quote)),
        filled_at=filled_at,
    )


def _sequence(
    *trades: tuple[OrderSide, int, str], contract: FuturesContract = _ES_DEC, prefix: str = "o"
) -> tuple[FuturesPaperFill, ...]:
    """One fill per trade, one day apart, in canonical order."""
    return tuple(
        _fill(f"{prefix}-{index:02d}", side, contracts, quote, _at(2 + index), contract=contract)
        for index, (side, contracts, quote) in enumerate(trades)
    )


def _pnl(
    fills: tuple[FuturesPaperFill, ...],
    economics: tuple[FuturesProductEconomics, ...] = (_ES_ECONOMICS,),
    as_of: PointInTime = _AS_OF,
) -> tuple[FuturesContractRealizedPnl, ...]:
    return CalculateFuturesRealizedPnlUseCase().execute(
        _PORTFOLIO, _STRATEGY, fills, economics, as_of
    )


def _realized(*trades: tuple[OrderSide, int, str]) -> Money:
    (result,) = _pnl(_sequence(*trades))
    return result.realized_pnl


def _usd(amount: str) -> Money:
    return Money(Decimal(amount), _USD)


def _position(fills: tuple[FuturesPaperFill, ...]) -> FuturesPosition | None:
    return (
        BuildFuturesPaperPortfolioUseCase()
        .execute(_PORTFOLIO, _STRATEGY, fills, _AS_OF)
        .get_position(_ES_DEC)
    )


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------


def test_no_visible_fills_realize_nothing() -> None:
    assert _pnl(()) == ()


def test_opening_fills_realize_an_explicit_zero() -> None:
    assert _pnl(_sequence((BUY, 2, "100"), (BUY, 1, "120"))) == (
        FuturesContractRealizedPnl(_ES_DEC, _usd("0")),
    )


# ---------------------------------------------------------------------------
# Long and short closes, point value 50 USD per point per contract
# ---------------------------------------------------------------------------


def test_a_long_partial_profit() -> None:
    fills = _sequence((BUY, 2, "100"), (SELL, 1, "110"))

    assert _realized((BUY, 2, "100"), (SELL, 1, "110")) == _usd("500")
    assert _position(fills) == FuturesPosition(_ES_DEC, 1, QuoteValue(Decimal("100")))


def test_a_long_partial_loss() -> None:
    assert _realized((BUY, 2, "100"), (SELL, 1, "90")) == _usd("-500")


def test_a_short_partial_profit() -> None:
    assert _realized((SELL, 2, "100"), (BUY, 1, "90")) == _usd("500")


def test_a_short_partial_loss() -> None:
    assert _realized((SELL, 2, "100"), (BUY, 1, "110")) == _usd("-500")


def test_a_full_close_realizes_the_whole_position_and_keeps_the_row() -> None:
    fills = _sequence((BUY, 2, "100"), (SELL, 2, "110"))

    assert _pnl(fills) == (FuturesContractRealizedPnl(_ES_DEC, _usd("1000")),)
    assert _position(fills) is None


def test_a_long_reversal_realizes_only_the_old_position() -> None:
    fills = _sequence((BUY, 2, "100"), (SELL, 3, "110"))

    assert _pnl(fills)[0].realized_pnl == _usd("1000")
    assert _position(fills) == FuturesPosition(_ES_DEC, -1, QuoteValue(Decimal("110")))
    # The new short was opened at 110, so closing it at 100 adds +500.
    assert _realized((BUY, 2, "100"), (SELL, 3, "110"), (BUY, 1, "100")) == _usd("1500")


def test_a_short_reversal_realizes_only_the_old_position() -> None:
    fills = _sequence((SELL, 2, "100"), (BUY, 3, "90"))

    assert _pnl(fills)[0].realized_pnl == _usd("1000")
    assert _position(fills) == FuturesPosition(_ES_DEC, 1, QuoteValue(Decimal("90")))
    assert _realized((SELL, 2, "100"), (BUY, 3, "90"), (SELL, 1, "100")) == _usd("1500")


def test_adding_realizes_nothing_and_later_closes_use_the_fold_average() -> None:
    """Closing at 115 realizes against the fold's 110, not a first lot at 100."""
    assert _realized((BUY, 1, "100"), (BUY, 1, "120")) == _usd("0")
    assert _position(_sequence((BUY, 1, "100"), (BUY, 1, "120"))).average_entry == QuoteValue(
        Decimal("110")
    )
    assert _realized((BUY, 1, "100"), (BUY, 1, "120"), (SELL, 1, "115")) == _usd("250")


def test_several_reductions_accumulate() -> None:
    trades = ((BUY, 3, "100"), (SELL, 1, "110"), (SELL, 1, "90"), (SELL, 1, "120"))

    assert _realized(*trades) == _usd("1000")


# ---------------------------------------------------------------------------
# Rounded basis and Decimal determinism
# ---------------------------------------------------------------------------

_ROUNDED_AVERAGE = _CONTEXT.divide(Decimal("302"), Decimal("3"))
_ROUNDED_EXPECTED = _CONTEXT.multiply(
    _CONTEXT.subtract(Decimal("110"), _ROUNDED_AVERAGE), Decimal("50")
)


@pytest.mark.parametrize("precision", [6, 28, 50])
@pytest.mark.parametrize("rounding", [ROUND_DOWN, ROUND_CEILING, ROUND_HALF_UP])
def test_the_rounded_reported_basis_is_closed_under_any_caller_context(
    precision: int, rounding: str
) -> None:
    trades = ((BUY, 2, "100"), (BUY, 1, "102"), (SELL, 1, "110"))

    with localcontext() as ambient:
        ambient.prec = precision
        ambient.rounding = rounding
        realized = _realized(*trades)
        assert (ambient.prec, ambient.rounding) == (precision, rounding)

    assert str(_ROUNDED_AVERAGE) == "100.6666666666666666666666667"
    assert _position(_sequence(*trades[:2])).average_entry == QuoteValue(_ROUNDED_AVERAGE)
    assert realized == _usd(str(_ROUNDED_EXPECTED))


_PRECISE_POINT = Decimal("1.234567890123456789012345678901")


@pytest.mark.parametrize("precision", [6, 50])
def test_the_point_value_conversion_is_owned_by_the_explicit_context(precision: int) -> None:
    economics = (FuturesProductEconomics(_ES, FuturesPointValue(_PRECISE_POINT, _USD)),)
    fills = _sequence((BUY, 1, "100"), (SELL, 1, "110"))

    with localcontext() as ambient:
        ambient.prec = precision
        (result,) = _pnl(fills, economics)

    exact = Context(prec=50).multiply(Decimal("10"), _PRECISE_POINT)
    assert result.realized_pnl.amount == _CONTEXT.multiply(Decimal("10"), _PRECISE_POINT)
    assert result.realized_pnl.amount != exact


def test_quote_points_accumulate_before_one_conversion() -> None:
    """Converting each close separately would round differently here."""
    point = Decimal("3.333333333333333333333333337")
    first = Decimal("0.1428571428571428571428571429")
    second = Decimal("3.666666666666666666666666667")
    economics = (FuturesProductEconomics(_ES, FuturesPointValue(point, _USD)),)
    fills = _sequence(
        (BUY, 1, "100"),
        (SELL, 1, "100.1428571428571428571428571429"),
        (BUY, 1, "100"),
        (SELL, 1, "103.666666666666666666666666667"),
    )

    (result,) = _pnl(fills, economics)
    once = _CONTEXT.multiply(_CONTEXT.add(first, second), point)
    per_close = _CONTEXT.add(_CONTEXT.multiply(first, point), _CONTEXT.multiply(second, point))

    assert once != per_close
    assert result.realized_pnl.amount == once


# ---------------------------------------------------------------------------
# Negative and zero quotes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("trades", "expected"),
    [
        (((BUY, 1, "-20"), (SELL, 1, "-10")), "500"),
        (((SELL, 1, "-20"), (BUY, 1, "-30")), "500"),
        (((BUY, 1, "10"), (SELL, 1, "-5")), "-750"),
        (((SELL, 1, "10"), (BUY, 1, "-5")), "750"),
        (((BUY, 1, "-10"), (SELL, 1, "5")), "750"),
        (((SELL, 1, "-10"), (BUY, 1, "5")), "-750"),
        (((BUY, 1, "10"), (SELL, 1, "0")), "-500"),
        (((SELL, 1, "10"), (BUY, 1, "0")), "500"),
    ],
    ids=[
        "long-neg-profit",
        "short-neg-profit",
        "long-cross-loss",
        "short-cross-profit",
        "long-cross-up-profit",
        "short-cross-up-loss",
        "long-to-zero-loss",
        "short-to-zero-profit",
    ],
)
def test_signed_quotes_and_zero_crossings_are_linear(trades, expected: str) -> None:
    assert _realized(*trades) == _usd(expected)


# ---------------------------------------------------------------------------
# Contracts and currencies
# ---------------------------------------------------------------------------


def test_expiries_sharing_economics_are_separate_results() -> None:
    fills = (
        _fill("a", BUY, 2, "100", _at(2), contract=_ES_MAR),
        _fill("b", BUY, 1, "100", _at(3), contract=_ES_DEC),
        _fill("c", SELL, 2, "90", _at(4), contract=_ES_MAR),
        _fill("d", SELL, 1, "130", _at(5), contract=_ES_DEC),
    )

    assert _pnl(fills) == (
        FuturesContractRealizedPnl(_ES_DEC, _usd("1500")),
        FuturesContractRealizedPnl(_ES_MAR, _usd("-1000")),
    )


def test_currencies_are_never_summed() -> None:
    fills = (
        _fill("a", BUY, 1, "100", _at(2)),
        _fill("b", BUY, 1, "5000", _at(3), contract=_FESX_DEC),
        _fill("c", SELL, 1, "110", _at(4)),
        _fill("d", SELL, 1, "5020", _at(5), contract=_FESX_DEC),
    )

    results = _pnl(fills, (_FESX_ECONOMICS, _ES_ECONOMICS))

    assert results == (
        FuturesContractRealizedPnl(_ES_DEC, _usd("500")),
        FuturesContractRealizedPnl(_FESX_DEC, Money(Decimal("200"), _EUR)),
    )
    assert {result.realized_pnl.currency for result in results} == {_USD, _EUR}


# ---------------------------------------------------------------------------
# As-of visibility
# ---------------------------------------------------------------------------


def test_fills_after_as_of_realize_nothing_yet() -> None:
    fills = _sequence((BUY, 2, "100"), (SELL, 1, "110"), (SELL, 1, "130"))

    assert _pnl(fills, as_of=_at(3))[0].realized_pnl == _usd("500")
    assert _pnl(fills, as_of=_at(4))[0].realized_pnl == _usd("2000")


def test_a_later_fill_cannot_change_an_earlier_basis() -> None:
    fills = _sequence((BUY, 1, "100"), (SELL, 1, "110"), (BUY, 9, "1"))

    assert _pnl(fills, as_of=_at(3)) == _pnl(fills[:2], as_of=_at(3))


def test_an_as_of_before_every_fill_is_empty_and_needs_no_economics() -> None:
    assert _pnl(_sequence((BUY, 1, "100")), economics=(), as_of=_at(1)) == ()


def test_a_product_only_filled_after_as_of_needs_no_economics() -> None:
    fills = (
        _fill("a", BUY, 1, "100", _at(2)),
        _fill("b", BUY, 1, "5000", _at(9), contract=_FESX_DEC),
    )

    assert _pnl(fills, as_of=_at(3)) == (FuturesContractRealizedPnl(_ES_DEC, _usd("0")),)


# ---------------------------------------------------------------------------
# Economics input
# ---------------------------------------------------------------------------


def test_missing_economics_for_a_visible_fill_fails() -> None:
    fills = (_fill("a", BUY, 1, "5000", _at(2), contract=_FESX_DEC),)

    with pytest.raises(InvalidFuturesProductEconomicsInputError, match="no product economics"):
        _pnl(fills)


def test_duplicate_economics_are_rejected_even_when_equal() -> None:
    with pytest.raises(InvalidFuturesProductEconomicsInputError, match="more than once"):
        _pnl(_sequence((BUY, 1, "100")), economics=(_ES_ECONOMICS, _ES_ECONOMICS))


def test_extra_economics_are_allowed() -> None:
    assert _pnl(_sequence((BUY, 1, "100")), economics=(_FESX_ECONOMICS, _ES_ECONOMICS)) == (
        FuturesContractRealizedPnl(_ES_DEC, _usd("0")),
    )


# ---------------------------------------------------------------------------
# Fill history semantics shared with the portfolio fold
# ---------------------------------------------------------------------------


def test_out_of_order_history_is_rejected() -> None:
    fills = _sequence((BUY, 1, "100"), (SELL, 1, "110"))

    with pytest.raises(InvalidFuturesPaperFillHistoryError, match="must be ordered"):
        _pnl((fills[1], fills[0]))


@pytest.mark.parametrize(
    "foreign",
    [
        _fill("x", BUY, 1, "1", _at(20), portfolio=PaperPortfolioIdentity("other")),
        _fill("x", BUY, 1, "1", _at(20), strategy=StrategyIdentity("other")),
    ],
    ids=["portfolio", "strategy"],
)
def test_foreign_fills_are_rejected_even_after_as_of(foreign: FuturesPaperFill) -> None:
    with pytest.raises(InvalidFuturesPaperFillHistoryError):
        _pnl((*_sequence((BUY, 1, "100")), foreign), as_of=_at(2))


def test_duplicate_fills_are_rejected() -> None:
    fill = _fill("a", BUY, 1, "100", _at(2))

    with pytest.raises(InvalidFuturesPaperFillHistoryError, match="duplicate fill identities"):
        _pnl((fill, fill))


def test_the_same_instant_order_identity_tie_break_is_load_bearing() -> None:
    opening = _fill("order-0", BUY, 1, "10", _at(2))
    sell_first = (
        opening,
        _fill("order-a", SELL, 2, "50", _at(3)),
        _fill("order-b", BUY, 2, "100", _at(3)),
    )
    buy_first = (
        opening,
        _fill("order-a", BUY, 2, "100", _at(3)),
        _fill("order-b", SELL, 2, "50", _at(3)),
    )

    # +1@10 -> SELL 2@50 realizes +40 -> short 1@50 -> BUY 2@100 realizes -50.
    assert _pnl(sell_first)[0].realized_pnl == _usd("-500")
    # +1@10 -> BUY 2@100 -> long 3@70 -> SELL 2@50 realizes -40.
    assert _pnl(buy_first)[0].realized_pnl == _usd("-2000")
    with pytest.raises(InvalidFuturesPaperFillHistoryError, match="then order identity"):
        _pnl((opening, sell_first[2], sell_first[1]))


# ---------------------------------------------------------------------------
# Input validation and result value
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("argument", "value", "message"),
    [
        (0, "p", "portfolio identity must be a PaperPortfolioIdentity"),
        (1, "s", "strategy identity must be a StrategyIdentity"),
        (2, [], "fills must be a tuple"),
        (2, ("fill",), "fills must contain FuturesPaperFill"),
        (3, [_ES_ECONOMICS], "economics must be a tuple"),
        (3, (_ES,), "economics must contain FuturesProductEconomics"),
        (4, "2026-10-30T21:00:00Z", "as-of instant must be a PointInTime"),
    ],
)
def test_inputs_are_type_checked(argument: int, value: object, message: str) -> None:
    arguments: list[object] = [_PORTFOLIO, _STRATEGY, (), (_ES_ECONOMICS,), _AS_OF]
    arguments[argument] = value

    with pytest.raises(TypeError, match=message):
        CalculateFuturesRealizedPnlUseCase().execute(*arguments)


def test_the_result_value_is_typed_immutable_and_comparable() -> None:
    result = FuturesContractRealizedPnl(_ES_DEC, _usd("1"))

    assert result == FuturesContractRealizedPnl(_ES_DEC, _usd("1.0"))
    assert hash(result) == hash(FuturesContractRealizedPnl(_ES_DEC, _usd("1")))
    with pytest.raises(AttributeError):
        result.realized_pnl = _usd("2")  # type: ignore[misc]
    with pytest.raises(TypeError, match="contract"):
        FuturesContractRealizedPnl(_ES, _usd("1"))
    with pytest.raises(TypeError, match="Money"):
        FuturesContractRealizedPnl(_ES_DEC, Decimal("1"))


def test_the_result_carries_no_valuation_facts() -> None:
    assert FuturesContractRealizedPnl.__slots__ == ("contract", "realized_pnl")


# ---------------------------------------------------------------------------
# Structure: one fold, no Money arithmetic, no market data
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("net", "side", "contracts", "closed", "after"),
    [
        (None, BUY, 2, 0, (2, "105")),
        (2, BUY, 1, 0, (3, "_avg")),
        (3, SELL, 1, 1, (2, "100")),
        (3, SELL, 3, 3, None),
        (3, SELL, 5, 3, (-2, "105")),
        (-3, BUY, 5, 3, (2, "105")),
    ],
    ids=["open", "add", "partial", "close", "reverse-long", "reverse-short"],
)
def test_the_shared_transition_reports_what_each_branch_closed(
    net: int | None, side: OrderSide, contracts: int, closed: int, after: tuple | None
) -> None:
    from northstar_application.application_services.build_futures_paper_portfolio import (
        _transition,
    )

    prior = None if net is None else FuturesPosition(_ES_DEC, net, QuoteValue(Decimal("100")))
    transition = _transition(prior, _fill("t", side, contracts, "105", _at(2)))

    assert transition.contracts_closed == closed
    assert transition.closing_average_entry == (QuoteValue(Decimal("100")) if closed else None)
    if after is None:
        assert transition.position is None
    else:
        expected_average = (
            _CONTEXT.divide(Decimal("305"), Decimal("3"))
            if after[1] == "_avg"
            else Decimal(after[1])
        )
        assert transition.position == FuturesPosition(
            _ES_DEC, after[0], QuoteValue(expected_average)
        )


_MODULE = "northstar_application.application_services.calculate_futures_realized_pnl"
_ARITHMETIC = (ast.Add, ast.Sub, ast.Mult, ast.Div)


def _tree() -> ast.Module:
    return ast.parse(Path(importlib.import_module(_MODULE).__file__).read_text(encoding="utf-8"))


def test_every_arithmetic_operation_runs_under_the_explicit_context() -> None:
    tree = _tree()
    guarded: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.With) and any(
            isinstance(item.context_expr, ast.Call)
            and getattr(item.context_expr.func, "id", None) == "localcontext"
            for item in node.items
        ):
            guarded.update(id(child) for child in ast.walk(node))

    arithmetic = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.BinOp) and isinstance(node.op, _ARITHMETIC)
    ]
    assert arithmetic
    assert all(id(node) in guarded for node in arithmetic)
    assert not [node for node in ast.walk(tree) if isinstance(node, ast.AugAssign)]


def test_money_is_only_constructed_from_finished_amounts() -> None:
    for node in ast.walk(_tree()):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "Money":
            assert not [
                arg for arg in node.args for inner in ast.walk(arg) if isinstance(inner, ast.BinOp)
            ]


def test_the_calculation_reuses_the_portfolio_fold_transition() -> None:
    modules = {
        node.module: {alias.name for alias in node.names}
        for node in ast.walk(_tree())
        if isinstance(node, ast.ImportFrom) and node.module
    }

    fold = "northstar_application.application_services.build_futures_paper_portfolio"
    assert "_transition" in modules[fold]
    assert "_validate_fill_history" in modules[fold]


def test_the_calculation_depends_on_no_market_data_or_outer_layer() -> None:
    names: set[str] = set()
    modules: set[str] = set()
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
            "time",
            "datetime",
        }
        assert not module.startswith("northstar_application.ports")
    assert not [name for name in names if "Repository" in name or "Store" in name]
    assert not names & {"FuturesOHLCVBar", "FuturesForwardResearchRecord", "Price"}


def test_the_public_surface_is_exported_without_helpers() -> None:
    import northstar_application.application_services as services

    for name in (
        "FuturesContractRealizedPnl",
        "CalculateFuturesRealizedPnlUseCase",
        "InvalidFuturesProductEconomicsInputError",
    ):
        assert name in services.__all__
    for private in ("_transition", "_validate_fill_history", "_PNL_CONTEXT", "_Transition"):
        assert not hasattr(services, private)
