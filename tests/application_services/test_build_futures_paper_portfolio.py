"""Tests for folding futures paper fills into a portfolio snapshot."""

from __future__ import annotations

import ast
import dataclasses
import importlib
from decimal import (
    ROUND_DOWN,
    ROUND_HALF_EVEN,
    ROUND_HALF_UP,
    ROUND_UP,
    Context,
    Decimal,
    localcontext,
)
from pathlib import Path

import pytest
from northstar_core.derivatives import ExpirationDate, QuoteValue
from northstar_core.foundation.value_objects import ExchangeCode, PointInTime, Symbol
from northstar_core.futures import FuturesContract, FuturesProductReference
from northstar_core.paper_trading import (
    FuturesContractCount,
    FuturesExecutionIntent,
    FuturesPaperFill,
    FuturesPaperPortfolio,
    FuturesPosition,
    OrderSide,
    PaperFillIdentity,
    PaperOrderIdentity,
    PaperPortfolioIdentity,
)
from northstar_core.strategy import StrategyIdentity

from northstar_application.application_services import (
    BuildFuturesPaperPortfolioUseCase,
    InvalidFuturesPaperFillHistoryError,
    InvalidPaperFillHistoryError,
)


def _contract(product: str = "ES", expiry: str = "2026-12-18") -> FuturesContract:
    return FuturesContract(
        FuturesProductReference(Symbol(product), ExchangeCode("CME")), ExpirationDate(expiry)
    )


_ES_DEC = _contract()
_ES_MAR = _contract(expiry="2027-03-19")
_MES_DEC = _contract("MES")
_PORTFOLIO = PaperPortfolioIdentity("futures-paper-1")
_STRATEGY = StrategyIdentity("futures-forward")
_DECIDED = PointInTime("2026-09-01T21:00:00Z")
_AS_OF = PointInTime("2026-10-30T21:00:00Z")

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
    fill_id: str | None = None,
) -> FuturesPaperFill:
    count = FuturesContractCount(contracts)
    return FuturesPaperFill(
        identity=PaperFillIdentity(fill_id or f"fill-{order_id}"),
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


def _sequence(*trades: tuple[OrderSide, int, str]) -> tuple[FuturesPaperFill, ...]:
    """One fill per trade on ES Dec, one day apart, in canonical order."""
    return tuple(
        _fill(f"order-{index:02d}", side, contracts, quote, _at(2 + index))
        for index, (side, contracts, quote) in enumerate(trades)
    )


def _build(
    fills: tuple[FuturesPaperFill, ...] = (),
    as_of: PointInTime = _AS_OF,
    *,
    portfolio: PaperPortfolioIdentity = _PORTFOLIO,
    strategy: StrategyIdentity = _STRATEGY,
) -> FuturesPaperPortfolio:
    return BuildFuturesPaperPortfolioUseCase().execute(portfolio, strategy, fills, as_of)


def _position(net: int, average: str, contract: FuturesContract = _ES_DEC) -> FuturesPosition:
    return FuturesPosition(contract, net, QuoteValue(Decimal(average)))


def _only(*trades: tuple[OrderSide, int, str]) -> FuturesPosition | None:
    return _build(_sequence(*trades)).get_position(_ES_DEC)


# ---------------------------------------------------------------------------
# Empty
# ---------------------------------------------------------------------------


def test_no_fills_produces_an_empty_portfolio_owned_by_the_strategy() -> None:
    assert _build(()) == FuturesPaperPortfolio(_PORTFOLIO, _STRATEGY, (), _AS_OF)


def test_as_of_is_the_supplied_instant_not_the_last_fill() -> None:
    portfolio = _build(_sequence((BUY, 1, "100")), as_of=PointInTime("2027-01-01T00:00:00Z"))

    assert portfolio.as_of == PointInTime("2027-01-01T00:00:00Z")


# ---------------------------------------------------------------------------
# Transitions: flat
# ---------------------------------------------------------------------------


def test_flat_buy_opens_a_long() -> None:
    assert _only((BUY, 3, "100")) == _position(3, "100")


def test_flat_sell_opens_a_short() -> None:
    """Unlike equity paper trading, a SELL from flat is a short, not an error."""
    assert _only((SELL, 3, "100")) == _position(-3, "100")


# ---------------------------------------------------------------------------
# Transitions: long
# ---------------------------------------------------------------------------


def test_long_buy_increases_at_the_weighted_average() -> None:
    assert _only((BUY, 2, "100"), (BUY, 1, "130")) == _position(3, "110")


def test_long_sell_partially_reduces_and_keeps_the_average() -> None:
    assert _only((BUY, 5, "100"), (SELL, 2, "250")) == _position(3, "100")


def test_long_sell_of_the_whole_position_closes_it() -> None:
    portfolio = _build(_sequence((BUY, 3, "100"), (SELL, 3, "120")))

    assert portfolio.positions == ()
    assert portfolio.get_position(_ES_DEC) is None


def test_long_sell_beyond_the_position_reverses_to_short_at_the_fill_quote() -> None:
    assert _only((BUY, 2, "100"), (SELL, 5, "90")) == _position(-3, "90")


# ---------------------------------------------------------------------------
# Transitions: short
# ---------------------------------------------------------------------------


def test_short_sell_increases_at_the_weighted_average() -> None:
    assert _only((SELL, 2, "100"), (SELL, 1, "130")) == _position(-3, "110")


def test_short_buy_partially_reduces_and_keeps_the_average() -> None:
    assert _only((SELL, 5, "100"), (BUY, 2, "40")) == _position(-3, "100")


def test_short_buy_of_the_whole_position_closes_it() -> None:
    assert _build(_sequence((SELL, 4, "100"), (BUY, 4, "80"))).positions == ()


def test_short_buy_beyond_the_position_reverses_to_long_at_the_fill_quote() -> None:
    assert _only((SELL, 2, "100"), (BUY, 5, "90")) == _position(3, "90")


def test_a_closed_position_reopens_at_the_new_quote() -> None:
    assert _only((BUY, 2, "100"), (SELL, 2, "110"), (SELL, 1, "120")) == _position(-1, "120")


def test_a_long_sequence_of_transitions() -> None:
    position = _only(
        (BUY, 1, "100"),  # +1 @ 100
        (BUY, 3, "120"),  # +4 @ 115
        (SELL, 1, "200"),  # +3 @ 115
        (SELL, 7, "90"),  # -4 @ 90
        (SELL, 1, "40"),  # -5 @ 80
        (BUY, 2, "10"),  # -3 @ 80
    )

    assert position == _position(-3, "80")


# ---------------------------------------------------------------------------
# Zero and negative quotes
# ---------------------------------------------------------------------------


def test_long_addition_across_zero() -> None:
    assert _only((BUY, 1, "-10"), (BUY, 1, "10")) == _position(2, "0")


def test_short_addition_with_negative_quotes() -> None:
    assert _only((SELL, 2, "-40"), (SELL, 2, "-20")) == _position(-4, "-30")


def test_reversal_at_zero() -> None:
    assert _only((BUY, 1, "5"), (SELL, 3, "0")) == _position(-2, "0")


def test_reversal_at_a_negative_quote() -> None:
    assert _only((SELL, 1, "18.27"), (BUY, 2, "-37.63")) == _position(1, "-37.63")


def test_partial_reduction_keeps_a_negative_average() -> None:
    assert _only((BUY, 3, "-37.63"), (SELL, 1, "5")) == _position(2, "-37.63")


# ---------------------------------------------------------------------------
# Explicit Decimal context
# ---------------------------------------------------------------------------

_EXPECTED_AVERAGE = QuoteValue(
    Context(prec=28, rounding=ROUND_HALF_EVEN).divide(Decimal("302"), Decimal("3"))
)


@pytest.mark.parametrize("precision", [6, 28, 50])
@pytest.mark.parametrize("rounding", [ROUND_DOWN, ROUND_UP, ROUND_HALF_UP, ROUND_HALF_EVEN])
def test_the_average_ignores_the_ambient_decimal_context(precision: int, rounding: str) -> None:
    fills = _sequence((BUY, 1, "100"), (BUY, 2, "101"))

    with localcontext() as ambient:
        ambient.prec = precision
        ambient.rounding = rounding
        position = _build(fills).get_position(_ES_DEC)

    assert position.average_entry == _EXPECTED_AVERAGE
    assert str(_EXPECTED_AVERAGE) == "100.6666666666666666666666667"


def test_the_ambient_context_is_left_untouched() -> None:
    with localcontext() as ambient:
        ambient.prec = 6
        ambient.rounding = ROUND_DOWN
        _build(_sequence((BUY, 1, "100"), (BUY, 2, "101")))

        assert ambient.prec == 6
        assert ambient.rounding == ROUND_DOWN


# ---------------------------------------------------------------------------
# Several contracts
# ---------------------------------------------------------------------------


def test_fills_mutate_only_their_exact_contract() -> None:
    fills = (
        _fill("o-1", BUY, 2, "100", _at(2), contract=_ES_DEC),
        _fill("o-2", SELL, 2, "200", _at(3), contract=_ES_MAR),
        _fill("o-3", BUY, 5, "10", _at(4), contract=_MES_DEC),
        _fill("o-4", SELL, 1, "300", _at(5), contract=_ES_MAR),
    )

    portfolio = _build(fills)
    short_average = Context(prec=28, rounding=ROUND_HALF_EVEN).divide(Decimal("700"), Decimal("3"))

    assert portfolio.positions == (
        _position(2, "100", _ES_DEC),
        FuturesPosition(_ES_MAR, -3, QuoteValue(short_average)),
        _position(5, "10", _MES_DEC),
    )


def test_expiries_are_never_pooled() -> None:
    """Pooled across expiry, BUY 2 Dec and SELL 2 Mar would net to flat."""
    fills = (
        _fill("o-1", BUY, 2, "100", _at(2), contract=_ES_DEC),
        _fill("o-2", SELL, 2, "100", _at(3), contract=_ES_MAR),
    )

    portfolio = _build(fills)

    assert portfolio.position_count == 2
    assert portfolio.get_position(_ES_DEC).net_contracts == 2
    assert portfolio.get_position(_ES_MAR).net_contracts == -2


def test_positions_are_returned_in_core_canonical_order() -> None:
    fills = (
        _fill("o-1", BUY, 1, "1", _at(2), contract=_MES_DEC),
        _fill("o-2", BUY, 1, "1", _at(3), contract=_ES_MAR),
        _fill("o-3", BUY, 1, "1", _at(4), contract=_ES_DEC),
    )

    contracts = [position.contract for position in _build(fills).positions]

    assert contracts == [_ES_DEC, _ES_MAR, _MES_DEC]


# ---------------------------------------------------------------------------
# As-of visibility
# ---------------------------------------------------------------------------


def test_fills_after_as_of_are_ignored() -> None:
    fills = _sequence((BUY, 2, "100"), (SELL, 5, "90"))

    assert _build(fills, as_of=_at(2)).get_position(_ES_DEC) == _position(2, "100")


def test_a_fill_exactly_at_as_of_is_visible() -> None:
    fills = _sequence((BUY, 2, "100"), (SELL, 5, "90"))

    assert _build(fills, as_of=_at(3)).get_position(_ES_DEC) == _position(-3, "90")


def test_an_offset_equivalent_as_of_sees_the_same_fills() -> None:
    fills = _sequence((BUY, 2, "100"), (SELL, 5, "90"))
    offset = PointInTime("2026-09-04T02:30:00+05:30")  # == 3 Sep 21:00Z

    assert _build(fills, as_of=offset).get_position(_ES_DEC) == _position(-3, "90")


def test_as_of_before_every_fill_is_empty_but_owned() -> None:
    portfolio = _build(_sequence((BUY, 2, "100")), as_of=_at(1))

    assert portfolio.positions == ()
    assert portfolio.strategy_identity == _STRATEGY


# ---------------------------------------------------------------------------
# Ordering
# ---------------------------------------------------------------------------


def test_out_of_order_fills_are_rejected_not_sorted() -> None:
    fills = _sequence((BUY, 2, "100"), (SELL, 1, "90"))

    with pytest.raises(InvalidFuturesPaperFillHistoryError, match="must be ordered"):
        _build((fills[1], fills[0]))


def test_semantic_order_is_accepted_where_text_order_disagrees() -> None:
    """'.5Z' sorts before 'Z' as text while being the later instant."""
    whole = _fill("o-1", BUY, 2, "100", _at(2))
    half = _fill("o-2", SELL, 5, "90", _at(2, ".5"))

    assert half.filled_at.value < whole.filled_at.value
    assert _build((whole, half)).get_position(_ES_DEC) == _position(-3, "90")
    with pytest.raises(InvalidFuturesPaperFillHistoryError, match="must be ordered"):
        _build((half, whole))


def test_same_instant_fills_are_ordered_by_order_identity() -> None:
    first = _fill("order-a", BUY, 1, "100", _at(2), fill_id="fill-z")
    second = _fill("order-b", BUY, 1, "100", _at(2), fill_id="fill-a")

    assert _build((first, second)).get_position(_ES_DEC).net_contracts == 2
    with pytest.raises(InvalidFuturesPaperFillHistoryError, match="then order identity"):
        _build((second, first))


def test_the_order_identity_tie_break_is_load_bearing() -> None:
    """Swapping which trade holds the lower order identity changes the outcome."""
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

    # +1 -> SELL 2 reverses to -1 @ 50 -> BUY 2 reverses to +1 @ 100
    assert _build(sell_first).get_position(_ES_DEC) == _position(1, "100")
    # +1 -> BUY 2 adds to +3 @ 70 -> SELL 2 partially reduces to +1 @ 70
    assert _build(buy_first).get_position(_ES_DEC) == _position(1, "70")


# ---------------------------------------------------------------------------
# Duplicates
# ---------------------------------------------------------------------------


def test_the_same_fill_twice_is_rejected() -> None:
    fill = _fill("o-1", BUY, 2, "100", _at(2))

    with pytest.raises(InvalidFuturesPaperFillHistoryError, match="duplicate fill identities"):
        _build((fill, fill))


def test_one_fill_identity_with_differing_payload_is_rejected() -> None:
    first = _fill("o-1", BUY, 2, "100", _at(2), fill_id="fill-x")
    second = _fill("o-2", BUY, 2, "101", _at(3), fill_id="fill-x")

    with pytest.raises(InvalidFuturesPaperFillHistoryError, match="duplicate fill identities"):
        _build((first, second))


def test_two_fills_for_one_order_are_rejected() -> None:
    first = _fill("o-1", BUY, 2, "100", _at(2), fill_id="fill-1")
    second = _fill("o-1", BUY, 2, "100", _at(3), fill_id="fill-2")

    with pytest.raises(InvalidFuturesPaperFillHistoryError, match="two fills for one order"):
        _build((first, second))


def test_duplicates_after_as_of_are_still_rejected() -> None:
    fill = _fill("o-1", BUY, 2, "100", _at(20))

    with pytest.raises(InvalidFuturesPaperFillHistoryError):
        _build((fill, fill), as_of=_at(2))


# ---------------------------------------------------------------------------
# Ownership
# ---------------------------------------------------------------------------


def test_a_fill_from_another_portfolio_is_rejected() -> None:
    foreign = _fill("o-1", BUY, 1, "100", _at(2), portfolio=PaperPortfolioIdentity("other"))

    with pytest.raises(InvalidFuturesPaperFillHistoryError, match="requested portfolio"):
        _build((foreign,))


def test_a_fill_from_another_strategy_is_rejected() -> None:
    foreign = _fill("o-1", BUY, 1, "100", _at(2), strategy=StrategyIdentity("other"))

    with pytest.raises(InvalidFuturesPaperFillHistoryError, match="portfolio's strategy"):
        _build((foreign,))


def test_ownership_is_checked_even_for_fills_after_as_of() -> None:
    foreign = _fill("o-1", BUY, 1, "100", _at(20), strategy=StrategyIdentity("other"))

    with pytest.raises(InvalidFuturesPaperFillHistoryError, match="portfolio's strategy"):
        _build((foreign,), as_of=_at(2))


def test_the_requested_strategy_owns_the_result() -> None:
    other = StrategyIdentity("other")
    fill = _fill("o-1", BUY, 1, "100", _at(2), strategy=other)

    assert _build((fill,), strategy=other).strategy_identity == other


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("args", "message"),
    [
        ((None, _STRATEGY, (), _AS_OF), "portfolio identity cannot be None"),
        (("p", _STRATEGY, (), _AS_OF), "portfolio identity must be a PaperPortfolioIdentity"),
        ((_PORTFOLIO, None, (), _AS_OF), "strategy identity cannot be None"),
        ((_PORTFOLIO, "s", (), _AS_OF), "strategy identity must be a StrategyIdentity"),
        ((_PORTFOLIO, _STRATEGY, None, _AS_OF), "fills cannot be None"),
        ((_PORTFOLIO, _STRATEGY, [], _AS_OF), "fills must be a tuple"),
        ((_PORTFOLIO, _STRATEGY, ("fill",), _AS_OF), "must contain FuturesPaperFill"),
        ((_PORTFOLIO, _STRATEGY, (), None), "as-of instant cannot be None"),
        ((_PORTFOLIO, _STRATEGY, (), "2026-10-30T21:00:00Z"), "as-of instant must be"),
    ],
)
def test_inputs_are_type_checked(args: tuple, message: str) -> None:
    with pytest.raises(TypeError, match=message):
        BuildFuturesPaperPortfolioUseCase().execute(*args)


def test_the_error_is_dedicated_to_futures_history() -> None:
    assert issubclass(InvalidFuturesPaperFillHistoryError, ValueError)
    assert not issubclass(InvalidFuturesPaperFillHistoryError, InvalidPaperFillHistoryError)
    assert not issubclass(InvalidPaperFillHistoryError, InvalidFuturesPaperFillHistoryError)


# ---------------------------------------------------------------------------
# No economics
# ---------------------------------------------------------------------------


def test_positions_carry_exposure_and_basis_only() -> None:
    position = _only((BUY, 2, "100"), (SELL, 1, "500"))

    assert [field.name for field in dataclasses.fields(position)] == [
        "contract",
        "net_contracts",
        "average_entry",
    ]
    assert position == _position(1, "100")


_MODULE_NAME = "northstar_application.application_services.build_futures_paper_portfolio"


def _tree() -> ast.Module:
    module = importlib.import_module(_MODULE_NAME)
    return ast.parse(Path(module.__file__).read_text(encoding="utf-8"))


def test_the_fold_reads_no_market_data_persistence_clock_or_economics() -> None:
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
            "time",
            "datetime",
            "random",
            "uuid",
        }
        assert not module.startswith("northstar_application")
        assert "market_data" not in module
    assert not names & {"Money", "Price", "Currency", "FuturesProductSpecification"}
    for node in ast.walk(_tree()):
        if isinstance(node, ast.Name | ast.Attribute):
            text = node.id if isinstance(node, ast.Name) else node.attr
            for forbidden in ("pnl", "profit", "notional", "margin", "multiplier"):
                assert forbidden not in text.lower()


def test_the_use_case_is_exported_without_its_helpers() -> None:
    import northstar_application.application_services as services

    assert "BuildFuturesPaperPortfolioUseCase" in services.__all__
    assert "InvalidFuturesPaperFillHistoryError" in services.__all__
    assert not hasattr(services, "_BASIS_CONTEXT")
