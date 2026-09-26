"""Tests for folding a paper fill history into a portfolio snapshot."""

from __future__ import annotations

from decimal import ROUND_DOWN, Decimal, getcontext, localcontext

import pytest
from northstar_core.domain.value_objects import ListingReference
from northstar_core.foundation.value_objects import (
    Currency,
    ExchangeCode,
    PointInTime,
    Price,
    Quantity,
    Symbol,
)
from northstar_core.paper_trading import (
    ExecutionIntent,
    OrderSide,
    PaperFill,
    PaperFillIdentity,
    PaperOrderIdentity,
    PaperPortfolio,
    PaperPortfolioIdentity,
    Position,
)
from northstar_core.strategy import StrategyIdentity

from northstar_application.application_services import (
    BuildPaperPortfolioUseCase,
    InvalidPaperFillHistoryError,
)

_USD = Currency("USD")
_EUR = Currency("EUR")
_NASDAQ = ExchangeCode("NASDAQ")
_LSE = ExchangeCode("LSE")
_AAPL = ListingReference(Symbol("AAPL"), _NASDAQ)
_MSFT = ListingReference(Symbol("MSFT"), _NASDAQ)
_AAPL_LSE = ListingReference(Symbol("AAPL"), _LSE)
_STRATEGY = StrategyIdentity("alpha")
_PORTFOLIO = PaperPortfolioIdentity("paper-1")
_OTHER_PORTFOLIO = PaperPortfolioIdentity("paper-2")
_AS_OF = PointInTime("2026-03-01T16:00:00Z")


def _instant(day: int, *, fraction: str = "") -> PointInTime:
    return PointInTime(f"2026-01-{day:02d}T16:00:00{fraction}Z")


def _fill(
    day: int,
    side: OrderSide = OrderSide.BUY,
    quantity: str = "10",
    price: str = "100",
    *,
    listing: ListingReference = _AAPL,
    currency: Currency = _USD,
    fill_id: str | None = None,
    order_id: str | None = None,
    portfolio: PaperPortfolioIdentity = _PORTFOLIO,
    filled_at: PointInTime | None = None,
    decided_at: PointInTime | None = None,
) -> PaperFill:
    instant = filled_at or _instant(day)
    intent = ExecutionIntent(
        portfolio_identity=portfolio,
        listing_reference=listing,
        side=side,
        quantity=Quantity(quantity),
        strategy_identity=_STRATEGY,
        decided_at=decided_at or instant,
    )
    return PaperFill(
        identity=PaperFillIdentity(fill_id or f"fill-{day:02d}"),
        order_identity=PaperOrderIdentity(order_id or f"order-{day:02d}"),
        intent=intent,
        quantity=Quantity(quantity),
        price=Price(price, currency),
        filled_at=instant,
    )


def _build(
    fills: tuple[PaperFill, ...] = (),
    as_of: PointInTime = _AS_OF,
    portfolio_identity: PaperPortfolioIdentity = _PORTFOLIO,
) -> PaperPortfolio:
    return BuildPaperPortfolioUseCase().execute(portfolio_identity, fills, as_of)


# ---------------------------------------------------------------------------
# Empty and single-fill folds
# ---------------------------------------------------------------------------


def test_no_fills_produces_an_empty_portfolio() -> None:
    portfolio = _build(())

    assert portfolio == PaperPortfolio(identity=_PORTFOLIO, positions=(), as_of=_AS_OF)
    assert portfolio.position_count == 0


def test_one_buy_opens_the_position_at_its_fill_price() -> None:
    portfolio = _build((_fill(20, quantity="10", price="100"),))

    assert portfolio.positions == (Position(_AAPL, Quantity("10"), Price("100", _USD)),)


def test_a_fractional_buy_is_preserved() -> None:
    portfolio = _build((_fill(20, quantity="0.5", price="123.45"),))

    assert portfolio.get_position(_AAPL) == Position(_AAPL, Quantity("0.5"), Price("123.45", _USD))


# ---------------------------------------------------------------------------
# BUY accounting
# ---------------------------------------------------------------------------


def test_two_buys_produce_a_weighted_average_cost() -> None:
    fills = (
        _fill(20, quantity="10", price="100"),
        _fill(21, quantity="10", price="200"),
    )

    position = _build(fills).get_position(_AAPL)

    assert position.quantity == Quantity("20")
    assert position.average_price == Price("150", _USD)


def test_weighted_average_respects_relative_size() -> None:
    fills = (
        _fill(20, quantity="30", price="100"),
        _fill(21, quantity="10", price="200"),
    )

    position = _build(fills).get_position(_AAPL)

    assert position.quantity == Quantity("40")
    assert position.average_price == Price("125", _USD)


def test_three_buys_accumulate_correctly() -> None:
    fills = (
        _fill(20, quantity="10", price="100"),
        _fill(21, quantity="10", price="200"),
        _fill(22, quantity="20", price="50"),
    )

    position = _build(fills).get_position(_AAPL)

    assert position.quantity == Quantity("40")
    # (10*100 + 10*200 + 20*50) / 40 = 4000 / 40
    assert position.average_price == Price("100", _USD)


def test_buy_in_a_different_currency_is_rejected() -> None:
    fills = (
        _fill(20, quantity="10", price="100"),
        _fill(21, quantity="10", price="100", currency=_EUR),
    )

    with pytest.raises(InvalidPaperFillHistoryError, match="cannot add a fill denominated in"):
        _build(fills)


# ---------------------------------------------------------------------------
# SELL accounting
# ---------------------------------------------------------------------------


def test_partial_sell_reduces_quantity_and_preserves_average_price() -> None:
    fills = (
        _fill(20, quantity="30", price="100"),
        _fill(21, quantity="10", price="200"),
        _fill(22, OrderSide.SELL, quantity="10", price="500"),
    )

    position = _build(fills).get_position(_AAPL)

    assert position.quantity == Quantity("30")
    assert position.average_price == Price("125", _USD)


def test_sell_price_never_changes_the_cost_basis() -> None:
    """No realised profit and loss is calculated, so cost basis is untouched."""
    for sell_price in ("1", "100", "99999"):
        fills = (
            _fill(20, quantity="10", price="100"),
            _fill(21, OrderSide.SELL, quantity="4", price=sell_price),
        )
        assert _build(fills).get_position(_AAPL).average_price == Price("100", _USD)


def test_full_sell_removes_the_position_entirely() -> None:
    fills = (
        _fill(20, quantity="10", price="100"),
        _fill(21, OrderSide.SELL, quantity="10", price="150"),
    )

    portfolio = _build(fills)

    assert portfolio.positions == ()
    assert portfolio.get_position(_AAPL) is None


def test_a_closed_out_position_is_never_held_at_zero() -> None:
    fills = (
        _fill(20, quantity="10", price="100"),
        _fill(21, OrderSide.SELL, quantity="10", price="150"),
    )

    assert all(position.quantity.value > 0 for position in _build(fills).positions)


def test_a_position_can_be_reopened_after_being_closed() -> None:
    fills = (
        _fill(20, quantity="10", price="100"),
        _fill(21, OrderSide.SELL, quantity="10", price="150"),
        _fill(22, quantity="5", price="300"),
    )

    position = _build(fills).get_position(_AAPL)

    assert position == Position(_AAPL, Quantity("5"), Price("300", _USD))


def test_sell_without_a_position_is_rejected() -> None:
    with pytest.raises(InvalidPaperFillHistoryError, match="which is not held"):
        _build((_fill(20, OrderSide.SELL, quantity="10", price="100"),))


def test_sell_of_a_different_listing_than_held_is_rejected() -> None:
    fills = (
        _fill(20, quantity="10", price="100", listing=_AAPL),
        _fill(21, OrderSide.SELL, quantity="1", price="100", listing=_MSFT),
    )

    with pytest.raises(InvalidPaperFillHistoryError, match="which is not held"):
        _build(fills)


def test_overselling_is_rejected() -> None:
    fills = (
        _fill(20, quantity="10", price="100"),
        _fill(21, OrderSide.SELL, quantity="11", price="100"),
    )

    with pytest.raises(InvalidPaperFillHistoryError, match="cannot sell more than is held"):
        _build(fills)


def test_overselling_by_the_smallest_fraction_is_rejected() -> None:
    fills = (
        _fill(20, quantity="10", price="100"),
        _fill(21, OrderSide.SELL, quantity="10.00000001", price="100"),
    )

    with pytest.raises(InvalidPaperFillHistoryError, match="does not permit short positions"):
        _build(fills)


def test_sell_in_a_different_currency_is_rejected() -> None:
    fills = (
        _fill(20, quantity="10", price="100"),
        _fill(21, OrderSide.SELL, quantity="5", price="100", currency=_EUR),
    )

    with pytest.raises(InvalidPaperFillHistoryError, match="cannot sell a position held in"):
        _build(fills)


# ---------------------------------------------------------------------------
# Listing independence
# ---------------------------------------------------------------------------


def test_independent_listings_remain_independent() -> None:
    fills = (
        _fill(20, quantity="10", price="100", listing=_AAPL),
        _fill(21, quantity="5", price="300", listing=_MSFT),
        _fill(22, OrderSide.SELL, quantity="10", price="110", listing=_AAPL),
    )

    portfolio = _build(fills)

    assert portfolio.get_position(_AAPL) is None
    assert portfolio.get_position(_MSFT) == Position(_MSFT, Quantity("5"), Price("300", _USD))


def test_the_same_symbol_on_different_exchanges_remains_independent() -> None:
    fills = (
        _fill(20, quantity="10", price="100", listing=_AAPL),
        _fill(21, quantity="4", price="200", listing=_AAPL_LSE),
    )

    portfolio = _build(fills)

    assert portfolio.get_position(_AAPL) == Position(_AAPL, Quantity("10"), Price("100", _USD))
    assert portfolio.get_position(_AAPL_LSE) == Position(
        _AAPL_LSE, Quantity("4"), Price("200", _USD)
    )


def test_selling_one_exchange_does_not_touch_the_other() -> None:
    fills = (
        _fill(20, quantity="10", price="100", listing=_AAPL),
        _fill(21, quantity="4", price="200", listing=_AAPL_LSE),
        _fill(22, OrderSide.SELL, quantity="4", price="250", listing=_AAPL_LSE),
    )

    portfolio = _build(fills)

    assert portfolio.get_position(_AAPL_LSE) is None
    assert portfolio.get_position(_AAPL).quantity == Quantity("10")


# ---------------------------------------------------------------------------
# Canonical ordering of the result
# ---------------------------------------------------------------------------


def test_final_positions_are_canonically_ordered() -> None:
    fills = (
        _fill(20, quantity="1", price="100", listing=_MSFT),
        _fill(21, quantity="1", price="100", listing=_AAPL),
        _fill(22, quantity="1", price="100", listing=_AAPL_LSE),
    )

    portfolio = _build(fills)

    assert [
        (p.listing_reference.symbol.value, p.listing_reference.exchange_code.value)
        for p in portfolio.positions
    ] == [("AAPL", "LSE"), ("AAPL", "NASDAQ"), ("MSFT", "NASDAQ")]


def test_canonical_order_does_not_follow_fill_order() -> None:
    reversed_fills = (
        _fill(20, quantity="1", price="100", listing=_MSFT),
        _fill(21, quantity="1", price="100", listing=_AAPL),
    )

    assert _build(reversed_fills).positions[0].listing_reference == _AAPL


# ---------------------------------------------------------------------------
# As-of semantics
# ---------------------------------------------------------------------------


def test_fills_after_the_as_of_instant_are_ignored() -> None:
    fills = (
        _fill(20, quantity="10", price="100"),
        _fill(25, quantity="10", price="200"),
    )

    position = _build(fills, as_of=_instant(21)).get_position(_AAPL)

    assert position == Position(_AAPL, Quantity("10"), Price("100", _USD))


def test_a_fill_exactly_at_the_as_of_instant_is_included() -> None:
    fills = (_fill(20, quantity="10", price="100"),)

    assert _build(fills, as_of=_instant(20)).get_position(_AAPL) is not None


def test_an_as_of_before_every_fill_produces_an_empty_portfolio() -> None:
    fills = (_fill(20, quantity="10", price="100"),)

    portfolio = _build(fills, as_of=_instant(19))

    assert portfolio.positions == ()
    assert portfolio.as_of == _instant(19)


def test_as_of_comparison_is_semantic_not_textual() -> None:
    """An offset spelling of the same instant still includes the fill."""
    fills = (_fill(20, quantity="10", price="100"),)
    offset_as_of = PointInTime("2026-01-20T21:30:00+05:30")

    assert offset_as_of.compare(_instant(20)) == 0
    assert _build(fills, as_of=offset_as_of).get_position(_AAPL) is not None


def test_sub_second_fill_is_excluded_by_a_whole_second_as_of() -> None:
    """Text ordering would put '.1Z' first; chronological comparison does not."""
    fractional = _instant(20, fraction=".1")
    fills = (_fill(20, quantity="10", price="100", filled_at=fractional),)

    assert fractional.value < _instant(20).value
    assert fractional.compare(_instant(20)) > 0
    assert _build(fills, as_of=_instant(20)).positions == ()


def test_sub_second_fill_is_included_at_its_own_instant() -> None:
    fractional = _instant(20, fraction=".1")
    fills = (_fill(20, quantity="10", price="100", filled_at=fractional),)

    assert _build(fills, as_of=fractional).position_count == 1


def test_an_earlier_as_of_rebuilds_an_earlier_snapshot() -> None:
    fills = (
        _fill(20, quantity="10", price="100"),
        _fill(21, quantity="10", price="200"),
        _fill(22, OrderSide.SELL, quantity="20", price="300"),
    )

    assert _build(fills, as_of=_instant(20)).get_position(_AAPL).quantity == Quantity("10")
    assert _build(fills, as_of=_instant(21)).get_position(_AAPL).quantity == Quantity("20")
    assert _build(fills, as_of=_instant(22)).positions == ()


# ---------------------------------------------------------------------------
# History coherence
# ---------------------------------------------------------------------------


def test_fills_from_another_portfolio_are_rejected() -> None:
    fills = (
        _fill(20, quantity="10", price="100"),
        _fill(21, quantity="10", price="100", portfolio=_OTHER_PORTFOLIO),
    )

    with pytest.raises(InvalidPaperFillHistoryError, match="must all belong to the requested"):
        _build(fills)


def test_a_history_for_the_wrong_portfolio_is_rejected_entirely() -> None:
    fills = (_fill(20, quantity="10", price="100", portfolio=_OTHER_PORTFOLIO),)

    with pytest.raises(InvalidPaperFillHistoryError, match="must all belong to the requested"):
        _build(fills)


def test_duplicate_fill_identity_is_rejected() -> None:
    fills = (
        _fill(20, quantity="10", price="100", fill_id="f-1", order_id="o-1"),
        _fill(21, quantity="10", price="100", fill_id="f-1", order_id="o-2"),
    )

    with pytest.raises(InvalidPaperFillHistoryError, match="duplicate fill identities"):
        _build(fills)


def test_duplicate_order_identity_is_rejected() -> None:
    """The current model allows exactly one fill per order."""
    fills = (
        _fill(20, quantity="10", price="100", fill_id="f-1", order_id="o-1"),
        _fill(21, quantity="10", price="100", fill_id="f-2", order_id="o-1"),
    )

    with pytest.raises(InvalidPaperFillHistoryError, match="two fills for one order"):
        _build(fills)


def test_unordered_fills_are_rejected() -> None:
    fills = (
        _fill(22, quantity="10", price="100"),
        _fill(20, quantity="10", price="100"),
    )

    with pytest.raises(InvalidPaperFillHistoryError, match="ordered by fill instant"):
        _build(fills)


def test_the_use_case_never_silently_reorders() -> None:
    unordered = (
        _fill(22, quantity="10", price="100"),
        _fill(20, quantity="10", price="100"),
    )

    with pytest.raises(InvalidPaperFillHistoryError):
        _build(unordered)

    ordered = tuple(reversed(unordered))
    assert _build(ordered).get_position(_AAPL).quantity == Quantity("20")


def test_equal_instant_fills_must_be_ordered_by_fill_identity() -> None:
    instant = _instant(20)
    good = (
        _fill(20, quantity="1", price="100", fill_id="a", order_id="o-a", filled_at=instant),
        _fill(20, quantity="1", price="100", fill_id="b", order_id="o-b", filled_at=instant),
    )
    bad = tuple(reversed(good))

    assert _build(good).get_position(_AAPL).quantity == Quantity("2")
    with pytest.raises(InvalidPaperFillHistoryError, match="then fill identity"):
        _build(bad)


def test_sub_second_ordering_is_chronological_not_textual() -> None:
    whole = _instant(20)
    fractional = _instant(20, fraction=".1")
    fills = (
        _fill(20, quantity="1", price="100", fill_id="f-1", order_id="o-1", filled_at=whole),
        _fill(20, quantity="1", price="100", fill_id="f-2", order_id="o-2", filled_at=fractional),
    )

    assert _build(fills).get_position(_AAPL).quantity == Quantity("2")
    with pytest.raises(InvalidPaperFillHistoryError, match="ordered by fill instant"):
        _build(tuple(reversed(fills)))


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_execute_validates_input_types() -> None:
    use_case = BuildPaperPortfolioUseCase()

    with pytest.raises(TypeError, match="portfolio identity cannot be None"):
        use_case.execute(None, (), _AS_OF)
    with pytest.raises(TypeError, match="portfolio identity must be a PaperPortfolioIdentity"):
        use_case.execute("paper-1", (), _AS_OF)
    with pytest.raises(TypeError, match="fills cannot be None"):
        use_case.execute(_PORTFOLIO, None, _AS_OF)
    with pytest.raises(TypeError, match="fills must be a tuple"):
        use_case.execute(_PORTFOLIO, [_fill(20)], _AS_OF)
    with pytest.raises(TypeError, match="fills must contain PaperFill values"):
        use_case.execute(_PORTFOLIO, ("fill",), _AS_OF)
    with pytest.raises(TypeError, match="as-of instant cannot be None"):
        use_case.execute(_PORTFOLIO, (), None)
    with pytest.raises(TypeError, match="as-of instant must be a PointInTime"):
        use_case.execute(_PORTFOLIO, (), "2026-03-01T16:00:00Z")


def test_history_error_is_not_a_historical_data_error() -> None:
    """Fill history has its own failure type, distinct from market-data errors."""
    from northstar_application.application_services import HistoricalDataContractViolationError

    assert not issubclass(InvalidPaperFillHistoryError, HistoricalDataContractViolationError)
    assert issubclass(InvalidPaperFillHistoryError, ValueError)


# ---------------------------------------------------------------------------
# Decimal determinism
# ---------------------------------------------------------------------------


def _recurring_fills() -> tuple[PaperFill, ...]:
    """Three buys whose weighted average recurs: 1000/3."""
    return (
        _fill(20, quantity="1", price="100"),
        _fill(21, quantity="1", price="200"),
        _fill(22, quantity="1", price="700"),
    )


def test_weighted_average_of_a_recurring_quotient_is_computed_at_full_precision() -> None:
    position = _build(_recurring_fills()).get_position(_AAPL)

    assert position.quantity == Quantity("3")
    assert position.average_price.amount == Decimal("333.3333333333333333333333333")


@pytest.mark.parametrize("precision", [6, 28, 50])
def test_fold_is_identical_under_any_ambient_precision(precision: int) -> None:
    fills = _recurring_fills()
    expected = _build(fills)

    with localcontext() as context:
        context.prec = precision

        assert _build(fills) == expected


def test_fold_agrees_across_every_ambient_precision() -> None:
    fills = _recurring_fills()

    results = []
    for precision in (6, 28, 50):
        with localcontext() as context:
            context.prec = precision
            results.append(_build(fills))

    assert results[0] == results[1] == results[2]


def test_partial_sell_quantity_is_also_precision_independent() -> None:
    fills = (
        _fill(20, quantity="1000000.0000000001", price="100"),
        _fill(21, OrderSide.SELL, quantity="0.0000000001", price="100"),
    )
    expected = _build(fills)

    with localcontext() as context:
        context.prec = 6

        assert _build(fills) == expected
    assert expected.get_position(_AAPL).quantity == Quantity("1000000")


def test_fold_does_not_modify_the_caller_decimal_context() -> None:
    caller_context = getcontext()
    precision_before = caller_context.prec
    rounding_before = caller_context.rounding

    _build(_recurring_fills())

    assert getcontext() is caller_context
    assert getcontext().prec == precision_before
    assert getcontext().rounding == rounding_before


def test_fold_restores_a_customised_caller_context() -> None:
    expected = _build(_recurring_fills())

    with localcontext() as context:
        context.prec = 6
        context.rounding = ROUND_DOWN

        assert _build(_recurring_fills()) == expected
        assert getcontext().prec == 6
        assert getcontext().rounding == ROUND_DOWN


def test_holdings_remain_decimal_backed() -> None:
    position = _build(_recurring_fills()).get_position(_AAPL)

    assert isinstance(position.quantity.value, Decimal)
    assert isinstance(position.average_price.amount, Decimal)


# ---------------------------------------------------------------------------
# Determinism and scope
# ---------------------------------------------------------------------------


def test_repeated_execution_returns_an_equal_portfolio() -> None:
    fills = (
        _fill(20, quantity="10", price="100"),
        _fill(21, quantity="5", price="300", listing=_MSFT),
        _fill(22, OrderSide.SELL, quantity="4", price="150"),
    )
    use_case = BuildPaperPortfolioUseCase()

    assert use_case.execute(_PORTFOLIO, fills, _AS_OF) == use_case.execute(
        _PORTFOLIO, fills, _AS_OF
    )


def test_separate_use_case_instances_agree() -> None:
    fills = _recurring_fills()

    assert BuildPaperPortfolioUseCase().execute(
        _PORTFOLIO, fills, _AS_OF
    ) == BuildPaperPortfolioUseCase().execute(_PORTFOLIO, fills, _AS_OF)


def test_the_use_case_needs_no_dependencies() -> None:
    assert not vars(BuildPaperPortfolioUseCase())


def test_the_snapshot_carries_the_requested_identity_and_instant() -> None:
    portfolio = _build((_fill(20),), as_of=_instant(25))

    assert portfolio.identity == _PORTFOLIO
    assert portfolio.as_of == _instant(25)


def test_the_fold_produces_no_cash_or_profit_and_loss() -> None:
    portfolio = _build((_fill(20),))

    for absent in ("cash", "balance", "realised_pnl", "unrealised_pnl", "commission"):
        assert not hasattr(portfolio, absent)
