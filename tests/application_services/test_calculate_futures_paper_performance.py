"""Tests for the read-only futures paper performance analysis (INDIA-8H-A).

Facts live in the snapshot tests' in-memory reference ports. Most scenarios
plant exact orders, fills and bars so every expected amount is hand-checkable
(ES: 50 USD per quote point); one acceptance scenario lets the production
paper session create the facts. Every reconciliation compares against the
existing valuation, never against a restatement of it.
"""

from __future__ import annotations

import ast
from datetime import timedelta
from decimal import ROUND_HALF_EVEN, Context, Decimal, localcontext
from pathlib import Path

import pytest
from northstar_core.derivatives import QuoteValue
from northstar_core.foundation.value_objects import Money
from northstar_core.paper_trading import OrderSide
from test_get_futures_paper_trading_snapshot import (
    _ALPHA,
    _BETA,
    _ES_DEC,
    _ES_MAR,
    _EUR,
    _FESX_DEC,
    _NIFTY_OCT,
    _NIFTY_OCT_ECONOMICS,
    _PORTFOLIO,
    _USD,
    Economics,
    Fills,
    Forward,
    Market,
    Orders,
    World,
    _at,
    _bar,
    _daily,
    _order,
    _phase_a,
    _plant,
    _record,
    _trade,
)

from northstar_application.application_services import (
    BuildFuturesPaperTradingValuationUseCase,
    CalculateFuturesPaperPerformanceUseCase,
    FuturesContractEconomicsNotFoundError,
    FuturesPaperPerformance,
    FuturesPaperProfitFactorUnavailableReason,
    FuturesPaperTradeDirection,
)
from northstar_application.application_services import (
    calculate_futures_paper_performance as module,
)

_LONG, _SHORT = FuturesPaperTradeDirection.LONG, FuturesPaperTradeDirection.SHORT
_BUY, _SELL = OrderSide.BUY, OrderSide.SELL
_DAY = timedelta(days=1)


def _usd(amount: str) -> Money:
    return Money(Decimal(amount), _USD)


# ---------------------------------------------------------------------------
# Building facts
# ---------------------------------------------------------------------------


def _world(*closes: str, contract=_ES_DEC) -> World:
    """Bars at sessions 1..n with the given closes."""
    return World(*(_bar(n, close, contract=contract) for n, close in enumerate(closes, 1)))


_count = 0


def _fill(
    world: World,
    side: OrderSide,
    contracts: int,
    quote: str | None,
    filled: int,
    *,
    contract=_ES_DEC,
    decided: int | None = None,
) -> None:
    """Plant one order decided at the session before ``filled`` and its fill there."""
    global _count
    _count += 1
    order = _order(f"order-{_count:04d}", contract, side, contracts, _at(decided or filled - 1))
    _trade(world, order, quote, _at(filled))


def _perf(
    world: World, through: int | object, *, contract=_ES_DEC, strategy=_ALPHA
) -> FuturesPaperPerformance:
    cutoff = _at(through) if isinstance(through, int) else through
    return CalculateFuturesPaperPerformanceUseCase(
        Forward(world), Orders(world), Fills(world), Market(world), Economics(world)
    ).execute(contract, strategy, _PORTFOLIO, cutoff)


def _valued(world: World, through: int, contract=_ES_DEC):
    valuation = BuildFuturesPaperTradingValuationUseCase(
        Orders(world), Fills(world), Market(world), Economics(world)
    ).execute(_PORTFOLIO, _ALPHA, _at(through))
    rows = [row for row in valuation.contracts if row.contract == contract]
    return rows[0] if rows else None


def _reconciles(performance: FuturesPaperPerformance, world: World, through: int) -> None:
    """Completed trades (+ open realized) equal the valuation's realized P&L exactly."""
    row = _valued(world, through, performance.contract)
    expected = row.realized_pnl.amount if row else Decimal(0)
    with localcontext() as context:
        context.prec = 28
        total = sum((t.realized_pnl.amount for t in performance.completed_trades), Decimal(0))
        if performance.open_exposure is not None:
            total += performance.open_exposure.realized_pnl.amount
    assert total == expected
    assert performance.realized_pnl.amount == expected


# ---------------------------------------------------------------------------
# A-C. Empty, HOLD-only and open-only histories
# ---------------------------------------------------------------------------


def test_no_decisions_and_no_fills() -> None:
    world = _world("100", "101", "99")

    result = _perf(world, 3)

    assert (result.decisions.decision_count, result.execution.fill_count) == (0, 0)
    assert result.completed_trades == () and result.open_exposure is None
    assert result.settlement_currency == _USD
    assert [p.total_pnl for p in result.equity_curve] == [_usd("0")] * 3
    assert result.max_drawdown.amount == _usd("0")
    assert result.max_drawdown.peak_instant == result.max_drawdown.trough_instant == _at(1)
    assert result.realized_pnl == _usd("0")


def test_hold_only_history() -> None:
    world = _world("100", "101", "102")
    for session in (1, 2, 3):
        _plant(_record(_at(session), "HOLD"), world)

    result = _perf(world, 3)

    assert result.decisions.decision_count == result.decisions.hold_count == 3
    assert (result.decisions.buy_count, result.decisions.sell_count) == (0, 0)
    assert result.execution.order_count == 0 and result.completed_trades == ()
    assert result.statistics.win_rate is None


def test_an_open_long_is_exposure_not_a_trade() -> None:
    world = _world("100", "100", "104")
    _fill(world, _BUY, 1, "100", 2)

    result = _perf(world, 3)

    assert result.completed_trades == ()
    exposure = result.open_exposure
    assert (exposure.direction, exposure.opened_at) == (_LONG, _at(2))
    assert exposure.position.net_contracts == 1
    assert exposure.position.average_entry == QuoteValue(Decimal("100"))
    assert (exposure.mark_quote, exposure.mark_instant) == (QuoteValue(Decimal("104")), _at(3))
    assert exposure.unrealized_pnl == _usd("200") and exposure.realized_pnl == _usd("0")
    stats = result.statistics
    assert (stats.completed_count, stats.winning_count, stats.losing_count) == (0, 0, 0)
    assert stats.win_rate is None and stats.average_trade_pnl is None
    assert stats.profit_factor is None
    assert (
        stats.profit_factor_unavailable_reason
        is FuturesPaperProfitFactorUnavailableReason.NO_COMPLETED_TRADES
    )
    assert stats.minimum_holding_duration is None and stats.median_holding_duration is None
    _reconciles(result, world, 3)


# ---------------------------------------------------------------------------
# D-J. Completed trades
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("opening", "closing", "quote", "direction", "contracts", "pnl"),
    [
        (_BUY, _SELL, "110", _LONG, 1, "500"),  # D profitable long
        (_BUY, _SELL, "95", _LONG, 1, "-250"),  # E losing long
        (_SELL, _BUY, "90", _SHORT, 2, "1000"),  # F profitable short
        (_BUY, _SELL, "100", _LONG, 1, "0"),  # J breakeven
    ],
    ids=["profitable-long", "losing-long", "profitable-short", "breakeven"],
)
def test_one_completed_trade(opening, closing, quote, direction, contracts, pnl) -> None:
    world = _world("100", "100", "100", "100")
    _fill(world, opening, contracts, "100", 2)
    _fill(world, closing, contracts, quote, 4)

    result = _perf(world, 4)

    (trade,) = result.completed_trades
    assert trade.direction is direction and trade.contracts == contracts
    assert (trade.opened_at, trade.closed_at) == (_at(2), _at(4))
    assert trade.closing_average_entry == QuoteValue(Decimal("100"))
    assert trade.average_exit == QuoteValue(Decimal(quote))
    assert trade.realized_pnl == _usd(pnl)
    assert trade.holding_duration == 2 * _DAY
    assert result.open_exposure is None and result.execution.net_contracts == 0
    _reconciles(result, world, 4)


def test_a_long_to_short_reversal_splits_one_fill() -> None:
    world = _world("100", "102", "104", "108", "105")
    _fill(world, _BUY, 1, "100", 2)
    _fill(world, _SELL, 3, "110", 4)

    result = _perf(world, 5)

    (trade,) = result.completed_trades
    assert (trade.direction, trade.contracts, trade.realized_pnl) == (_LONG, 1, _usd("500"))
    assert trade.closed_at == _at(4)
    exposure = result.open_exposure
    assert (exposure.direction, exposure.opened_at) == (_SHORT, _at(4))
    assert exposure.position.net_contracts == -2
    assert exposure.position.average_entry == QuoteValue(Decimal("110"))
    assert exposure.unrealized_pnl == _usd("500")  # (105 - 110) * -2 * 50
    assert result.execution.fill_count == 2  # nothing fabricated
    _reconciles(result, world, 5)


def test_a_short_to_long_reversal_splits_one_fill() -> None:
    world = _world("110", "110", "110", "100", "101")
    _fill(world, _SELL, 2, "110", 2)
    _fill(world, _BUY, 3, "100", 4)

    result = _perf(world, 5)

    (trade,) = result.completed_trades
    assert (trade.direction, trade.contracts, trade.realized_pnl) == (_SHORT, 2, _usd("1000"))
    exposure = result.open_exposure
    assert (exposure.direction, exposure.position.net_contracts) == (_LONG, 1)
    assert exposure.unrealized_pnl == _usd("50")
    _reconciles(result, world, 5)


def test_adds_and_partial_closes_stay_one_episode() -> None:
    world = _world(*["100"] * 7)
    _fill(world, _BUY, 1, "100", 2)
    _fill(world, _BUY, 1, "104", 3)  # average 102
    _fill(world, _SELL, 1, "110", 4)  # partial close: +8 points
    _fill(world, _SELL, 1, "106", 5)  # completes: +4 points

    result = _perf(world, 6)

    (trade,) = result.completed_trades
    assert (trade.opened_at, trade.closed_at, trade.contracts) == (_at(2), _at(5), 2)
    assert trade.closing_average_entry == QuoteValue(Decimal("102"))
    assert trade.average_exit == QuoteValue(Decimal("108"))
    assert trade.realized_pnl == _usd("600")
    _reconciles(result, world, 6)


def test_a_partial_close_of_a_still_open_episode_is_open_realized() -> None:
    world = _world("100", "100", "100", "103")
    _fill(world, _BUY, 2, "100", 2)
    _fill(world, _SELL, 1, "110", 3)

    result = _perf(world, 4)

    assert result.completed_trades == ()
    assert result.open_exposure.realized_pnl == _usd("500")
    assert result.open_exposure.unrealized_pnl == _usd("150")
    assert result.statistics.completed_count == 0  # never a win while open
    _reconciles(result, world, 4)


def _three_trades() -> World:
    """A +500 long (2 days), a -250 short (3 days, over a weekend), a breakeven long (2 days)."""
    world = _world(*["100"] * 10)
    _fill(world, _BUY, 1, "100", 2)
    _fill(world, _SELL, 1, "110", 4)
    _fill(world, _SELL, 1, "100", 5)
    _fill(world, _BUY, 1, "105", 6)
    _fill(world, _BUY, 1, "100", 7)
    _fill(world, _SELL, 1, "100", 9)
    return world


def test_multiple_completed_episodes() -> None:
    world = _three_trades()

    result = _perf(world, 10)

    assert [(t.direction, t.realized_pnl) for t in result.completed_trades] == [
        (_LONG, _usd("500")),
        (_SHORT, _usd("-250")),
        (_LONG, _usd("0")),
    ]
    _reconciles(result, world, 10)


# ---------------------------------------------------------------------------
# M-N. Decisions and execution
# ---------------------------------------------------------------------------


def test_decisions_come_from_frozen_records_only() -> None:
    world = _world(*["100"] * 6)
    for session, action in ((1, "BUY"), (2, "HOLD"), (3, "SELL"), (4, "HOLD"), (6, "BUY")):
        _plant(_record(_at(session), action), world)
    _plant(_record(_at(2), "BUY", strategy=_BETA), world)  # another strategy
    _plant(_record(_at(3), "SELL", contract=_ES_MAR), world)  # another contract
    _fill(world, _BUY, 1, "100", 2)  # orders never become decisions

    result = _perf(world, 5)  # session 6 is after the cutoff

    decisions = result.decisions
    assert (decisions.decision_count, decisions.buy_count) == (4, 1)
    assert (decisions.sell_count, decisions.hold_count) == (1, 2)


def test_contracts_bought_sold_and_turnover() -> None:
    world = _world(*["100"] * 8)
    _fill(world, _BUY, 2, "100", 2)
    _fill(world, _SELL, 5, "101", 4)
    _fill(world, _BUY, 1, "99", 6)
    _fill(world, _BUY, 4, None, 7)  # pending: an order, no fill
    _fill(world, _SELL, 9, "100", 3, contract=_ES_MAR)  # another contract

    execution = _perf(world, 8).execution

    assert (execution.order_count, execution.fill_count) == (4, 3)
    assert (execution.contracts_bought, execution.contracts_sold) == (3, 5)
    assert execution.net_contracts == -2
    assert execution.turnover_contracts == 8


# ---------------------------------------------------------------------------
# O-P. Reconciliation with the existing valuation
# ---------------------------------------------------------------------------


def test_open_exposure_reconciles_with_the_valuation() -> None:
    world = _world("100", "102", "104", "108", "105")
    _fill(world, _BUY, 1, "100", 2)
    _fill(world, _SELL, 3, "110", 4)

    exposure = _perf(world, 5).open_exposure
    row = _valued(world, 5)

    assert exposure.position == row.position
    assert (exposure.mark_quote, exposure.mark_instant) == (row.mark_quote, row.mark_instant)
    assert exposure.unrealized_pnl == row.unrealized_pnl


def test_an_unmarked_open_position_has_no_unrealized_value() -> None:
    world = World()  # no stored bar at all: nothing to mark with
    _fill(world, _BUY, 1, "100", 2)

    result = _perf(world, 3)

    exposure = result.open_exposure
    assert exposure.position.net_contracts == 1
    assert (exposure.mark_quote, exposure.mark_instant, exposure.unrealized_pnl) == (
        None,
        None,
        None,
    )
    assert result.equity_curve == () and result.max_drawdown is None


def test_a_stale_mark_is_the_latest_close_by_the_cutoff() -> None:
    world = World(_bar(1, "103"))
    _fill(world, _BUY, 1, "100", 2)

    exposure = _perf(world, 3).open_exposure

    assert exposure.mark_instant == _at(1) and exposure.unrealized_pnl == _usd("150")
    assert exposure.unrealized_pnl == _valued(world, 3).unrealized_pnl


# ---------------------------------------------------------------------------
# Q-S. Equity curve, lookahead and drawdown
# ---------------------------------------------------------------------------


def test_the_equity_curve_has_exact_values() -> None:
    world = _world("100", "102", "104", "108", "105")
    _fill(world, _BUY, 1, "100", 2)
    _fill(world, _SELL, 3, "110", 4)

    curve = _perf(world, 5).equity_curve

    assert [p.instant for p in curve] == [_at(n) for n in range(1, 6)]
    assert [p.net_contracts for p in curve] == [0, 1, 1, -2, -2]
    assert [(p.realized_pnl, p.unrealized_pnl, p.total_pnl) for p in curve] == [
        (_usd("0"), _usd("0"), _usd("0")),
        (_usd("0"), _usd("100"), _usd("100")),
        (_usd("0"), _usd("200"), _usd("200")),
        (_usd("500"), _usd("200"), _usd("700")),
        (_usd("500"), _usd("500"), _usd("1000")),
    ]


def test_a_cutoff_excludes_later_fills_and_poison_bars() -> None:
    def facts(*, later: bool) -> World:
        world = _world("100", "102", "104")
        _fill(world, _BUY, 1, "100", 2)
        if later:
            world.bars.append(_bar(4, "1"))  # poison close
            world.bars.append(_bar(5, "99999"))
            _fill(world, _SELL, 1, "1", 5, decided=4)
            _plant(_record(_at(4), "SELL"), world)
        return world

    clean, poisoned = _perf(facts(later=False), 3), _perf(facts(later=True), 3)

    assert poisoned == clean
    assert clean.equity_curve[-1].instant == _at(3)
    assert clean.open_exposure.mark_quote == QuoteValue(Decimal("104"))


def test_the_same_facts_give_an_equal_result() -> None:
    world = _three_trades()

    assert _perf(world, 10) == _perf(world, 10)


def test_the_absolute_max_drawdown() -> None:
    world = _world("100", "102", "106", "101", "104", "98", "103")
    _fill(world, _BUY, 1, "100", 2)

    result = _perf(world, 7)

    totals = [p.total_pnl for p in result.equity_curve]
    assert totals == [_usd(v) for v in ("0", "100", "300", "50", "200", "-100", "150")]
    drawdown = result.max_drawdown
    assert (drawdown.peak_instant, drawdown.peak_pnl) == (_at(3), _usd("300"))
    assert (drawdown.trough_instant, drawdown.trough_pnl) == (_at(6), _usd("-100"))
    assert drawdown.amount == _usd("400")


def test_a_rising_curve_has_zero_drawdown() -> None:
    world = _world("100", "101", "102")
    _fill(world, _BUY, 1, "100", 2)

    drawdown = _perf(world, 3).max_drawdown

    assert drawdown.amount == _usd("0")
    assert drawdown.peak_instant == drawdown.trough_instant == _at(1)


# ---------------------------------------------------------------------------
# T-Y. Statistics
# ---------------------------------------------------------------------------


def test_win_loss_breakeven_average_and_profit_factor() -> None:
    stats = _perf(_three_trades(), 10).statistics

    assert (stats.completed_count, stats.winning_count) == (3, 1)
    assert (stats.losing_count, stats.breakeven_count) == (1, 1)
    assert stats.win_rate == Decimal("0.3333333333333333333333333333")
    assert stats.total_realized_pnl == _usd("250")
    assert stats.average_trade_pnl == _usd("83.33333333333333333333333333")
    assert (stats.gross_profit, stats.gross_loss) == (_usd("500"), _usd("250"))
    assert stats.profit_factor == Decimal("2")
    assert stats.profit_factor_unavailable_reason is None


def test_profit_factor_without_losses_or_wins() -> None:
    world = _world(*["100"] * 6)
    _fill(world, _BUY, 1, "100", 2)
    _fill(world, _SELL, 1, "110", 3)
    _fill(world, _BUY, 1, "100", 4)
    _fill(world, _SELL, 1, "100", 5)
    no_loss = _perf(world, 6).statistics

    assert no_loss.profit_factor is None
    assert (
        no_loss.profit_factor_unavailable_reason
        is FuturesPaperProfitFactorUnavailableReason.NO_LOSING_TRADES
    )

    losses = _world(*["100"] * 4)
    _fill(losses, _BUY, 1, "100", 2)
    _fill(losses, _SELL, 1, "90", 3)
    only_losses = _perf(losses, 4).statistics
    assert only_losses.profit_factor == Decimal("0")
    assert only_losses.win_rate == Decimal("0")


def test_holding_duration_aggregation() -> None:
    stats = _perf(_three_trades(), 10).statistics  # 2, 3 and 2 days

    assert stats.minimum_holding_duration == 2 * _DAY
    assert stats.maximum_holding_duration == 3 * _DAY
    assert stats.average_holding_duration == timedelta(days=7) / 3
    assert stats.median_holding_duration == 2 * _DAY

    world = _world(*["100"] * 8)
    _fill(world, _BUY, 1, "100", 2)
    _fill(world, _SELL, 1, "100", 3)  # 1 day
    _fill(world, _BUY, 1, "100", 4)
    _fill(world, _SELL, 1, "100", 7)  # Jun 4 -> Jun 9: 5 days
    even = _perf(world, 8).statistics
    assert even.median_holding_duration == 3 * _DAY


# ---------------------------------------------------------------------------
# Canonical realized P&L versus the per-trade decomposition
# ---------------------------------------------------------------------------

_PNL = Context(prec=28, rounding=ROUND_HALF_EVEN)


def test_average_entry_rounding_leaves_a_trailing_digit_aggregation_difference() -> None:
    """Two short episodes, each built from adds of 1 and 2 at different prices.

    Each average entry repeats (111.66... and 110.66...), so each trade's 28-digit
    P&L is rounded on its own, while the canonical fold sums all points first and
    rounds once. The difference is left visible: nothing is reallocated.
    """
    world = World(
        *(_bar(n, "110", contract=_NIFTY_OCT) for n in range(1, 8)),
        economics=(_NIFTY_OCT_ECONOMICS,),  # 65 per quote point
    )
    for side, contracts, quote, session in (
        (_SELL, 1, "115", 2),
        (_SELL, 2, "110", 3),
        (_BUY, 3, "104", 4),
        (_SELL, 1, "110", 5),
        (_SELL, 2, "111", 6),
        (_BUY, 3, "124", 7),
    ):
        _fill(world, side, contracts, quote, session, contract=_NIFTY_OCT)

    result = _perf(world, 7, contract=_NIFTY_OCT)
    first, second = result.completed_trades

    # Each trade from its own facts, under the fold's own context and formula.
    with localcontext(_PNL):
        entry_one = (Decimal(115) + 2 * Decimal(110)) / 3
        entry_two = (Decimal(110) + 2 * Decimal(111)) / 3
        expected_one = (Decimal(104) - entry_one) * -1 * 3 * 65
        expected_two = (Decimal(124) - entry_two) * -1 * 3 * 65
    assert first.closing_average_entry.value == entry_one
    assert second.closing_average_entry.value == entry_two
    assert first.realized_pnl.amount == expected_one == Decimal("1495.000000000000000000000006")
    assert second.realized_pnl.amount == expected_two == Decimal("-2599.999999999999999999999994")

    # The canonical portfolio amount is the realized fold's, unchanged.
    canonical = _valued(world, 7, _NIFTY_OCT).realized_pnl.amount
    assert canonical == Decimal("-1104.999999999999999999999987")
    assert result.realized_pnl.amount == canonical

    with localcontext(_PNL):
        trades_total = first.realized_pnl.amount + second.realized_pnl.amount
        difference = trades_total - canonical
    assert trades_total == Decimal("-1104.999999999999999999999988")
    # Non-zero, and exactly one unit in the canonical amount's 28th significant digit.
    last_place = Decimal(1).scaleb(canonical.adjusted() - 27)
    assert difference != 0
    assert abs(difference) == last_place == Decimal("1E-24")
    assert not [name for name in dir(module) if "toleran" in name.lower()]


def test_completed_trades_exclude_what_an_open_episode_already_realized() -> None:
    world = _world(*["100"] * 7)
    _fill(world, _BUY, 1, "100", 2)
    _fill(world, _SELL, 1, "110", 3)  # completed: +500
    _fill(world, _BUY, 2, "100", 4)
    _fill(world, _SELL, 1, "120", 5)  # partial close, episode stays open: +1000

    result = _perf(world, 6)

    (trade,) = result.completed_trades
    assert trade.realized_pnl == _usd("500")
    assert result.open_exposure.position.net_contracts == 1
    assert result.open_exposure.realized_pnl == _usd("1000")
    assert result.realized_pnl == _usd("1500") == _valued(world, 6).realized_pnl
    # Completed trades alone do not, and are not expected to, reconcile.
    assert trade.realized_pnl != result.realized_pnl
    assert result.statistics.total_realized_pnl == _usd("500")
    _reconciles(result, world, 6)


# ---------------------------------------------------------------------------
# Z. Currency and economics safety
# ---------------------------------------------------------------------------


def test_each_contract_is_analysed_in_its_own_currency_and_point_value() -> None:
    world = World(
        *(_bar(n, "100") for n in range(1, 5)),
        *(_bar(n, "100", contract=_FESX_DEC) for n in range(1, 5)),
    )
    _fill(world, _BUY, 1, "100", 2)
    _fill(world, _SELL, 1, "110", 3)
    _fill(world, _BUY, 1, "100", 2, contract=_FESX_DEC, decided=1)
    _fill(world, _SELL, 1, "110", 4, contract=_FESX_DEC, decided=3)

    es, fesx = _perf(world, 4), _perf(world, 4, contract=_FESX_DEC)

    assert es.settlement_currency == _USD and fesx.settlement_currency == _EUR
    assert es.completed_trades[0].realized_pnl == _usd("500")
    assert fesx.completed_trades[0].realized_pnl == Money(Decimal("100"), _EUR)
    assert es.execution.fill_count == fesx.execution.fill_count == 2
    _reconciles(es, world, 4)
    _reconciles(fesx, world, 4)


def test_a_contract_without_economics_fails_clearly() -> None:
    world = World(_bar(1, "100"), economics=())

    with pytest.raises(FuturesContractEconomicsNotFoundError) as raised:
        _perf(world, 1)
    assert raised.value.contract == _ES_DEC


# ---------------------------------------------------------------------------
# Acceptance over production paper facts
# ---------------------------------------------------------------------------


def test_production_paper_facts_reconcile_with_the_valuation() -> None:
    world = _daily(World(*_phase_a()), 29)

    result = _perf(world, 29)

    assert result.decisions.decision_count == len(world.records)
    assert result.execution.fill_count == len(world.fills)
    assert result.execution.order_count == len(world.orders)
    assert result.completed_trades  # the scripted path closes at least one trade
    _reconciles(result, world, 29)
    row = _valued(world, 29)
    if row.position is not None:
        assert result.open_exposure.unrealized_pnl == row.unrealized_pnl
    final = result.equity_curve[-1]
    with localcontext() as context:
        context.prec = 28
        valuation_total = row.realized_pnl.amount + (row.unrealized_pnl or _usd("0")).amount
    assert final.total_pnl.amount == valuation_total
    # Every earlier cutoff agrees with its own valuation too.
    for session in range(25, 29):
        earlier = _perf(world, session)
        _reconciles(earlier, world, session)


# ---------------------------------------------------------------------------
# Boundary
# ---------------------------------------------------------------------------


def test_the_analysis_is_generic_and_clock_free() -> None:
    source = Path(module.__file__).read_text(encoding="utf-8")
    imported = {
        node.module
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert not any(name.startswith("northstar_infrastructure") for name in imported)
    for forbidden in ("NSE", "NIFTY", "Upstox", "CME", ".now(", "utcnow", "time.time"):
        assert forbidden not in source
