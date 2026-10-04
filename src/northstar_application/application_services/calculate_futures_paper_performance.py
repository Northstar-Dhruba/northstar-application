"""Application read-only performance analysis of one futures paper strategy.

The analysis covers one exact FuturesContract, one paper portfolio and its one
strategy at one explicit cutoff, ``available_through``. It is derived entirely
from persisted facts -- frozen forward decisions, paper orders and fills, stored
daily bars and the contract's own economics -- and persists nothing, reads no
clock and contacts no provider. Only facts observable at or before the cutoff
participate: decisions and orders decided by it, fills filled by it and bars
completed by it. The same facts and cutoff always give an equal result.

It restates no economics. Positions come from the portfolio fold's single
transition rule, realized P&L from CalculateFuturesRealizedPnlUseCase and marks
from ValueFuturesPaperPortfolioUseCase, all under the futures P&L context
(precision 28, ROUND_HALF_EVEN). Amounts are gross: no fees, commission,
slippage, financing or taxes. There is one settlement currency, the analysed
contract's own; nothing is ever summed across currencies.

Decisions
---------
Decisions are the strategy's frozen forward records for the contract, never
inferred from orders. Each is BUY, SELL or HOLD by its recorded action.

Execution
---------
Orders are those decided for the contract by the cutoff; fills those filled by
it. Turnover is in whole contracts only: contracts bought plus contracts sold.
There is no notional turnover.

Trades
------
A trade is one position episode in the contract, replayed fill by fill:

- it opens when a fill moves the position from flat to non-zero, or when a
  reversal fill opens the opposite position with the contracts left after
  closing the old one;
- adding in its direction and partially closing it keep it open;
- it completes when a fill takes the position to flat, or when a reversal fill
  closes it in full. A reversal is one persisted fill whose closing portion
  completes the old trade and whose remainder opens the new one at the same
  instant and quote; no fill is fabricated or split in storage.

A completed trade's realized P&L is every closing portion of the episode,
including partial closes, valued exactly as the realized fold values them:
``(fill quote - closing average entry) * direction * contracts closed`` quote
points, then times the contract's point value. A still-open episode is not a
trade: it is reported as open exposure, never as a win or a loss. Whatever it
has already realized through partial closes is reported with it, so

    sum(completed trade P&L) + open exposure realized P&L
        == the portfolio's realized P&L for the contract at the cutoff

and when the open episode has realized nothing -- always the case without
partial closes -- the completed trades alone reconcile. When an episode is
still open after a partial close, what it has realized is not completed-trade
P&L, so the completed trades alone are not expected to reconcile.

Canonical versus analytical P&L: the portfolio's realized P&L, from the
realized fold, is canonical for portfolio valuation. Per-trade realized P&L
is an analytical decomposition of it, each trade valued from its own facts
under the same 28-digit ROUND_HALF_EVEN context. The fold sums quote points
over the whole contract and converts once, while each trade converts its own
points, so when average-entry rounding occurs (adds at different prices
leaving a repeating average) finite Decimal precision can make the sum of
trades differ from the canonical amount in its trailing digits. That is an
expected aggregation effect, not a defect: no economic P&L is reassigned
between trades to force reconciliation, and no tolerance is applied.

Statistics
----------
Over completed trades only: winners have P&L > 0, losers < 0, breakeven == 0.
Win rate is winners / completed trades, a Decimal fraction in [0, 1]. The
average trade is total completed P&L / completed trades. Gross profit is the
sum of positive trade P&L and gross loss the absolute sum of negative trade
P&L; profit factor is gross profit / gross loss when gross loss is positive.
With no completed trade every ratio and average is None, never zero. With
completed trades but no loss the profit factor is None with reason
NO_LOSING_TRADES: no infinity is invented. Holding duration is elapsed time
between the opening and completing fill instants, not trading sessions.

Equity curve and drawdown
-------------------------
One point per stored daily bar of the contract completed by the cutoff: the
realized P&L of the fills visible at that bar's instant, the unrealized P&L of
the position then open marked at that bar's close, and their total. Before the
first fill every amount is zero. There is no account capital, so drawdown is
absolute, on the cumulative gross P&L curve:

    drawdown_t = running peak of total P&L - total P&L_t
    max drawdown = max over t of drawdown_t

reported as an amount with its peak and trough; never as a percentage.

Expiry governance is deliberately not analysed here: classifying decisions
against the pre-expiry window needs a venue calendar, which belongs to the
read-model composition, not to this generic analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import ROUND_HALF_EVEN, Context, Decimal, localcontext
from enum import StrEnum

from northstar_core.derivatives import QuoteValue
from northstar_core.foundation.value_objects import Currency, Money, PointInTime, Timeframe
from northstar_core.futures import FuturesContract, FuturesContractEconomics, FuturesOHLCVBar
from northstar_core.paper_trading import (
    FuturesPaperFill,
    FuturesPaperOrder,
    FuturesPosition,
    OrderSide,
    PaperPortfolioIdentity,
)
from northstar_core.strategy import StrategyIdentity

from northstar_application.application_services._futures_repository_output import (
    validate_futures_repository_bars,
)
from northstar_application.application_services.build_futures_paper_portfolio import (
    BuildFuturesPaperPortfolioUseCase,
    _transition,
)
from northstar_application.application_services.calculate_futures_realized_pnl import (
    CalculateFuturesRealizedPnlUseCase,
)
from northstar_application.application_services.futures_forward_research_record import (
    FuturesForwardResearchRecord,
)
from northstar_application.application_services.run_futures_paper_trading_decision import (
    _load_history,
    _validate_forward_output,
)
from northstar_application.application_services.value_futures_paper_portfolio import (
    ValueFuturesPaperPortfolioUseCase,
    _economics_for,
)
from northstar_application.ports import (
    FuturesContractEconomicsRepository,
    FuturesForwardResearchRecordQuery,
    FuturesForwardResearchRecordRepository,
    FuturesHistoricalMarketDataQuery,
    FuturesHistoricalMarketDataRepository,
    FuturesPaperFillRepository,
    FuturesPaperOrderRepository,
)

_DAILY = Timeframe("1d")
_PNL_CONTEXT = Context(prec=28, rounding=ROUND_HALF_EVEN)
_SUBJECT = "CalculateFuturesPaperPerformanceUseCase"
_ACTIONS = ("BUY", "SELL", "HOLD")


class FuturesPaperTradeDirection(StrEnum):
    """Which way a futures paper position episode was exposed."""

    LONG = "LONG"
    SHORT = "SHORT"


class FuturesPaperProfitFactorUnavailableReason(StrEnum):
    """Why a profit factor is undefined; it is never forced to zero or infinity."""

    NO_COMPLETED_TRADES = "NO_COMPLETED_TRADES"
    NO_LOSING_TRADES = "NO_LOSING_TRADES"


def _instant(value: PointInTime) -> datetime:
    return datetime.fromisoformat(value.value.replace("Z", "+00:00")).astimezone(UTC)


def _require(subject: str, name: str, value: object, expected: type) -> None:
    if not isinstance(value, expected):
        raise TypeError(f"{subject} {name} must be a {expected.__name__}.")


def _require_count(subject: str, name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{subject} {name} must be a non-negative integer.")


def _require_currency(subject: str, currency: Currency, *amounts: Money | None) -> None:
    for amount in amounts:
        if amount is not None and amount.currency != currency:
            raise ValueError(f"{subject} amounts must all be in the settlement currency.")


# ---------------------------------------------------------------------------
# Result values
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FuturesPaperDecisionDistribution:
    """How many frozen decisions the strategy took on the contract, by action."""

    decision_count: int
    buy_count: int
    sell_count: int
    hold_count: int

    def __post_init__(self) -> None:
        subject = "FuturesPaperDecisionDistribution"
        for name in ("decision_count", "buy_count", "sell_count", "hold_count"):
            _require_count(subject, name, getattr(self, name))
        if self.decision_count != self.buy_count + self.sell_count + self.hold_count:
            raise ValueError(f"{subject} decisions must be exactly BUY, SELL or HOLD.")


@dataclass(frozen=True, slots=True)
class FuturesPaperExecutionActivity:
    """Orders and fills on the contract, in whole contracts.

    ``net_contracts`` is the signed position at the cutoff, zero when flat.
    Turnover is ``contracts_bought + contracts_sold``; there is no notional.
    """

    order_count: int
    fill_count: int
    contracts_bought: int
    contracts_sold: int
    net_contracts: int

    def __post_init__(self) -> None:
        subject = "FuturesPaperExecutionActivity"
        for name in ("order_count", "fill_count", "contracts_bought", "contracts_sold"):
            _require_count(subject, name, getattr(self, name))
        if isinstance(self.net_contracts, bool) or not isinstance(self.net_contracts, int):
            raise TypeError(f"{subject} net contracts must be an integer.")
        if self.fill_count > self.order_count:
            raise ValueError(f"{subject} cannot have more fills than orders.")
        if self.net_contracts != self.contracts_bought - self.contracts_sold:
            raise ValueError(f"{subject} net contracts must be bought minus sold.")

    @property
    def turnover_contracts(self) -> int:
        """Return contracts bought plus contracts sold."""
        return self.contracts_bought + self.contracts_sold


@dataclass(frozen=True, slots=True)
class FuturesPaperCompletedTrade:
    """One completed position episode in the contract.

    ``contracts`` is every contract the episode closed, which equals every
    contract it opened. ``closing_average_entry`` is the fold's basis that the
    completing fill closed against; ``average_exit`` is the contract-weighted
    average quote of all the episode's closing portions.

    ``realized_pnl`` is analytical: valued from this episode's own facts, it
    is never adjusted to make trades sum to the canonical portfolio amount.
    """

    contract: FuturesContract
    direction: FuturesPaperTradeDirection
    opened_at: PointInTime
    closed_at: PointInTime
    contracts: int
    closing_average_entry: QuoteValue
    average_exit: QuoteValue
    realized_pnl: Money

    def __post_init__(self) -> None:
        subject = "FuturesPaperCompletedTrade"
        _require(subject, "contract", self.contract, FuturesContract)
        _require(subject, "direction", self.direction, FuturesPaperTradeDirection)
        _require(subject, "opened_at", self.opened_at, PointInTime)
        _require(subject, "closed_at", self.closed_at, PointInTime)
        _require(subject, "closing_average_entry", self.closing_average_entry, QuoteValue)
        _require(subject, "average_exit", self.average_exit, QuoteValue)
        _require(subject, "realized_pnl", self.realized_pnl, Money)
        _require_count(subject, "contracts", self.contracts)
        if self.contracts == 0:
            raise ValueError(f"{subject} must close at least one contract.")
        if self.closed_at.compare(self.opened_at) < 0:
            raise ValueError(f"{subject} cannot close before it opens.")

    @property
    def holding_duration(self) -> timedelta:
        """Return the elapsed time from the opening to the completing fill."""
        return _instant(self.closed_at) - _instant(self.opened_at)


@dataclass(frozen=True, slots=True)
class FuturesPaperOpenExposure:
    """The position episode still open in the contract at the cutoff.

    ``position``, the mark and ``unrealized_pnl`` are exactly what the paper
    valuation reports; the mark fields and ``unrealized_pnl`` are None
    together when no stored daily close was observable by the cutoff.
    ``realized_pnl`` is what partial closes of this episode already realized.
    """

    position: FuturesPosition
    direction: FuturesPaperTradeDirection
    opened_at: PointInTime
    realized_pnl: Money
    mark_quote: QuoteValue | None
    mark_instant: PointInTime | None
    unrealized_pnl: Money | None

    def __post_init__(self) -> None:
        subject = "FuturesPaperOpenExposure"
        _require(subject, "position", self.position, FuturesPosition)
        _require(subject, "direction", self.direction, FuturesPaperTradeDirection)
        _require(subject, "opened_at", self.opened_at, PointInTime)
        _require(subject, "realized_pnl", self.realized_pnl, Money)
        expected = (
            FuturesPaperTradeDirection.LONG
            if self.position.net_contracts > 0
            else FuturesPaperTradeDirection.SHORT
        )
        if self.direction is not expected:
            raise ValueError(f"{subject} direction must match the position's sign.")
        marks = (self.mark_quote, self.mark_instant, self.unrealized_pnl)
        if any(value is None for value in marks) and any(value is not None for value in marks):
            raise ValueError(f"{subject} mark fields must be present together or not at all.")
        _require_currency(subject, self.realized_pnl.currency, self.unrealized_pnl)


@dataclass(frozen=True, slots=True)
class FuturesPaperTradeStatistics:
    """Win/loss, average, profit-factor and holding statistics of completed trades.

    Every ratio, average and duration is None when there is no completed trade.
    ``profit_factor`` and ``profit_factor_unavailable_reason`` are exactly one
    present. ``win_rate`` is a fraction in [0, 1], not a percentage.
    """

    completed_count: int
    winning_count: int
    losing_count: int
    breakeven_count: int
    win_rate: Decimal | None
    total_realized_pnl: Money
    average_trade_pnl: Money | None
    gross_profit: Money
    gross_loss: Money
    profit_factor: Decimal | None
    profit_factor_unavailable_reason: FuturesPaperProfitFactorUnavailableReason | None
    minimum_holding_duration: timedelta | None
    maximum_holding_duration: timedelta | None
    average_holding_duration: timedelta | None
    median_holding_duration: timedelta | None

    def __post_init__(self) -> None:
        subject = "FuturesPaperTradeStatistics"
        for name in ("completed_count", "winning_count", "losing_count", "breakeven_count"):
            _require_count(subject, name, getattr(self, name))
        if self.completed_count != self.winning_count + self.losing_count + self.breakeven_count:
            raise ValueError(f"{subject} trades must be exactly winning, losing or breakeven.")
        _require_currency(
            subject,
            self.total_realized_pnl.currency,
            self.average_trade_pnl,
            self.gross_profit,
            self.gross_loss,
        )
        if (self.profit_factor is None) == (self.profit_factor_unavailable_reason is None):
            raise ValueError(f"{subject} needs exactly one of profit factor or its reason.")
        undefined = (
            self.win_rate,
            self.average_trade_pnl,
            self.minimum_holding_duration,
            self.maximum_holding_duration,
            self.average_holding_duration,
            self.median_holding_duration,
        )
        if self.completed_count == 0 and any(value is not None for value in undefined):
            raise ValueError(f"{subject} without completed trades has no ratio or average.")
        if self.completed_count and any(value is None for value in undefined):
            raise ValueError(f"{subject} with completed trades must carry every statistic.")


@dataclass(frozen=True, slots=True)
class FuturesPaperEquityPoint:
    """Gross P&L of the contract at one stored daily close.

    ``total_pnl`` is ``realized_pnl + unrealized_pnl`` under the P&L context.
    """

    instant: PointInTime
    mark_quote: QuoteValue
    net_contracts: int
    realized_pnl: Money
    unrealized_pnl: Money
    total_pnl: Money

    def __post_init__(self) -> None:
        subject = "FuturesPaperEquityPoint"
        _require(subject, "instant", self.instant, PointInTime)
        _require(subject, "mark_quote", self.mark_quote, QuoteValue)
        for name in ("realized_pnl", "unrealized_pnl", "total_pnl"):
            _require(subject, name, getattr(self, name), Money)
        if isinstance(self.net_contracts, bool) or not isinstance(self.net_contracts, int):
            raise TypeError(f"{subject} net contracts must be an integer.")
        _require_currency(subject, self.realized_pnl.currency, self.unrealized_pnl, self.total_pnl)


@dataclass(frozen=True, slots=True)
class FuturesPaperDrawdown:
    """The largest absolute fall of total gross P&L from a running peak.

    The trough is the earliest point reaching the maximum drawdown and the
    peak the earliest point holding the running peak before it. With no fall
    at all the amount is zero and peak and trough are the first point.
    """

    peak_instant: PointInTime
    peak_pnl: Money
    trough_instant: PointInTime
    trough_pnl: Money
    amount: Money

    def __post_init__(self) -> None:
        subject = "FuturesPaperDrawdown"
        _require(subject, "peak_instant", self.peak_instant, PointInTime)
        _require(subject, "trough_instant", self.trough_instant, PointInTime)
        for name in ("peak_pnl", "trough_pnl", "amount"):
            _require(subject, name, getattr(self, name), Money)
        _require_currency(subject, self.amount.currency, self.peak_pnl, self.trough_pnl)
        if self.amount.amount < 0:
            raise ValueError(f"{subject} amount cannot be negative.")
        if self.trough_instant.compare(self.peak_instant) < 0:
            raise ValueError(f"{subject} trough cannot precede its peak.")


@dataclass(frozen=True, slots=True)
class FuturesPaperPerformance:
    """Performance of one strategy's paper trading on one contract at one cutoff.

    ``realized_pnl`` is the portfolio's realized P&L for the contract at the
    cutoff, from the realized fold itself, and is canonical. The completed
    trades plus the open exposure's realized P&L decompose it analytically;
    with average-entry rounding their sum can differ in the trailing Decimal
    digits, which is left visible rather than reallocated.
    """

    contract: FuturesContract
    strategy_identity: StrategyIdentity
    portfolio_identity: PaperPortfolioIdentity
    available_through: PointInTime
    settlement_currency: Currency
    decisions: FuturesPaperDecisionDistribution
    execution: FuturesPaperExecutionActivity
    completed_trades: tuple[FuturesPaperCompletedTrade, ...]
    open_exposure: FuturesPaperOpenExposure | None
    statistics: FuturesPaperTradeStatistics
    realized_pnl: Money
    equity_curve: tuple[FuturesPaperEquityPoint, ...]
    max_drawdown: FuturesPaperDrawdown | None

    def __post_init__(self) -> None:
        subject = "FuturesPaperPerformance"
        for name, expected in (
            ("contract", FuturesContract),
            ("strategy_identity", StrategyIdentity),
            ("portfolio_identity", PaperPortfolioIdentity),
            ("available_through", PointInTime),
            ("settlement_currency", Currency),
            ("decisions", FuturesPaperDecisionDistribution),
            ("execution", FuturesPaperExecutionActivity),
            ("statistics", FuturesPaperTradeStatistics),
            ("realized_pnl", Money),
        ):
            _require(subject, name, getattr(self, name), expected)
        for name, expected in (
            ("completed_trades", FuturesPaperCompletedTrade),
            ("equity_curve", FuturesPaperEquityPoint),
        ):
            values = getattr(self, name)
            if not isinstance(values, tuple) or not all(isinstance(v, expected) for v in values):
                raise TypeError(f"{subject} {name} must be a tuple of {expected.__name__}.")
        if self.open_exposure is not None:
            _require(subject, "open_exposure", self.open_exposure, FuturesPaperOpenExposure)
        if self.max_drawdown is not None:
            _require(subject, "max_drawdown", self.max_drawdown, FuturesPaperDrawdown)
        if (self.max_drawdown is None) != (not self.equity_curve):
            raise ValueError(f"{subject} has a max drawdown exactly when it has an equity curve.")

        currency = self.settlement_currency
        _require_currency(
            subject,
            currency,
            self.realized_pnl,
            self.statistics.total_realized_pnl,
            *(trade.realized_pnl for trade in self.completed_trades),
            *(point.total_pnl for point in self.equity_curve),
            self.max_drawdown.amount if self.max_drawdown else None,
            self.open_exposure.realized_pnl if self.open_exposure else None,
        )
        for trade in self.completed_trades:
            if trade.contract != self.contract:
                raise ValueError(f"{subject} trades must all be on the analysed contract.")
            if trade.closed_at.compare(self.available_through) > 0:
                raise ValueError(f"{subject} trades must complete by the cutoff.")
        if self.statistics.completed_count != len(self.completed_trades):
            raise ValueError(f"{subject} statistics must cover exactly the completed trades.")
        for point in self.equity_curve:
            if point.instant.compare(self.available_through) > 0:
                raise ValueError(f"{subject} equity points must lie at or before the cutoff.")
        if self.open_exposure is not None:
            if self.open_exposure.position.contract != self.contract:
                raise ValueError(f"{subject} open exposure must be on the analysed contract.")
            if self.open_exposure.position.net_contracts != self.execution.net_contracts:
                raise ValueError(f"{subject} open exposure must hold the net contracts.")
        elif self.execution.net_contracts != 0:
            raise ValueError(f"{subject} with net contracts must report open exposure.")


# ---------------------------------------------------------------------------
# The use case
# ---------------------------------------------------------------------------


class _LoadedBars(FuturesHistoricalMarketDataRepository):
    """Serve already loaded, validated bars to the existing mark use case.

    The equity curve marks once per stored close; answering from the one
    validated load keeps every mark identical to the valuation's rule without
    reading storage again per point.
    """

    def __init__(self, bars: tuple[FuturesOHLCVBar, ...]) -> None:
        self._bars = bars

    def get_bars(self, query: FuturesHistoricalMarketDataQuery) -> tuple[FuturesOHLCVBar, ...]:
        return tuple(
            bar
            for bar in self._bars
            if bar.contract == query.contract
            and bar.timeframe == query.timeframe
            and query.covers(bar.point_in_time)
        )


@dataclass(slots=True)
class _Episode:
    direction: FuturesPaperTradeDirection
    opened_at: PointInTime
    points: Decimal
    closed: int
    exit_weight: Decimal


class CalculateFuturesPaperPerformanceUseCase:
    """Analyse one strategy's paper performance on one contract at one cutoff."""

    def __init__(
        self,
        forward_repository: FuturesForwardResearchRecordRepository,
        order_repository: FuturesPaperOrderRepository,
        fill_repository: FuturesPaperFillRepository,
        market_repository: FuturesHistoricalMarketDataRepository,
        economics_repository: FuturesContractEconomicsRepository,
    ) -> None:
        for name, value, expected in (
            ("forward_repository", forward_repository, FuturesForwardResearchRecordRepository),
            ("order_repository", order_repository, FuturesPaperOrderRepository),
            ("fill_repository", fill_repository, FuturesPaperFillRepository),
            ("market_repository", market_repository, FuturesHistoricalMarketDataRepository),
            ("economics_repository", economics_repository, FuturesContractEconomicsRepository),
        ):
            _require(_SUBJECT, name, value, expected)
        self._forward_repository = forward_repository
        self._order_repository = order_repository
        self._fill_repository = fill_repository
        self._market_repository = market_repository
        self._economics_repository = economics_repository
        self._build_portfolio = BuildFuturesPaperPortfolioUseCase()
        self._realize = CalculateFuturesRealizedPnlUseCase()

    def execute(
        self,
        contract: FuturesContract,
        strategy_identity: StrategyIdentity,
        portfolio_identity: PaperPortfolioIdentity,
        available_through: PointInTime,
    ) -> FuturesPaperPerformance:
        """Return the contract's paper performance as of ``available_through``.

        The analysed contract must have economics: they fix the settlement
        currency of every amount, even before the first fill.
        """
        for name, value, expected in (
            ("contract", contract, FuturesContract),
            ("strategy identity", strategy_identity, StrategyIdentity),
            ("portfolio identity", portfolio_identity, PaperPortfolioIdentity),
            ("available-through", available_through, PointInTime),
        ):
            _require(_SUBJECT, name, value, expected)

        # The complete history is validated, facts after the cutoff included.
        orders, all_fills = _load_history(
            self._order_repository, self._fill_repository, portfolio_identity, strategy_identity
        )
        economics = _economics_for(_SUBJECT, self._economics_repository, contract)
        # A subsequence of canonical history is still canonical history.
        fills = tuple(fill for fill in all_fills if fill.contract == contract)
        visible_fills = tuple(f for f in fills if f.filled_at.compare(available_through) <= 0)
        visible_orders = tuple(
            order
            for order in orders
            if order.intent.contract == contract
            and order.intent.decided_at.compare(available_through) <= 0
        )
        bars = self._bars(contract, available_through)

        trades, open_episode = self._episodes(visible_fills, economics)
        realized = self._realized(
            portfolio_identity, strategy_identity, fills, economics, available_through
        )
        open_exposure = self._open_exposure(
            portfolio_identity,
            strategy_identity,
            fills,
            economics,
            bars,
            open_episode,
            available_through,
        )
        curve = self._equity_curve(portfolio_identity, strategy_identity, fills, economics, bars)
        currency = economics.point_value.currency
        return FuturesPaperPerformance(
            contract=contract,
            strategy_identity=strategy_identity,
            portfolio_identity=portfolio_identity,
            available_through=available_through,
            settlement_currency=currency,
            decisions=self._decisions(contract, strategy_identity, available_through),
            execution=self._execution(visible_orders, visible_fills),
            completed_trades=trades,
            open_exposure=open_exposure,
            statistics=_statistics(trades, currency),
            realized_pnl=realized,
            equity_curve=curve,
            max_drawdown=_max_drawdown(curve),
        )

    # -- facts -----------------------------------------------------------------

    def _decisions(
        self,
        contract: FuturesContract,
        strategy_identity: StrategyIdentity,
        available_through: PointInTime,
    ) -> FuturesPaperDecisionDistribution:
        query = FuturesForwardResearchRecordQuery(contract, _DAILY)
        records = self._forward_repository.get_records(query)
        _validate_forward_output(records, query)
        actions = [
            _action(record)
            for record in records
            if record.strategy_identity == strategy_identity
            and record.decision_instant.compare(available_through) <= 0
        ]
        return FuturesPaperDecisionDistribution(
            decision_count=len(actions),
            buy_count=actions.count("BUY"),
            sell_count=actions.count("SELL"),
            hold_count=actions.count("HOLD"),
        )

    def _bars(
        self, contract: FuturesContract, available_through: PointInTime
    ) -> tuple[FuturesOHLCVBar, ...]:
        query = FuturesHistoricalMarketDataQuery(
            contract=contract, timeframe=_DAILY, start=None, end=available_through
        )
        bars = validate_futures_repository_bars(self._market_repository.get_bars(query), query)
        return tuple(bar for bar in bars if bar.point_in_time.compare(available_through) <= 0)

    @staticmethod
    def _execution(
        orders: tuple[FuturesPaperOrder, ...], fills: tuple[FuturesPaperFill, ...]
    ) -> FuturesPaperExecutionActivity:
        bought = sum(fill.contracts.value for fill in fills if fill.side is OrderSide.BUY)
        sold = sum(fill.contracts.value for fill in fills if fill.side is OrderSide.SELL)
        return FuturesPaperExecutionActivity(
            order_count=len(orders),
            fill_count=len(fills),
            contracts_bought=bought,
            contracts_sold=sold,
            net_contracts=bought - sold,
        )

    # -- economics, all through the existing folds ----------------------------

    def _realized(
        self,
        portfolio_identity: PaperPortfolioIdentity,
        strategy_identity: StrategyIdentity,
        fills: tuple[FuturesPaperFill, ...],
        economics: FuturesContractEconomics,
        at: PointInTime,
    ) -> Money:
        rows = self._realize.execute(portfolio_identity, strategy_identity, fills, (economics,), at)
        if not rows:
            return Money(Decimal(0), economics.point_value.currency)
        (row,) = rows
        return row.realized_pnl

    def _marked(
        self,
        portfolio_identity: PaperPortfolioIdentity,
        strategy_identity: StrategyIdentity,
        fills: tuple[FuturesPaperFill, ...],
        bars: tuple[FuturesOHLCVBar, ...],
        at: PointInTime,
    ):
        portfolio = self._build_portfolio.execute(portfolio_identity, strategy_identity, fills, at)
        if not portfolio.positions:
            return None
        mark = ValueFuturesPaperPortfolioUseCase(_LoadedBars(bars), self._economics_repository)
        (valued,) = mark.execute(portfolio, at)
        return valued

    def _open_exposure(
        self,
        portfolio_identity: PaperPortfolioIdentity,
        strategy_identity: StrategyIdentity,
        fills: tuple[FuturesPaperFill, ...],
        economics: FuturesContractEconomics,
        bars: tuple[FuturesOHLCVBar, ...],
        episode: _Episode | None,
        available_through: PointInTime,
    ) -> FuturesPaperOpenExposure | None:
        valued = self._marked(portfolio_identity, strategy_identity, fills, bars, available_through)
        if valued is None:
            return None
        if episode is None:
            raise ValueError(f"{_SUBJECT} found an open position without an open episode.")
        return FuturesPaperOpenExposure(
            position=valued.position,
            direction=episode.direction,
            opened_at=episode.opened_at,
            realized_pnl=_money(episode.points, economics),
            mark_quote=valued.mark_quote,
            mark_instant=valued.mark_instant,
            unrealized_pnl=valued.unrealized_pnl,
        )

    def _equity_curve(
        self,
        portfolio_identity: PaperPortfolioIdentity,
        strategy_identity: StrategyIdentity,
        fills: tuple[FuturesPaperFill, ...],
        economics: FuturesContractEconomics,
        bars: tuple[FuturesOHLCVBar, ...],
    ) -> tuple[FuturesPaperEquityPoint, ...]:
        currency = economics.point_value.currency
        points: list[FuturesPaperEquityPoint] = []
        for bar in bars:
            at = bar.point_in_time
            realized = self._realized(portfolio_identity, strategy_identity, fills, economics, at)
            valued = self._marked(portfolio_identity, strategy_identity, fills, bars, at)
            if valued is None:
                net, unrealized = 0, Money(Decimal(0), currency)
            else:
                # The bar is stored and observable at its own instant, so it is the mark.
                net, unrealized = valued.position.net_contracts, valued.unrealized_pnl
            with localcontext(_PNL_CONTEXT):
                total = realized.amount + unrealized.amount
            points.append(
                FuturesPaperEquityPoint(
                    instant=at,
                    mark_quote=bar.close,
                    net_contracts=net,
                    realized_pnl=realized,
                    unrealized_pnl=unrealized,
                    total_pnl=Money(total, currency),
                )
            )
        return tuple(points)

    @staticmethod
    def _episodes(
        fills: tuple[FuturesPaperFill, ...], economics: FuturesContractEconomics
    ) -> tuple[tuple[FuturesPaperCompletedTrade, ...], _Episode | None]:
        """Replay the fills through the fold's transition and cut them into episodes."""
        trades: list[FuturesPaperCompletedTrade] = []
        position: FuturesPosition | None = None
        episode: _Episode | None = None
        for fill in fills:
            transition = _transition(position, fill)
            quote = fill.fill_quote.value
            if transition.contracts_closed:
                sign = 1 if position.net_contracts > 0 else -1
                with localcontext(_PNL_CONTEXT):
                    episode.points += (
                        (quote - transition.closing_average_entry.value)
                        * sign
                        * transition.contracts_closed
                    )
                    episode.exit_weight += quote * transition.contracts_closed
                episode.closed += transition.contracts_closed

            after = transition.position
            completed = episode is not None and (
                after is None or (after.net_contracts > 0) != (position.net_contracts > 0)
            )
            if completed:
                with localcontext(_PNL_CONTEXT):
                    exit_average = episode.exit_weight / episode.closed
                trades.append(
                    FuturesPaperCompletedTrade(
                        contract=fill.contract,
                        direction=episode.direction,
                        opened_at=episode.opened_at,
                        closed_at=fill.filled_at,
                        contracts=episode.closed,
                        closing_average_entry=transition.closing_average_entry,
                        average_exit=QuoteValue(exit_average),
                        realized_pnl=_money(episode.points, economics),
                    )
                )
                episode = None
            if after is not None and episode is None:
                # A fresh position, or the opposite remainder of a reversal.
                episode = _Episode(
                    direction=(
                        FuturesPaperTradeDirection.LONG
                        if after.net_contracts > 0
                        else FuturesPaperTradeDirection.SHORT
                    ),
                    opened_at=fill.filled_at,
                    points=Decimal(0),
                    closed=0,
                    exit_weight=Decimal(0),
                )
            position = after
        return tuple(trades), episode


# ---------------------------------------------------------------------------
# Pure derivations
# ---------------------------------------------------------------------------


def _action(record: FuturesForwardResearchRecord) -> str:
    action = record.result.recommendation.action.value
    if action not in _ACTIONS:
        raise ValueError(f"{_SUBJECT} cannot classify recommendation action {action!r}.")
    return action


def _money(points: Decimal, economics: FuturesContractEconomics) -> Money:
    """Convert quote points to money exactly as the realized fold does."""
    point_value = economics.point_value
    with localcontext(_PNL_CONTEXT):
        amount = points * point_value.amount
    return Money(amount, point_value.currency)


def _statistics(
    trades: tuple[FuturesPaperCompletedTrade, ...], currency: Currency
) -> FuturesPaperTradeStatistics:
    zero = Decimal(0)
    amounts = [trade.realized_pnl.amount for trade in trades]
    with localcontext(_PNL_CONTEXT):
        total = sum(amounts, zero)
        profit = sum((amount for amount in amounts if amount > 0), zero)
        loss = -sum((amount for amount in amounts if amount < 0), zero)
        count = len(trades)
        winning = sum(1 for amount in amounts if amount > 0)
        win_rate = Decimal(winning) / count if count else None
        average = total / count if count else None
        factor = profit / loss if loss > 0 else None
    if factor is not None:
        reason = None
    elif count == 0:
        reason = FuturesPaperProfitFactorUnavailableReason.NO_COMPLETED_TRADES
    else:
        reason = FuturesPaperProfitFactorUnavailableReason.NO_LOSING_TRADES

    durations = sorted(trade.holding_duration for trade in trades)
    middle = len(durations) // 2
    if not durations:
        median = None
    elif len(durations) % 2:
        median = durations[middle]
    else:
        median = (durations[middle - 1] + durations[middle]) / 2
    return FuturesPaperTradeStatistics(
        completed_count=count,
        winning_count=winning,
        losing_count=sum(1 for amount in amounts if amount < 0),
        breakeven_count=sum(1 for amount in amounts if amount == 0),
        win_rate=win_rate,
        total_realized_pnl=Money(total, currency),
        average_trade_pnl=Money(average, currency) if average is not None else None,
        gross_profit=Money(profit, currency),
        gross_loss=Money(loss, currency),
        profit_factor=factor,
        profit_factor_unavailable_reason=reason,
        minimum_holding_duration=durations[0] if durations else None,
        maximum_holding_duration=durations[-1] if durations else None,
        average_holding_duration=(
            sum(durations, timedelta()) / len(durations) if durations else None
        ),
        median_holding_duration=median,
    )


def _max_drawdown(curve: tuple[FuturesPaperEquityPoint, ...]) -> FuturesPaperDrawdown | None:
    if not curve:
        return None
    currency = curve[0].total_pnl.currency
    peak = trough = best_peak = curve[0]
    worst = Decimal(0)
    for point in curve:
        if point.total_pnl.amount > peak.total_pnl.amount:
            peak = point
        with localcontext(_PNL_CONTEXT):
            drawdown = peak.total_pnl.amount - point.total_pnl.amount
        if drawdown > worst:
            worst, best_peak, trough = drawdown, peak, point
    return FuturesPaperDrawdown(
        peak_instant=best_peak.instant,
        peak_pnl=best_peak.total_pnl,
        trough_instant=trough.instant,
        trough_pnl=trough.total_pnl,
        amount=Money(worst, currency),
    )
