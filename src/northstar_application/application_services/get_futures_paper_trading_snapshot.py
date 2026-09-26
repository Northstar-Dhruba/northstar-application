"""Canonical read of futures paper-trading operational state at one cutoff.

One snapshot answers, for an explicit ``available_through``, two scopes that
must not be confused:

- the SELECTED CONTRACT (optional): its latest persisted daily bar and its
  most recent frozen research decisions for the portfolio's strategy;
- the WHOLE PAPER PORTFOLIO: its visible orders and fills, pending orders, the
  portfolio fold and, when product economics allow, its gross simulated
  valuation. Other contracts held by the portfolio are always included.

Everything is read from persisted facts through existing ports. Nothing is
frozen, executed, acquired, valued differently or stored, and no clock is read.
A recommendation is only ever a persisted FuturesForwardResearchRecord; the
analysis generator and strategy are never invoked here.

Research versus execution: a decision's action is its record's recommendation;
its paper state is the order persisted for that decision, if any. No-action
outcomes (HOLD, TARGET_ALREADY_MET) are not persisted, so a decision without an
order is reported as having none -- never as a reason inferred from absence.

Missing economics: valuation stops at the first visible product without
economics. The snapshot then carries no valuation and names that one product;
every other section is still returned. Integrity failures -- corrupt storage,
malformed port output, a mixed-strategy portfolio -- propagate unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

from northstar_core.foundation.value_objects import PointInTime, Timeframe
from northstar_core.futures import FuturesContract, FuturesOHLCVBar, FuturesProductReference
from northstar_core.paper_trading import (
    FuturesPaperFill,
    FuturesPaperOrder,
    FuturesPaperPortfolio,
    PaperPortfolioIdentity,
)
from northstar_core.strategy import StrategyIdentity

from northstar_application.application_services._futures_repository_output import (
    validate_futures_repository_bars,
)
from northstar_application.application_services.build_futures_paper_portfolio import (
    BuildFuturesPaperPortfolioUseCase,
)
from northstar_application.application_services.build_futures_paper_trading_valuation import (
    BuildFuturesPaperTradingValuationUseCase,
    FuturesPaperTradingValuation,
)
from northstar_application.application_services.futures_forward_research_record import (
    FuturesForwardResearchRecord,
)
from northstar_application.application_services.futures_paper_execution_identities import (
    FuturesPaperExecutionIdentityService,
)
from northstar_application.application_services.run_futures_paper_trading_decision import (
    _load_history,
    _validate_forward_output,
)
from northstar_application.application_services.value_futures_paper_portfolio import (
    FuturesProductEconomicsNotFoundError,
)
from northstar_application.ports import (
    FuturesForwardResearchRecordQuery,
    FuturesForwardResearchRecordRepository,
    FuturesHistoricalMarketDataQuery,
    FuturesHistoricalMarketDataRepository,
    FuturesPaperFillRepository,
    FuturesPaperOrderRepository,
    FuturesProductEconomicsRepository,
)

_DAILY = Timeframe("1d")
_RECENT_DECISION_LIMIT = 10
_SUBJECT = "GetFuturesPaperTradingSnapshotUseCase"


def _visible(instant: PointInTime, cutoff: PointInTime) -> bool:
    return instant.compare(cutoff) <= 0


@dataclass(frozen=True, slots=True)
class FuturesPaperDecisionSnapshot:
    """One frozen research decision and the paper order persisted for it, if any.

    ``fill`` is present only when visible at the snapshot cutoff; an order
    without one is pending there. No order means none was persisted for the
    decision -- the reason is not persisted and is not inferred.
    """

    record: FuturesForwardResearchRecord
    order: FuturesPaperOrder | None
    fill: FuturesPaperFill | None

    def __post_init__(self) -> None:
        subject = "FuturesPaperDecisionSnapshot"
        if not isinstance(self.record, FuturesForwardResearchRecord):
            raise TypeError(f"{subject} record must be a FuturesForwardResearchRecord.")
        if self.order is not None and not isinstance(self.order, FuturesPaperOrder):
            raise TypeError(f"{subject} order must be a FuturesPaperOrder or None.")
        if self.fill is not None and not isinstance(self.fill, FuturesPaperFill):
            raise TypeError(f"{subject} fill must be a FuturesPaperFill or None.")
        if self.order is None:
            if self.fill is not None:
                raise ValueError(f"{subject} cannot carry a fill without its order.")
            return
        intent = self.order.intent
        if (
            intent.contract != self.record.contract
            or intent.strategy_identity != self.record.strategy_identity
            or intent.decided_at.compare(self.record.decision_instant) != 0
        ):
            raise ValueError(f"{subject} order must execute its own decision.")
        if self.fill is not None and self.fill.order_identity != self.order.identity:
            raise ValueError(f"{subject} fill must fill its own order.")


@dataclass(frozen=True, slots=True)
class FuturesPaperTradingSnapshot:
    """Persisted operational state of one paper portfolio at one cutoff.

    Portfolio identity, strategy and cutoff are the portfolio's own. ``contract``
    selects the research scope and is None when only the whole portfolio is
    requested; the market bar and decisions then stay empty. ``orders`` and
    ``fills`` are those visible at the cutoff. ``recent_decisions`` are newest
    first and bounded. Exactly one of ``valuation`` and ``missing_economics``
    is present.
    """

    contract: FuturesContract | None
    portfolio: FuturesPaperPortfolio
    latest_market_bar: FuturesOHLCVBar | None
    recent_decisions: tuple[FuturesPaperDecisionSnapshot, ...]
    orders: tuple[FuturesPaperOrder, ...]
    fills: tuple[FuturesPaperFill, ...]
    valuation: FuturesPaperTradingValuation | None
    missing_economics: FuturesProductReference | None

    def __post_init__(self) -> None:
        self._validate_types()
        self._validate_selected_contract()
        self._validate_execution()
        self._validate_valuation()

    @property
    def portfolio_identity(self) -> PaperPortfolioIdentity:
        return self.portfolio.identity

    @property
    def strategy_identity(self) -> StrategyIdentity:
        return self.portfolio.strategy_identity

    @property
    def available_through(self) -> PointInTime:
        return self.portfolio.as_of

    @property
    def latest_forward_record(self) -> FuturesForwardResearchRecord | None:
        """Return the latest frozen decision for the selected contract, if any."""
        return self.recent_decisions[0].record if self.recent_decisions else None

    @property
    def pending_orders(self) -> tuple[FuturesPaperOrder, ...]:
        """Return visible orders with no visible fill, across the whole portfolio."""
        filled = {fill.order_identity for fill in self.fills}
        return tuple(order for order in self.orders if order.identity not in filled)

    def _validate_types(self) -> None:
        subject = "FuturesPaperTradingSnapshot"
        if self.contract is not None and not isinstance(self.contract, FuturesContract):
            raise TypeError(f"{subject} contract must be a FuturesContract or None.")
        if not isinstance(self.portfolio, FuturesPaperPortfolio):
            raise TypeError(f"{subject} portfolio must be a FuturesPaperPortfolio.")
        if self.latest_market_bar is not None and not isinstance(
            self.latest_market_bar, FuturesOHLCVBar
        ):
            raise TypeError(f"{subject} latest market bar must be a FuturesOHLCVBar or None.")
        for name, expected in (
            ("recent_decisions", FuturesPaperDecisionSnapshot),
            ("orders", FuturesPaperOrder),
            ("fills", FuturesPaperFill),
        ):
            value = getattr(self, name)
            if not isinstance(value, tuple) or not all(isinstance(v, expected) for v in value):
                raise TypeError(f"{subject} {name} must be a tuple of {expected.__name__}.")
        if self.valuation is not None and not isinstance(
            self.valuation, FuturesPaperTradingValuation
        ):
            raise TypeError(f"{subject} valuation must be a FuturesPaperTradingValuation or None.")
        if self.missing_economics is not None and not isinstance(
            self.missing_economics, FuturesProductReference
        ):
            raise TypeError(
                f"{subject} missing economics must be a FuturesProductReference or None."
            )

    def _validate_selected_contract(self) -> None:
        subject = "FuturesPaperTradingSnapshot"
        cutoff = self.available_through
        if self.contract is None:
            if self.latest_market_bar is not None or self.recent_decisions:
                raise ValueError(f"{subject} without a selected contract has no research scope.")
            return
        bar = self.latest_market_bar
        if bar is not None and (
            bar.contract != self.contract
            or bar.timeframe != _DAILY
            or not _visible(bar.point_in_time, cutoff)
        ):
            raise ValueError(
                f"{subject} market bar must be a visible daily bar of the selected contract."
            )
        if len(self.recent_decisions) > _RECENT_DECISION_LIMIT:
            raise ValueError(f"{subject} holds at most {_RECENT_DECISION_LIMIT} decisions.")
        previous: PointInTime | None = None
        for decision in self.recent_decisions:
            record = decision.record
            if (
                record.contract != self.contract
                or record.timeframe != _DAILY
                or record.strategy_identity != self.strategy_identity
                or not _visible(record.decision_instant, cutoff)
            ):
                raise ValueError(
                    f"{subject} decisions must be visible daily decisions of the selected "
                    "contract and the portfolio's strategy."
                )
            if previous is not None and record.decision_instant.compare(previous) >= 0:
                raise ValueError(f"{subject} decisions must be strictly newest first.")
            previous = record.decision_instant
            if decision.order is not None and decision.order not in self.orders:
                raise ValueError(f"{subject} decision orders must be visible orders.")
            if decision.fill is not None and decision.fill not in self.fills:
                raise ValueError(f"{subject} decision fills must be visible fills.")

    def _validate_execution(self) -> None:
        subject = "FuturesPaperTradingSnapshot"
        cutoff = self.available_through
        for order in self.orders:
            if order.intent.portfolio_identity != self.portfolio_identity or not _visible(
                order.intent.decided_at, cutoff
            ):
                raise ValueError(f"{subject} orders must be the portfolio's visible orders.")
        identities = {order.identity for order in self.orders}
        for fill in self.fills:
            if fill.portfolio_identity != self.portfolio_identity or not _visible(
                fill.filled_at, cutoff
            ):
                raise ValueError(f"{subject} fills must be the portfolio's visible fills.")
            if fill.order_identity not in identities:
                raise ValueError(f"{subject} every visible fill must have its visible order.")

    def _validate_valuation(self) -> None:
        subject = "FuturesPaperTradingSnapshot"
        if (self.valuation is None) == (self.missing_economics is None):
            raise ValueError(
                f"{subject} carries a valuation or the missing product economics, not both."
            )
        if self.valuation is not None and self.valuation.portfolio != self.portfolio:
            raise ValueError(f"{subject} valuation must value the snapshot portfolio.")


class GetFuturesPaperTradingSnapshotUseCase:
    """Read the persisted operational state of one paper portfolio at one cutoff."""

    def __init__(
        self,
        market_repository: FuturesHistoricalMarketDataRepository,
        forward_repository: FuturesForwardResearchRecordRepository,
        order_repository: FuturesPaperOrderRepository,
        fill_repository: FuturesPaperFillRepository,
        economics_repository: FuturesProductEconomicsRepository,
    ) -> None:
        for name, value, expected in (
            ("market_repository", market_repository, FuturesHistoricalMarketDataRepository),
            ("forward_repository", forward_repository, FuturesForwardResearchRecordRepository),
            ("order_repository", order_repository, FuturesPaperOrderRepository),
            ("fill_repository", fill_repository, FuturesPaperFillRepository),
            ("economics_repository", economics_repository, FuturesProductEconomicsRepository),
        ):
            if not isinstance(value, expected):
                raise TypeError(f"{_SUBJECT} {name} must be a {expected.__name__}.")
        self._market_repository = market_repository
        self._forward_repository = forward_repository
        self._order_repository = order_repository
        self._fill_repository = fill_repository
        self._build_portfolio = BuildFuturesPaperPortfolioUseCase()
        self._valuation = BuildFuturesPaperTradingValuationUseCase(
            order_repository=order_repository,
            fill_repository=fill_repository,
            market_repository=market_repository,
            economics_repository=economics_repository,
        )
        self._identities = FuturesPaperExecutionIdentityService()

    def execute(
        self,
        contract: FuturesContract | None,
        strategy_identity: StrategyIdentity,
        portfolio_identity: PaperPortfolioIdentity,
        available_through: PointInTime,
    ) -> FuturesPaperTradingSnapshot:
        """Return the snapshot; ``contract`` None reads the whole-portfolio scope only."""
        if contract is not None and not isinstance(contract, FuturesContract):
            raise TypeError(f"{_SUBJECT} contract must be a FuturesContract or None.")
        for name, value, expected in (
            ("strategy identity", strategy_identity, StrategyIdentity),
            ("portfolio identity", portfolio_identity, PaperPortfolioIdentity),
            ("available-through", available_through, PointInTime),
        ):
            if not isinstance(value, expected):
                raise TypeError(f"{_SUBJECT} {name} must be a {expected.__name__}.")

        # Validates the complete history, including the one-strategy invariant, first.
        orders, fills = _load_history(
            self._order_repository, self._fill_repository, portfolio_identity, strategy_identity
        )
        portfolio = self._build_portfolio.execute(
            portfolio_identity, strategy_identity, fills, available_through
        )
        visible_orders = tuple(
            o for o in orders if _visible(o.intent.decided_at, available_through)
        )
        visible_fills = tuple(f for f in fills if _visible(f.filled_at, available_through))

        try:
            valuation = self._valuation.execute(
                portfolio_identity, strategy_identity, available_through
            )
            missing = None
        except FuturesProductEconomicsNotFoundError as error:
            valuation, missing = None, error.reference

        bar, decisions = None, ()
        if contract is not None:
            bar = self._latest_bar(contract, available_through)
            decisions = self._recent_decisions(
                contract,
                strategy_identity,
                portfolio_identity,
                available_through,
                visible_orders,
                visible_fills,
            )
        return FuturesPaperTradingSnapshot(
            contract=contract,
            portfolio=portfolio,
            latest_market_bar=bar,
            recent_decisions=decisions,
            orders=visible_orders,
            fills=visible_fills,
            valuation=valuation,
            missing_economics=missing,
        )

    def _latest_bar(
        self, contract: FuturesContract, available_through: PointInTime
    ) -> FuturesOHLCVBar | None:
        query = FuturesHistoricalMarketDataQuery(
            contract=contract, timeframe=_DAILY, start=None, end=available_through
        )
        bars = validate_futures_repository_bars(self._market_repository.get_bars(query), query)
        latest: FuturesOHLCVBar | None = None
        for bar in bars:
            if _visible(bar.point_in_time, available_through) and (
                latest is None or bar.point_in_time.compare(latest.point_in_time) > 0
            ):
                latest = bar
        return latest

    def _recent_decisions(
        self,
        contract: FuturesContract,
        strategy_identity: StrategyIdentity,
        portfolio_identity: PaperPortfolioIdentity,
        available_through: PointInTime,
        orders: tuple[FuturesPaperOrder, ...],
        fills: tuple[FuturesPaperFill, ...],
    ) -> tuple[FuturesPaperDecisionSnapshot, ...]:
        query = FuturesForwardResearchRecordQuery(contract, _DAILY)
        stored = self._forward_repository.get_records(query)
        _validate_forward_output(stored, query)
        # Validated output is ordered by decision instant, so the tail is the newest.
        mine = [
            record
            for record in stored
            if record.strategy_identity == strategy_identity
            and _visible(record.decision_instant, available_through)
        ][-_RECENT_DECISION_LIMIT:]

        by_identity = {order.identity: order for order in orders}
        fill_of = {fill.order_identity: fill for fill in fills}
        decisions: list[FuturesPaperDecisionSnapshot] = []
        for record in reversed(mine):
            order = by_identity.get(self._identities.order_identity(record, portfolio_identity))
            fill = fill_of.get(order.identity) if order is not None else None
            decisions.append(FuturesPaperDecisionSnapshot(record, order, fill))
        return tuple(decisions)
