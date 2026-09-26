"""Application orchestration of one futures paper-trading run.

A run executes every frozen forward decision for ONE strategy on one queried
contract, decided at or before an explicit cutoff, into one paper portfolio,
then reports that portfolio as of the cutoff. Unlike a forward research run it
never mixes strategies: a paper portfolio belongs to exactly one strategy.

Each selected decision is executed, oldest first, through
RunFuturesPaperTradingDecisionUseCase with the RUN's cutoff, not the decision's
own instant, so older pending orders whose next bar is now visible settle.
Idempotency, conflict detection and position logic stay in that one place; the
run adds selection, sequencing and the final as-of portfolio.

What a run at one cutoff does and does not guarantee
----------------------------------------------------
Market bars stored after the cutoff, and decisions frozen with an instant after
it, cannot change the run. A later cutoff is a different view: pending orders
may fill and the final portfolio may change, but persisted orders and fills are
never rewritten. Two things can change a re-run at the same cutoff, both
accepted because a run is re-derived rather than persisted:

- a decision frozen later with an instant at or before the cutoff joins the run
  and may execute;
- a market bar back-filled before an already persisted fill's bar makes that
  decision conflict (FuturesPaperFillConflictError), exactly as it would when
  run on its own.

The run reads no clock, touches no provider or calendar, and calculates no
profit and loss.
"""

from __future__ import annotations

from dataclasses import dataclass

from northstar_core.foundation.value_objects import PointInTime
from northstar_core.paper_trading import (
    FuturesContractCount,
    FuturesPaperFill,
    FuturesPaperOrder,
    FuturesPaperPortfolio,
    PaperPortfolioIdentity,
)
from northstar_core.strategy import StrategyIdentity

from northstar_application.application_services.build_futures_paper_portfolio import (
    BuildFuturesPaperPortfolioUseCase,
)
from northstar_application.application_services.run_futures_paper_trading_decision import (
    FuturesPaperTradingDecisionResult,
    RunFuturesPaperTradingDecisionUseCase,
    _load_history,
    _validate_forward_output,
)
from northstar_application.ports import (
    FuturesForwardResearchRecordQuery,
    FuturesForwardResearchRecordRepository,
    FuturesHistoricalMarketDataRepository,
    FuturesPaperFillRepository,
    FuturesPaperFillStore,
    FuturesPaperOrderRepository,
    FuturesPaperOrderStore,
)


@dataclass(frozen=True, slots=True)
class FuturesPaperTradingRun:
    """Every selected frozen decision of one strategy, executed into one portfolio.

    ``results`` holds one result per frozen decision for ``strategy_identity``
    on the query's contract and timeframe, decided at or before
    ``available_through``, oldest first. ``portfolio`` is the portfolio as of
    ``available_through`` -- not the last decision's ``portfolio_before`` -- so
    it includes fills that landed after the last decision and positions in
    other contracts held from earlier history.
    """

    query: FuturesForwardResearchRecordQuery
    portfolio_identity: PaperPortfolioIdentity
    strategy_identity: StrategyIdentity
    target_contracts: FuturesContractCount
    available_through: PointInTime
    results: tuple[FuturesPaperTradingDecisionResult, ...]
    portfolio: FuturesPaperPortfolio

    def __post_init__(self) -> None:
        self._validate_types()
        self._validate_results()

        portfolio = self.portfolio
        if portfolio.identity != self.portfolio_identity:
            raise ValueError("FuturesPaperTradingRun portfolio must be the run's portfolio.")
        if portfolio.strategy_identity != self.strategy_identity:
            raise ValueError("FuturesPaperTradingRun portfolio must belong to the run's strategy.")
        if portfolio.as_of.compare(self.available_through) != 0:
            raise ValueError("FuturesPaperTradingRun portfolio must be as of the run cutoff.")

    def _validate_types(self) -> None:
        subject = "FuturesPaperTradingRun"
        for name, expected in (
            ("query", FuturesForwardResearchRecordQuery),
            ("portfolio_identity", PaperPortfolioIdentity),
            ("strategy_identity", StrategyIdentity),
            ("target_contracts", FuturesContractCount),
            ("available_through", PointInTime),
            ("portfolio", FuturesPaperPortfolio),
        ):
            if not isinstance(getattr(self, name), expected):
                raise TypeError(f"{subject} {name} must be a {expected.__name__}.")
        if not isinstance(self.results, tuple):
            raise TypeError(f"{subject} results must be a tuple.")
        if not all(isinstance(r, FuturesPaperTradingDecisionResult) for r in self.results):
            raise TypeError(
                f"{subject} results must contain FuturesPaperTradingDecisionResult values."
            )

    def _validate_results(self) -> None:
        orders: set[str] = set()
        previous: FuturesPaperTradingDecisionResult | None = None
        for result in self.results:
            record = result.record
            if record.contract != self.query.contract or record.timeframe != self.query.timeframe:
                raise ValueError(
                    "FuturesPaperTradingRun results must match the query contract and timeframe."
                )
            if record.strategy_identity != self.strategy_identity:
                raise ValueError("FuturesPaperTradingRun results must belong to the run strategy.")
            if result.available_through.compare(self.available_through) != 0:
                raise ValueError(
                    "FuturesPaperTradingRun results must all be executed at the run cutoff."
                )
            if result.portfolio_before.identity != self.portfolio_identity:
                raise ValueError(
                    "FuturesPaperTradingRun results must belong to the run's portfolio."
                )
            # Contract, timeframe and strategy are fixed, so a strictly later
            # instant also rules out a repeated natural key.
            if previous is not None and (
                previous.record.decision_instant.compare(record.decision_instant) >= 0
            ):
                raise ValueError(
                    "FuturesPaperTradingRun results must be strictly ordered by decision instant."
                )
            previous = result
            if result.order is not None:
                if result.order.identity.identity in orders:
                    raise ValueError("FuturesPaperTradingRun results cannot share one paper order.")
                orders.add(result.order.identity.identity)

    @property
    def decision_count(self) -> int:
        """Return how many frozen decisions the run executed."""
        return len(self.results)

    @property
    def orders(self) -> tuple[FuturesPaperOrder, ...]:
        """Return the paper orders the run's decisions produced, in decision order."""
        return tuple(result.order for result in self.results if result.order is not None)

    @property
    def fills(self) -> tuple[FuturesPaperFill, ...]:
        """Return the fills of those orders visible at the cutoff, in decision order."""
        return tuple(result.fill for result in self.results if result.fill is not None)

    @property
    def first_decision_instant(self) -> PointInTime | None:
        """Return the earliest executed decision instant, or None when empty."""
        return self.results[0].record.decision_instant if self.results else None

    @property
    def last_decision_instant(self) -> PointInTime | None:
        """Return the latest executed decision instant, or None when empty."""
        return self.results[-1].record.decision_instant if self.results else None


class RunFuturesPaperTradingUseCase:
    """Execute one strategy's frozen futures decisions into one paper portfolio."""

    def __init__(
        self,
        forward_repository: FuturesForwardResearchRecordRepository,
        market_repository: FuturesHistoricalMarketDataRepository,
        order_store: FuturesPaperOrderStore,
        order_repository: FuturesPaperOrderRepository,
        fill_store: FuturesPaperFillStore,
        fill_repository: FuturesPaperFillRepository,
    ) -> None:
        for name, value, expected in (
            ("forward_repository", forward_repository, FuturesForwardResearchRecordRepository),
            ("market_repository", market_repository, FuturesHistoricalMarketDataRepository),
            ("order_store", order_store, FuturesPaperOrderStore),
            ("order_repository", order_repository, FuturesPaperOrderRepository),
            ("fill_store", fill_store, FuturesPaperFillStore),
            ("fill_repository", fill_repository, FuturesPaperFillRepository),
        ):
            if not isinstance(value, expected):
                raise TypeError(
                    f"RunFuturesPaperTradingUseCase {name} must be a {expected.__name__}."
                )

        self._forward_repository = forward_repository
        self._order_repository = order_repository
        self._fill_repository = fill_repository
        self._decide = RunFuturesPaperTradingDecisionUseCase(
            forward_repository=forward_repository,
            market_repository=market_repository,
            order_store=order_store,
            order_repository=order_repository,
            fill_store=fill_store,
            fill_repository=fill_repository,
        )
        self._build_portfolio = BuildFuturesPaperPortfolioUseCase()

    def execute(
        self,
        query: FuturesForwardResearchRecordQuery,
        portfolio_identity: PaperPortfolioIdentity,
        strategy_identity: StrategyIdentity,
        target_contracts: FuturesContractCount,
        available_through: PointInTime,
    ) -> FuturesPaperTradingRun:
        """Execute every selected frozen decision and report the portfolio at the cutoff."""
        subject = "RunFuturesPaperTradingUseCase"
        for name, value, expected in (
            ("query", query, FuturesForwardResearchRecordQuery),
            ("portfolio identity", portfolio_identity, PaperPortfolioIdentity),
            ("strategy identity", strategy_identity, StrategyIdentity),
            ("target contracts", target_contracts, FuturesContractCount),
            ("available-through", available_through, PointInTime),
        ):
            if not isinstance(value, expected):
                raise TypeError(f"{subject} {name} must be a {expected.__name__}.")

        stored = self._forward_repository.get_records(query)
        _validate_forward_output(stored, query)
        # Other strategies and later decisions are valid repository output, not this run.
        selected = tuple(
            record
            for record in stored
            if record.strategy_identity == strategy_identity
            and record.decision_instant.compare(available_through) <= 0
        )

        # Reject a mixed-strategy portfolio before any decision can write.
        _load_history(
            self._order_repository, self._fill_repository, portfolio_identity, strategy_identity
        )

        results = tuple(
            self._decide.execute(record, portfolio_identity, target_contracts, available_through)
            for record in selected
        )

        _, fills = _load_history(
            self._order_repository, self._fill_repository, portfolio_identity, strategy_identity
        )
        portfolio = self._build_portfolio.execute(
            portfolio_identity, strategy_identity, fills, available_through
        )
        return FuturesPaperTradingRun(
            query=query,
            portfolio_identity=portfolio_identity,
            strategy_identity=strategy_identity,
            target_contracts=target_contracts,
            available_through=available_through,
            results=results,
            portfolio=portfolio,
        )
