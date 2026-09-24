"""Application operational session for daily futures paper trading.

A session is one operator's processing of one contract, strategy and paper
portfolio through one explicit cutoff, over market data already persisted:

1. freeze -- or idempotently reproduce -- the latest forward decision visible
   at the cutoff, through FreezeFuturesForwardResearchDecisionUseCase;
2. run paper trading through the same cutoff, through
   RunFuturesPaperTradingUseCase, which replays every frozen decision of that
   strategy on that contract and settles pending orders as bars become visible;
3. build the execution report from that run.

It composes those use cases and restates none of their rules. It acquires no
market data, reads no clock, and values nothing: execution needs no product
economics, so missing economics can never block a legitimate decision, order or
fill. Valuation is a separate read-only step.

The strategy is the single built-in directional policy; the StrategyIdentity is
its label, not a selector between implementations. Timeframe is session-daily.
"""

from __future__ import annotations

from dataclasses import dataclass

from northstar_core.foundation.value_objects import PointInTime, Timeframe
from northstar_core.futures import FuturesContract
from northstar_core.paper_trading import FuturesContractCount, PaperPortfolioIdentity
from northstar_core.strategy import FuturesAssetAnalysisGenerator, Strategy, StrategyIdentity

from northstar_application.application_services.build_futures_paper_trading_report import (
    BuildFuturesPaperTradingReportUseCase,
    FuturesPaperTradingReport,
)
from northstar_application.application_services.freeze_futures_forward_research_decision import (
    FreezeFuturesForwardResearchDecisionUseCase,
)
from northstar_application.application_services.futures_forward_research_record import (
    FuturesForwardResearchRecord,
)
from northstar_application.application_services.run_futures_paper_trading import (
    FuturesPaperTradingRun,
    RunFuturesPaperTradingUseCase,
)
from northstar_application.ports import (
    FuturesForwardResearchRecordQuery,
    FuturesForwardResearchRecordRepository,
    FuturesForwardResearchRecordStore,
    FuturesHistoricalMarketDataRepository,
    FuturesPaperFillRepository,
    FuturesPaperFillStore,
    FuturesPaperOrderRepository,
    FuturesPaperOrderStore,
)

_DAILY = Timeframe("1d")
_SUBJECT = "RunFuturesPaperTradingSessionUseCase"


@dataclass(frozen=True, slots=True)
class FuturesPaperTradingSessionResult:
    """What one operational session froze, executed and reported.

    ``frozen_record`` is the record this session's freeze returned -- newly
    stored or an equal one already frozen -- or None when persisted history is
    still in warm-up. It is always one of the run's executed decisions.
    Context such as contract, strategy, portfolio, target and cutoff is read
    from ``run``.
    """

    frozen_record: FuturesForwardResearchRecord | None
    run: FuturesPaperTradingRun
    report: FuturesPaperTradingReport

    def __post_init__(self) -> None:
        subject = "FuturesPaperTradingSessionResult"
        if self.frozen_record is not None and not isinstance(
            self.frozen_record, FuturesForwardResearchRecord
        ):
            raise TypeError(f"{subject} frozen record must be a FuturesForwardResearchRecord.")
        if not isinstance(self.run, FuturesPaperTradingRun):
            raise TypeError(f"{subject} run must be a FuturesPaperTradingRun.")
        if not isinstance(self.report, FuturesPaperTradingReport):
            raise TypeError(f"{subject} report must be a FuturesPaperTradingReport.")
        if self.report.run != self.run:
            raise ValueError(f"{subject} report must describe the session's run.")

        record = self.frozen_record
        if record is None:
            return
        run = self.run
        if record.contract != run.query.contract or record.timeframe != run.query.timeframe:
            raise ValueError(
                f"{subject} frozen record must be for the run's contract and timeframe."
            )
        if record.strategy_identity != run.strategy_identity:
            raise ValueError(f"{subject} frozen record must belong to the run's strategy.")
        if record.decision_instant.compare(run.available_through) > 0:
            raise ValueError(f"{subject} frozen record cannot be decided after the run cutoff.")
        if record not in (result.record for result in run.results):
            raise ValueError(f"{subject} frozen record must be one of the run's decisions.")


class RunFuturesPaperTradingSessionUseCase:
    """Freeze the latest daily decision, then paper trade and report through one cutoff."""

    def __init__(
        self,
        market_repository: FuturesHistoricalMarketDataRepository,
        forward_store: FuturesForwardResearchRecordStore,
        forward_repository: FuturesForwardResearchRecordRepository,
        order_store: FuturesPaperOrderStore,
        order_repository: FuturesPaperOrderRepository,
        fill_store: FuturesPaperFillStore,
        fill_repository: FuturesPaperFillRepository,
    ) -> None:
        for name, value, expected in (
            ("market_repository", market_repository, FuturesHistoricalMarketDataRepository),
            ("forward_store", forward_store, FuturesForwardResearchRecordStore),
            ("forward_repository", forward_repository, FuturesForwardResearchRecordRepository),
            ("order_store", order_store, FuturesPaperOrderStore),
            ("order_repository", order_repository, FuturesPaperOrderRepository),
            ("fill_store", fill_store, FuturesPaperFillStore),
            ("fill_repository", fill_repository, FuturesPaperFillRepository),
        ):
            if not isinstance(value, expected):
                raise TypeError(f"{_SUBJECT} {name} must be a {expected.__name__}.")

        self._freeze = FreezeFuturesForwardResearchDecisionUseCase(
            market_repository, FuturesAssetAnalysisGenerator(), forward_store
        )
        self._run = RunFuturesPaperTradingUseCase(
            forward_repository=forward_repository,
            market_repository=market_repository,
            order_store=order_store,
            order_repository=order_repository,
            fill_store=fill_store,
            fill_repository=fill_repository,
        )
        self._report = BuildFuturesPaperTradingReportUseCase()

    def execute(
        self,
        contract: FuturesContract,
        strategy_identity: StrategyIdentity,
        portfolio_identity: PaperPortfolioIdentity,
        target_contracts: FuturesContractCount,
        available_through: PointInTime,
    ) -> FuturesPaperTradingSessionResult:
        """Process one contract, strategy and portfolio through ``available_through``."""
        for name, value, expected in (
            ("contract", contract, FuturesContract),
            ("strategy identity", strategy_identity, StrategyIdentity),
            ("portfolio identity", portfolio_identity, PaperPortfolioIdentity),
            ("target contracts", target_contracts, FuturesContractCount),
            ("available-through", available_through, PointInTime),
        ):
            if not isinstance(value, expected):
                raise TypeError(f"{_SUBJECT} {name} must be a {expected.__name__}.")

        frozen = self._freeze.execute(
            contract, _DAILY, Strategy(strategy_identity), available_through
        )
        # Run even without a new decision: earlier frozen decisions still execute.
        run = self._run.execute(
            FuturesForwardResearchRecordQuery(contract, _DAILY),
            portfolio_identity,
            strategy_identity,
            target_contracts,
            available_through,
        )
        return FuturesPaperTradingSessionResult(
            frozen_record=frozen, run=run, report=self._report.execute(run)
        )
