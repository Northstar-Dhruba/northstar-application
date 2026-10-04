"""Application read-only analysis of one futures strategy on one contract.

The analysis composes existing use cases at one explicit cutoff and adds no
formula of its own:

- CalculateFuturesPaperPerformanceUseCase -- the paper portfolio's decisions,
  execution, trades, canonical P&L, equity curve and drawdown;
- RunFuturesHistoricalResearchUseCase -- the same built-in strategy replayed
  over the contract's stored daily history, each decision measured at every
  requested horizon, using only evidence available through the cutoff;
- CalculateFuturesHistoricalResearchMetricsUseCase -- per-horizon summaries,
  applied once to the whole run and once to each action's part of it.

The research strategy is ``Strategy(strategy_identity)`` with the built-in
analysis generator, exactly what the paper session freezes decisions with.

Forward returns are market movement after a decision, whichever way the
decision leaned. Grouping by BUY, SELL and HOLD selects which decisions are
summarised; it never turns a forward return into a directional or strategy
return, and nothing here judges a result.

Missing contract economics follow the snapshot precedent: the paper part is
absent and the dated contract is named, while research, which needs no
economics, is still returned. Nothing is persisted and no clock is read.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from northstar_core.foundation.value_objects import PointInTime, Timeframe
from northstar_core.futures import FuturesContract
from northstar_core.paper_trading import PaperPortfolioIdentity
from northstar_core.strategy import (
    FuturesAssetAnalysisGenerator,
    ResearchHorizon,
    Strategy,
    StrategyIdentity,
)

from northstar_application.application_services.calculate_futures_paper_performance import (
    CalculateFuturesPaperPerformanceUseCase,
    FuturesPaperPerformance,
)
from northstar_application.application_services.calculate_futures_research_metrics import (
    CalculateFuturesHistoricalResearchMetricsUseCase,
    FuturesHistoricalResearchStrategyHorizonMetrics,
)
from northstar_application.application_services.run_futures_historical_research import (
    FuturesHistoricalResearchRun,
    RunFuturesHistoricalResearchUseCase,
)
from northstar_application.application_services.value_futures_paper_portfolio import (
    FuturesContractEconomicsNotFoundError,
)
from northstar_application.ports import (
    FuturesContractEconomicsRepository,
    FuturesForwardResearchRecordRepository,
    FuturesHistoricalMarketDataRepository,
    FuturesPaperFillRepository,
    FuturesPaperOrderRepository,
)

_DAILY = Timeframe("1d")
_ACTIONS = ("BUY", "SELL", "HOLD")
_SUBJECT = "CalculateFuturesAnalysisUseCase"


@dataclass(frozen=True, slots=True)
class FuturesResearchActionMetrics:
    """Historical research metrics of the decisions that took one action.

    ``decision_count`` is how many research decisions took ``action``;
    ``horizons`` summarises their outcomes per horizon, in run horizon order,
    with the same meaning as the whole run's metrics.
    """

    action: str
    decision_count: int
    horizons: tuple[FuturesHistoricalResearchStrategyHorizonMetrics, ...]

    def __post_init__(self) -> None:
        subject = "FuturesResearchActionMetrics"
        if self.action not in _ACTIONS:
            raise ValueError(f"{subject} action must be one of {', '.join(_ACTIONS)}.")
        count = self.decision_count
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(f"{subject} decision count must be a non-negative integer.")
        if not isinstance(self.horizons, tuple) or not all(
            isinstance(metrics, FuturesHistoricalResearchStrategyHorizonMetrics)
            for metrics in self.horizons
        ):
            raise TypeError(
                f"{subject} horizons must be a tuple of "
                "FuturesHistoricalResearchStrategyHorizonMetrics."
            )
        if any(metrics.total_count != count for metrics in self.horizons):
            raise ValueError(f"{subject} every horizon covers each of its decisions once.")


@dataclass(frozen=True, slots=True)
class FuturesAnalysis:
    """Paper performance and historical research of one strategy on one contract.

    Exactly one of ``performance`` and ``missing_economics`` is present.
    ``action_metrics`` always holds BUY, SELL and HOLD, in that order, and
    their decision counts partition ``research_decision_count``.
    """

    contract: FuturesContract
    strategy_identity: StrategyIdentity
    portfolio_identity: PaperPortfolioIdentity
    available_through: PointInTime
    horizons: tuple[ResearchHorizon, ...]
    performance: FuturesPaperPerformance | None
    missing_economics: FuturesContract | None
    research_decision_count: int
    research_metrics: tuple[FuturesHistoricalResearchStrategyHorizonMetrics, ...]
    action_metrics: tuple[FuturesResearchActionMetrics, ...]

    def __post_init__(self) -> None:
        subject = "FuturesAnalysis"
        for name, expected in (
            ("contract", FuturesContract),
            ("strategy_identity", StrategyIdentity),
            ("portfolio_identity", PaperPortfolioIdentity),
            ("available_through", PointInTime),
        ):
            if not isinstance(getattr(self, name), expected):
                raise TypeError(f"{subject} {name} must be a {expected.__name__}.")
        if (self.performance is None) == (self.missing_economics is None):
            raise ValueError(f"{subject} needs exactly one of performance or missing economics.")
        if self.performance is not None:
            if not isinstance(self.performance, FuturesPaperPerformance):
                raise TypeError(f"{subject} performance must be a FuturesPaperPerformance.")
            if (
                self.performance.contract != self.contract
                or self.performance.available_through.compare(self.available_through) != 0
            ):
                raise ValueError(f"{subject} performance must be of its contract and cutoff.")
        if tuple(metrics.action for metrics in self.action_metrics) != _ACTIONS:
            raise ValueError(f"{subject} action metrics must be BUY, SELL and HOLD in order.")
        if sum(m.decision_count for m in self.action_metrics) != self.research_decision_count:
            raise ValueError(f"{subject} action decision counts must partition the research.")
        if tuple(m.horizon for m in self.research_metrics) != self.horizons:
            raise ValueError(f"{subject} research metrics must follow the horizons.")


class CalculateFuturesAnalysisUseCase:
    """Compose paper performance and historical research at one explicit cutoff."""

    def __init__(
        self,
        forward_repository: FuturesForwardResearchRecordRepository,
        order_repository: FuturesPaperOrderRepository,
        fill_repository: FuturesPaperFillRepository,
        market_repository: FuturesHistoricalMarketDataRepository,
        economics_repository: FuturesContractEconomicsRepository,
        analysis_generator: FuturesAssetAnalysisGenerator,
    ) -> None:
        if not isinstance(analysis_generator, FuturesAssetAnalysisGenerator):
            raise TypeError(
                f"{_SUBJECT} analysis_generator must be a FuturesAssetAnalysisGenerator."
            )
        # The composed use cases validate their own repositories.
        self._performance = CalculateFuturesPaperPerformanceUseCase(
            forward_repository,
            order_repository,
            fill_repository,
            market_repository,
            economics_repository,
        )
        self._research = RunFuturesHistoricalResearchUseCase(market_repository, analysis_generator)
        self._metrics = CalculateFuturesHistoricalResearchMetricsUseCase()

    def execute(
        self,
        contract: FuturesContract,
        strategy_identity: StrategyIdentity,
        portfolio_identity: PaperPortfolioIdentity,
        horizons: tuple[ResearchHorizon, ...],
        available_through: PointInTime,
    ) -> FuturesAnalysis:
        """Return the analysis as of ``available_through`` at the given horizons."""
        if not isinstance(strategy_identity, StrategyIdentity):
            raise TypeError(f"{_SUBJECT} strategy identity must be a StrategyIdentity.")
        try:
            performance = self._performance.execute(
                contract, strategy_identity, portfolio_identity, available_through
            )
            missing = None
        except FuturesContractEconomicsNotFoundError as error:
            performance, missing = None, error.contract

        run = self._research.execute(
            contract, _DAILY, Strategy(strategy_identity), horizons, available_through
        )
        return FuturesAnalysis(
            contract=contract,
            strategy_identity=strategy_identity,
            portfolio_identity=portfolio_identity,
            available_through=available_through,
            horizons=horizons,
            performance=performance,
            missing_economics=missing,
            research_decision_count=len(run.analysis_results),
            research_metrics=self._metrics.execute(run),
            action_metrics=tuple(self._action_metrics(run, action) for action in _ACTIONS),
        )

    def _action_metrics(
        self, run: FuturesHistoricalResearchRun, action: str
    ) -> FuturesResearchActionMetrics:
        """Summarise one action's decisions with the run's own metrics.

        The part of the run is itself a valid run -- the same subject, cutoff
        and horizons, with only that action's results and their outcomes, in
        their original order -- so the existing metrics apply unchanged.
        """
        unknown = {result.recommendation.action.value for result in run.analysis_results} - set(
            _ACTIONS
        )
        if unknown:
            raise ValueError(f"{_SUBJECT} cannot group recommendation actions {sorted(unknown)}.")
        part = replace(
            run,
            analysis_results=tuple(
                result
                for result in run.analysis_results
                if result.recommendation.action.value == action
            ),
            outcomes=tuple(
                outcome for outcome in run.outcomes if outcome.recommendation.action.value == action
            ),
        )
        return FuturesResearchActionMetrics(
            action=action,
            decision_count=len(part.analysis_results),
            horizons=self._metrics.execute(part),
        )
