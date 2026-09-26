"""Factual execution counts of one futures paper-trading run.

Metrics count what the run's decisions did: how many declined to act and why,
how many produced orders, how many of those filled by the cutoff, how many
whole contracts the fills bought and sold, and the resulting net exposure. They
express no return, profit and loss, win rate, notional or margin.

Executed side comes from each fill's intent, never from the research action: a
research BUY against an over-target long is a SELL trade and counts as sold
contracts. A pending order contributes no executed contracts.

Metrics are query-scoped, following the paper and forward precedents: one entry
per contract the run's decisions concern, which for one query is at most one.
Positions the portfolio holds in other contracts from earlier history are not
metrics of this run; the report exposes them through the run's portfolio.
"""

from __future__ import annotations

from dataclasses import dataclass

from northstar_core.futures import FuturesContract
from northstar_core.paper_trading import OrderSide
from northstar_core.strategy import StrategyIdentity

from northstar_application.application_services.create_futures_execution_intent import (
    FuturesExecutionIntentNoIntentReason,
)
from northstar_application.application_services.run_futures_paper_trading import (
    FuturesPaperTradingRun,
)
from northstar_application.application_services.run_futures_paper_trading_decision import (
    FuturesPaperTradingDecisionResult,
)

_COUNT_FIELDS = (
    "decision_count",
    "hold_count",
    "target_already_met_count",
    "order_count",
    "filled_order_count",
    "pending_order_count",
    "contracts_bought",
    "contracts_sold",
)


@dataclass(frozen=True, slots=True)
class FuturesPaperTradingStrategyContractMetrics:
    """What one strategy's decisions executed on one concrete futures contract.

    Every decision is exactly one of a hold, a target already met, or an order,
    and every order is exactly one of filled or pending at the run cutoff.
    ``current_net_contracts`` is the signed exposure in the run's final
    portfolio, zero when flat.
    """

    contract: FuturesContract
    strategy_identity: StrategyIdentity
    decision_count: int
    hold_count: int
    target_already_met_count: int
    order_count: int
    filled_order_count: int
    pending_order_count: int
    contracts_bought: int
    contracts_sold: int
    current_net_contracts: int

    def __post_init__(self) -> None:
        subject = "FuturesPaperTradingStrategyContractMetrics"
        if not isinstance(self.contract, FuturesContract):
            raise TypeError(f"{subject} contract must be a FuturesContract.")
        if not isinstance(self.strategy_identity, StrategyIdentity):
            raise TypeError(f"{subject} strategy identity must be a StrategyIdentity.")
        for name in (*_COUNT_FIELDS, "current_net_contracts"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{subject} {name} must be an integer.")
        for name in _COUNT_FIELDS:
            if getattr(self, name) < 0:
                raise ValueError(f"{subject} {name} cannot be negative.")

        if self.order_count != self.filled_order_count + self.pending_order_count:
            raise ValueError(f"{subject} orders must be exactly filled or pending.")
        if self.decision_count != (
            self.hold_count + self.target_already_met_count + self.order_count
        ):
            raise ValueError(
                f"{subject} decisions must be exactly holds, targets already met, or orders."
            )


class CalculateFuturesPaperTradingMetricsUseCase:
    """Count one futures paper-trading run's execution facts per contract."""

    def execute(
        self, run: FuturesPaperTradingRun
    ) -> tuple[FuturesPaperTradingStrategyContractMetrics, ...]:
        """Return one metrics value per decided contract, by contract natural key."""
        if not isinstance(run, FuturesPaperTradingRun):
            raise TypeError(
                "CalculateFuturesPaperTradingMetricsUseCase run must be a FuturesPaperTradingRun."
            )

        grouped: dict[FuturesContract, list[FuturesPaperTradingDecisionResult]] = {}
        for result in run.results:
            grouped.setdefault(result.record.contract, []).append(result)

        return tuple(
            self._metrics(run, contract, grouped[contract])
            for contract in sorted(grouped, key=lambda contract: contract.natural_key)
        )

    @staticmethod
    def _metrics(
        run: FuturesPaperTradingRun,
        contract: FuturesContract,
        results: list[FuturesPaperTradingDecisionResult],
    ) -> FuturesPaperTradingStrategyContractMetrics:
        reasons = [result.decision.no_intent_reason for result in results]
        orders = {result.order.identity for result in results if result.order is not None}
        fills = [result.fill for result in results if result.fill is not None]
        position = run.portfolio.get_position(contract)
        return FuturesPaperTradingStrategyContractMetrics(
            contract=contract,
            strategy_identity=run.strategy_identity,
            decision_count=len(results),
            hold_count=reasons.count(FuturesExecutionIntentNoIntentReason.HOLD),
            target_already_met_count=reasons.count(
                FuturesExecutionIntentNoIntentReason.TARGET_ALREADY_MET
            ),
            order_count=len(orders),
            filled_order_count=len(fills),
            pending_order_count=len(orders) - len(fills),
            contracts_bought=sum(f.contracts.value for f in fills if f.side is OrderSide.BUY),
            contracts_sold=sum(f.contracts.value for f in fills if f.side is OrderSide.SELL),
            current_net_contracts=0 if position is None else position.net_contracts,
        )
