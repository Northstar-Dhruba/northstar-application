"""Application orchestration of one frozen futures decision through paper trading.

For one frozen forward record and one paper portfolio, a run:

1. verifies the record is exactly the one frozen in the forward repository;
2. loads and validates the portfolio's persisted orders and fills, including
   that every one belongs to the record's strategy;
3. settles every pending order decided at or before ``available_through``
   whose next stored daily bar is now visible, and persists those fills;
4. folds the fills into the portfolio as of the decision instant;
5. applies the target-position policy to that pre-decision portfolio;
6. persists the decision's deterministic order when the policy intends one;
7. settles that order immediately if its next bar is already visible.

Why retries converge
--------------------
The order identity is derived from the frozen record and the portfolio, never
from side or size, and a fill always lands strictly after its decision. The
pre-decision portfolio therefore never contains the decision's own fill: a
retry recomputes the same intent, stores an equal order idempotently and finds
the equal fill already settled. If pre-decision history changed so the policy
now wants a different trade -- or no trade -- under the same order identity,
the run raises FuturesPaperOrderConflictError instead of rewriting history. A
back-filled earlier bar that would move a persisted fill raises
FuturesPaperFillConflictError the same way.

Accepted limitations
--------------------
No-action outcomes are not persisted. If a first run produced no order and
pre-decision history later changes, a re-run may produce an order for that
decision. There is no cross-port transaction either: each store call commits
independently. An order persisted before a failed fill write simply remains
pending and a later run settles it, and fills settled for older orders stay
valid facts even if the current decision then fails.

The run reads no clock, touches no provider or calendar, and calculates no
profit and loss. Futures paper trading is session-daily only.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cmp_to_key

from northstar_core.foundation.value_objects import PointInTime
from northstar_core.paper_trading import (
    FuturesContractCount,
    FuturesPaperFill,
    FuturesPaperOrder,
    FuturesPaperPortfolio,
    PaperOrderIdentity,
    PaperPortfolioIdentity,
)
from northstar_core.strategy import StrategyIdentity

from northstar_application.application_services.build_futures_paper_portfolio import (
    BuildFuturesPaperPortfolioUseCase,
)
from northstar_application.application_services.create_futures_execution_intent import (
    CreateFuturesExecutionIntentUseCase,
    FuturesExecutionIntentDecision,
    FuturesExecutionIntentNoIntentReason,
)
from northstar_application.application_services.futures_forward_research_record import (
    FuturesForwardResearchRecord,
)
from northstar_application.application_services.futures_paper_execution_identities import (
    FuturesPaperExecutionIdentityService,
)
from northstar_application.application_services.record_forward_research_decision import (
    ForwardResearchContractViolationError,
)
from northstar_application.application_services.run_futures_forward_research import (
    _canonical_order as _canonical_record_order,
)
from northstar_application.application_services.simulate_futures_paper_order_fill import (
    SimulateFuturesPaperOrderFillUseCase,
)
from northstar_application.ports import (
    FuturesForwardResearchRecordQuery,
    FuturesForwardResearchRecordRepository,
    FuturesHistoricalMarketDataRepository,
    FuturesPaperFillConflictError,
    FuturesPaperFillQuery,
    FuturesPaperFillRepository,
    FuturesPaperFillStore,
    FuturesPaperOrderConflictError,
    FuturesPaperOrderQuery,
    FuturesPaperOrderRepository,
    FuturesPaperOrderStore,
)

_HOLD_ACTION = "HOLD"


class FuturesPaperTradingContractViolationError(ValueError):
    """Raised when a futures paper-trading port violates its contract."""


def _compare_text(left: str, right: str) -> int:
    return (left > right) - (left < right)


def _compare_orders(left: FuturesPaperOrder, right: FuturesPaperOrder) -> int:
    instant = left.intent.decided_at.compare(right.intent.decided_at)
    return instant or _compare_text(left.identity.identity, right.identity.identity)


def _compare_fills(left: FuturesPaperFill, right: FuturesPaperFill) -> int:
    instant = left.filled_at.compare(right.filled_at)
    return instant or _compare_text(left.order_identity.identity, right.order_identity.identity)


def _validate_forward_output(records: object, query: FuturesForwardResearchRecordQuery) -> None:
    """Require forward repository output to honour its complete contract."""
    if not isinstance(records, tuple):
        raise ForwardResearchContractViolationError(
            "FuturesForwardResearchRecordRepository must return a tuple of "
            "FuturesForwardResearchRecord."
        )
    seen: set[tuple] = set()
    previous: FuturesForwardResearchRecord | None = None
    for index, item in enumerate(records):
        if not isinstance(item, FuturesForwardResearchRecord):
            raise ForwardResearchContractViolationError(
                f"FuturesForwardResearchRecordRepository record {index} must be a "
                "FuturesForwardResearchRecord."
            )
        if item.contract != query.contract or item.timeframe != query.timeframe:
            raise ForwardResearchContractViolationError(
                f"FuturesForwardResearchRecordRepository record {index} is not for the "
                "queried contract and timeframe."
            )
        if item.natural_key in seen:
            raise ForwardResearchContractViolationError(
                f"FuturesForwardResearchRecordRepository record {index} repeats a natural key."
            )
        seen.add(item.natural_key)
        if previous is not None and not _canonical_record_order(previous, item):
            raise ForwardResearchContractViolationError(
                "FuturesForwardResearchRecordRepository records must be ordered by decision "
                f"instant, then strategy identity; record {index} is out of order."
            )
        previous = item


def _load_orders(
    repository: FuturesPaperOrderRepository, portfolio_identity: PaperPortfolioIdentity
) -> tuple[FuturesPaperOrder, ...]:
    orders = repository.get_orders(FuturesPaperOrderQuery(portfolio_identity))
    if not isinstance(orders, tuple):
        raise FuturesPaperTradingContractViolationError(
            "FuturesPaperOrderRepository must return a tuple of FuturesPaperOrder."
        )
    previous: FuturesPaperOrder | None = None
    for index, order in enumerate(orders):
        if not isinstance(order, FuturesPaperOrder):
            raise FuturesPaperTradingContractViolationError(
                f"FuturesPaperOrderRepository order {index} must be a FuturesPaperOrder."
            )
        if order.intent.portfolio_identity != portfolio_identity:
            raise FuturesPaperTradingContractViolationError(
                f"FuturesPaperOrderRepository order {index} belongs to another portfolio."
            )
        # Strictly increasing, which also rejects a repeated order identity.
        if previous is not None and _compare_orders(previous, order) >= 0:
            raise FuturesPaperTradingContractViolationError(
                "FuturesPaperOrderRepository orders must be uniquely ordered by decision "
                f"instant, then order identity; order {index} is out of order."
            )
        previous = order
    return orders


def _load_fills(
    repository: FuturesPaperFillRepository, portfolio_identity: PaperPortfolioIdentity
) -> tuple[FuturesPaperFill, ...]:
    fills = repository.get_fills(FuturesPaperFillQuery(portfolio_identity))
    if not isinstance(fills, tuple):
        raise FuturesPaperTradingContractViolationError(
            "FuturesPaperFillRepository must return a tuple of FuturesPaperFill."
        )
    seen: set[str] = set()
    previous: FuturesPaperFill | None = None
    for index, fill in enumerate(fills):
        if not isinstance(fill, FuturesPaperFill):
            raise FuturesPaperTradingContractViolationError(
                f"FuturesPaperFillRepository fill {index} must be a FuturesPaperFill."
            )
        if fill.portfolio_identity != portfolio_identity:
            raise FuturesPaperTradingContractViolationError(
                f"FuturesPaperFillRepository fill {index} belongs to another portfolio."
            )
        if fill.identity.identity in seen:
            raise FuturesPaperTradingContractViolationError(
                f"FuturesPaperFillRepository fill {index} repeats a fill identity."
            )
        seen.add(fill.identity.identity)
        # Strictly increasing, which also rejects two fills for one order.
        if previous is not None and _compare_fills(previous, fill) >= 0:
            raise FuturesPaperTradingContractViolationError(
                "FuturesPaperFillRepository fills must be uniquely ordered by fill instant, "
                f"then order identity; fill {index} is out of order."
            )
        previous = fill
    return fills


def _load_history(
    order_repository: FuturesPaperOrderRepository,
    fill_repository: FuturesPaperFillRepository,
    portfolio_identity: PaperPortfolioIdentity,
    strategy_identity: StrategyIdentity,
) -> tuple[tuple[FuturesPaperOrder, ...], tuple[FuturesPaperFill, ...]]:
    """Load one portfolio's orders and fills and require one strategy's coherent history."""
    orders = _load_orders(order_repository, portfolio_identity)
    fills = _load_fills(fill_repository, portfolio_identity)
    for order in orders:
        if order.intent.strategy_identity != strategy_identity:
            raise ValueError(
                f"Futures paper portfolio {portfolio_identity} already holds orders for "
                f"strategy {order.intent.strategy_identity}; one paper portfolio belongs to "
                f"one strategy, not {strategy_identity}."
            )
    by_identity = {order.identity: order for order in orders}
    for fill in fills:
        order = by_identity.get(fill.order_identity)
        if order is None:
            raise FuturesPaperTradingContractViolationError(
                f"FuturesPaperFillRepository fill {fill.identity} has no loaded order."
            )
        if fill.intent != order.intent:
            raise FuturesPaperTradingContractViolationError(
                f"FuturesPaperFillRepository fill {fill.identity} does not carry its "
                "order's intent."
            )
    return orders, fills


@dataclass(frozen=True, slots=True)
class FuturesPaperTradingDecisionResult:
    """The outcome of running one frozen futures decision through paper trading.

    The outcome's state is its shape:

    - no ``order``: no action (``decision.no_intent_reason`` says why);
    - ``order`` without ``fill``: the order is pending at ``available_through``;
    - ``order`` with ``fill``: the order is filled.

    ``portfolio_before`` is the portfolio as of the decision instant -- the
    exposure the policy was judged against -- and never contains this
    decision's own fill.
    """

    record: FuturesForwardResearchRecord
    portfolio_before: FuturesPaperPortfolio
    decision: FuturesExecutionIntentDecision
    order: FuturesPaperOrder | None
    fill: FuturesPaperFill | None
    available_through: PointInTime

    def __post_init__(self) -> None:
        self._validate_types()
        record = self.record

        if self.available_through.compare(record.decision_instant) < 0:
            raise ValueError(
                "FuturesPaperTradingDecisionResult available-through cannot precede the "
                "decision instant."
            )
        portfolio = self.portfolio_before
        if portfolio.strategy_identity != record.strategy_identity:
            raise ValueError(
                "FuturesPaperTradingDecisionResult portfolio must belong to the decision strategy."
            )
        if portfolio.as_of.compare(record.decision_instant) != 0:
            raise ValueError(
                "FuturesPaperTradingDecisionResult portfolio must be as of the decision instant."
            )

        self._validate_decision_follows_the_action()
        self._validate_execution()

    def _validate_types(self) -> None:
        subject = "FuturesPaperTradingDecisionResult"
        if not isinstance(self.record, FuturesForwardResearchRecord):
            raise TypeError(f"{subject} record must be a FuturesForwardResearchRecord.")
        if not isinstance(self.portfolio_before, FuturesPaperPortfolio):
            raise TypeError(f"{subject} portfolio before must be a FuturesPaperPortfolio.")
        if not isinstance(self.decision, FuturesExecutionIntentDecision):
            raise TypeError(f"{subject} decision must be a FuturesExecutionIntentDecision.")
        if self.order is not None and not isinstance(self.order, FuturesPaperOrder):
            raise TypeError(f"{subject} order must be a FuturesPaperOrder or None.")
        if self.fill is not None and not isinstance(self.fill, FuturesPaperFill):
            raise TypeError(f"{subject} fill must be a FuturesPaperFill or None.")
        if not isinstance(self.available_through, PointInTime):
            raise TypeError(f"{subject} available-through must be a PointInTime.")

    def _validate_decision_follows_the_action(self) -> None:
        is_hold = self.record.result.recommendation.action.value == _HOLD_ACTION
        reason = self.decision.no_intent_reason
        if is_hold and reason is not FuturesExecutionIntentNoIntentReason.HOLD:
            raise ValueError(
                "FuturesPaperTradingDecisionResult must decline a HOLD decision as a hold."
            )
        if not is_hold and reason is FuturesExecutionIntentNoIntentReason.HOLD:
            raise ValueError(
                "FuturesPaperTradingDecisionResult cannot decline a directional decision as a hold."
            )

    def _validate_execution(self) -> None:
        intent = self.decision.intent
        if intent is None:
            if self.order is not None or self.fill is not None:
                raise ValueError(
                    "FuturesPaperTradingDecisionResult cannot carry an order or fill for a "
                    "decision that intended none."
                )
            return

        if self.order is None:
            raise ValueError(
                "FuturesPaperTradingDecisionResult must carry an order for an intended decision."
            )
        if self.order.intent != intent:
            raise ValueError(
                "FuturesPaperTradingDecisionResult order must carry the decision's own intent."
            )
        if intent.portfolio_identity != self.portfolio_before.identity:
            raise ValueError(
                "FuturesPaperTradingDecisionResult intent must belong to the portfolio."
            )
        if (
            intent.contract != self.record.contract
            or intent.strategy_identity != self.record.strategy_identity
            or intent.decided_at.compare(self.record.decision_instant) != 0
        ):
            raise ValueError(
                "FuturesPaperTradingDecisionResult intent must describe the record's decision."
            )

        if self.fill is None:
            return
        if self.fill.order_identity != self.order.identity or self.fill.intent != intent:
            raise ValueError("FuturesPaperTradingDecisionResult fill must fill its own order.")
        if self.fill.filled_at.compare(self.available_through) > 0:
            raise ValueError(
                "FuturesPaperTradingDecisionResult fill must be visible at available-through."
            )


class RunFuturesPaperTradingDecisionUseCase:
    """Run one frozen futures decision through settlement, policy and persistence."""

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
                    f"RunFuturesPaperTradingDecisionUseCase {name} must be a {expected.__name__}."
                )

        self._forward_repository = forward_repository
        self._order_store = order_store
        self._order_repository = order_repository
        self._fill_store = fill_store
        self._fill_repository = fill_repository
        self._simulate = SimulateFuturesPaperOrderFillUseCase(market_repository)
        self._policy = CreateFuturesExecutionIntentUseCase()
        self._identities = FuturesPaperExecutionIdentityService()
        self._build_portfolio = BuildFuturesPaperPortfolioUseCase()

    def execute(
        self,
        record: FuturesForwardResearchRecord,
        portfolio_identity: PaperPortfolioIdentity,
        target_contracts: FuturesContractCount,
        available_through: PointInTime,
    ) -> FuturesPaperTradingDecisionResult:
        """Run one frozen decision for one paper portfolio as of one evidence cutoff."""
        self._validate_inputs(record, portfolio_identity, target_contracts, available_through)
        self._verify_frozen(record)

        orders, fills = _load_history(
            self._order_repository,
            self._fill_repository,
            portfolio_identity,
            record.strategy_identity,
        )

        fills = self._settle_pending(orders, fills, available_through)

        portfolio_before = self._build_portfolio.execute(
            portfolio_identity, record.strategy_identity, fills, record.decision_instant
        )
        decision = self._policy.execute(record, portfolio_before, target_contracts)

        order_identity = self._identities.order_identity(record, portfolio_identity)
        existing = next((order for order in orders if order.identity == order_identity), None)

        if decision.intent is None:
            if existing is not None:
                raise FuturesPaperOrderConflictError(
                    "A futures paper order is already persisted for this decision, which now "
                    "intends no execution."
                )
            return FuturesPaperTradingDecisionResult(
                record=record,
                portfolio_before=portfolio_before,
                decision=decision,
                order=None,
                fill=None,
                available_through=available_through,
            )

        order = FuturesPaperOrder(identity=order_identity, intent=decision.intent)
        if existing is not None and existing != order:
            raise FuturesPaperOrderConflictError(
                "A different futures paper order is already persisted for this decision."
            )
        self._require_count(self._order_store.store((order,)), 1, "FuturesPaperOrderStore")

        known = next((fill for fill in fills if fill.order_identity == order_identity), None)
        fill = self._settle_current(order, known, available_through)

        return FuturesPaperTradingDecisionResult(
            record=record,
            portfolio_before=portfolio_before,
            decision=decision,
            order=order,
            fill=fill,
            available_through=available_through,
        )

    # -- forward record ----------------------------------------------------

    def _verify_frozen(self, record: FuturesForwardResearchRecord) -> None:
        """Require the supplied record to be exactly the persisted frozen decision."""
        query = FuturesForwardResearchRecordQuery(
            contract=record.contract, timeframe=record.timeframe
        )
        stored = self._forward_repository.get_records(query)
        _validate_forward_output(stored, query)

        candidate = next((item for item in stored if item.natural_key == record.natural_key), None)
        if candidate is None:
            raise ValueError(
                "RunFuturesPaperTradingDecisionUseCase can only execute a frozen forward "
                "research record; this record is not persisted."
            )
        if candidate != record:
            raise ValueError(
                "RunFuturesPaperTradingDecisionUseCase record differs from the frozen forward "
                "research record persisted under its natural key."
            )

    # -- settlement --------------------------------------------------------

    def _settle_pending(
        self,
        orders: tuple[FuturesPaperOrder, ...],
        fills: tuple[FuturesPaperFill, ...],
        available_through: PointInTime,
    ) -> tuple[FuturesPaperFill, ...]:
        """Fill every visible pending order, persist the fills and merge them in."""
        filled: set[PaperOrderIdentity] = {fill.order_identity for fill in fills}
        settled: list[FuturesPaperFill] = []
        for order in orders:
            if order.identity in filled:
                continue
            if order.intent.decided_at.compare(available_through) > 0:
                continue
            fill = self._simulate.execute(order, available_through)
            if fill is not None:
                settled.append(fill)

        if not settled:
            return fills
        self._require_count(
            self._fill_store.store(tuple(settled)), len(settled), "FuturesPaperFillStore"
        )
        return tuple(sorted((*fills, *settled), key=cmp_to_key(_compare_fills)))

    def _settle_current(
        self,
        order: FuturesPaperOrder,
        known: FuturesPaperFill | None,
        available_through: PointInTime,
    ) -> FuturesPaperFill | None:
        """Return this decision's fill, persisting it when newly visible.

        A fill already persisted for the order is authoritative: re-simulation
        must reproduce it exactly, or the run conflicts rather than moving it.
        """
        simulated = self._simulate.execute(order, available_through)
        if known is None:
            if simulated is None:
                return None
            self._require_count(self._fill_store.store((simulated,)), 1, "FuturesPaperFillStore")
            return simulated

        if simulated is None:
            if known.filled_at.compare(available_through) > 0:
                return None
            raise FuturesPaperFillConflictError(
                "The persisted futures paper fill for this decision is no longer reproducible "
                "from the visible market data."
            )
        if simulated != known:
            raise FuturesPaperFillConflictError(
                "A different futures paper fill is already persisted for this decision's order."
            )
        return known

    @staticmethod
    def _require_count(stored: object, expected: int, port: str) -> None:
        if isinstance(stored, bool) or not isinstance(stored, int):
            raise FuturesPaperTradingContractViolationError(
                f"{port} must return an integer count of accepted values."
            )
        if stored != expected:
            raise FuturesPaperTradingContractViolationError(
                f"{port} must accept exactly {expected}, but reported {stored}."
            )

    # -- validation --------------------------------------------------------

    @staticmethod
    def _validate_inputs(
        record: FuturesForwardResearchRecord,
        portfolio_identity: PaperPortfolioIdentity,
        target_contracts: FuturesContractCount,
        available_through: PointInTime,
    ) -> None:
        subject = "RunFuturesPaperTradingDecisionUseCase"
        if not isinstance(record, FuturesForwardResearchRecord):
            raise TypeError(f"{subject} record must be a FuturesForwardResearchRecord.")
        if not isinstance(portfolio_identity, PaperPortfolioIdentity):
            raise TypeError(f"{subject} portfolio identity must be a PaperPortfolioIdentity.")
        if not isinstance(target_contracts, FuturesContractCount):
            raise TypeError(f"{subject} target contracts must be a FuturesContractCount.")
        if not isinstance(available_through, PointInTime):
            raise TypeError(f"{subject} available-through must be a PointInTime.")
        if available_through.compare(record.decision_instant) < 0:
            raise ValueError(
                f"{subject} available-through {available_through} precedes the decision "
                f"instant {record.decision_instant}."
            )
