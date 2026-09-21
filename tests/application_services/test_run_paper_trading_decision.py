"""Tests for running one decision end to end through paper trading.

Every collaborator is a real Application use case. Only the two ports are
doubled, by an in-memory store and repository pair that honours the documented
persistence semantics. Recommendations come from the real strategy wherever the
action can be provoked from market evidence.
"""

from __future__ import annotations

from functools import cmp_to_key

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
    OrderSide,
    PaperFill,
    PaperPortfolioIdentity,
)
from northstar_core.strategy import (
    AssetAnalysisGenerator,
    MarketObservationContext,
    Strategy,
    StrategyIdentity,
)

from northstar_application.application_services import (
    AnalyzeAssetResult,
    AnalyzeMarketObservationContextService,
    BuildPaperPortfolioUseCase,
    CreateExecutionIntentUseCase,
    CreatePaperExecutionIdentitiesUseCase,
    ExecutionIntentNoIntentReason,
    InvalidPaperFillHistoryError,
    PaperTradingContractViolationError,
    PaperTradingResult,
    RunPaperTradingDecisionUseCase,
    SimulatePaperExecutionUseCase,
)
from northstar_application.ports import (
    PaperFillConflictError,
    PaperFillQuery,
    PaperFillRepository,
    PaperFillStore,
)

_USD = Currency("USD")
_LISTING = ListingReference(Symbol("AAPL"), ExchangeCode("NASDAQ"))
_STRATEGY = StrategyIdentity("alpha")
_PORTFOLIO = PaperPortfolioIdentity("paper-1")
_AT = PointInTime("2026-01-20T16:00:00Z")
_LATER = PointInTime("2026-01-25T16:00:00Z")


# ---------------------------------------------------------------------------
# Real strategy evidence
# ---------------------------------------------------------------------------


def _context(
    closes: list[int],
    latest: str,
    previous: str,
    *,
    volume: str = "5000",
    observed_at: PointInTime = _AT,
    listing: ListingReference = _LISTING,
) -> MarketObservationContext:
    return MarketObservationContext(
        listing,
        observed_at,
        Price(latest, _USD),
        Price(previous, _USD),
        Quantity(volume),
        Price("9000", _USD),
        Price("1", _USD),
        tuple(Price(str(close), _USD) for close in closes),
        tuple(Quantity("1000") for _ in range(20)),
    )


def _analyse(
    context: MarketObservationContext, strategy: StrategyIdentity = _STRATEGY
) -> AnalyzeAssetResult:
    return AnalyzeMarketObservationContextService(
        strategy=Strategy(strategy),
        analysis_generator=AssetAnalysisGenerator(),
    ).execute(context)


def _buy_result(**kwargs: object) -> AnalyzeAssetResult:
    """A rising series on elevated volume: the real strategy answers BUY."""
    return _analyse(_context(list(range(180, 200)), "250", "199", **kwargs))


def _sell_result(**kwargs: object) -> AnalyzeAssetResult:
    """A declining series on elevated volume: the real strategy answers SELL."""
    return _analyse(_context(list(range(200, 180, -1)), "150", "181", **kwargs))


def _hold_result(**kwargs: object) -> AnalyzeAssetResult:
    """A flat series on thin volume: the real strategy answers HOLD."""
    return _analyse(_context([100] * 20, "100", "100", volume="100", **kwargs))


# ---------------------------------------------------------------------------
# Port doubles
# ---------------------------------------------------------------------------


def _compare(left: PaperFill, right: PaperFill) -> int:
    instant = left.filled_at.compare(right.filled_at)
    if instant:
        return instant
    a, b = left.identity.identity, right.identity.identity
    return (a > b) - (a < b)


class InMemoryFills:
    """Shared storage honouring the append-only persistence contract."""

    def __init__(self) -> None:
        self.by_fill: dict[str, PaperFill] = {}
        self.by_order: dict[str, str] = {}


_HONEST_COUNT = object()
"""Sentinel: None is a value a misbehaving store might genuinely return."""


class MemoryStore(PaperFillStore):
    def __init__(self, state: InMemoryFills, return_value: object = _HONEST_COUNT) -> None:
        self.state = state
        self.calls: list[tuple[PaperFill, ...]] = []
        self._return_value = return_value

    def store(self, fills: tuple[PaperFill, ...]) -> int:
        self.calls.append(fills)
        if self._return_value is not _HONEST_COUNT:
            return self._return_value
        if not fills:
            return 0
        for fill in fills:
            key = fill.identity.identity
            existing = self.state.by_fill.get(key)
            if existing is not None:
                if existing != fill:
                    raise PaperFillConflictError(
                        "A different paper fill is already stored under this fill identity."
                    )
                continue
            attached = self.state.by_order.get(fill.order_identity.identity)
            if attached is not None and attached != key:
                raise PaperFillConflictError(
                    "This order is already attached to a different paper fill."
                )
            self.state.by_fill[key] = fill
            self.state.by_order[fill.order_identity.identity] = key
        return len(fills)


class MemoryRepository(PaperFillRepository):
    def __init__(self, state: InMemoryFills, override: object = None) -> None:
        self.state = state
        self.queries: list[PaperFillQuery] = []
        self._override = override

    def get_fills(self, query: PaperFillQuery) -> tuple[PaperFill, ...]:
        self.queries.append(query)
        if self._override is not None:
            return self._override
        owned = [
            fill
            for fill in self.state.by_fill.values()
            if fill.portfolio_identity == query.portfolio_identity
        ]
        return tuple(sorted(owned, key=cmp_to_key(_compare)))


def _use_case(
    state: InMemoryFills | None = None,
    *,
    store: PaperFillStore | None = None,
    repository: PaperFillRepository | None = None,
) -> RunPaperTradingDecisionUseCase:
    shared = state if state is not None else InMemoryFills()
    return RunPaperTradingDecisionUseCase(
        repository if repository is not None else MemoryRepository(shared),
        store if store is not None else MemoryStore(shared),
    )


def _run(
    use_case: RunPaperTradingDecisionUseCase,
    result: AnalyzeAssetResult,
    quantity: str = "10",
) -> PaperTradingResult:
    return use_case.execute(result, _PORTFOLIO, Quantity(quantity))


# ---------------------------------------------------------------------------
# The real strategy reaches every action
# ---------------------------------------------------------------------------


def test_the_real_strategy_produces_buy_hold_and_sell() -> None:
    actions = {
        _buy_result().recommendation.action.value,
        _hold_result().recommendation.action.value,
        _sell_result().recommendation.action.value,
    }

    assert actions == {"BUY", "HOLD", "SELL"}


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def test_buy_from_an_empty_portfolio_opens_a_position() -> None:
    state = InMemoryFills()

    outcome = _run(_use_case(state), _buy_result(), quantity="10")

    assert outcome.decision.has_intent
    assert outcome.decision.intent.side is OrderSide.BUY
    assert outcome.execution is not None
    assert outcome.execution.fill.quantity == Quantity("10")
    assert outcome.execution.fill.price == Price("250", _USD)
    assert len(state.by_fill) == 1
    assert outcome.portfolio.get_position(_LISTING).quantity == Quantity("10")


def test_hold_stores_nothing_and_leaves_the_portfolio_unchanged() -> None:
    state = InMemoryFills()
    store = MemoryStore(state)
    use_case = _use_case(state, store=store)
    _run(use_case, _buy_result(), quantity="10")

    outcome = _run(use_case, _hold_result(observed_at=_LATER))

    assert outcome.decision.no_intent_reason is ExecutionIntentNoIntentReason.HOLD
    assert outcome.execution is None
    assert len(store.calls) == 1  # only the earlier BUY
    assert outcome.portfolio.get_position(_LISTING).quantity == Quantity("10")


def test_sell_against_a_position_reduces_it() -> None:
    state = InMemoryFills()
    use_case = _use_case(state)
    _run(use_case, _buy_result(), quantity="10")

    outcome = _run(use_case, _sell_result(observed_at=_LATER), quantity="4")

    assert outcome.decision.intent.side is OrderSide.SELL
    assert outcome.portfolio.get_position(_LISTING).quantity == Quantity("6")
    assert len(state.by_fill) == 2


def test_a_full_sell_removes_the_position() -> None:
    state = InMemoryFills()
    use_case = _use_case(state)
    _run(use_case, _buy_result(), quantity="10")

    outcome = _run(use_case, _sell_result(observed_at=_LATER), quantity="10")

    assert outcome.portfolio.positions == ()
    assert outcome.portfolio.get_position(_LISTING) is None


def test_an_insufficient_sell_produces_an_explicit_no_intent() -> None:
    state = InMemoryFills()
    store = MemoryStore(state)

    outcome = _run(_use_case(state, store=store), _sell_result(), quantity="10")

    assert outcome.decision.has_intent is False
    assert outcome.decision.no_intent_reason is ExecutionIntentNoIntentReason.INSUFFICIENT_POSITION
    assert outcome.execution is None
    assert store.calls == []
    assert outcome.portfolio.positions == ()


def test_overselling_an_existing_position_produces_no_intent() -> None:
    state = InMemoryFills()
    use_case = _use_case(state)
    _run(use_case, _buy_result(), quantity="5")

    outcome = _run(use_case, _sell_result(observed_at=_LATER), quantity="6")

    assert outcome.decision.no_intent_reason is ExecutionIntentNoIntentReason.INSUFFICIENT_POSITION
    assert len(state.by_fill) == 1


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


def test_rerunning_an_identical_buy_changes_nothing() -> None:
    state = InMemoryFills()
    use_case = _use_case(state)
    result = _buy_result()

    first = _run(use_case, result, quantity="10")
    second = _run(use_case, result, quantity="10")

    assert first == second
    assert len(state.by_fill) == 1
    assert second.portfolio.get_position(_LISTING).quantity == Quantity("10")


def test_rerunning_an_identical_sell_succeeds_idempotently() -> None:
    """The decisive retry case: the persisted SELL must not block its own retry."""
    state = InMemoryFills()
    use_case = _use_case(state)
    _run(use_case, _buy_result(), quantity="10")
    sell = _sell_result(observed_at=_LATER)

    first = _run(use_case, sell, quantity="10")
    second = _run(use_case, sell, quantity="10")

    assert first == second
    assert len(state.by_fill) == 2
    assert second.portfolio.positions == ()


def test_a_repeated_sell_is_judged_against_the_pre_decision_portfolio() -> None:
    """Without excluding its own fill, the retry would see a flat holding and refuse."""
    state = InMemoryFills()
    use_case = _use_case(state)
    _run(use_case, _buy_result(), quantity="10")
    sell = _sell_result(observed_at=_LATER)
    _run(use_case, sell, quantity="10")

    retry = _run(use_case, sell, quantity="10")

    assert retry.decision.has_intent
    assert retry.decision.no_intent_reason is None
    assert retry.execution is not None


def test_a_partial_sell_retry_sees_the_original_holding() -> None:
    state = InMemoryFills()
    use_case = _use_case(state)
    _run(use_case, _buy_result(), quantity="10")
    sell = _sell_result(observed_at=_LATER)
    _run(use_case, sell, quantity="7")

    retry = _run(use_case, sell, quantity="7")

    assert retry.decision.has_intent
    assert retry.portfolio.get_position(_LISTING).quantity == Quantity("3")


def test_a_changed_quantity_at_one_decision_boundary_conflicts() -> None:
    state = InMemoryFills()
    use_case = _use_case(state)
    result = _buy_result()
    _run(use_case, result, quantity="10")

    with pytest.raises(PaperFillConflictError):
        _run(use_case, result, quantity="11")

    assert len(state.by_fill) == 1
    assert state.by_fill[next(iter(state.by_fill))].quantity == Quantity("10")


def test_a_changed_action_at_one_decision_boundary_conflicts() -> None:
    """Same portfolio, listing, strategy and instant, but now advising a sale."""
    state = InMemoryFills()
    use_case = _use_case(state)
    _run(use_case, _buy_result(), quantity="10")

    with pytest.raises(PaperFillConflictError):
        _run(use_case, _sell_result(), quantity="5")

    assert len(state.by_fill) == 1


def test_a_decision_that_now_intends_nothing_conflicts_with_its_persisted_fill() -> None:
    state = InMemoryFills()
    store = MemoryStore(state)
    use_case = _use_case(state, store=store)
    _run(use_case, _buy_result(), quantity="10")

    with pytest.raises(PaperFillConflictError, match="now intends no execution"):
        _run(use_case, _hold_result(), quantity="10")

    assert len(store.calls) == 1


def test_identities_are_stable_across_retries() -> None:
    state = InMemoryFills()
    use_case = _use_case(state)
    result = _buy_result()

    first = _run(use_case, result, quantity="10")
    second = _run(use_case, result, quantity="10")

    assert first.execution.order.identity == second.execution.order.identity
    assert first.execution.fill.identity == second.execution.fill.identity


# ---------------------------------------------------------------------------
# Temporal correctness
# ---------------------------------------------------------------------------


def test_later_fills_do_not_contaminate_an_older_decision() -> None:
    state = InMemoryFills()
    use_case = _use_case(state)
    _run(use_case, _buy_result(), quantity="10")
    _run(use_case, _buy_result(observed_at=_LATER), quantity="5")

    rebuilt = _run(use_case, _buy_result(), quantity="10")

    assert rebuilt.portfolio.as_of == _AT
    assert rebuilt.portfolio.get_position(_LISTING).quantity == Quantity("10")
    assert len(state.by_fill) == 2


def test_a_sell_is_judged_only_on_holdings_at_its_decision_instant() -> None:
    state = InMemoryFills()
    use_case = _use_case(state)
    _run(use_case, _buy_result(observed_at=_LATER), quantity="100")

    outcome = _run(use_case, _sell_result(), quantity="1")

    assert outcome.decision.no_intent_reason is ExecutionIntentNoIntentReason.INSUFFICIENT_POSITION


def test_offset_equivalent_decision_instants_behave_identically() -> None:
    offset = PointInTime("2026-01-20T21:30:00+05:30")
    state = InMemoryFills()
    use_case = _use_case(state)

    _run(use_case, _buy_result(), quantity="10")
    retry = _run(use_case, _buy_result(observed_at=offset), quantity="10")

    assert offset.compare(_AT) == 0
    assert len(state.by_fill) == 1
    assert retry.execution.fill.filled_at == _AT


def test_no_market_data_port_is_consulted() -> None:
    """Execution prices come from the decision evidence, never a lookup."""
    import ast
    from pathlib import Path

    from northstar_application.application_services import run_paper_trading_decision

    tree = ast.parse(Path(run_paper_trading_decision.__file__).read_text(encoding="utf-8"))
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
        for alias in node.names
    }

    assert not {name for name in imported if "MarketData" in name or "Historical" in name}


# ---------------------------------------------------------------------------
# Port contracts
# ---------------------------------------------------------------------------


def test_ports_must_be_injected() -> None:
    state = InMemoryFills()

    with pytest.raises(TypeError, match="fill repository cannot be None"):
        RunPaperTradingDecisionUseCase(None, MemoryStore(state))
    with pytest.raises(TypeError, match="fill repository must be a PaperFillRepository"):
        RunPaperTradingDecisionUseCase("repository", MemoryStore(state))
    with pytest.raises(TypeError, match="fill store cannot be None"):
        RunPaperTradingDecisionUseCase(MemoryRepository(state), None)
    with pytest.raises(TypeError, match="fill store must be a PaperFillStore"):
        RunPaperTradingDecisionUseCase(MemoryRepository(state), "store")


def test_dependency_free_use_cases_default_but_are_type_checked() -> None:
    state = InMemoryFills()

    assert RunPaperTradingDecisionUseCase(MemoryRepository(state), MemoryStore(state))
    with pytest.raises(TypeError, match="simulate_execution must be a"):
        RunPaperTradingDecisionUseCase(
            MemoryRepository(state), MemoryStore(state), simulate_execution="simulate"
        )
    with pytest.raises(TypeError, match="build_portfolio must be a"):
        RunPaperTradingDecisionUseCase(
            MemoryRepository(state), MemoryStore(state), build_portfolio="build"
        )


def test_explicit_dependencies_are_accepted() -> None:
    state = InMemoryFills()
    use_case = RunPaperTradingDecisionUseCase(
        MemoryRepository(state),
        MemoryStore(state),
        CreateExecutionIntentUseCase(),
        CreatePaperExecutionIdentitiesUseCase(),
        SimulatePaperExecutionUseCase(),
        BuildPaperPortfolioUseCase(),
    )

    assert _run(use_case, _buy_result()).execution is not None


def test_execute_validates_its_inputs() -> None:
    use_case = _use_case()

    with pytest.raises(TypeError, match="result cannot be None"):
        use_case.execute(None, _PORTFOLIO, Quantity("1"))
    with pytest.raises(TypeError, match="result must be an AnalyzeAssetResult"):
        use_case.execute("result", _PORTFOLIO, Quantity("1"))
    with pytest.raises(TypeError, match="portfolio identity must be a PaperPortfolioIdentity"):
        use_case.execute(_buy_result(), "paper-1", Quantity("1"))
    with pytest.raises(TypeError, match="quantity must be a Quantity"):
        use_case.execute(_buy_result(), _PORTFOLIO, 10)


@pytest.mark.parametrize("returned", ["1", 1.0, None])
def test_a_non_integer_store_count_is_rejected(returned: object) -> None:
    state = InMemoryFills()
    store = MemoryStore(state, return_value=returned)

    with pytest.raises(PaperTradingContractViolationError, match="integer count"):
        _run(_use_case(state, store=store), _buy_result())


def test_a_boolean_store_count_is_rejected() -> None:
    state = InMemoryFills()
    store = MemoryStore(state, return_value=True)

    with pytest.raises(PaperTradingContractViolationError, match="integer count"):
        _run(_use_case(state, store=store), _buy_result())


@pytest.mark.parametrize("returned", [0, 2, 7])
def test_a_wrong_store_count_is_rejected(returned: int) -> None:
    state = InMemoryFills()
    store = MemoryStore(state, return_value=returned)

    with pytest.raises(PaperTradingContractViolationError, match="exactly one fill"):
        _run(_use_case(state, store=store), _buy_result())


def test_a_non_tuple_repository_result_is_rejected() -> None:
    state = InMemoryFills()
    repository = MemoryRepository(state, override=[])

    with pytest.raises(PaperTradingContractViolationError, match="must return a tuple"):
        _run(_use_case(state, repository=repository), _buy_result())


def test_a_non_fill_repository_entry_is_rejected() -> None:
    state = InMemoryFills()
    repository = MemoryRepository(state, override=("fill",))

    with pytest.raises(PaperTradingContractViolationError, match="PaperFill instances"):
        _run(_use_case(state, repository=repository), _buy_result())


def test_an_unordered_repository_result_is_rejected() -> None:
    state = InMemoryFills()
    seed = _use_case(state)
    _run(seed, _buy_result(), quantity="10")
    _run(seed, _buy_result(observed_at=_LATER), quantity="5")
    ordered = MemoryRepository(state).get_fills(PaperFillQuery(_PORTFOLIO))
    repository = MemoryRepository(state, override=tuple(reversed(ordered)))

    with pytest.raises(InvalidPaperFillHistoryError, match="ordered by fill instant"):
        _run(_use_case(state, repository=repository), _buy_result())


def test_a_foreign_portfolio_fill_is_rejected() -> None:
    state = InMemoryFills()
    seed = _use_case(state)
    _run(seed, _buy_result(), quantity="10")
    borrowed = MemoryRepository(state).get_fills(PaperFillQuery(_PORTFOLIO))
    repository = MemoryRepository(state, override=borrowed)
    use_case = RunPaperTradingDecisionUseCase(repository, MemoryStore(state))

    with pytest.raises(InvalidPaperFillHistoryError, match="must all belong to the requested"):
        use_case.execute(_buy_result(), PaperPortfolioIdentity("paper-2"), Quantity("1"))


# ---------------------------------------------------------------------------
# Call flow
# ---------------------------------------------------------------------------


def test_an_executed_decision_reads_twice_and_writes_once() -> None:
    state = InMemoryFills()
    store = MemoryStore(state)
    repository = MemoryRepository(state)

    _run(_use_case(state, store=store, repository=repository), _buy_result())

    assert len(repository.queries) == 2
    assert len(store.calls) == 1
    assert len(store.calls[0]) == 1


def test_a_no_intent_decision_reads_once_and_never_writes() -> None:
    state = InMemoryFills()
    store = MemoryStore(state)
    repository = MemoryRepository(state)

    _run(_use_case(state, store=store, repository=repository), _hold_result())

    assert len(repository.queries) == 1
    assert store.calls == []


def test_every_query_asks_for_the_requested_portfolio() -> None:
    state = InMemoryFills()
    repository = MemoryRepository(state)

    _run(_use_case(state, repository=repository), _buy_result())

    assert all(query == PaperFillQuery(_PORTFOLIO) for query in repository.queries)


def test_exactly_the_simulated_fill_is_stored() -> None:
    state = InMemoryFills()
    store = MemoryStore(state)

    outcome = _run(_use_case(state, store=store), _buy_result())

    assert store.calls == [(outcome.execution.fill,)]


# ---------------------------------------------------------------------------
# Result coherence
# ---------------------------------------------------------------------------


def test_result_requires_an_execution_for_an_intended_decision() -> None:
    outcome = _run(_use_case(), _buy_result())

    with pytest.raises(ValueError, match="must carry an execution"):
        PaperTradingResult(
            portfolio_identity=_PORTFOLIO,
            result=outcome.result,
            decision=outcome.decision,
            execution=None,
            portfolio=outcome.portfolio,
        )


def test_result_forbids_an_execution_without_an_intent() -> None:
    executed = _run(_use_case(), _buy_result())
    held = _run(_use_case(), _hold_result())

    with pytest.raises(ValueError, match="cannot carry an execution"):
        PaperTradingResult(
            portfolio_identity=_PORTFOLIO,
            result=held.result,
            decision=held.decision,
            execution=executed.execution,
            portfolio=held.portfolio,
        )


def test_result_requires_the_portfolio_to_belong_to_the_decision() -> None:
    outcome = _run(_use_case(), _buy_result())
    foreign = BuildPaperPortfolioUseCase().execute(PaperPortfolioIdentity("paper-2"), (), _AT)

    with pytest.raises(ValueError, match="must belong to the requested paper portfolio"):
        PaperTradingResult(
            portfolio_identity=_PORTFOLIO,
            result=outcome.result,
            decision=outcome.decision,
            execution=outcome.execution,
            portfolio=foreign,
        )


def test_result_rejects_invalid_member_types() -> None:
    outcome = _run(_use_case(), _buy_result())

    with pytest.raises(TypeError, match="portfolio identity must be a PaperPortfolioIdentity"):
        PaperTradingResult(
            portfolio_identity="paper-1",
            result=outcome.result,
            decision=outcome.decision,
            execution=outcome.execution,
            portfolio=outcome.portfolio,
        )
    with pytest.raises(TypeError, match="result must be an AnalyzeAssetResult"):
        PaperTradingResult(
            portfolio_identity=_PORTFOLIO,
            result="result",
            decision=outcome.decision,
            execution=outcome.execution,
            portfolio=outcome.portfolio,
        )
    with pytest.raises(TypeError, match="decision must be an ExecutionIntentDecision"):
        PaperTradingResult(
            portfolio_identity=_PORTFOLIO,
            result=outcome.result,
            decision="d",
            execution=None,
            portfolio=outcome.portfolio,
        )
    with pytest.raises(TypeError, match="execution must be a PaperExecution or None"):
        PaperTradingResult(
            portfolio_identity=_PORTFOLIO,
            result=outcome.result,
            decision=outcome.decision,
            execution="e",
            portfolio=outcome.portfolio,
        )
    with pytest.raises(TypeError, match="portfolio must be a PaperPortfolio"):
        PaperTradingResult(
            portfolio_identity=_PORTFOLIO,
            result=outcome.result,
            decision=outcome.decision,
            execution=outcome.execution,
            portfolio="p",
        )


def test_result_is_immutable() -> None:
    outcome = _run(_use_case(), _buy_result())

    with pytest.raises(AttributeError):
        outcome.execution = None


def test_repeated_runs_on_separate_stores_produce_equal_results() -> None:
    result = _buy_result()

    first = _run(_use_case(), result, quantity="10")
    second = _run(_use_case(), result, quantity="10")

    assert first == second


# ---------------------------------------------------------------------------
# Requested portfolio attribution
# ---------------------------------------------------------------------------


_FOREIGN = PaperPortfolioIdentity("paper-2")


def _empty_portfolio(identity: PaperPortfolioIdentity = _PORTFOLIO):
    return BuildPaperPortfolioUseCase().execute(identity, (), _AT)


def test_a_result_records_the_requested_portfolio_identity() -> None:
    outcome = _run(_use_case(), _buy_result())

    assert outcome.portfolio_identity == _PORTFOLIO
    assert outcome.portfolio.identity == _PORTFOLIO


def test_a_no_intent_result_also_records_the_requested_portfolio() -> None:
    outcome = _run(_use_case(), _hold_result())

    assert outcome.portfolio_identity == _PORTFOLIO
    assert outcome.decision.has_intent is False


def test_a_no_intent_result_with_a_matching_portfolio_is_accepted() -> None:
    held = _run(_use_case(), _hold_result())

    accepted = PaperTradingResult(
        portfolio_identity=_PORTFOLIO,
        result=held.result,
        decision=held.decision,
        execution=None,
        portfolio=_empty_portfolio(),
    )

    assert accepted.portfolio_identity == _PORTFOLIO
    assert accepted.execution is None


def test_a_no_intent_result_with_a_foreign_portfolio_is_rejected() -> None:
    """The gap this follow-up closes: a hold previously carried no ownership claim."""
    held = _run(_use_case(), _hold_result())

    with pytest.raises(ValueError, match="must belong to the requested paper portfolio"):
        PaperTradingResult(
            portfolio_identity=_PORTFOLIO,
            result=held.result,
            decision=held.decision,
            execution=None,
            portfolio=_empty_portfolio(_FOREIGN),
        )


def test_an_insufficient_position_result_with_a_foreign_portfolio_is_rejected() -> None:
    refused = _run(_use_case(), _sell_result(), quantity="10")

    assert refused.decision.no_intent_reason is ExecutionIntentNoIntentReason.INSUFFICIENT_POSITION
    with pytest.raises(ValueError, match="must belong to the requested paper portfolio"):
        PaperTradingResult(
            portfolio_identity=_PORTFOLIO,
            result=refused.result,
            decision=refused.decision,
            execution=None,
            portfolio=_empty_portfolio(_FOREIGN),
        )


def test_an_intended_result_with_an_intent_from_a_foreign_portfolio_is_rejected() -> None:
    outcome = _run(_use_case(), _buy_result())

    with pytest.raises(ValueError, match="must intend execution in the requested"):
        PaperTradingResult(
            portfolio_identity=_FOREIGN,
            result=outcome.result,
            decision=outcome.decision,
            execution=outcome.execution,
            portfolio=_empty_portfolio(_FOREIGN),
        )


def test_an_intended_result_with_a_foreign_portfolio_snapshot_is_rejected() -> None:
    outcome = _run(_use_case(), _buy_result())

    with pytest.raises(ValueError, match="must belong to the requested paper portfolio"):
        PaperTradingResult(
            portfolio_identity=_PORTFOLIO,
            result=outcome.result,
            decision=outcome.decision,
            execution=outcome.execution,
            portfolio=_empty_portfolio(_FOREIGN),
        )


def test_a_coherent_intended_result_is_accepted() -> None:
    outcome = _run(_use_case(), _buy_result())

    rebuilt = PaperTradingResult(
        portfolio_identity=_PORTFOLIO,
        result=outcome.result,
        decision=outcome.decision,
        execution=outcome.execution,
        portfolio=outcome.portfolio,
    )

    assert rebuilt == outcome


# ---------------------------------------------------------------------------
# Retained decision evidence
# ---------------------------------------------------------------------------


def test_a_hold_retains_the_original_analysis_result() -> None:
    """Reporting needs listing, strategy and instant even when nothing executed."""
    source = _hold_result()

    outcome = _run(_use_case(), source)

    assert outcome.result is source
    assert outcome.result.recommendation.action.value == "HOLD"
    assert outcome.result.recommendation.asset_analysis.listing_reference == _LISTING
    assert outcome.result.recommendation.strategy_identity == _STRATEGY
    assert outcome.result.recommendation.point_in_time == _AT


def test_an_insufficient_sell_retains_the_original_analysis_result() -> None:
    source = _sell_result()

    outcome = _run(_use_case(), source, quantity="10")

    assert outcome.result is source
    assert outcome.decision.no_intent_reason is ExecutionIntentNoIntentReason.INSUFFICIENT_POSITION
    assert outcome.result.recommendation.action.value == "SELL"


def test_a_buy_retains_the_exact_result_object_passed_in() -> None:
    source = _buy_result()

    outcome = _run(_use_case(), source)

    assert outcome.result is source


def test_a_sell_retains_the_exact_result_object_passed_in() -> None:
    state = InMemoryFills()
    use_case = _use_case(state)
    _run(use_case, _buy_result(), quantity="10")
    source = _sell_result(observed_at=_LATER)

    outcome = _run(use_case, source, quantity="4")

    assert outcome.result is source
    assert outcome.decision.intent.side is OrderSide.SELL


def test_a_hold_decision_paired_with_a_buy_source_is_rejected() -> None:
    held = _run(_use_case(), _hold_result())
    bought = _run(_use_case(), _buy_result())

    with pytest.raises(ValueError, match="must intend execution for a BUY recommendation"):
        PaperTradingResult(
            portfolio_identity=_PORTFOLIO,
            result=bought.result,
            decision=held.decision,
            execution=None,
            portfolio=_empty_portfolio(),
        )


def test_an_intended_decision_paired_with_a_hold_source_is_rejected() -> None:
    bought = _run(_use_case(), _buy_result())
    held = _run(_use_case(), _hold_result())

    with pytest.raises(ValueError, match="cannot intend execution for a HOLD recommendation"):
        PaperTradingResult(
            portfolio_identity=_PORTFOLIO,
            result=held.result,
            decision=bought.decision,
            execution=bought.execution,
            portfolio=bought.portfolio,
        )


def test_an_insufficient_position_paired_with_a_hold_source_is_rejected() -> None:
    refused = _run(_use_case(), _sell_result(), quantity="10")
    held = _run(_use_case(), _hold_result())

    with pytest.raises(ValueError, match="must decline a HOLD recommendation as a hold"):
        PaperTradingResult(
            portfolio_identity=_PORTFOLIO,
            result=held.result,
            decision=refused.decision,
            execution=None,
            portfolio=_empty_portfolio(),
        )


def test_an_insufficient_position_paired_with_a_buy_source_is_rejected() -> None:
    refused = _run(_use_case(), _sell_result(), quantity="10")
    bought = _run(_use_case(), _buy_result())

    with pytest.raises(ValueError, match="must intend execution for a BUY recommendation"):
        PaperTradingResult(
            portfolio_identity=_PORTFOLIO,
            result=bought.result,
            decision=refused.decision,
            execution=None,
            portfolio=_empty_portfolio(),
        )


def test_a_hold_decline_paired_with_a_sell_source_is_rejected() -> None:
    """A SELL may only be declined for want of a position, never as a hold."""
    held = _run(_use_case(), _hold_result())
    sold = _run(_use_case(), _sell_result(), quantity="10")

    with pytest.raises(ValueError, match="may only decline a SELL recommendation"):
        PaperTradingResult(
            portfolio_identity=_PORTFOLIO,
            result=sold.result,
            decision=held.decision,
            execution=None,
            portfolio=_empty_portfolio(),
        )


def test_an_intent_for_a_foreign_listing_is_rejected() -> None:
    other_listing = ListingReference(Symbol("MSFT"), ExchangeCode("NASDAQ"))
    bought = _run(_use_case(), _buy_result())
    elsewhere = _run(_use_case(), _buy_result(listing=other_listing))

    with pytest.raises(ValueError, match="intent listing must match the recommendation listing"):
        PaperTradingResult(
            portfolio_identity=_PORTFOLIO,
            result=elsewhere.result,
            decision=bought.decision,
            execution=bought.execution,
            portfolio=bought.portfolio,
        )


def test_an_intent_for_a_foreign_strategy_is_rejected() -> None:
    bought = _run(_use_case(), _buy_result())
    other = _analyse(
        _context(list(range(180, 200)), "250", "199"), strategy=StrategyIdentity("zeta")
    )

    with pytest.raises(ValueError, match="intent strategy must match the recommendation strategy"):
        PaperTradingResult(
            portfolio_identity=_PORTFOLIO,
            result=other,
            decision=bought.decision,
            execution=bought.execution,
            portfolio=bought.portfolio,
        )


def test_an_intent_at_a_different_instant_is_rejected() -> None:
    bought = _run(_use_case(), _buy_result())
    later = _run(_use_case(), _buy_result(observed_at=_LATER))

    with pytest.raises(ValueError, match="intent decision instant must match"):
        PaperTradingResult(
            portfolio_identity=_PORTFOLIO,
            result=later.result,
            decision=bought.decision,
            execution=bought.execution,
            portfolio=bought.portfolio,
        )


def test_an_offset_equivalent_instant_remains_coherent() -> None:
    """Instant correspondence is semantic, so an offset spelling still matches."""
    offset_source = _buy_result(observed_at=PointInTime("2026-01-20T21:30:00+05:30"))
    bought = _run(_use_case(), _buy_result())

    accepted = PaperTradingResult(
        portfolio_identity=_PORTFOLIO,
        result=offset_source,
        decision=bought.decision,
        execution=bought.execution,
        portfolio=bought.portfolio,
    )

    assert accepted.result is offset_source
