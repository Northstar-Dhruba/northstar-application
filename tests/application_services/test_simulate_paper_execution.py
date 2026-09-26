"""Tests for simulated execution of one approved paper-trading intent."""

from __future__ import annotations

import ast
from pathlib import Path

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
    InvalidPaperFillError,
    OrderSide,
    PaperFill,
    PaperFillIdentity,
    PaperOrder,
    PaperOrderIdentity,
    PaperOrderStatus,
    PaperPortfolioIdentity,
)
from northstar_core.strategy import (
    AssetAnalysis,
    ExplanationReason,
    MarketObservationContext,
    Recommendation,
    RecommendationAction,
    RecommendationExplanation,
    StrategyIdentity,
)

from northstar_application.application_services import (
    AnalyzeAssetResult,
    PaperExecution,
    SimulatePaperExecutionUseCase,
)

_USD = Currency("USD")
_LISTING = ListingReference(Symbol("AAPL"), ExchangeCode("NASDAQ"))
_OTHER_LISTING = ListingReference(Symbol("MSFT"), ExchangeCode("NASDAQ"))
_STRATEGY = StrategyIdentity("alpha")
_PORTFOLIO = PaperPortfolioIdentity("paper-1")
_OBSERVED_AT = PointInTime("2026-01-20T16:00:00Z")
_ORDER_IDENTITY = PaperOrderIdentity("order-1")
_FILL_IDENTITY = PaperFillIdentity("fill-1")


def _result(
    action: str = "BUY",
    *,
    latest_price: Price | None = None,
    observed_at: PointInTime = _OBSERVED_AT,
    listing_reference: ListingReference = _LISTING,
    strategy: StrategyIdentity = _STRATEGY,
    analysis_listing: ListingReference | None = None,
    analysis_instant: PointInTime | None = None,
    recommendation_instant: PointInTime | None = None,
) -> AnalyzeAssetResult:
    price = latest_price if latest_price is not None else Price("100", _USD)
    # The context derives its currency from latest_price, so every other price
    # in it must be denominated the same way.
    currency = price.currency
    analysis = AssetAnalysis(
        analysis_listing or listing_reference,
        analysis_instant or observed_at,
        ("signal",),
    )
    recommendation = Recommendation(
        RecommendationAction(action),
        analysis,
        strategy,
        recommendation_instant or observed_at,
    )
    context = MarketObservationContext(
        listing_reference,
        observed_at,
        price,
        Price("99", currency),
        Quantity("1000"),
        Price("900", currency),
        Price("1", currency),
        tuple(Price("100", currency) for _ in range(20)),
        tuple(Quantity("1000") for _ in range(20)),
    )
    return AnalyzeAssetResult(
        recommendation=recommendation,
        explanation=RecommendationExplanation(
            recommendation=recommendation,
            reasons=(ExplanationReason("because", ("signal",)),),
        ),
        market_observation_context=context,
    )


def _intent(
    side: OrderSide = OrderSide.BUY,
    *,
    listing_reference: ListingReference = _LISTING,
    strategy: StrategyIdentity = _STRATEGY,
    quantity: str = "10",
    decided_at: PointInTime = _OBSERVED_AT,
    portfolio_identity: PaperPortfolioIdentity = _PORTFOLIO,
) -> ExecutionIntent:
    return ExecutionIntent(
        portfolio_identity=portfolio_identity,
        listing_reference=listing_reference,
        side=side,
        quantity=Quantity(quantity),
        strategy_identity=strategy,
        decided_at=decided_at,
    )


def _execute(
    intent: ExecutionIntent | None = None,
    result: AnalyzeAssetResult | None = None,
    order_identity: PaperOrderIdentity = _ORDER_IDENTITY,
    fill_identity: PaperFillIdentity = _FILL_IDENTITY,
) -> PaperExecution:
    return SimulatePaperExecutionUseCase().execute(
        intent if intent is not None else _intent(),
        result if result is not None else _result(),
        order_identity,
        fill_identity,
    )


# ---------------------------------------------------------------------------
# Execution contract
# ---------------------------------------------------------------------------


def test_buy_intent_produces_a_filled_order_and_exact_fill() -> None:
    execution = _execute(_intent(OrderSide.BUY), _result("BUY"))

    assert execution.order == PaperOrder(
        identity=_ORDER_IDENTITY,
        intent=_intent(OrderSide.BUY),
        status=PaperOrderStatus.FILLED,
    )
    assert execution.fill == PaperFill(
        identity=_FILL_IDENTITY,
        order_identity=_ORDER_IDENTITY,
        intent=_intent(OrderSide.BUY),
        quantity=Quantity("10"),
        price=Price("100", _USD),
        filled_at=_OBSERVED_AT,
    )


def test_sell_intent_produces_a_filled_order_and_exact_fill() -> None:
    execution = _execute(_intent(OrderSide.SELL), _result("SELL"))

    assert execution.order.status is PaperOrderStatus.FILLED
    assert execution.fill.side is OrderSide.SELL
    assert execution.fill.quantity == Quantity("10")
    assert execution.fill.price == Price("100", _USD)


def test_every_valid_executable_intent_is_filled() -> None:
    """The initial simulator has no rejection policy."""
    for side, action in ((OrderSide.BUY, "BUY"), (OrderSide.SELL, "SELL")):
        assert _execute(_intent(side), _result(action)).order.status is PaperOrderStatus.FILLED


def test_simulation_never_produces_a_rejected_order() -> None:
    execution = _execute()

    assert execution.order.status is not PaperOrderStatus.REJECTED
    assert PaperOrderStatus.REJECTED in PaperOrderStatus  # reserved for a later policy


# ---------------------------------------------------------------------------
# Price and time semantics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("amount", ["100", "0.01", "1234.56789012345678901234"])
def test_fill_uses_the_exact_decision_time_latest_price(amount: str) -> None:
    result = _result("BUY", latest_price=Price(amount, _USD))

    assert _execute(result=result).fill.price == Price(amount, _USD)


def test_fill_price_is_not_the_previous_close_or_daily_range() -> None:
    """Only latest_price is used; neighbouring context prices are not."""
    execution = _execute(result=_result("BUY", latest_price=Price("123.45", _USD)))

    assert execution.fill.price == Price("123.45", _USD)
    assert execution.fill.price != Price("99", _USD)
    assert execution.fill.price != Price("900", _USD)
    assert execution.fill.price != Price("1", _USD)


def test_fill_price_currency_follows_the_observation() -> None:
    result = _result("BUY", latest_price=Price("100", Currency("EUR")))

    assert _execute(result=result).fill.price.currency == Currency("EUR")


def test_filled_at_is_the_exact_observed_instant() -> None:
    observed = PointInTime("2019-07-04T13:30:00Z")
    result = _result("BUY", observed_at=observed)

    execution = _execute(_intent(decided_at=observed), result)

    assert execution.fill.filled_at == observed


def test_execution_has_no_delay_between_decision_and_fill() -> None:
    execution = _execute()

    assert execution.fill.filled_at.compare(execution.fill.decided_at) == 0


def test_quantity_is_preserved_exactly() -> None:
    for quantity in ("1", "0.5", "7.25", "1000"):
        execution = _execute(_intent(quantity=quantity))
        assert execution.fill.quantity == Quantity(quantity)
        assert execution.fill.quantity == execution.order.intent.quantity


# ---------------------------------------------------------------------------
# Linkage and reachability
# ---------------------------------------------------------------------------


def test_order_and_fill_identities_are_linked() -> None:
    execution = _execute(
        order_identity=PaperOrderIdentity("o-77"), fill_identity=PaperFillIdentity("f-88")
    )

    assert execution.order.identity == PaperOrderIdentity("o-77")
    assert execution.fill.identity == PaperFillIdentity("f-88")
    assert execution.fill.order_identity == execution.order.identity


def test_order_and_fill_share_one_intent_object() -> None:
    intent = _intent()

    execution = _execute(intent)

    assert execution.order.intent is intent
    assert execution.fill.intent is intent


def test_portfolio_listing_side_and_strategy_remain_reachable_through_the_fill() -> None:
    intent = _intent(
        OrderSide.SELL,
        portfolio_identity=PaperPortfolioIdentity("pf-9"),
        strategy=StrategyIdentity("zeta"),
    )

    fill = _execute(intent, _result("SELL", strategy=StrategyIdentity("zeta"))).fill

    assert fill.portfolio_identity == PaperPortfolioIdentity("pf-9")
    assert fill.listing_reference == _LISTING
    assert fill.side is OrderSide.SELL
    assert fill.strategy_identity == StrategyIdentity("zeta")
    assert fill.decided_at == _OBSERVED_AT


# ---------------------------------------------------------------------------
# PaperExecution coherence
# ---------------------------------------------------------------------------


def test_execution_rejects_invalid_member_types() -> None:
    execution = _execute()

    with pytest.raises(TypeError, match="order must be a PaperOrder"):
        PaperExecution(order="order", fill=execution.fill)
    with pytest.raises(TypeError, match="fill must be a PaperFill"):
        PaperExecution(order=execution.order, fill="fill")


def test_execution_rejects_a_rejected_order() -> None:
    execution = _execute()
    rejected = PaperOrder(
        identity=execution.order.identity,
        intent=execution.order.intent,
        status=PaperOrderStatus.REJECTED,
    )

    with pytest.raises(ValueError, match="order must be filled"):
        PaperExecution(order=rejected, fill=execution.fill)


def test_execution_rejects_a_fill_for_another_order() -> None:
    execution = _execute()
    foreign = PaperFill(
        identity=execution.fill.identity,
        order_identity=PaperOrderIdentity("other-order"),
        intent=execution.order.intent,
        quantity=execution.fill.quantity,
        price=execution.fill.price,
        filled_at=execution.fill.filled_at,
    )

    with pytest.raises(ValueError, match="must reference its own order identity"):
        PaperExecution(order=execution.order, fill=foreign)


def test_execution_rejects_a_fill_for_another_intent() -> None:
    execution = _execute()
    other_intent = _intent(quantity="99")
    foreign = PaperFill(
        identity=execution.fill.identity,
        order_identity=execution.order.identity,
        intent=other_intent,
        quantity=other_intent.quantity,
        price=execution.fill.price,
        filled_at=execution.fill.filled_at,
    )

    with pytest.raises(ValueError, match="must execute the order's own intent"):
        PaperExecution(order=execution.order, fill=foreign)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_execute_validates_input_types() -> None:
    use_case = SimulatePaperExecutionUseCase()

    with pytest.raises(TypeError, match="intent cannot be None"):
        use_case.execute(None, _result(), _ORDER_IDENTITY, _FILL_IDENTITY)
    with pytest.raises(TypeError, match="intent must be an ExecutionIntent"):
        use_case.execute("intent", _result(), _ORDER_IDENTITY, _FILL_IDENTITY)
    with pytest.raises(TypeError, match="result cannot be None"):
        use_case.execute(_intent(), None, _ORDER_IDENTITY, _FILL_IDENTITY)
    with pytest.raises(TypeError, match="result must be an AnalyzeAssetResult"):
        use_case.execute(_intent(), "result", _ORDER_IDENTITY, _FILL_IDENTITY)
    with pytest.raises(TypeError, match="order identity cannot be None"):
        use_case.execute(_intent(), _result(), None, _FILL_IDENTITY)
    with pytest.raises(TypeError, match="order identity must be a PaperOrderIdentity"):
        use_case.execute(_intent(), _result(), "order-1", _FILL_IDENTITY)
    with pytest.raises(TypeError, match="fill identity cannot be None"):
        use_case.execute(_intent(), _result(), _ORDER_IDENTITY, None)
    with pytest.raises(TypeError, match="fill identity must be a PaperFillIdentity"):
        use_case.execute(_intent(), _result(), _ORDER_IDENTITY, "fill-1")


def test_a_fill_identity_is_not_an_order_identity() -> None:
    use_case = SimulatePaperExecutionUseCase()

    with pytest.raises(TypeError, match="order identity must be a PaperOrderIdentity"):
        use_case.execute(_intent(), _result(), PaperFillIdentity("x"), _FILL_IDENTITY)


# ---------------------------------------------------------------------------
# Result coherence
# ---------------------------------------------------------------------------


def test_incoherent_recommendation_instant_is_rejected() -> None:
    result = _result(recommendation_instant=PointInTime("2026-01-21T16:00:00Z"))

    with pytest.raises(ValueError, match="recommendation instant must match"):
        _execute(result=result)


def test_incoherent_analysis_instant_is_rejected() -> None:
    result = _result(analysis_instant=PointInTime("2026-01-21T16:00:00Z"))

    with pytest.raises(ValueError, match="asset analysis instant must match"):
        _execute(result=result)


def test_incoherent_analysis_listing_is_rejected() -> None:
    result = _result(analysis_listing=_OTHER_LISTING)

    with pytest.raises(ValueError, match="asset analysis listing must match"):
        _execute(result=result)


# ---------------------------------------------------------------------------
# Intent must correspond to the evidence
# ---------------------------------------------------------------------------


def test_wrong_listing_is_rejected() -> None:
    with pytest.raises(ValueError, match="intent listing must match"):
        _execute(_intent(listing_reference=_OTHER_LISTING), _result("BUY"))


def test_wrong_strategy_is_rejected() -> None:
    with pytest.raises(ValueError, match="intent strategy must match"):
        _execute(_intent(strategy=StrategyIdentity("zeta")), _result("BUY"))


def test_wrong_decision_instant_is_rejected() -> None:
    with pytest.raises(ValueError, match="intent decision instant must match"):
        _execute(_intent(decided_at=PointInTime("2026-01-21T16:00:00Z")), _result("BUY"))


def test_offset_equivalent_decision_instant_is_accepted() -> None:
    """Correspondence uses compare(), so an offset spelling is the same instant."""
    intent = _intent(decided_at=PointInTime("2026-01-20T21:30:00+05:30"))

    execution = _execute(intent, _result("BUY"))

    assert execution.fill.filled_at == _OBSERVED_AT
    assert execution.order.intent.decided_at.compare(_OBSERVED_AT) == 0


# ---------------------------------------------------------------------------
# Side correspondence
# ---------------------------------------------------------------------------


def test_hold_result_cannot_execute() -> None:
    with pytest.raises(ValueError, match="cannot execute a HOLD recommendation"):
        _execute(_intent(OrderSide.BUY), _result("HOLD"))


def test_hold_is_rejected_for_either_side() -> None:
    for side in (OrderSide.BUY, OrderSide.SELL):
        with pytest.raises(ValueError, match="cannot execute a HOLD recommendation"):
            _execute(_intent(side), _result("HOLD"))


def test_buy_result_with_a_sell_intent_is_rejected() -> None:
    with pytest.raises(ValueError, match="intent side must match"):
        _execute(_intent(OrderSide.SELL), _result("BUY"))


def test_sell_result_with_a_buy_intent_is_rejected() -> None:
    with pytest.raises(ValueError, match="intent side must match"):
        _execute(_intent(OrderSide.BUY), _result("SELL"))


@pytest.mark.parametrize(("action", "side"), [("BUY", OrderSide.BUY), ("SELL", OrderSide.SELL)])
def test_matching_action_and_side_execute(action: str, side: OrderSide) -> None:
    assert _execute(_intent(side), _result(action)).fill.side is side


# ---------------------------------------------------------------------------
# Failure rather than a manufactured rejection
# ---------------------------------------------------------------------------


def test_zero_execution_price_fails_explicitly() -> None:
    """An unusable price is an error, not a rejected order."""
    result = _result("BUY", latest_price=Price("0", _USD))

    with pytest.raises(InvalidPaperFillError, match="price must be greater than zero"):
        _execute(result=result)


def test_the_simulator_has_no_rejection_path_at_all() -> None:
    """REJECTED is never constructed in code, only named in prose.

    This is what makes the failure above a genuine error rather than a
    manufactured rejection: the simulator has no way to build a rejected order.
    """
    rejected_uses = [
        node
        for node in ast.walk(ast.parse(_source()))
        if isinstance(node, ast.Attribute) and node.attr == "REJECTED"
    ]

    assert rejected_uses == []


# ---------------------------------------------------------------------------
# Determinism and isolation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("side", "action"), [(OrderSide.BUY, "BUY"), (OrderSide.SELL, "SELL")])
def test_repeated_execution_is_deterministic(side: OrderSide, action: str) -> None:
    intent = _intent(side)
    result = _result(action)
    use_case = SimulatePaperExecutionUseCase()

    first = use_case.execute(intent, result, _ORDER_IDENTITY, _FILL_IDENTITY)
    second = use_case.execute(intent, result, _ORDER_IDENTITY, _FILL_IDENTITY)

    assert first == second


def test_separate_use_case_instances_agree() -> None:
    intent = _intent()
    result = _result()

    assert SimulatePaperExecutionUseCase().execute(
        intent, result, _ORDER_IDENTITY, _FILL_IDENTITY
    ) == SimulatePaperExecutionUseCase().execute(intent, result, _ORDER_IDENTITY, _FILL_IDENTITY)


def test_the_use_case_needs_no_dependencies() -> None:
    """No repository, no clock, nothing to inject."""
    use_case = SimulatePaperExecutionUseCase()

    assert not vars(use_case)
    assert _execute() is not None


def test_execution_is_immutable() -> None:
    execution = _execute()

    with pytest.raises(AttributeError):
        execution.order = None


def test_execution_persists_nothing_and_holds_no_portfolio() -> None:
    execution = _execute()

    for absent in ("save", "store", "persist", "portfolio", "position", "pnl"):
        assert not hasattr(execution, absent)


# ---------------------------------------------------------------------------
# Module boundaries
# ---------------------------------------------------------------------------


_MODULE = Path(
    __import__(
        "northstar_application.application_services.simulate_paper_execution",
        fromlist=["__file__"],
    ).__file__
)
_FORBIDDEN_MODULE_ROOTS = frozenset({"datetime", "time", "random", "uuid", "secrets", "os"})
_FORBIDDEN_MODULE_PREFIXES = (
    "northstar_core.orders",
    "northstar_core.trades",
    "northstar_core.portfolio",
    "northstar_core.domain.listing",
    "northstar_application.execution",
    "northstar_application.ports",
)


def _source() -> str:
    return _MODULE.read_text(encoding="utf-8")


def _imported_modules() -> set[str]:
    modules: set[str] = set()
    for node in ast.walk(ast.parse(_source())):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.add(node.module)
    return modules


def test_no_clock_or_randomness_is_imported() -> None:
    for module in _imported_modules():
        assert module.split(".")[0] not in _FORBIDDEN_MODULE_ROOTS


def test_no_legacy_or_port_package_is_imported() -> None:
    """No market-data repository is reachable, so no lookup can occur."""
    for module in _imported_modules():
        for prefix in _FORBIDDEN_MODULE_PREFIXES:
            assert not module.startswith(prefix), f"imports {module}"


def test_no_market_data_lookup_is_performed() -> None:
    source = _source()

    for forbidden in ("get_history", "HistoricalMarketData", "Repository", "Store"):
        assert forbidden not in source


def test_no_call_reads_a_clock() -> None:
    for node in ast.walk(ast.parse(_source())):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert node.func.attr not in {"now", "utcnow", "today", "monotonic"}


def test_paper_trading_contracts_come_from_core() -> None:
    for contract in (ExecutionIntent, OrderSide, PaperOrder, PaperFill):
        assert contract.__module__.startswith("northstar_core.paper_trading")
