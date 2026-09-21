"""Tests for translating one recommendation into a paper-trading decision."""

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
    InvalidExecutionIntentError,
    OrderSide,
    PaperPortfolioIdentity,
    Position,
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
    CreateExecutionIntentUseCase,
    ExecutionIntentDecision,
    ExecutionIntentNoIntentReason,
)

_USD = Currency("USD")
_SYMBOL = Symbol("AAPL")
_EXCHANGE = ExchangeCode("NASDAQ")
_LISTING = ListingReference(_SYMBOL, _EXCHANGE)
_OTHER_LISTING = ListingReference(Symbol("MSFT"), _EXCHANGE)
_STRATEGY = StrategyIdentity("alpha")
_PORTFOLIO = PaperPortfolioIdentity("paper-1")
_OBSERVED_AT = PointInTime("2026-01-20T16:00:00Z")


def _context(
    *,
    listing_reference: ListingReference = _LISTING,
    observed_at: PointInTime = _OBSERVED_AT,
) -> MarketObservationContext:
    return MarketObservationContext(
        listing_reference,
        observed_at,
        Price("100", _USD),
        Price("99", _USD),
        Quantity("1000"),
        Price("900", _USD),
        Price("1", _USD),
        tuple(Price("100", _USD) for _ in range(20)),
        tuple(Quantity("1000") for _ in range(20)),
    )


def _result(
    action: str = "BUY",
    *,
    listing_reference: ListingReference = _LISTING,
    observed_at: PointInTime = _OBSERVED_AT,
    analysis_listing: ListingReference | None = None,
    analysis_instant: PointInTime | None = None,
    recommendation_instant: PointInTime | None = None,
    strategy: StrategyIdentity = _STRATEGY,
) -> AnalyzeAssetResult:
    """Build a result with a chosen action, bypassing strategy signal policy."""
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
    return AnalyzeAssetResult(
        recommendation=recommendation,
        explanation=RecommendationExplanation(
            recommendation=recommendation,
            reasons=(ExplanationReason("because", ("signal",)),),
        ),
        market_observation_context=_context(
            listing_reference=listing_reference, observed_at=observed_at
        ),
    )


def _position(listing_reference: ListingReference = _LISTING, quantity: str = "100") -> Position:
    return Position(listing_reference, Quantity(quantity), Price("90", _USD))


def _execute(
    result: AnalyzeAssetResult | None = None,
    quantity: str = "10",
    position: Position | None = None,
    portfolio_identity: PaperPortfolioIdentity = _PORTFOLIO,
) -> ExecutionIntentDecision:
    return CreateExecutionIntentUseCase().execute(
        result if result is not None else _result(),
        portfolio_identity,
        Quantity(quantity),
        position,
    )


# ---------------------------------------------------------------------------
# Decision contract
# ---------------------------------------------------------------------------


def test_decision_requires_exactly_one_of_intent_or_reason() -> None:
    with pytest.raises(ValueError, match="exactly one of intent or no-intent reason"):
        ExecutionIntentDecision()
    with pytest.raises(ValueError, match="exactly one of intent or no-intent reason"):
        ExecutionIntentDecision(
            intent=_execute().intent,
            no_intent_reason=ExecutionIntentNoIntentReason.HOLD,
        )


def test_decision_rejects_invalid_member_types() -> None:
    with pytest.raises(TypeError, match="intent must be an ExecutionIntent or None"):
        ExecutionIntentDecision(intent="intent")
    with pytest.raises(TypeError, match="must be an ExecutionIntentNoIntentReason or None"):
        ExecutionIntentDecision(no_intent_reason="HOLD")


def test_has_intent_reflects_the_decision() -> None:
    assert _execute().has_intent is True
    assert _execute(_result("HOLD")).has_intent is False


def test_decision_is_immutable() -> None:
    decision = _execute()

    with pytest.raises(AttributeError):
        decision.intent = None


def test_no_intent_vocabulary_is_closed() -> None:
    assert [reason.value for reason in ExecutionIntentNoIntentReason] == [
        "HOLD",
        "INSUFFICIENT_POSITION",
    ]
    with pytest.raises(ValueError):
        ExecutionIntentNoIntentReason("REJECTED")


# ---------------------------------------------------------------------------
# BUY
# ---------------------------------------------------------------------------


def test_buy_creates_the_exact_intent() -> None:
    decision = _execute(_result("BUY"), quantity="10")

    assert decision.has_intent
    assert decision.no_intent_reason is None
    assert decision.intent == ExecutionIntent(
        portfolio_identity=_PORTFOLIO,
        listing_reference=_LISTING,
        side=OrderSide.BUY,
        quantity=Quantity("10"),
        strategy_identity=_STRATEGY,
        decided_at=_OBSERVED_AT,
    )


def test_buy_ignores_the_position_entirely() -> None:
    """Position is irrelevant to a purchase, including a mismatched one."""
    without = _execute(_result("BUY"))
    with_matching = _execute(_result("BUY"), position=_position())
    with_foreign = _execute(_result("BUY"), position=_position(_OTHER_LISTING, "1"))

    assert without.intent == with_matching.intent == with_foreign.intent


def test_buy_is_unaffected_by_an_empty_holding() -> None:
    assert _execute(_result("BUY"), quantity="999").has_intent


# ---------------------------------------------------------------------------
# HOLD
# ---------------------------------------------------------------------------


def test_hold_creates_an_explicit_no_intent_decision() -> None:
    decision = _execute(_result("HOLD"))

    assert decision.has_intent is False
    assert decision.intent is None
    assert decision.no_intent_reason is ExecutionIntentNoIntentReason.HOLD


def test_hold_can_never_carry_an_order_side() -> None:
    """There is no OrderSide for HOLD, so no intent can escape carrying one."""
    decision = _execute(_result("HOLD"))

    assert decision.intent is None
    assert not hasattr(ExecutionIntentNoIntentReason.HOLD, "side")
    assert "HOLD" not in OrderSide.__members__


def test_hold_ignores_the_position() -> None:
    assert (
        _execute(_result("HOLD"), position=_position()).no_intent_reason
        is ExecutionIntentNoIntentReason.HOLD
    )
    assert (
        _execute(_result("HOLD"), position=None).no_intent_reason
        is ExecutionIntentNoIntentReason.HOLD
    )


def test_hold_short_circuits_before_any_intent_invariant() -> None:
    """A hold never constructs an intent, so intent invariants never apply."""
    assert (
        _execute(_result("HOLD"), quantity="0").no_intent_reason
        is ExecutionIntentNoIntentReason.HOLD
    )


# ---------------------------------------------------------------------------
# SELL
# ---------------------------------------------------------------------------


def test_sell_with_a_sufficient_matching_position_creates_an_intent() -> None:
    decision = _execute(_result("SELL"), quantity="10", position=_position(quantity="100"))

    assert decision.has_intent
    assert decision.intent.side is OrderSide.SELL
    assert decision.intent.quantity == Quantity("10")
    assert decision.intent.listing_reference == _LISTING


def test_sell_without_a_position_is_insufficient() -> None:
    decision = _execute(_result("SELL"), position=None)

    assert decision.has_intent is False
    assert decision.no_intent_reason is ExecutionIntentNoIntentReason.INSUFFICIENT_POSITION


def test_sell_against_a_wrong_listing_position_is_insufficient() -> None:
    """A holding in another asset cannot fund this sale."""
    decision = _execute(_result("SELL"), quantity="1", position=_position(_OTHER_LISTING, "999"))

    assert decision.no_intent_reason is ExecutionIntentNoIntentReason.INSUFFICIENT_POSITION


def test_sell_of_more_than_held_is_insufficient() -> None:
    decision = _execute(_result("SELL"), quantity="101", position=_position(quantity="100"))

    assert decision.no_intent_reason is ExecutionIntentNoIntentReason.INSUFFICIENT_POSITION


def test_sell_of_the_entire_holding_is_allowed() -> None:
    decision = _execute(_result("SELL"), quantity="100", position=_position(quantity="100"))

    assert decision.has_intent
    assert decision.intent.quantity == Quantity("100")


def test_sell_boundary_is_exact_to_the_smallest_fraction() -> None:
    position = _position(quantity="100")

    assert _execute(_result("SELL"), quantity="99.99999999", position=position).has_intent
    assert (
        _execute(_result("SELL"), quantity="100.00000001", position=position).no_intent_reason
        is ExecutionIntentNoIntentReason.INSUFFICIENT_POSITION
    )


def test_sell_compares_quantities_numerically_not_textually() -> None:
    position = _position(quantity="100.0")

    assert _execute(_result("SELL"), quantity="100", position=position).has_intent


def test_sell_never_opens_a_short() -> None:
    """Every accepted sale is covered by an existing holding."""
    for quantity in ("1", "50", "100"):
        decision = _execute(_result("SELL"), quantity=quantity, position=_position(quantity="100"))
        assert decision.has_intent
        assert decision.intent.quantity <= Quantity("100")


def test_sell_does_not_mutate_the_position() -> None:
    position = _position(quantity="100")

    _execute(_result("SELL"), quantity="40", position=position)

    assert position == _position(quantity="100")
    assert position.quantity == Quantity("100")


def test_sell_with_zero_quantity_fails_the_intent_invariant() -> None:
    """A sale is attempted, so the intent's own positivity rule applies."""
    with pytest.raises(InvalidExecutionIntentError, match="must be greater than zero"):
        _execute(_result("SELL"), quantity="0", position=_position())


# ---------------------------------------------------------------------------
# Explicit action to side conversion
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("action", "side"), [("BUY", OrderSide.BUY), ("SELL", OrderSide.SELL)])
def test_recommendation_action_converts_to_the_matching_side(action: str, side: OrderSide) -> None:
    decision = _execute(_result(action), position=_position(quantity="1000"))

    assert decision.intent.side is side


def test_the_two_vocabularies_are_distinct_types() -> None:
    """RecommendationAction and OrderSide share spellings but are not the same type."""
    decision = _execute(_result("BUY"))

    assert isinstance(decision.intent.side, OrderSide)
    assert not isinstance(RecommendationAction("BUY"), OrderSide)
    assert decision.intent.side != RecommendationAction("BUY")


def test_lowercase_actions_are_normalized_by_core_and_still_translate() -> None:
    decision = _execute(_result("buy"))

    assert decision.intent.side is OrderSide.BUY


# ---------------------------------------------------------------------------
# Input and coherence validation
# ---------------------------------------------------------------------------


def test_execute_validates_input_types() -> None:
    use_case = CreateExecutionIntentUseCase()

    with pytest.raises(TypeError, match="result cannot be None"):
        use_case.execute(None, _PORTFOLIO, Quantity("1"))
    with pytest.raises(TypeError, match="result must be an AnalyzeAssetResult"):
        use_case.execute("result", _PORTFOLIO, Quantity("1"))
    with pytest.raises(TypeError, match="portfolio identity cannot be None"):
        use_case.execute(_result(), None, Quantity("1"))
    with pytest.raises(TypeError, match="portfolio identity must be a PaperPortfolioIdentity"):
        use_case.execute(_result(), "paper-1", Quantity("1"))
    with pytest.raises(TypeError, match="quantity cannot be None"):
        use_case.execute(_result(), _PORTFOLIO, None)
    with pytest.raises(TypeError, match="quantity must be a Quantity"):
        use_case.execute(_result(), _PORTFOLIO, 10)
    with pytest.raises(TypeError, match="position must be a Position or None"):
        use_case.execute(_result(), _PORTFOLIO, Quantity("1"), "position")


def test_incoherent_recommendation_instant_is_rejected() -> None:
    result = _result(recommendation_instant=PointInTime("2026-01-21T16:00:00Z"))

    with pytest.raises(ValueError, match="recommendation instant must match"):
        _execute(result)


def test_incoherent_analysis_instant_is_rejected() -> None:
    result = _result(analysis_instant=PointInTime("2026-01-21T16:00:00Z"))

    with pytest.raises(ValueError, match="asset analysis instant must match"):
        _execute(result)


def test_incoherent_analysis_listing_is_rejected() -> None:
    result = _result(analysis_listing=_OTHER_LISTING)

    with pytest.raises(ValueError, match="asset analysis listing must match"):
        _execute(result)


def test_coherence_is_checked_before_any_translation() -> None:
    """Even a HOLD, which creates nothing, must rest on coherent evidence."""
    result = _result("HOLD", analysis_listing=_OTHER_LISTING)

    with pytest.raises(ValueError, match="asset analysis listing must match"):
        _execute(result)


def test_offset_equivalent_instants_remain_coherent() -> None:
    """Coherence uses compare(), so an offset spelling is the same instant."""
    result = _result(recommendation_instant=PointInTime("2026-01-20T21:30:00+05:30"))

    assert _execute(result).has_intent


# ---------------------------------------------------------------------------
# Preservation and determinism
# ---------------------------------------------------------------------------


def test_intent_preserves_every_translated_value() -> None:
    result = _result("BUY", strategy=StrategyIdentity("zeta"))

    decision = _execute(result, quantity="7.5", portfolio_identity=PaperPortfolioIdentity("pf-9"))
    intent = decision.intent

    assert intent.portfolio_identity == PaperPortfolioIdentity("pf-9")
    assert intent.listing_reference == _LISTING
    assert intent.strategy_identity == StrategyIdentity("zeta")
    assert intent.quantity == Quantity("7.5")
    assert intent.decided_at == _OBSERVED_AT


def test_decision_instant_comes_from_the_recommendation_not_a_clock() -> None:
    result = _result("BUY", observed_at=PointInTime("2019-07-04T13:30:00Z"))

    assert _execute(result).intent.decided_at == PointInTime("2019-07-04T13:30:00Z")


@pytest.mark.parametrize("action", ["BUY", "SELL", "HOLD"])
def test_repeated_execution_is_deterministic(action: str) -> None:
    result = _result(action)
    position = _position(quantity="1000")
    use_case = CreateExecutionIntentUseCase()

    first = use_case.execute(result, _PORTFOLIO, Quantity("10"), position)
    second = use_case.execute(result, _PORTFOLIO, Quantity("10"), position)

    assert first == second


def test_separate_use_case_instances_agree() -> None:
    result = _result("BUY")

    assert CreateExecutionIntentUseCase().execute(
        result, _PORTFOLIO, Quantity("10")
    ) == CreateExecutionIntentUseCase().execute(result, _PORTFOLIO, Quantity("10"))


def test_translation_creates_no_order_fill_or_holding() -> None:
    decision = _execute(_result("BUY"))

    assert isinstance(decision.intent, ExecutionIntent)
    for absent in ("order", "fill", "position", "portfolio"):
        assert not hasattr(decision, absent)


# ---------------------------------------------------------------------------
# Module boundaries
# ---------------------------------------------------------------------------


_MODULE = Path(
    __import__(
        "northstar_application.application_services.create_execution_intent",
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
)


def _imported_modules() -> set[str]:
    tree = ast.parse(_MODULE.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.add(node.module)
    return modules


def test_no_clock_or_randomness_is_imported() -> None:
    for module in _imported_modules():
        assert module.split(".")[0] not in _FORBIDDEN_MODULE_ROOTS


def test_no_legacy_package_is_imported() -> None:
    for module in _imported_modules():
        for prefix in _FORBIDDEN_MODULE_PREFIXES:
            assert not module.startswith(prefix), f"imports legacy {module}"


def test_no_call_reads_a_clock() -> None:
    for node in ast.walk(ast.parse(_MODULE.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert node.func.attr not in {"now", "utcnow", "today", "monotonic"}


def test_paper_trading_contracts_come_from_core() -> None:
    assert ExecutionIntent.__module__.startswith("northstar_core.paper_trading")
    assert OrderSide.__module__.startswith("northstar_core.paper_trading")
    assert Position.__module__.startswith("northstar_core.paper_trading")
