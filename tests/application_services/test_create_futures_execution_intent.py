"""Tests for the futures target-position execution policy."""

from __future__ import annotations

import ast
import importlib
from decimal import Decimal
from pathlib import Path

import pytest
from northstar_core.derivatives import ExpirationDate, QuoteValue
from northstar_core.foundation.value_objects import (
    ExchangeCode,
    PointInTime,
    Quantity,
    Symbol,
    Timeframe,
)
from northstar_core.futures import FuturesContract, FuturesProductReference
from northstar_core.paper_trading import (
    FuturesContractCount,
    FuturesExecutionIntent,
    FuturesPaperPortfolio,
    FuturesPosition,
    OrderSide,
    PaperPortfolioIdentity,
)
from northstar_core.strategy import (
    FuturesAssetAnalysis,
    FuturesMarketObservationContext,
    FuturesRecommendation,
    RecommendationAction,
    StrategyIdentity,
)

from northstar_application.application_services import (
    CreateFuturesExecutionIntentUseCase,
    ExecutionIntentNoIntentReason,
    FuturesAnalysisResult,
    FuturesExecutionIntentDecision,
    FuturesExecutionIntentNoIntentReason,
    FuturesForwardResearchRecord,
)


def _contract(product: str = "ES", exchange: str = "CME", expiry: str = "2026-12-18"):
    return FuturesContract(
        FuturesProductReference(Symbol(product), ExchangeCode(exchange)), ExpirationDate(expiry)
    )


_ES_DEC = _contract()
_ES_MAR = _contract(expiry="2027-03-19")
_MES_DEC = _contract("MES")
_PORTFOLIO = PaperPortfolioIdentity("futures-paper-1")
_STRATEGY = StrategyIdentity("futures-forward")
_INSTANT = PointInTime("2026-09-15T21:00:00Z")
_AS_OF = PointInTime("2026-09-15T21:00:00Z")


def _quote(value: str) -> QuoteValue:
    return QuoteValue(Decimal(value))


def _result(
    action: str,
    *,
    contract: FuturesContract = _ES_DEC,
    instant: PointInTime = _INSTANT,
    strategy: StrategyIdentity = _STRATEGY,
) -> FuturesAnalysisResult:
    context = FuturesMarketObservationContext(
        contract=contract,
        timeframe=Timeframe("1d"),
        observed_at=instant,
        latest_quote=_quote("7663.25"),
        previous_close=_quote("7650"),
        latest_volume=Quantity(Decimal("1250")),
        session_high=_quote("7700"),
        session_low=_quote("7500"),
        recent_closes=tuple(_quote(str(7600 + index)) for index in range(20)),
        recent_volumes=tuple(Quantity(Decimal(1000 + index)) for index in range(20)),
    )
    recommendation = FuturesRecommendation(
        action=RecommendationAction(action),
        asset_analysis=FuturesAssetAnalysis(contract, instant, ("signal",)),
        strategy_identity=strategy,
        point_in_time=instant,
    )
    return FuturesAnalysisResult(recommendation, context)


def _record(action: str = "BUY", **overrides: object) -> FuturesForwardResearchRecord:
    return FuturesForwardResearchRecord(_result(action, **overrides))  # type: ignore[arg-type]


def _portfolio(
    *positions: FuturesPosition,
    strategy: StrategyIdentity = _STRATEGY,
    identity: PaperPortfolioIdentity = _PORTFOLIO,
) -> FuturesPaperPortfolio:
    return FuturesPaperPortfolio(identity, strategy, tuple(positions), _AS_OF)


def _holding(net: int, contract: FuturesContract = _ES_DEC) -> FuturesPosition:
    return FuturesPosition(contract, net, _quote("7600"))


def _portfolio_at(current: int) -> FuturesPaperPortfolio:
    return _portfolio() if current == 0 else _portfolio(_holding(current))


def _decide(
    action: str,
    current: int,
    target: int = 2,
) -> FuturesExecutionIntentDecision:
    return CreateFuturesExecutionIntentUseCase().execute(
        _record(action), _portfolio_at(current), FuturesContractCount(target)
    )


_HOLD = FuturesExecutionIntentNoIntentReason.HOLD
_MET = FuturesExecutionIntentNoIntentReason.TARGET_ALREADY_MET


# ---------------------------------------------------------------------------
# The exposure transition matrix, N = 2
# ---------------------------------------------------------------------------

_MATRIX = [
    # current, action, expected (side, contracts) or no-action reason
    (0, "BUY", (OrderSide.BUY, 2)),
    (0, "HOLD", _HOLD),
    (0, "SELL", (OrderSide.SELL, 2)),
    (1, "BUY", (OrderSide.BUY, 1)),
    (1, "HOLD", _HOLD),
    (1, "SELL", (OrderSide.SELL, 3)),
    (2, "BUY", _MET),
    (2, "HOLD", _HOLD),
    (2, "SELL", (OrderSide.SELL, 4)),
    (3, "BUY", (OrderSide.SELL, 1)),
    (3, "HOLD", _HOLD),
    (3, "SELL", (OrderSide.SELL, 5)),
    (-1, "BUY", (OrderSide.BUY, 3)),
    (-1, "HOLD", _HOLD),
    (-1, "SELL", (OrderSide.SELL, 1)),
    (-2, "BUY", (OrderSide.BUY, 4)),
    (-2, "HOLD", _HOLD),
    (-2, "SELL", _MET),
    (-3, "BUY", (OrderSide.BUY, 5)),
    (-3, "HOLD", _HOLD),
    (-3, "SELL", (OrderSide.BUY, 1)),
]


@pytest.mark.parametrize(
    ("current", "action", "expected"),
    _MATRIX,
    ids=[f"{current:+d}-{action}" for current, action, _ in _MATRIX],
)
def test_the_exposure_transition_matrix(current: int, action: str, expected: object) -> None:
    decision = _decide(action, current)

    if isinstance(expected, FuturesExecutionIntentNoIntentReason):
        assert decision.intent is None
        assert decision.no_intent_reason is expected
        assert not decision.has_intent
    else:
        side, contracts = expected
        assert decision.no_intent_reason is None
        assert decision.has_intent
        assert decision.intent.side is side
        assert decision.intent.contracts == FuturesContractCount(contracts)


@pytest.mark.parametrize(("current", "action", "expected"), _MATRIX)
def test_every_trade_lands_exactly_on_the_target(
    current: int, action: str, expected: object
) -> None:
    decision = _decide(action, current)
    if not decision.has_intent:
        return
    signed = decision.intent.contracts.value * (1 if decision.intent.side is OrderSide.BUY else -1)

    assert current + signed == {"BUY": 2, "SELL": -2}[action]


def test_reducing_an_over_target_long_on_buy_is_a_sell_trade() -> None:
    """Research action and trade side are not the same concept."""
    decision = _decide("BUY", 3)

    assert decision.intent.side is OrderSide.SELL
    assert decision.intent.contracts.value == 1


def test_reducing_an_over_target_short_on_sell_is_a_buy_trade() -> None:
    decision = _decide("SELL", -3)

    assert decision.intent.side is OrderSide.BUY
    assert decision.intent.contracts.value == 1


@pytest.mark.parametrize(
    ("current", "action", "side", "contracts"),
    [(5, "SELL", OrderSide.SELL, 7), (-5, "BUY", OrderSide.BUY, 7)],
)
def test_a_reversal_is_one_trade_through_zero(
    current: int, action: str, side: OrderSide, contracts: int
) -> None:
    decision = _decide(action, current)

    assert decision.intent.side is side
    assert decision.intent.contracts.value == contracts


def test_the_target_size_is_caller_supplied() -> None:
    assert _decide("BUY", 0, target=1).intent.contracts.value == 1
    assert _decide("BUY", 0, target=50).intent.contracts.value == 50
    assert _decide("BUY", 50, target=50).no_intent_reason is _MET


def test_large_sizes_use_exact_integer_arithmetic() -> None:
    big = 10**20

    decision = CreateFuturesExecutionIntentUseCase().execute(
        _record("SELL"), _portfolio(_holding(big)), FuturesContractCount(big)
    )

    assert decision.intent.contracts.value == 2 * big
    assert type(decision.intent.contracts.value) is int


# ---------------------------------------------------------------------------
# Only the decided contract matters
# ---------------------------------------------------------------------------


def test_other_contracts_are_ignored() -> None:
    portfolio = _portfolio(_holding(9, _ES_MAR), _holding(-9, _MES_DEC))

    decision = CreateFuturesExecutionIntentUseCase().execute(
        _record("BUY"), portfolio, FuturesContractCount(2)
    )

    assert decision.intent.side is OrderSide.BUY
    assert decision.intent.contracts.value == 2


def test_other_expiries_are_not_aggregated() -> None:
    portfolio = _portfolio(_holding(2, _ES_DEC), _holding(5, _ES_MAR))

    decision = CreateFuturesExecutionIntentUseCase().execute(
        _record("BUY"), portfolio, FuturesContractCount(2)
    )

    assert decision.no_intent_reason is _MET


# ---------------------------------------------------------------------------
# Intent construction
# ---------------------------------------------------------------------------


def test_the_intent_copies_every_fact_from_the_record_and_portfolio() -> None:
    portfolio_identity = PaperPortfolioIdentity("another-portfolio")
    instant = PointInTime("2026-09-16T21:00:00Z")
    record = _record("SELL", contract=_MES_DEC, instant=instant)

    decision = CreateFuturesExecutionIntentUseCase().execute(
        record, _portfolio(identity=portfolio_identity), FuturesContractCount(3)
    )

    assert decision.intent == FuturesExecutionIntent(
        portfolio_identity=portfolio_identity,
        contract=_MES_DEC,
        side=OrderSide.SELL,
        contracts=FuturesContractCount(3),
        strategy_identity=_STRATEGY,
        decided_at=instant,
    )


def test_the_decision_instant_is_the_records() -> None:
    offset = PointInTime("2026-09-16T02:30:00+05:30")

    decision = CreateFuturesExecutionIntentUseCase().execute(
        _record("BUY", instant=offset), _portfolio(), FuturesContractCount(1)
    )

    assert decision.intent.decided_at == _INSTANT


def test_equivalent_inputs_produce_equal_decisions() -> None:
    first = _decide("BUY", -1)
    second = _decide("BUY", -1)

    assert first == second
    assert hash(first) == hash(second)
    assert _decide("HOLD", 0) == _decide("HOLD", 0)


# ---------------------------------------------------------------------------
# Strategy / portfolio coherence
# ---------------------------------------------------------------------------


def test_a_cross_strategy_portfolio_is_rejected() -> None:
    with pytest.raises(ValueError, match="does not match the decision strategy"):
        CreateFuturesExecutionIntentUseCase().execute(
            _record("BUY"),
            _portfolio(strategy=StrategyIdentity("other")),
            FuturesContractCount(1),
        )


def test_a_cross_strategy_hold_is_also_rejected() -> None:
    """Coherence is checked before the action, so no decision escapes it."""
    with pytest.raises(ValueError, match="does not match the decision strategy"):
        CreateFuturesExecutionIntentUseCase().execute(
            _record("HOLD"),
            _portfolio(strategy=StrategyIdentity("other")),
            FuturesContractCount(1),
        )


def test_any_portfolio_identity_is_accepted() -> None:
    for identity in ("p", "futures-paper-2", "x" * 64):
        portfolio = _portfolio(identity=PaperPortfolioIdentity(identity))

        decision = CreateFuturesExecutionIntentUseCase().execute(
            _record("BUY"), portfolio, FuturesContractCount(1)
        )

        assert decision.intent.portfolio_identity == PaperPortfolioIdentity(identity)


# ---------------------------------------------------------------------------
# Input validation: the frozen record is the execution boundary
# ---------------------------------------------------------------------------


def test_an_unfrozen_result_cannot_be_executed() -> None:
    with pytest.raises(TypeError, match="must be a FuturesForwardResearchRecord"):
        CreateFuturesExecutionIntentUseCase().execute(
            _result("BUY"), _portfolio(), FuturesContractCount(1)
        )


def test_a_bare_recommendation_cannot_be_executed() -> None:
    with pytest.raises(TypeError, match="must be a FuturesForwardResearchRecord"):
        CreateFuturesExecutionIntentUseCase().execute(
            _result("BUY").recommendation, _portfolio(), FuturesContractCount(1)
        )


@pytest.mark.parametrize(
    ("record", "portfolio", "target", "message"),
    [
        (None, "portfolio", "target", "record cannot be None"),
        ("record", "portfolio", "target", "record must be a FuturesForwardResearchRecord"),
        ("valid", None, "target", "portfolio cannot be None"),
        ("valid", "portfolio", "target", "portfolio must be a FuturesPaperPortfolio"),
        ("valid", "valid", None, "target contracts cannot be None"),
        ("valid", "valid", 2, "target contracts must be a FuturesContractCount"),
        ("valid", "valid", Quantity("2"), "target contracts must be a FuturesContractCount"),
    ],
)
def test_inputs_are_validated(
    record: object, portfolio: object, target: object, message: str
) -> None:
    record = _record() if record == "valid" else record
    portfolio = _portfolio() if portfolio == "valid" else portfolio

    with pytest.raises(TypeError, match=message):
        CreateFuturesExecutionIntentUseCase().execute(record, portfolio, target)


# ---------------------------------------------------------------------------
# Decision value
# ---------------------------------------------------------------------------


def test_the_no_intent_vocabulary_is_exactly_hold_and_target_already_met() -> None:
    assert [reason.value for reason in FuturesExecutionIntentNoIntentReason] == [
        "HOLD",
        "TARGET_ALREADY_MET",
    ]
    assert "INSUFFICIENT_POSITION" not in FuturesExecutionIntentNoIntentReason.__members__


def test_the_futures_reason_is_not_the_equity_reason() -> None:
    assert FuturesExecutionIntentNoIntentReason is not ExecutionIntentNoIntentReason


def test_a_decision_must_carry_exactly_one_outcome() -> None:
    intent = _decide("BUY", 0).intent

    with pytest.raises(ValueError, match="exactly one"):
        FuturesExecutionIntentDecision()
    with pytest.raises(ValueError, match="exactly one"):
        FuturesExecutionIntentDecision(intent=intent, no_intent_reason=_HOLD)


def test_a_decision_validates_its_member_types() -> None:
    with pytest.raises(TypeError, match="must be a FuturesExecutionIntent"):
        FuturesExecutionIntentDecision(intent="intent")
    with pytest.raises(TypeError, match="FuturesExecutionIntentNoIntentReason"):
        FuturesExecutionIntentDecision(no_intent_reason="HOLD")
    with pytest.raises(TypeError, match="FuturesExecutionIntentNoIntentReason"):
        FuturesExecutionIntentDecision(no_intent_reason=ExecutionIntentNoIntentReason.HOLD)


def test_a_decision_is_immutable() -> None:
    decision = _decide("HOLD", 0)

    with pytest.raises(AttributeError):
        decision.no_intent_reason = _MET  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Module boundaries
# ---------------------------------------------------------------------------

_MODULE_NAME = "northstar_application.application_services.create_futures_execution_intent"


def _tree() -> ast.Module:
    module = importlib.import_module(_MODULE_NAME)
    return ast.parse(Path(module.__file__).read_text(encoding="utf-8"))


def _imported_modules() -> set[str]:
    modules: set[str] = set()
    for node in ast.walk(_tree()):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.add(node.module)
    return modules


def test_the_policy_uses_no_decimal_clock_market_data_or_persistence() -> None:
    for module in _imported_modules():
        assert module.split(".")[0] not in {
            "decimal",
            "northstar_infrastructure",
            "sqlite3",
            "databento",
            "exchange_calendars",
            "requests",
            "urllib",
            "socket",
            "time",
            "datetime",
            "random",
            "uuid",
        }
        assert not module.startswith("northstar_application.ports")
        assert "market_data" not in module
        assert "acquire" not in module


def test_the_policy_does_not_reuse_the_equity_action_to_side_mapping() -> None:
    """The equity mapping reads SELL as a trade side; here it is a target sign."""
    assert "northstar_application.application_services._recommendation_action" not in (
        _imported_modules()
    )


def test_the_policy_reads_no_clock() -> None:
    for node in ast.walk(_tree()):
        if isinstance(node, ast.Attribute):
            assert node.attr not in {"now", "today", "utcnow"}


def test_the_policy_is_exported() -> None:
    import northstar_application.application_services as services

    for name in (
        "CreateFuturesExecutionIntentUseCase",
        "FuturesExecutionIntentDecision",
        "FuturesExecutionIntentNoIntentReason",
    ):
        assert name in services.__all__
    assert not hasattr(services, "_TARGET_SIGNS")
