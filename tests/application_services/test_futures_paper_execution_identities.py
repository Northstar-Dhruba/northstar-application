"""Tests for deterministic futures paper order and fill identities."""

from __future__ import annotations

import ast
import hashlib
import importlib
import inspect
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
    FuturesPaperPortfolio,
    FuturesPosition,
    PaperFillIdentity,
    PaperOrderIdentity,
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
    FuturesAnalysisResult,
    FuturesForwardResearchRecord,
    FuturesPaperExecutionIdentityService,
)


def _contract(product: str = "ES", exchange: str = "CME", expiry: str = "2026-12-18"):
    return FuturesContract(
        FuturesProductReference(Symbol(product), ExchangeCode(exchange)), ExpirationDate(expiry)
    )


_ES_DEC = _contract()
_PORTFOLIO = PaperPortfolioIdentity("futures-paper-1")
_STRATEGY = StrategyIdentity("futures-forward")
_INSTANT = "2026-09-15T21:00:00Z"


def _quote(value: str) -> QuoteValue:
    return QuoteValue(Decimal(value))


def _result(
    *,
    action: str = "BUY",
    contract: FuturesContract = _ES_DEC,
    instant: str = _INSTANT,
    strategy: str = "futures-forward",
    timeframe: str = "1d",
    latest: str = "7663.25",
) -> FuturesAnalysisResult:
    observed_at = PointInTime(instant)
    context = FuturesMarketObservationContext(
        contract=contract,
        timeframe=Timeframe(timeframe),
        observed_at=observed_at,
        latest_quote=_quote(latest),
        previous_close=_quote("7650"),
        latest_volume=Quantity(Decimal("1250")),
        session_high=_quote("7700"),
        session_low=_quote("7500"),
        recent_closes=tuple(_quote(str(7600 + index)) for index in range(20)),
        recent_volumes=tuple(Quantity(Decimal(1000 + index)) for index in range(20)),
    )
    recommendation = FuturesRecommendation(
        action=RecommendationAction(action),
        asset_analysis=FuturesAssetAnalysis(contract, observed_at, ("signal",)),
        strategy_identity=StrategyIdentity(strategy),
        point_in_time=observed_at,
    )
    return FuturesAnalysisResult(recommendation, context)


def _record(**overrides: object) -> FuturesForwardResearchRecord:
    return FuturesForwardResearchRecord(_result(**overrides))  # type: ignore[arg-type]


def _order_id(
    record: FuturesForwardResearchRecord | None = None,
    portfolio: PaperPortfolioIdentity = _PORTFOLIO,
) -> PaperOrderIdentity:
    return FuturesPaperExecutionIdentityService().order_identity(
        record if record is not None else _record(), portfolio
    )


def _fill_id(order_identity: PaperOrderIdentity | None = None) -> PaperFillIdentity:
    return FuturesPaperExecutionIdentityService().fill_identity(
        order_identity if order_identity is not None else _order_id()
    )


def _recipe(namespace: str, key: tuple[str, ...]) -> str:
    encoded = bytearray()
    for part in (namespace, *key):
        raw = part.encode("utf-8")
        encoded += str(len(raw)).encode("ascii") + b":" + raw
    return hashlib.sha256(bytes(encoded)).hexdigest()


_ORDER_NAMESPACE = "northstar.futures-paper-order.v1"
_FILL_NAMESPACE = "northstar.futures-paper-fill.v1"
_BASELINE_KEY = (
    "futures-paper-1",
    "ES",
    "CME",
    "2026-12-18",
    "1d",
    "2026-09-15T21:00:00Z",
    "futures-forward",
)


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------


def test_identities_reuse_the_existing_paper_identity_types() -> None:
    assert type(_order_id()) is PaperOrderIdentity
    assert type(_fill_id()) is PaperFillIdentity


@pytest.mark.parametrize(
    ("record", "portfolio", "message"),
    [
        (None, _PORTFOLIO, "record cannot be None"),
        ("record", _PORTFOLIO, "record must be a FuturesForwardResearchRecord"),
        ("valid", None, "portfolio identity cannot be None"),
        ("valid", "futures-paper-1", "portfolio identity must be a PaperPortfolioIdentity"),
        ("valid", StrategyIdentity("x"), "portfolio identity must be a PaperPortfolioIdentity"),
    ],
)
def test_order_identity_inputs_are_validated(
    record: object, portfolio: object, message: str
) -> None:
    record = _record() if record == "valid" else record

    with pytest.raises(TypeError, match=message):
        FuturesPaperExecutionIdentityService().order_identity(record, portfolio)


def test_an_unfrozen_result_has_no_order_identity() -> None:
    with pytest.raises(TypeError, match="must be a FuturesForwardResearchRecord"):
        FuturesPaperExecutionIdentityService().order_identity(_result(), _PORTFOLIO)


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (None, "order identity cannot be None"),
        ("order", "order identity must be a PaperOrderIdentity"),
        (PaperFillIdentity("order"), "order identity must be a PaperOrderIdentity"),
    ],
)
def test_fill_identity_input_is_validated(value: object, message: str) -> None:
    with pytest.raises(TypeError, match=message):
        FuturesPaperExecutionIdentityService().fill_identity(value)


# ---------------------------------------------------------------------------
# Pinned scheme
# ---------------------------------------------------------------------------

_PINNED_ORDER = "1228097694eff9d599f7e14e159f46def04055e69d6072a6110662bf4296d177"
_PINNED_FILL = "0caf5a36437ec9b19891f69875f014d8e7130aff2ac33eeffc4dc219c018745e"


def test_a_known_fixture_has_a_pinned_output() -> None:
    """Changing the scheme is a breaking change and must fail loudly here."""
    assert _order_id().identity == _PINNED_ORDER
    assert _fill_id().identity == _PINNED_FILL


def test_the_order_identity_follows_the_documented_recipe() -> None:
    """Namespace, then portfolio + the frozen natural key, in this exact order."""
    assert _recipe(_ORDER_NAMESPACE, _BASELINE_KEY) == _PINNED_ORDER


def test_the_fill_identity_is_derived_from_the_order_identity_alone() -> None:
    assert _recipe(_FILL_NAMESPACE, (_PINNED_ORDER,)) == _PINNED_FILL


def test_any_order_identity_maps_to_its_fill_under_the_fill_namespace() -> None:
    arbitrary = PaperOrderIdentity("any opaque order")

    assert _fill_id(arbitrary).identity == _recipe(_FILL_NAMESPACE, ("any opaque order",))


def test_identities_are_sha256_hex_digests() -> None:
    for identity in (_order_id().identity, _fill_id().identity):
        assert len(identity) == 64
        assert set(identity) <= set("0123456789abcdef")


def test_the_namespaces_differ_from_the_equity_scheme() -> None:
    """Futures and equity identities can never collide on one key."""
    module = importlib.import_module(
        "northstar_application.application_services.futures_paper_execution_identities"
    )
    equity = importlib.import_module(
        "northstar_application.application_services.create_paper_execution_identities"
    )

    assert module._ORDER_NAMESPACE == _ORDER_NAMESPACE
    assert module._FILL_NAMESPACE == _FILL_NAMESPACE
    assert {module._ORDER_NAMESPACE, module._FILL_NAMESPACE}.isdisjoint(
        {equity._ORDER_NAMESPACE, equity._FILL_NAMESPACE}
    )


# ---------------------------------------------------------------------------
# Determinism and stability
# ---------------------------------------------------------------------------


def test_repeated_derivation_is_stable() -> None:
    assert len({_order_id() for _ in range(20)}) == 1
    assert len({_fill_id() for _ in range(20)}) == 1


def test_a_structurally_equal_fresh_record_yields_the_same_identities() -> None:
    first, second = _record(), _record()

    assert first is not second
    assert _order_id(first) == _order_id(second)
    assert _fill_id(_order_id(first)) == _fill_id(_order_id(second))


def test_identities_are_stable_across_service_instances() -> None:
    record = _record()

    assert FuturesPaperExecutionIdentityService().order_identity(
        record, _PORTFOLIO
    ) == FuturesPaperExecutionIdentityService().order_identity(record, _PORTFOLIO)


def test_an_offset_equivalent_decision_instant_yields_the_same_identities() -> None:
    offset = "2026-09-16T02:30:00+05:30"

    assert PointInTime(offset).compare(PointInTime(_INSTANT)) == 0
    assert _order_id(_record(instant=offset)) == _order_id()


def test_whitespace_normalized_identities_yield_the_same_identities() -> None:
    padded = _record(strategy="  futures-forward ")

    assert _order_id(padded, PaperPortfolioIdentity(" futures-paper-1 ")) == _order_id()


# ---------------------------------------------------------------------------
# What the order key distinguishes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        {"contract": _contract("MES")},
        {"contract": _contract(exchange="CBOT")},
        {"contract": _contract(expiry="2027-03-19")},
        {"instant": "2026-09-16T21:00:00Z"},
        {"instant": "2026-09-15T21:00:00.1Z"},
        {"strategy": "other-strategy"},
    ],
    ids=["product-ES-vs-MES", "exchange", "expiry", "decision-day", "sub-second", "strategy"],
)
def test_a_different_natural_key_component_yields_a_different_order(overrides: dict) -> None:
    changed = _order_id(_record(**overrides))

    assert changed != _order_id()
    assert _fill_id(changed) != _fill_id()


def test_a_different_portfolio_yields_a_different_order() -> None:
    assert _order_id(portfolio=PaperPortfolioIdentity("futures-paper-2")) != _order_id()


def test_the_timeframe_component_is_load_bearing() -> None:
    """Records are 1d-only, so the guard is bypassed here purely to prove the
    service reads the timeframe rather than assuming it."""
    hourly = object.__new__(FuturesForwardResearchRecord)
    object.__setattr__(hourly, "result", _result(timeframe="1h"))

    assert hourly.timeframe == Timeframe("1h")
    assert _order_id(hourly) != _order_id()
    assert _order_id(hourly).identity == _recipe(
        _ORDER_NAMESPACE, (*_BASELINE_KEY[:4], "1h", *_BASELINE_KEY[5:])
    )


def test_key_components_cannot_be_forged_across_field_boundaries() -> None:
    left = _order_id(_record(strategy="bc"), PaperPortfolioIdentity("a"))
    right = _order_id(_record(strategy="c"), PaperPortfolioIdentity("ab"))

    assert left != right


# ---------------------------------------------------------------------------
# What the order key deliberately ignores
# ---------------------------------------------------------------------------


def test_research_evidence_beyond_the_natural_key_is_not_identity() -> None:
    """A changed action or quote under one natural key is a conflict, not a new order."""
    assert _order_id(_record(action="SELL")) == _order_id()
    assert _order_id(_record(latest="7501.25")) == _order_id()


def test_a_recomputed_side_and_size_keep_the_same_identity() -> None:
    """Same frozen decision, different current exposure: the payload differs,
    the logical identity does not, so persistence reports a conflict."""
    record = _record(action="BUY")
    flat = FuturesPaperPortfolio(_PORTFOLIO, _STRATEGY, (), PointInTime(_INSTANT))
    long_five = FuturesPaperPortfolio(
        _PORTFOLIO,
        _STRATEGY,
        (FuturesPosition(_ES_DEC, 5, _quote("7600")),),
        PointInTime(_INSTANT),
    )
    policy = CreateFuturesExecutionIntentUseCase()

    first = policy.execute(record, flat, FuturesContractCount(2)).intent
    second = policy.execute(record, long_five, FuturesContractCount(2)).intent

    assert (first.side, first.contracts) != (second.side, second.contracts)
    assert _order_id(record, first.portfolio_identity) == _order_id(
        record, second.portfolio_identity
    )


def test_the_fill_identity_takes_no_payload() -> None:
    """Side, contracts, fill quote and fill instant cannot reach the fill key."""
    parameters = inspect.signature(FuturesPaperExecutionIdentityService.fill_identity).parameters

    assert list(parameters) == ["self", "order_identity"]


# ---------------------------------------------------------------------------
# Namespace separation
# ---------------------------------------------------------------------------


def test_no_fill_identity_equals_any_order_identity() -> None:
    records = [
        _record(),
        _record(contract=_contract("MES")),
        _record(strategy="zeta"),
        _record(instant="2026-10-01T21:00:00Z"),
    ]
    orders = {_order_id(record).identity for record in records}
    fills = {_fill_id(_order_id(record)).identity for record in records}

    assert len(orders) == len(fills) == 4
    assert orders.isdisjoint(fills)


def test_the_futures_fill_of_an_order_is_not_its_equity_style_fill() -> None:
    """Fill keys are the order identity only, never the decision key again."""
    assert _fill_id().identity != _recipe(_FILL_NAMESPACE, _BASELINE_KEY)


# ---------------------------------------------------------------------------
# Module boundaries
# ---------------------------------------------------------------------------

_MODULE_NAME = "northstar_application.application_services.futures_paper_execution_identities"


def _tree() -> ast.Module:
    module = importlib.import_module(_MODULE_NAME)
    return ast.parse(Path(module.__file__).read_text(encoding="utf-8"))


def test_no_nondeterministic_or_persistent_dependency_is_imported() -> None:
    modules: set[str] = set()
    for node in ast.walk(_tree()):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.add(node.module)

    roots = {module.split(".")[0] for module in modules}
    for forbidden in (
        "uuid",
        "random",
        "time",
        "datetime",
        "os",
        "secrets",
        "sqlite3",
        "northstar_infrastructure",
        "databento",
    ):
        assert forbidden not in roots
    assert not [module for module in modules if module.startswith("northstar_application.ports")]
    assert "northstar_application.application_services.create_paper_execution_identities" not in (
        modules
    )


def test_neither_builtin_hash_nor_repr_is_used() -> None:
    names = {
        node.func.id
        for node in ast.walk(_tree())
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }

    assert "hash" not in names
    assert "repr" not in names


def test_the_service_is_exported_without_its_helpers() -> None:
    import northstar_application.application_services as services

    assert "FuturesPaperExecutionIdentityService" in services.__all__
    for private in ("_digest", "_canonical", "_order_key", "_ORDER_NAMESPACE"):
        assert private not in services.__all__
        assert not hasattr(services, private)
