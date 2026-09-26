"""Tests for deterministic derivation of paper execution identities."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import pytest
from northstar_core.domain.value_objects import ListingReference
from northstar_core.foundation.value_objects import (
    ExchangeCode,
    PointInTime,
    Quantity,
    Symbol,
)
from northstar_core.paper_trading import (
    ExecutionIntent,
    OrderSide,
    PaperFillIdentity,
    PaperOrderIdentity,
    PaperPortfolioIdentity,
)
from northstar_core.strategy import StrategyIdentity

from northstar_application.application_services import (
    CreatePaperExecutionIdentitiesUseCase,
    PaperExecutionIdentities,
)

_PORTFOLIO = PaperPortfolioIdentity("paper-1")
_LISTING = ListingReference(Symbol("AAPL"), ExchangeCode("NASDAQ"))
_STRATEGY = StrategyIdentity("alpha")
_DECIDED_AT = PointInTime("2026-01-20T16:00:00Z")


def _intent(**overrides: object) -> ExecutionIntent:
    values: dict[str, object] = {
        "portfolio_identity": _PORTFOLIO,
        "listing_reference": _LISTING,
        "side": OrderSide.BUY,
        "quantity": Quantity("10"),
        "strategy_identity": _STRATEGY,
        "decided_at": _DECIDED_AT,
    }
    values.update(overrides)
    return ExecutionIntent(**values)


def _identities(intent: ExecutionIntent | None = None) -> PaperExecutionIdentities:
    return CreatePaperExecutionIdentitiesUseCase().execute(
        intent if intent is not None else _intent()
    )


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------


def test_identities_are_the_expected_types() -> None:
    identities = _identities()

    assert isinstance(identities.order_identity, PaperOrderIdentity)
    assert isinstance(identities.fill_identity, PaperFillIdentity)


def test_the_value_rejects_invalid_member_types() -> None:
    identities = _identities()

    with pytest.raises(TypeError, match="order identity must be a PaperOrderIdentity"):
        PaperExecutionIdentities(order_identity="order-1", fill_identity=identities.fill_identity)
    with pytest.raises(TypeError, match="fill identity must be a PaperFillIdentity"):
        PaperExecutionIdentities(order_identity=identities.order_identity, fill_identity="fill-1")


def test_an_order_identity_is_not_accepted_as_a_fill_identity() -> None:
    identities = _identities()

    with pytest.raises(TypeError, match="fill identity must be a PaperFillIdentity"):
        PaperExecutionIdentities(
            order_identity=identities.order_identity,
            fill_identity=PaperOrderIdentity("x"),
        )


def test_the_value_is_immutable() -> None:
    identities = _identities()

    with pytest.raises(AttributeError):
        identities.order_identity = PaperOrderIdentity("other")


def test_execute_validates_its_input() -> None:
    use_case = CreatePaperExecutionIdentitiesUseCase()

    with pytest.raises(TypeError, match="intent cannot be None"):
        use_case.execute(None)
    with pytest.raises(TypeError, match="intent must be an ExecutionIntent"):
        use_case.execute("intent")


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_the_same_intent_yields_the_same_identities() -> None:
    intent = _intent()
    use_case = CreatePaperExecutionIdentitiesUseCase()

    assert use_case.execute(intent) == use_case.execute(intent)


def test_a_structurally_equal_fresh_intent_yields_the_same_identities() -> None:
    """Value equality decides, never object memory identity."""
    first, second = _intent(), _intent()

    assert first is not second
    assert _identities(first) == _identities(second)


def test_identities_are_stable_across_use_case_instances() -> None:
    intent = _intent()

    assert CreatePaperExecutionIdentitiesUseCase().execute(
        intent
    ) == CreatePaperExecutionIdentitiesUseCase().execute(intent)


def test_repeated_derivation_is_stable_over_many_calls() -> None:
    intent = _intent()
    results = {_identities(intent) for _ in range(20)}

    assert len(results) == 1


def test_an_offset_equivalent_decision_instant_yields_the_same_identities() -> None:
    """PointInTime.value is canonical UTC, so the spelling cannot leak in."""
    offset = PointInTime("2026-01-20T21:30:00+05:30")

    assert offset.compare(_DECIDED_AT) == 0
    assert _identities(_intent(decided_at=offset)) == _identities()


def test_whitespace_normalized_identities_yield_the_same_identities() -> None:
    """Identity value objects normalize on construction, so derivation inherits it."""
    padded = _intent(
        portfolio_identity=PaperPortfolioIdentity("  paper-1  "),
        strategy_identity=StrategyIdentity(" alpha "),
    )

    assert _identities(padded) == _identities()


# ---------------------------------------------------------------------------
# What the key deliberately ignores
# ---------------------------------------------------------------------------


def test_a_different_quantity_yields_the_same_identities() -> None:
    """A resized retry must collide so persistence reports a conflict."""
    assert _identities(_intent(quantity=Quantity("999"))) == _identities()


def test_buy_and_sell_yield_the_same_identities() -> None:
    """A reversed retry must collide too; one decision executes once."""
    assert _identities(_intent(side=OrderSide.SELL)) == _identities(_intent(side=OrderSide.BUY))


def test_the_key_ignores_side_and_quantity_together() -> None:
    changed = _intent(side=OrderSide.SELL, quantity=Quantity("0.5"))

    assert _identities(changed) == _identities()


# ---------------------------------------------------------------------------
# What the key distinguishes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("portfolio_identity", PaperPortfolioIdentity("paper-2")),
        ("listing_reference", ListingReference(Symbol("MSFT"), ExchangeCode("NASDAQ"))),
        ("listing_reference", ListingReference(Symbol("AAPL"), ExchangeCode("LSE"))),
        ("strategy_identity", StrategyIdentity("zeta")),
        ("decided_at", PointInTime("2026-01-21T16:00:00Z")),
    ],
)
def test_a_different_key_component_yields_different_identities(field: str, value: object) -> None:
    changed = _identities(_intent(**{field: value}))
    baseline = _identities()

    assert changed.order_identity != baseline.order_identity
    assert changed.fill_identity != baseline.fill_identity


def test_a_sub_second_decision_instant_is_distinct() -> None:
    fractional = PointInTime("2026-01-20T16:00:00.1Z")

    assert _identities(_intent(decided_at=fractional)) != _identities()


def test_the_key_components_cannot_be_forged_across_field_boundaries() -> None:
    """Length-prefixed encoding stops one field borrowing from the next."""
    left = _intent(
        portfolio_identity=PaperPortfolioIdentity("a"),
        strategy_identity=StrategyIdentity("bc"),
    )
    right = _intent(
        portfolio_identity=PaperPortfolioIdentity("ab"),
        strategy_identity=StrategyIdentity("c"),
    )

    assert _identities(left) != _identities(right)


# ---------------------------------------------------------------------------
# Namespace separation
# ---------------------------------------------------------------------------


def test_the_order_and_fill_identities_differ() -> None:
    identities = _identities()

    assert identities.order_identity.identity != identities.fill_identity.identity


def test_no_intent_produces_a_fill_identity_equal_to_any_order_identity() -> None:
    intents = [
        _intent(),
        _intent(portfolio_identity=PaperPortfolioIdentity("paper-2")),
        _intent(strategy_identity=StrategyIdentity("zeta")),
        _intent(decided_at=PointInTime("2026-02-01T16:00:00Z")),
    ]
    orders = {_identities(intent).order_identity.identity for intent in intents}
    fills = {_identities(intent).fill_identity.identity for intent in intents}

    assert len(orders) == len(fills) == 4
    assert orders.isdisjoint(fills)


# ---------------------------------------------------------------------------
# Pinned derivation
# ---------------------------------------------------------------------------

_PINNED_ORDER = "83cec3e75320d04d8e2f7809dc0ed0382dd034b96a9296fbfa526a9045fe9641"
_PINNED_FILL = "daca3950032c35a2c4ae44d2ec6482c33e2edd53235f8c705073dfe2d82f3d43"


def test_a_known_fixture_has_a_pinned_output() -> None:
    """Changing the scheme is a breaking change and must fail loudly here."""
    identities = _identities()

    assert identities.order_identity.identity == _PINNED_ORDER
    assert identities.fill_identity.identity == _PINNED_FILL


def test_the_pinned_output_follows_the_documented_recipe() -> None:
    """Rebuild the digest from first principles, so the scheme is specified."""
    key = ("paper-1", "AAPL", "NASDAQ", "alpha", "2026-01-20T16:00:00Z")

    def recipe(namespace: str) -> str:
        encoded = bytearray()
        for part in (namespace, *key):
            raw = part.encode("utf-8")
            encoded += str(len(raw)).encode("ascii") + b":" + raw
        return hashlib.sha256(bytes(encoded)).hexdigest()

    assert recipe("northstar.paper-order.v1") == _PINNED_ORDER
    assert recipe("northstar.paper-fill.v1") == _PINNED_FILL


def test_identities_are_sha256_hex_digests() -> None:
    identities = _identities()

    for identity in (identities.order_identity.identity, identities.fill_identity.identity):
        assert len(identity) == 64
        assert set(identity) <= set("0123456789abcdef")


# ---------------------------------------------------------------------------
# Module boundaries
# ---------------------------------------------------------------------------

_MODULE = Path(
    __import__(
        "northstar_application.application_services.create_paper_execution_identities",
        fromlist=["__file__"],
    ).__file__
)


def _tree() -> ast.Module:
    return ast.parse(_MODULE.read_text(encoding="utf-8"))


def test_the_builtin_hash_is_never_used() -> None:
    """Python's hash() is salted per process and would not be reproducible."""
    calls = [
        node
        for node in ast.walk(_tree())
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "hash"
    ]

    assert calls == []


def test_sha256_is_used() -> None:
    attributes = {
        node.func.attr
        for node in ast.walk(_tree())
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }

    assert "sha256" in attributes


def test_no_nondeterministic_or_persistent_dependency_is_imported() -> None:
    modules: set[str] = set()
    for node in ast.walk(_tree()):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.add(node.module)

    roots = {module.split(".")[0] for module in modules}
    for forbidden in ("uuid", "random", "time", "datetime", "os", "secrets", "sqlite3"):
        assert forbidden not in roots
    assert not [module for module in modules if module.startswith("northstar_application.ports")]


def test_repr_is_never_used_in_derivation() -> None:
    """repr() can embed memory addresses and is not a stable encoding."""
    calls = [
        node
        for node in ast.walk(_tree())
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "repr"
    ]

    assert calls == []


def test_the_use_case_needs_no_dependencies() -> None:
    assert not vars(CreatePaperExecutionIdentitiesUseCase())
