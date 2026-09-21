"""Contract tests for the PaperFillStore Application port.

The port cannot enforce its own semantics, so the obligations are pinned here
against a conforming in-memory reference implementation. An adapter that fails
these behaviours is not a PaperFillStore, whatever its storage engine.
"""

from __future__ import annotations

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
    OrderSide,
    PaperFill,
    PaperFillIdentity,
    PaperOrderIdentity,
    PaperPortfolioIdentity,
)
from northstar_core.strategy import StrategyIdentity

from northstar_application.ports import PaperFillConflictError, PaperFillStore

_USD = Currency("USD")
_LISTING = ListingReference(Symbol("AAPL"), ExchangeCode("NASDAQ"))
_PORTFOLIO = PaperPortfolioIdentity("paper-1")
_STRATEGY = StrategyIdentity("alpha")


def _instant(day: int) -> PointInTime:
    return PointInTime(f"2026-01-{day:02d}T16:00:00Z")


def _fill(
    fill_id: str = "fill-1",
    order_id: str = "order-1",
    *,
    day: int = 20,
    quantity: str = "10",
    price: str = "100",
    side: OrderSide = OrderSide.BUY,
) -> PaperFill:
    instant = _instant(day)
    return PaperFill(
        identity=PaperFillIdentity(fill_id),
        order_identity=PaperOrderIdentity(order_id),
        intent=ExecutionIntent(
            portfolio_identity=_PORTFOLIO,
            listing_reference=_LISTING,
            side=side,
            quantity=Quantity(quantity),
            strategy_identity=_STRATEGY,
            decided_at=instant,
        ),
        quantity=Quantity(quantity),
        price=Price(price, _USD),
        filled_at=instant,
    )


class ReferencePaperFillStore(PaperFillStore):
    """A minimal store honouring every documented obligation."""

    def __init__(self) -> None:
        self.by_fill: dict[str, PaperFill] = {}
        self.by_order: dict[str, str] = {}

    def store(self, fills: tuple[PaperFill, ...]) -> int:
        if not fills:
            return 0

        batch_fills: set[str] = set()
        batch_orders: set[str] = set()
        for fill in fills:
            if fill.identity.identity in batch_fills:
                raise PaperFillConflictError("Batch contains two fills sharing one fill identity.")
            batch_fills.add(fill.identity.identity)
            if fill.order_identity.identity in batch_orders:
                raise PaperFillConflictError("Batch contains two fills for one order.")
            batch_orders.add(fill.order_identity.identity)

        # Stage everything, so a failure late in the batch persists nothing.
        staged_fills = dict(self.by_fill)
        staged_orders = dict(self.by_order)
        for fill in fills:
            key = fill.identity.identity
            order_key = fill.order_identity.identity
            existing = staged_fills.get(key)
            if existing is not None:
                if existing != fill:
                    raise PaperFillConflictError(
                        "A different paper fill is already stored under this fill identity."
                    )
                continue
            attached = staged_orders.get(order_key)
            if attached is not None and attached != key:
                raise PaperFillConflictError(
                    "This order is already attached to a different paper fill."
                )
            staged_fills[key] = fill
            staged_orders[order_key] = key

        self.by_fill = staged_fills
        self.by_order = staged_orders
        return len(fills)


@pytest.fixture
def store() -> ReferencePaperFillStore:
    return ReferencePaperFillStore()


# ---------------------------------------------------------------------------
# Append-only persistence
# ---------------------------------------------------------------------------


def test_an_empty_batch_is_a_safe_no_op(store: ReferencePaperFillStore) -> None:
    assert store.store(()) == 0
    assert store.by_fill == {}


def test_a_new_fill_is_persisted(store: ReferencePaperFillStore) -> None:
    fill = _fill()

    assert store.store((fill,)) == 1
    assert store.by_fill["fill-1"] == fill


def test_a_batch_of_new_fills_is_persisted(store: ReferencePaperFillStore) -> None:
    fills = (_fill("f-1", "o-1"), _fill("f-2", "o-2", day=21))

    assert store.store(fills) == 2
    assert len(store.by_fill) == 2


def test_identical_retry_is_idempotent(store: ReferencePaperFillStore) -> None:
    fill = _fill()
    store.store((fill,))

    assert store.store((fill,)) == 1
    assert len(store.by_fill) == 1


def test_a_structurally_equal_rebuild_is_idempotent(store: ReferencePaperFillStore) -> None:
    """Equality, not object identity, decides whether a retry is a retry."""
    store.store((_fill(),))

    assert store.store((_fill(),)) == 1
    assert len(store.by_fill) == 1


def test_accepted_and_idempotent_fills_both_count(store: ReferencePaperFillStore) -> None:
    store.store((_fill("f-1", "o-1"),))

    assert store.store((_fill("f-1", "o-1"), _fill("f-2", "o-2", day=21))) == 2


# ---------------------------------------------------------------------------
# Fill identity conflicts
# ---------------------------------------------------------------------------


def test_a_different_fill_under_one_identity_is_a_conflict(
    store: ReferencePaperFillStore,
) -> None:
    store.store((_fill("f-1", "o-1", quantity="10"),))

    with pytest.raises(PaperFillConflictError, match="already stored under this fill identity"):
        store.store((_fill("f-1", "o-1", quantity="99"),))


def test_a_conflicting_write_never_overwrites(store: ReferencePaperFillStore) -> None:
    original = _fill("f-1", "o-1", price="100")
    store.store((original,))

    with pytest.raises(PaperFillConflictError):
        store.store((_fill("f-1", "o-1", price="999"),))

    assert store.by_fill["f-1"] == original


@pytest.mark.parametrize(
    ("field", "kwargs"),
    [
        ("quantity", {"quantity": "11"}),
        ("price", {"price": "101"}),
        ("instant", {"day": 21}),
        ("side", {"side": OrderSide.SELL}),
    ],
)
def test_any_difference_makes_a_retry_a_conflict(
    store: ReferencePaperFillStore, field: str, kwargs: object
) -> None:
    store.store((_fill("f-1", "o-1"),))

    with pytest.raises(PaperFillConflictError):
        store.store((_fill("f-1", "o-1", **kwargs),))


def test_a_batch_sharing_one_fill_identity_is_rejected(
    store: ReferencePaperFillStore,
) -> None:
    with pytest.raises(PaperFillConflictError, match="two fills sharing one fill identity"):
        store.store((_fill("f-1", "o-1"), _fill("f-1", "o-2")))


# ---------------------------------------------------------------------------
# Order identity conflicts
# ---------------------------------------------------------------------------


def test_one_order_cannot_produce_a_second_fill(store: ReferencePaperFillStore) -> None:
    """The current model permits exactly one fill per order."""
    store.store((_fill("f-1", "o-1"),))

    with pytest.raises(PaperFillConflictError, match="already attached to a different paper fill"):
        store.store((_fill("f-2", "o-1", day=21),))


def test_a_batch_sharing_one_order_identity_is_rejected(
    store: ReferencePaperFillStore,
) -> None:
    with pytest.raises(PaperFillConflictError, match="two fills for one order"):
        store.store((_fill("f-1", "o-1"), _fill("f-2", "o-1")))


def test_distinct_orders_coexist(store: ReferencePaperFillStore) -> None:
    assert store.store((_fill("f-1", "o-1"), _fill("f-2", "o-2", day=21))) == 2
    assert len(store.by_order) == 2


# ---------------------------------------------------------------------------
# Batch atomicity
# ---------------------------------------------------------------------------


def test_a_failing_batch_persists_nothing(store: ReferencePaperFillStore) -> None:
    store.store((_fill("f-1", "o-1"),))

    with pytest.raises(PaperFillConflictError):
        store.store((_fill("f-2", "o-2", day=21), _fill("f-1", "o-1", price="999")))

    assert set(store.by_fill) == {"f-1"}
    assert store.by_fill["f-1"] == _fill("f-1", "o-1")


def test_partial_batch_success_is_not_permitted(store: ReferencePaperFillStore) -> None:
    with pytest.raises(PaperFillConflictError):
        store.store((_fill("f-1", "o-1"), _fill("f-2", "o-1", day=21)))

    assert store.by_fill == {}


# ---------------------------------------------------------------------------
# Port surface
# ---------------------------------------------------------------------------


def test_the_port_is_abstract() -> None:
    with pytest.raises(TypeError):
        PaperFillStore()


def test_the_conflict_error_is_dedicated() -> None:
    from northstar_application.ports import ForwardResearchRecordConflictError

    assert issubclass(PaperFillConflictError, ValueError)
    assert not issubclass(PaperFillConflictError, ForwardResearchRecordConflictError)
    assert not issubclass(ForwardResearchRecordConflictError, PaperFillConflictError)


def test_the_port_exposes_only_store() -> None:
    public = [name for name in vars(PaperFillStore) if not name.startswith("_")]

    assert public == ["store"]


def test_no_holdings_persistence_port_exists() -> None:
    """Positions and portfolios are folded from fills, never stored."""
    import northstar_application.ports as ports

    for forbidden in ("Position", "PaperPortfolioStore", "PaperPortfolioRepository", "PaperOrder"):
        assert not [name for name in ports.__all__ if forbidden in name]
