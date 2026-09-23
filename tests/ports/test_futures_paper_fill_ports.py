"""Contract tests for the futures paper fill store and repository ports.

One FuturesPaperOrder has zero or one FuturesPaperFill. The obligations are
pinned against minimal in-memory reference implementations whose fixtures
include the sub-second instants that defeat text ordering.
"""

from __future__ import annotations

import ast
import importlib
from decimal import Decimal
from functools import cmp_to_key
from pathlib import Path
from typing import get_type_hints

import pytest
from northstar_core.derivatives import ExpirationDate, QuoteValue
from northstar_core.foundation.value_objects import ExchangeCode, PointInTime, Symbol
from northstar_core.futures import FuturesContract, FuturesProductReference
from northstar_core.paper_trading import (
    FuturesContractCount,
    FuturesExecutionIntent,
    FuturesPaperFill,
    OrderSide,
    PaperFillIdentity,
    PaperOrderIdentity,
    PaperPortfolioIdentity,
)
from northstar_core.strategy import StrategyIdentity

from northstar_application.application_services import FuturesPaperExecutionIdentityService
from northstar_application.ports import (
    FuturesPaperFillConflictError,
    FuturesPaperFillQuery,
    FuturesPaperFillRepository,
    FuturesPaperFillStore,
    FuturesPaperOrderConflictError,
    InvalidFuturesPaperFillQueryError,
    PaperFillConflictError,
)


def _contract(product: str = "ES", expiry: str = "2026-12-18") -> FuturesContract:
    return FuturesContract(
        FuturesProductReference(Symbol(product), ExchangeCode("CME")), ExpirationDate(expiry)
    )


_PORTFOLIO = PaperPortfolioIdentity("futures-paper-1")
_OTHER_PORTFOLIO = PaperPortfolioIdentity("futures-paper-2")
_STRATEGY = StrategyIdentity("futures-forward")
_DECIDED_AT = PointInTime("2026-09-14T21:00:00Z")
_WHOLE_SECOND = PointInTime("2026-09-15T21:00:00Z")
_SUB_SECOND = PointInTime("2026-09-15T21:00:00.1Z")
_NEXT_DAY = PointInTime("2026-09-16T21:00:00Z")


def _fill(
    fill_id: str = "fill-1",
    order_id: str = "order-1",
    *,
    portfolio: PaperPortfolioIdentity = _PORTFOLIO,
    contract: FuturesContract | None = None,
    strategy: StrategyIdentity = _STRATEGY,
    side: OrderSide = OrderSide.BUY,
    contracts: int = 2,
    quote: str = "7663.25",
    filled_at: PointInTime = _WHOLE_SECOND,
) -> FuturesPaperFill:
    count = FuturesContractCount(contracts)
    return FuturesPaperFill(
        identity=PaperFillIdentity(fill_id),
        order_identity=PaperOrderIdentity(order_id),
        intent=FuturesExecutionIntent(
            portfolio_identity=portfolio,
            contract=contract if contract is not None else _contract(),
            side=side,
            contracts=count,
            strategy_identity=strategy,
            decided_at=_DECIDED_AT,
        ),
        contracts=count,
        fill_quote=QuoteValue(Decimal(quote)),
        filled_at=filled_at,
    )


class ReferenceFuturesPaperFillStore(FuturesPaperFillStore):
    """A minimal store honouring every documented obligation."""

    def __init__(self) -> None:
        self.by_fill: dict[str, FuturesPaperFill] = {}
        self.by_order: dict[str, str] = {}

    def store(self, fills: tuple[FuturesPaperFill, ...]) -> int:
        if not fills:
            return 0

        batch_fills: set[str] = set()
        batch_orders: set[str] = set()
        for fill in fills:
            if fill.identity.identity in batch_fills:
                raise FuturesPaperFillConflictError(
                    "Batch contains two fills sharing one fill identity."
                )
            batch_fills.add(fill.identity.identity)
            if fill.order_identity.identity in batch_orders:
                raise FuturesPaperFillConflictError("Batch contains two fills for one order.")
            batch_orders.add(fill.order_identity.identity)

        staged_fills = dict(self.by_fill)
        staged_orders = dict(self.by_order)
        for fill in fills:
            key = fill.identity.identity
            order_key = fill.order_identity.identity
            existing = staged_fills.get(key)
            if existing is not None:
                if existing != fill:
                    raise FuturesPaperFillConflictError(
                        "A different futures paper fill is already stored under this identity."
                    )
                continue
            attached = staged_orders.get(order_key)
            if attached is not None and attached != key:
                raise FuturesPaperFillConflictError(
                    "This order is already attached to a different futures paper fill."
                )
            staged_fills[key] = fill
            staged_orders[order_key] = key

        self.by_fill = staged_fills
        self.by_order = staged_orders
        return len(fills)


def _compare(left: FuturesPaperFill, right: FuturesPaperFill) -> int:
    instant = left.filled_at.compare(right.filled_at)
    if instant:
        return instant
    a, b = left.order_identity.identity, right.order_identity.identity
    return (a > b) - (a < b)


class ReferenceFuturesPaperFillRepository(FuturesPaperFillRepository):
    """A minimal repository honouring the documented ordering obligation."""

    def __init__(self, fills: tuple[FuturesPaperFill, ...] = ()) -> None:
        self.fills = fills

    def get_fills(self, query: FuturesPaperFillQuery) -> tuple[FuturesPaperFill, ...]:
        owned = [f for f in self.fills if f.portfolio_identity == query.portfolio_identity]
        return tuple(sorted(owned, key=cmp_to_key(_compare)))


@pytest.fixture
def store() -> ReferenceFuturesPaperFillStore:
    return ReferenceFuturesPaperFillStore()


def _query(portfolio: PaperPortfolioIdentity = _PORTFOLIO) -> FuturesPaperFillQuery:
    return FuturesPaperFillQuery(portfolio)


# ---------------------------------------------------------------------------
# Store: immutable, idempotent persistence
# ---------------------------------------------------------------------------


def test_an_empty_batch_is_a_safe_no_op(store: ReferenceFuturesPaperFillStore) -> None:
    assert store.store(()) == 0
    assert store.by_fill == {}


def test_a_new_fill_is_persisted(store: ReferenceFuturesPaperFillStore) -> None:
    assert store.store((_fill(),)) == 1
    assert store.by_fill["fill-1"] == _fill()


def test_an_equal_retry_is_idempotent(store: ReferenceFuturesPaperFillStore) -> None:
    store.store((_fill(),))

    assert store.store((_fill(),)) == 1
    assert len(store.by_fill) == 1


@pytest.mark.parametrize(
    "overrides",
    [
        {"quote": "7663.5"},
        {"quote": "-37.63"},
        {"filled_at": _NEXT_DAY},
        {"side": OrderSide.SELL},
        {"contracts": 3},
    ],
    ids=["quote", "negative-quote", "fill-instant", "side", "contracts"],
)
def test_a_different_fill_under_one_identity_is_a_conflict(
    store: ReferenceFuturesPaperFillStore, overrides: dict
) -> None:
    original = _fill()
    store.store((original,))

    with pytest.raises(FuturesPaperFillConflictError, match="already stored"):
        store.store((_fill(**overrides),))

    assert store.by_fill["fill-1"] == original


def test_zero_and_negative_quotes_are_storable(store: ReferenceFuturesPaperFillStore) -> None:
    fills = (_fill("f-1", "o-1", quote="0"), _fill("f-2", "o-2", quote="-37.63"))

    assert store.store(fills) == 2


# ---------------------------------------------------------------------------
# Store: zero or one fill per order
# ---------------------------------------------------------------------------


def test_an_order_without_a_fill_is_simply_absent(store: ReferenceFuturesPaperFillStore) -> None:
    store.store((_fill("f-1", "o-1"),))

    assert "o-2" not in store.by_order


def test_a_second_fill_for_one_order_is_a_conflict_not_a_partial_fill(
    store: ReferenceFuturesPaperFillStore,
) -> None:
    store.store((_fill("f-1", "o-1"),))

    with pytest.raises(FuturesPaperFillConflictError, match="already attached"):
        store.store((_fill("f-2", "o-1", filled_at=_NEXT_DAY),))

    assert set(store.by_fill) == {"f-1"}


def test_derived_identities_make_a_changed_retry_a_conflict(
    store: ReferenceFuturesPaperFillStore,
) -> None:
    """Fill identity comes from the order identity, so a re-simulated fill with a
    different quote lands on the same key and cannot be stored beside the first."""
    service = FuturesPaperExecutionIdentityService()
    order_identity = PaperOrderIdentity("derived-order")
    fill_identity = service.fill_identity(order_identity).identity

    store.store((_fill(fill_identity, order_identity.identity, quote="100"),))

    with pytest.raises(FuturesPaperFillConflictError):
        store.store((_fill(fill_identity, order_identity.identity, quote="101"),))


@pytest.mark.parametrize(
    ("fills", "message"),
    [
        ((_fill("f-1", "o-1"), _fill("f-1", "o-2")), "two fills sharing one fill identity"),
        ((_fill("f-1", "o-1"), _fill("f-1", "o-1")), "two fills sharing one fill identity"),
        ((_fill("f-1", "o-1"), _fill("f-2", "o-1")), "two fills for one order"),
    ],
    ids=["same-fill-id", "equal-duplicate", "same-order"],
)
def test_duplicate_batch_identities_are_rejected(
    store: ReferenceFuturesPaperFillStore, fills: tuple, message: str
) -> None:
    with pytest.raises(FuturesPaperFillConflictError, match=message):
        store.store(fills)

    assert store.by_fill == {}


def test_a_failing_batch_persists_nothing(store: ReferenceFuturesPaperFillStore) -> None:
    store.store((_fill("f-1", "o-1"),))

    with pytest.raises(FuturesPaperFillConflictError):
        store.store((_fill("f-2", "o-2"), _fill("f-1", "o-1", quote="999")))

    assert set(store.by_fill) == {"f-1"}


def test_fills_across_portfolios_strategies_and_contracts_coexist(
    store: ReferenceFuturesPaperFillStore,
) -> None:
    fills = (
        _fill("f-1", "o-1"),
        _fill("f-2", "o-2", portfolio=_OTHER_PORTFOLIO),
        _fill("f-3", "o-3", strategy=StrategyIdentity("other")),
        _fill("f-4", "o-4", contract=_contract("MES")),
        _fill("f-5", "o-5", contract=_contract(expiry="2027-03-19")),
    )

    assert store.store(fills) == 5


# ---------------------------------------------------------------------------
# Repository: query contract
# ---------------------------------------------------------------------------


def test_the_query_carries_only_the_portfolio_identity() -> None:
    assert _query().portfolio_identity == _PORTFOLIO
    assert FuturesPaperFillQuery.__slots__ == ("portfolio_identity",)
    for absent in ("status", "contract", "as_of", "start", "end", "timeframe"):
        assert not hasattr(_query(), absent)


@pytest.mark.parametrize("value", [None, "futures-paper-1", PaperFillIdentity("x"), _STRATEGY])
def test_the_query_validates_its_portfolio_identity(value: object) -> None:
    with pytest.raises(InvalidFuturesPaperFillQueryError, match="PaperPortfolioIdentity"):
        FuturesPaperFillQuery(value)


def test_the_query_is_immutable_and_compares_by_value() -> None:
    assert _query() == _query()
    assert hash(_query()) == hash(_query())
    assert _query() != _query(_OTHER_PORTFOLIO)

    with pytest.raises(AttributeError):
        _query().portfolio_identity = _OTHER_PORTFOLIO


# ---------------------------------------------------------------------------
# Repository: retrieval and ordering
# ---------------------------------------------------------------------------


def test_an_empty_repository_returns_an_empty_tuple() -> None:
    assert ReferenceFuturesPaperFillRepository().get_fills(_query()) == ()


def test_only_the_queried_portfolio_is_returned_as_a_tuple() -> None:
    fills = (
        _fill("f-1", "o-1"),
        _fill("f-2", "o-2", portfolio=_OTHER_PORTFOLIO),
        _fill("f-3", "o-3", contract=_contract("MES")),
    )

    retrieved = ReferenceFuturesPaperFillRepository(fills).get_fills(_query())

    assert isinstance(retrieved, tuple)
    assert [fill.identity.identity for fill in retrieved] == ["f-1", "f-3"]


def test_fills_are_ordered_by_fill_instant_then_order_identity() -> None:
    fills = (
        _fill("f-a", "order-z", filled_at=_NEXT_DAY),
        _fill("f-b", "order-b", filled_at=_WHOLE_SECOND),
        _fill("f-c", "order-a", filled_at=_WHOLE_SECOND),
    )

    retrieved = ReferenceFuturesPaperFillRepository(fills).get_fills(_query())

    assert [fill.order_identity.identity for fill in retrieved] == [
        "order-a",
        "order-b",
        "order-z",
    ]


def test_the_tie_break_is_order_identity_not_fill_identity() -> None:
    fills = (
        _fill("f-a", "order-b", filled_at=_WHOLE_SECOND),
        _fill("f-b", "order-a", filled_at=_WHOLE_SECOND),
    )

    retrieved = ReferenceFuturesPaperFillRepository(fills).get_fills(_query())

    assert [fill.identity.identity for fill in retrieved] == ["f-b", "f-a"]


def test_ordering_is_chronological_not_textual() -> None:
    """'.1Z' sorts before 'Z' as text while being the later instant."""
    fills = (
        _fill("f-1", "o-1", filled_at=_SUB_SECOND),
        _fill("f-2", "o-2", filled_at=_WHOLE_SECOND),
    )

    retrieved = ReferenceFuturesPaperFillRepository(fills).get_fills(_query())
    text_ordered = sorted(fills, key=lambda fill: fill.filled_at.value)

    assert _SUB_SECOND.value < _WHOLE_SECOND.value
    assert [fill.filled_at for fill in retrieved] == [_WHOLE_SECOND, _SUB_SECOND]
    assert list(retrieved) != text_ordered


def test_offset_equivalent_fill_instants_tie_break_by_order_identity() -> None:
    offset = PointInTime("2026-09-16T02:30:00+05:30")
    fills = (_fill("f-1", "order-z", filled_at=offset), _fill("f-2", "order-a"))

    retrieved = ReferenceFuturesPaperFillRepository(fills).get_fills(_query())

    assert offset.compare(_WHOLE_SECOND) == 0
    assert [fill.order_identity.identity for fill in retrieved] == ["order-a", "order-z"]


# ---------------------------------------------------------------------------
# Port surface
# ---------------------------------------------------------------------------


def test_the_ports_are_abstract() -> None:
    with pytest.raises(TypeError):
        FuturesPaperFillStore()
    with pytest.raises(TypeError):
        FuturesPaperFillRepository()


def test_the_ports_expose_only_their_single_operation() -> None:
    assert [n for n in vars(FuturesPaperFillStore) if not n.startswith("_")] == ["store"]
    assert [n for n in vars(FuturesPaperFillRepository) if not n.startswith("_")] == ["get_fills"]


def test_the_conflict_error_is_dedicated() -> None:
    assert issubclass(FuturesPaperFillConflictError, ValueError)
    for other in (FuturesPaperOrderConflictError, PaperFillConflictError):
        assert not issubclass(FuturesPaperFillConflictError, other)
        assert not issubclass(other, FuturesPaperFillConflictError)


def test_the_ports_are_typed_on_futures_fills_not_equity_fills() -> None:
    assert get_type_hints(FuturesPaperFillStore.store)["fills"] == tuple[FuturesPaperFill, ...]
    assert (
        get_type_hints(FuturesPaperFillRepository.get_fills)["return"]
        == (tuple[FuturesPaperFill, ...])
    )


_MODULES = (
    "northstar_application.ports.futures_paper_fill_store",
    "northstar_application.ports.futures_paper_fill_repository",
)


@pytest.mark.parametrize("module_name", _MODULES)
def test_the_ports_depend_on_no_storage_provider_clock_or_economics(module_name: str) -> None:
    module = importlib.import_module(module_name)
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    modules = {
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module
    } | {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    names = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }

    for module_path in modules:
        assert module_path.split(".")[0] not in {
            "northstar_infrastructure",
            "sqlite3",
            "databento",
            "exchange_calendars",
            "requests",
            "time",
            "datetime",
        }
    assert not names & {"PaperOrder", "PaperFill", "PaperOrderStatus", "Price", "Money"}


def test_no_order_status_or_holdings_port_is_exported() -> None:
    import northstar_application.ports as ports

    for forbidden in ("Status", "State", "Pending", "FuturesPosition", "FuturesPaperPortfolio"):
        assert not [name for name in ports.__all__ if forbidden in name]
    for name in (
        "FuturesPaperOrderStore",
        "FuturesPaperOrderRepository",
        "FuturesPaperOrderQuery",
        "FuturesPaperOrderConflictError",
        "InvalidFuturesPaperOrderQueryError",
        "FuturesPaperFillStore",
        "FuturesPaperFillRepository",
        "FuturesPaperFillQuery",
        "FuturesPaperFillConflictError",
        "InvalidFuturesPaperFillQueryError",
    ):
        assert name in ports.__all__
