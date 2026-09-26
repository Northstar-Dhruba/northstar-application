"""Contract tests for the futures paper order store and repository ports.

The ports cannot enforce their own semantics, so the obligations are pinned
against minimal in-memory reference implementations. An adapter that fails
these behaviours does not honour the port, whatever its storage engine.
"""

from __future__ import annotations

import ast
import importlib
from functools import cmp_to_key
from pathlib import Path

import pytest
from northstar_core.derivatives import ExpirationDate
from northstar_core.foundation.value_objects import ExchangeCode, PointInTime, Symbol
from northstar_core.futures import FuturesContract, FuturesProductReference
from northstar_core.paper_trading import (
    FuturesContractCount,
    FuturesExecutionIntent,
    FuturesPaperOrder,
    OrderSide,
    PaperFillIdentity,
    PaperOrderIdentity,
    PaperPortfolioIdentity,
)
from northstar_core.strategy import StrategyIdentity

from northstar_application.ports import (
    FuturesForwardResearchRecordConflictError,
    FuturesPaperFillConflictError,
    FuturesPaperOrderConflictError,
    FuturesPaperOrderQuery,
    FuturesPaperOrderRepository,
    FuturesPaperOrderStore,
    InvalidFuturesPaperOrderQueryError,
    PaperFillConflictError,
)


def _contract(product: str = "ES", expiry: str = "2026-12-18") -> FuturesContract:
    return FuturesContract(
        FuturesProductReference(Symbol(product), ExchangeCode("CME")), ExpirationDate(expiry)
    )


_PORTFOLIO = PaperPortfolioIdentity("futures-paper-1")
_OTHER_PORTFOLIO = PaperPortfolioIdentity("futures-paper-2")
_STRATEGY = StrategyIdentity("futures-forward")
_WHOLE_SECOND = PointInTime("2026-09-15T21:00:00Z")
_SUB_SECOND = PointInTime("2026-09-15T21:00:00.1Z")
_NEXT_DAY = PointInTime("2026-09-16T21:00:00Z")


def _order(
    order_id: str = "order-1",
    *,
    portfolio: PaperPortfolioIdentity = _PORTFOLIO,
    contract: FuturesContract | None = None,
    side: OrderSide = OrderSide.BUY,
    contracts: int = 2,
    strategy: StrategyIdentity = _STRATEGY,
    decided_at: PointInTime = _WHOLE_SECOND,
) -> FuturesPaperOrder:
    return FuturesPaperOrder(
        identity=PaperOrderIdentity(order_id),
        intent=FuturesExecutionIntent(
            portfolio_identity=portfolio,
            contract=contract if contract is not None else _contract(),
            side=side,
            contracts=FuturesContractCount(contracts),
            strategy_identity=strategy,
            decided_at=decided_at,
        ),
    )


class ReferenceFuturesPaperOrderStore(FuturesPaperOrderStore):
    """A minimal store honouring every documented obligation."""

    def __init__(self) -> None:
        self.orders: dict[str, FuturesPaperOrder] = {}

    def store(self, orders: tuple[FuturesPaperOrder, ...]) -> int:
        if not orders:
            return 0

        seen: set[str] = set()
        for order in orders:
            if order.identity.identity in seen:
                raise FuturesPaperOrderConflictError(
                    "Batch contains two orders sharing one order identity."
                )
            seen.add(order.identity.identity)

        staged = dict(self.orders)
        for order in orders:
            existing = staged.get(order.identity.identity)
            if existing is not None and existing != order:
                raise FuturesPaperOrderConflictError(
                    "A different futures paper order is already stored under this identity."
                )
            staged[order.identity.identity] = order

        self.orders = staged
        return len(orders)


def _compare(left: FuturesPaperOrder, right: FuturesPaperOrder) -> int:
    instant = left.intent.decided_at.compare(right.intent.decided_at)
    if instant:
        return instant
    a, b = left.identity.identity, right.identity.identity
    return (a > b) - (a < b)


class ReferenceFuturesPaperOrderRepository(FuturesPaperOrderRepository):
    """A minimal repository honouring the documented ordering obligation."""

    def __init__(self, orders: tuple[FuturesPaperOrder, ...] = ()) -> None:
        self.orders = orders

    def get_orders(self, query: FuturesPaperOrderQuery) -> tuple[FuturesPaperOrder, ...]:
        owned = [o for o in self.orders if o.intent.portfolio_identity == query.portfolio_identity]
        return tuple(sorted(owned, key=cmp_to_key(_compare)))


@pytest.fixture
def store() -> ReferenceFuturesPaperOrderStore:
    return ReferenceFuturesPaperOrderStore()


# ---------------------------------------------------------------------------
# Store: immutable, idempotent persistence
# ---------------------------------------------------------------------------


def test_an_empty_batch_is_a_safe_no_op(store: ReferenceFuturesPaperOrderStore) -> None:
    assert store.store(()) == 0
    assert store.orders == {}


def test_a_new_order_is_persisted(store: ReferenceFuturesPaperOrderStore) -> None:
    order = _order()

    assert store.store((order,)) == 1
    assert store.orders["order-1"] == order


def test_an_equal_retry_is_idempotent(store: ReferenceFuturesPaperOrderStore) -> None:
    store.store((_order(),))

    assert store.store((_order(),)) == 1
    assert len(store.orders) == 1


def test_accepted_and_idempotent_orders_both_count(
    store: ReferenceFuturesPaperOrderStore,
) -> None:
    store.store((_order("o-1"),))

    assert store.store((_order("o-1"), _order("o-2", decided_at=_NEXT_DAY))) == 2


@pytest.mark.parametrize(
    "overrides",
    [
        {"side": OrderSide.SELL},
        {"contracts": 3},
        {"decided_at": _NEXT_DAY},
        {"contract": _contract("MES")},
        {"strategy": StrategyIdentity("other")},
        {"portfolio": _OTHER_PORTFOLIO},
    ],
    ids=["side", "contracts", "instant", "contract", "strategy", "portfolio"],
)
def test_a_different_order_under_one_identity_is_a_conflict(
    store: ReferenceFuturesPaperOrderStore, overrides: dict
) -> None:
    original = _order()
    store.store((original,))

    with pytest.raises(FuturesPaperOrderConflictError, match="already stored"):
        store.store((_order(**overrides),))

    assert store.orders["order-1"] == original


def test_a_batch_sharing_one_identity_is_rejected_even_when_equal(
    store: ReferenceFuturesPaperOrderStore,
) -> None:
    with pytest.raises(FuturesPaperOrderConflictError, match="two orders sharing"):
        store.store((_order(), _order()))

    assert store.orders == {}


def test_a_failing_batch_persists_nothing(store: ReferenceFuturesPaperOrderStore) -> None:
    store.store((_order("o-1"),))

    with pytest.raises(FuturesPaperOrderConflictError):
        store.store((_order("o-2"), _order("o-1", contracts=9)))

    assert set(store.orders) == {"o-1"}


def test_orders_across_portfolios_strategies_and_contracts_coexist(
    store: ReferenceFuturesPaperOrderStore,
) -> None:
    orders = (
        _order("o-1"),
        _order("o-2", portfolio=_OTHER_PORTFOLIO),
        _order("o-3", strategy=StrategyIdentity("other")),
        _order("o-4", contract=_contract("MES")),
        _order("o-5", contract=_contract(expiry="2027-03-19")),
    )

    assert store.store(orders) == 5
    assert len(store.orders) == 5


# ---------------------------------------------------------------------------
# Repository: query contract
# ---------------------------------------------------------------------------


def test_the_query_carries_only_the_portfolio_identity() -> None:
    query = FuturesPaperOrderQuery(_PORTFOLIO)

    assert query.portfolio_identity == _PORTFOLIO
    assert FuturesPaperOrderQuery.__slots__ == ("portfolio_identity",)
    for absent in ("status", "state", "pending", "contract", "as_of", "timeframe"):
        assert not hasattr(query, absent)


@pytest.mark.parametrize(
    "value", [None, "futures-paper-1", PaperOrderIdentity("futures-paper-1"), _STRATEGY]
)
def test_the_query_validates_its_portfolio_identity(value: object) -> None:
    with pytest.raises(InvalidFuturesPaperOrderQueryError, match="PaperPortfolioIdentity"):
        FuturesPaperOrderQuery(value)


def test_the_query_is_immutable_and_compares_by_value() -> None:
    assert FuturesPaperOrderQuery(_PORTFOLIO) == FuturesPaperOrderQuery(_PORTFOLIO)
    assert hash(FuturesPaperOrderQuery(_PORTFOLIO)) == hash(FuturesPaperOrderQuery(_PORTFOLIO))

    with pytest.raises(AttributeError):
        FuturesPaperOrderQuery(_PORTFOLIO).portfolio_identity = _OTHER_PORTFOLIO


# ---------------------------------------------------------------------------
# Repository: retrieval and ordering
# ---------------------------------------------------------------------------


def test_an_empty_repository_returns_an_empty_tuple() -> None:
    assert (
        ReferenceFuturesPaperOrderRepository().get_orders(FuturesPaperOrderQuery(_PORTFOLIO)) == ()
    )


def test_only_the_queried_portfolio_is_returned_as_a_tuple() -> None:
    orders = (
        _order("o-1"),
        _order("o-2", portfolio=_OTHER_PORTFOLIO),
        _order("o-3", contract=_contract("MES"), strategy=StrategyIdentity("other")),
    )

    retrieved = ReferenceFuturesPaperOrderRepository(orders).get_orders(
        FuturesPaperOrderQuery(_PORTFOLIO)
    )

    assert isinstance(retrieved, tuple)
    assert [order.identity.identity for order in retrieved] == ["o-1", "o-3"]


def test_orders_are_ordered_by_decision_then_identity() -> None:
    orders = (
        _order("zeta", decided_at=_NEXT_DAY),
        _order("beta", decided_at=_WHOLE_SECOND),
        _order("alpha", decided_at=_WHOLE_SECOND),
    )

    retrieved = ReferenceFuturesPaperOrderRepository(orders).get_orders(
        FuturesPaperOrderQuery(_PORTFOLIO)
    )

    assert [order.identity.identity for order in retrieved] == ["alpha", "beta", "zeta"]


def test_ordering_is_chronological_not_textual() -> None:
    """'.1Z' sorts before 'Z' as text while being the later instant."""
    orders = (_order("a", decided_at=_SUB_SECOND), _order("b", decided_at=_WHOLE_SECOND))

    retrieved = ReferenceFuturesPaperOrderRepository(orders).get_orders(
        FuturesPaperOrderQuery(_PORTFOLIO)
    )
    text_ordered = sorted(orders, key=lambda order: order.intent.decided_at.value)

    assert _SUB_SECOND.value < _WHOLE_SECOND.value
    assert [order.intent.decided_at for order in retrieved] == [_WHOLE_SECOND, _SUB_SECOND]
    assert list(retrieved) != text_ordered


# ---------------------------------------------------------------------------
# Port surface
# ---------------------------------------------------------------------------


def test_the_ports_are_abstract() -> None:
    with pytest.raises(TypeError):
        FuturesPaperOrderStore()
    with pytest.raises(TypeError):
        FuturesPaperOrderRepository()


def test_the_ports_expose_only_their_single_operation() -> None:
    assert [n for n in vars(FuturesPaperOrderStore) if not n.startswith("_")] == ["store"]
    assert [n for n in vars(FuturesPaperOrderRepository) if not n.startswith("_")] == ["get_orders"]


def test_the_conflict_error_is_dedicated() -> None:
    assert issubclass(FuturesPaperOrderConflictError, ValueError)
    for other in (
        FuturesPaperFillConflictError,
        PaperFillConflictError,
        FuturesForwardResearchRecordConflictError,
    ):
        assert not issubclass(FuturesPaperOrderConflictError, other)
        assert not issubclass(other, FuturesPaperOrderConflictError)


def test_the_ports_are_typed_on_futures_orders_not_equity_orders() -> None:
    from typing import get_type_hints

    store_hints = get_type_hints(FuturesPaperOrderStore.store)
    repository_hints = get_type_hints(FuturesPaperOrderRepository.get_orders)

    assert store_hints["orders"] == tuple[FuturesPaperOrder, ...]
    assert repository_hints["return"] == tuple[FuturesPaperOrder, ...]


_MODULES = (
    "northstar_application.ports.futures_paper_order_store",
    "northstar_application.ports.futures_paper_order_repository",
)


@pytest.mark.parametrize("module_name", _MODULES)
def test_the_ports_depend_on_no_storage_provider_clock_or_status(module_name: str) -> None:
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
    assert not names & {"PaperOrder", "PaperFill", "PaperOrderStatus"}


def test_a_fill_identity_is_not_an_order_query_identity() -> None:
    with pytest.raises(InvalidFuturesPaperOrderQueryError):
        FuturesPaperOrderQuery(PaperFillIdentity("futures-paper-1"))
