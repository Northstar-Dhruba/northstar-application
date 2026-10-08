"""Contract tests for the option contract economics store and repository ports.

The obligations are pinned against minimal in-memory reference implementations:
exact five-part lookup, insert-only storage, idempotent equal re-stores,
conflicts on change, in-batch duplicates and all-or-nothing batches.
"""

from __future__ import annotations

import ast
import importlib
import inspect
from decimal import Decimal
from pathlib import Path
from typing import get_type_hints

import pytest
from northstar_core.derivatives import ExpirationDate
from northstar_core.foundation.value_objects import Currency, ExchangeCode, Symbol
from northstar_core.options import (
    OptionContract,
    OptionContractEconomics,
    OptionPointValue,
    OptionProductReference,
    OptionRight,
    OptionStrike,
)

import northstar_application.ports as ports
from northstar_application.ports import (
    FuturesContractEconomicsConflictError,
    OptionContractEconomicsConflictError,
    OptionContractEconomicsRepository,
    OptionContractEconomicsStore,
)

_NIFTY = OptionProductReference(Symbol("NIFTY"), ExchangeCode("NSE"))
_INR = Currency("INR")


def _contract(
    expiration: str = "2026-10-27",
    strike: str = "25000",
    right: OptionRight = OptionRight.CALL,
    product: OptionProductReference = _NIFTY,
) -> OptionContract:
    return OptionContract(product, ExpirationDate(expiration), OptionStrike(Decimal(strike)), right)


def _economics(
    contract: OptionContract | None = None, amount: str = "65"
) -> OptionContractEconomics:
    return OptionContractEconomics(
        _contract() if contract is None else contract, OptionPointValue(Decimal(amount), _INR)
    )


class InMemoryOptionContractEconomics(
    OptionContractEconomicsStore, OptionContractEconomicsRepository
):
    """Reference implementation of both port contracts."""

    def __init__(self) -> None:
        self.rows: dict[OptionContract, OptionContractEconomics] = {}

    def store(self, economics: tuple[OptionContractEconomics, ...]) -> int:
        if not isinstance(economics, tuple):
            raise TypeError("economics must be a tuple.")
        seen: set[OptionContract] = set()
        for entry in economics:
            if not isinstance(entry, OptionContractEconomics):
                raise TypeError("economics must be OptionContractEconomics values.")
            if entry.contract in seen:
                raise OptionContractEconomicsConflictError("Batch repeats one contract.")
            seen.add(entry.contract)
            stored = self.rows.get(entry.contract)
            if stored is not None and stored != entry:
                raise OptionContractEconomicsConflictError(f"Conflict for {entry.contract}.")
        for entry in economics:
            self.rows.setdefault(entry.contract, entry)
        return len(economics)

    def get_economics(self, contract: OptionContract) -> OptionContractEconomics | None:
        return self.rows.get(contract)


# ---------------------------------------------------------------------------
# Port shape
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("port", [OptionContractEconomicsRepository, OptionContractEconomicsStore])
def test_the_ports_are_abstract(port: type) -> None:
    assert inspect.isabstract(port)
    with pytest.raises(TypeError):
        port()


def test_the_abstract_methods_are_exactly_one_per_port() -> None:
    assert OptionContractEconomicsRepository.__abstractmethods__ == frozenset({"get_economics"})
    assert OptionContractEconomicsStore.__abstractmethods__ == frozenset({"store"})


def test_the_signatures_are_option_typed() -> None:
    lookup = get_type_hints(OptionContractEconomicsRepository.get_economics)
    store = get_type_hints(OptionContractEconomicsStore.store)

    assert lookup["contract"] is OptionContract
    assert lookup["return"] == OptionContractEconomics | None
    assert store["economics"] == tuple[OptionContractEconomics, ...]
    assert store["return"] is int


def test_the_ports_and_error_are_exported() -> None:
    for name in (
        "OptionContractEconomicsConflictError",
        "OptionContractEconomicsRepository",
        "OptionContractEconomicsStore",
    ):
        assert name in ports.__all__
        assert getattr(ports, name) is not None


def test_the_conflict_error_is_a_value_error_of_its_own() -> None:
    assert issubclass(OptionContractEconomicsConflictError, ValueError)
    assert not issubclass(
        OptionContractEconomicsConflictError, FuturesContractEconomicsConflictError
    )
    assert not issubclass(
        FuturesContractEconomicsConflictError, OptionContractEconomicsConflictError
    )


@pytest.mark.parametrize(
    "module",
    [
        "northstar_application.ports.option_contract_economics_repository",
        "northstar_application.ports.option_contract_economics_store",
    ],
)
def test_the_option_ports_depend_on_no_futures_or_provider(module: str) -> None:
    source = Path(importlib.import_module(module).__file__).read_text(encoding="utf-8")
    imported = {
        node.module
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ImportFrom) and node.module
    }

    for name in imported:
        assert not name.startswith(("northstar_core.futures", "northstar_infrastructure"))
        assert name.split(".")[0] not in {"sqlite3", "requests", "httpx", "time", "datetime"}


# ---------------------------------------------------------------------------
# Store contract
# ---------------------------------------------------------------------------


def test_an_absent_contract_is_stored() -> None:
    store = InMemoryOptionContractEconomics()

    assert store.store((_economics(),)) == 1
    assert store.get_economics(_contract()) == _economics()


def test_an_equal_re_store_is_idempotent_and_counted() -> None:
    store = InMemoryOptionContractEconomics()
    store.store((_economics(),))

    assert store.store((_economics(amount="65.000"),)) == 1
    assert store.get_economics(_contract()) == _economics()


def test_different_economics_for_a_stored_contract_conflict_and_change_nothing() -> None:
    store = InMemoryOptionContractEconomics()
    store.store((_economics(),))

    with pytest.raises(OptionContractEconomicsConflictError):
        store.store((_economics(amount="75"),))
    assert store.get_economics(_contract()) == _economics()


@pytest.mark.parametrize(
    "second", [_economics(), _economics(amount="75")], ids=["equal", "different"]
)
def test_a_batch_repeating_one_contract_conflicts(second: OptionContractEconomics) -> None:
    store = InMemoryOptionContractEconomics()

    with pytest.raises(OptionContractEconomicsConflictError):
        store.store((_economics(), second))
    assert store.rows == {}


def test_an_empty_batch_returns_zero() -> None:
    store = InMemoryOptionContractEconomics()

    assert store.store(()) == 0
    assert store.rows == {}


def test_a_batch_is_all_or_nothing() -> None:
    store = InMemoryOptionContractEconomics()
    store.store((_economics(),))
    fresh = _economics(_contract(strike="25050"))

    with pytest.raises(OptionContractEconomicsConflictError):
        store.store((fresh, _economics(amount="75")))
    assert store.get_economics(fresh.contract) is None


def test_success_returns_the_batch_length() -> None:
    store = InMemoryOptionContractEconomics()
    batch = (
        _economics(_contract(right=OptionRight.CALL)),
        _economics(_contract(right=OptionRight.PUT)),
        _economics(_contract(strike="25050")),
    )

    assert store.store(batch) == 3


# ---------------------------------------------------------------------------
# Repository contract
# ---------------------------------------------------------------------------


def test_a_missing_contract_is_none() -> None:
    assert InMemoryOptionContractEconomics().get_economics(_contract()) is None


@pytest.mark.parametrize(
    "neighbour",
    [
        _contract(strike="25050"),
        _contract(right=OptionRight.PUT),
        _contract(expiration="2026-10-20"),
        _contract(product=OptionProductReference(Symbol("BANKNIFTY"), ExchangeCode("NSE"))),
        _contract(product=OptionProductReference(Symbol("NIFTY"), ExchangeCode("BSE"))),
    ],
    ids=["strike", "right", "expiry", "product", "exchange"],
)
def test_a_neighbouring_contract_never_answers(neighbour: OptionContract) -> None:
    store = InMemoryOptionContractEconomics()
    store.store((_economics(),))

    assert store.get_economics(neighbour) is None
