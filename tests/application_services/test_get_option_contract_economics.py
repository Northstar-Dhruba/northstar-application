"""Tests for the fail-closed option contract economics query."""

from __future__ import annotations

import ast
from decimal import Decimal
from pathlib import Path

import pytest
from northstar_core.derivatives import ExpirationDate
from northstar_core.foundation.value_objects import Currency, ExchangeCode, Symbol
from northstar_core.futures import (
    FuturesContract,
    FuturesContractEconomics,
    FuturesPointValue,
    FuturesProductReference,
)
from northstar_core.options import (
    OptionContract,
    OptionContractEconomics,
    OptionPointValue,
    OptionProductReference,
    OptionRight,
    OptionStrike,
)

import northstar_application.application_services as services
import northstar_application.application_services.get_option_contract_economics as module
from northstar_application.application_services import (
    FuturesContractEconomicsNotFoundError,
    GetOptionContractEconomicsUseCase,
    OptionContractEconomicsContractViolationError,
    OptionContractEconomicsNotFoundError,
)
from northstar_application.ports import OptionContractEconomicsRepository

_NSE = ExchangeCode("NSE")
_NIFTY = OptionProductReference(Symbol("NIFTY"), _NSE)
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


class RecordingRepository(OptionContractEconomicsRepository):
    """Answers every lookup with one fixed value and records what was asked."""

    def __init__(self, answer: object) -> None:
        self.answer = answer
        self.calls: list[OptionContract] = []

    def get_economics(self, contract: OptionContract) -> OptionContractEconomics | None:
        self.calls.append(contract)
        return self.answer  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Exact answer
# ---------------------------------------------------------------------------


def test_the_requested_contracts_economics_are_returned() -> None:
    economics = _economics()
    repository = RecordingRepository(economics)

    assert GetOptionContractEconomicsUseCase(repository).execute(_contract()) is economics


def test_the_repository_is_asked_once_for_exactly_the_requested_contract() -> None:
    repository = RecordingRepository(_economics())

    GetOptionContractEconomicsUseCase(repository).execute(_contract())

    assert repository.calls == [_contract()]


# ---------------------------------------------------------------------------
# Fail closed
# ---------------------------------------------------------------------------


def test_missing_economics_raise_not_found_with_the_contract() -> None:
    repository = RecordingRepository(None)

    with pytest.raises(OptionContractEconomicsNotFoundError) as raised:
        GetOptionContractEconomicsUseCase(repository).execute(_contract())

    assert raised.value.contract == _contract()
    assert str(raised.value) == (
        "Option contract economics not configured for NIFTY@NSE 2026-10-27 25000 CALL."
    )


def test_a_missing_contract_is_not_retried_against_any_neighbour() -> None:
    """There is no fallback: one lookup, for the requested contract, then failure."""
    repository = RecordingRepository(None)

    with pytest.raises(OptionContractEconomicsNotFoundError):
        GetOptionContractEconomicsUseCase(repository).execute(_contract(strike="25050"))

    assert repository.calls == [_contract(strike="25050")]


@pytest.mark.parametrize(
    "answer",
    [
        FuturesContractEconomics(
            FuturesContract(
                FuturesProductReference(Symbol("NIFTY"), _NSE), ExpirationDate("2026-10-27")
            ),
            FuturesPointValue(Decimal("65"), _INR),
        ),
        OptionPointValue(Decimal("65"), _INR),
        "NIFTY@NSE 2026-10-27 25000 CALL 65 INR/premium-point/contract",
        Decimal("65"),
    ],
    ids=["futures-economics", "point-value", "text", "decimal"],
)
def test_a_wrong_type_answer_is_a_contract_violation(answer: object) -> None:
    with pytest.raises(OptionContractEconomicsContractViolationError, match="must return"):
        GetOptionContractEconomicsUseCase(RecordingRepository(answer)).execute(_contract())


@pytest.mark.parametrize(
    "other",
    [
        _contract(product=OptionProductReference(Symbol("BANKNIFTY"), _NSE)),
        _contract(product=OptionProductReference(Symbol("NIFTY"), ExchangeCode("BSE"))),
        _contract(expiration="2026-10-20"),
        _contract(strike="25050"),
        _contract(right=OptionRight.PUT),
    ],
    ids=["product", "exchange", "expiry", "strike", "right"],
)
def test_economics_for_any_other_contract_are_a_contract_violation(other: OptionContract) -> None:
    repository = RecordingRepository(_economics(other))

    with pytest.raises(OptionContractEconomicsContractViolationError) as raised:
        GetOptionContractEconomicsUseCase(repository).execute(_contract())

    assert str(other) in str(raised.value)
    assert str(_contract()) in str(raised.value)


# ---------------------------------------------------------------------------
# Input and construction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "contract",
    [
        None,
        "NIFTY@NSE 2026-10-27 25000 CALL",
        FuturesContract(
            FuturesProductReference(Symbol("NIFTY"), _NSE), ExpirationDate("2026-10-27")
        ),
    ],
    ids=["none", "text", "futures-contract"],
)
def test_a_non_option_contract_is_rejected_before_any_lookup(contract: object) -> None:
    repository = RecordingRepository(_economics())

    with pytest.raises(TypeError, match="must be an OptionContract"):
        GetOptionContractEconomicsUseCase(repository).execute(contract)  # type: ignore[arg-type]
    assert repository.calls == []


@pytest.mark.parametrize("repository", [None, object(), {}])
def test_the_repository_must_be_the_option_port(repository: object) -> None:
    with pytest.raises(TypeError, match="OptionContractEconomicsRepository"):
        GetOptionContractEconomicsUseCase(repository)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Error types and surface
# ---------------------------------------------------------------------------


def test_the_errors_are_value_errors_distinct_from_futures() -> None:
    for error in (
        OptionContractEconomicsNotFoundError,
        OptionContractEconomicsContractViolationError,
    ):
        assert issubclass(error, ValueError)
        assert not issubclass(error, FuturesContractEconomicsNotFoundError)
    assert not issubclass(
        OptionContractEconomicsNotFoundError, OptionContractEconomicsContractViolationError
    )


def test_the_not_found_contract_defaults_to_none() -> None:
    assert OptionContractEconomicsNotFoundError("missing").contract is None


def test_the_use_case_and_errors_are_exported() -> None:
    for name in (
        "GetOptionContractEconomicsUseCase",
        "OptionContractEconomicsContractViolationError",
        "OptionContractEconomicsNotFoundError",
    ):
        assert name in services.__all__


def test_the_module_depends_on_no_futures_infrastructure_or_clock() -> None:
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    imported = {
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module
    } | {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }

    for name in imported:
        assert not name.startswith(("northstar_core.futures", "northstar_infrastructure"))
        assert name.split(".")[0] not in {"sqlite3", "time", "datetime", "random", "uuid"}
