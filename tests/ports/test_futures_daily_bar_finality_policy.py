"""Contract tests for FuturesDailyBarFinality and its Application policy port.

The port separates "the session exists and has ended" (the resolver's answer)
from "the provider's daily bar for it is final" (this port's answer). These
tests pin the assessment value and the port's shape; the deterministic
Application policies are tested alongside the application services.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import re
from datetime import date
from pathlib import Path

import pytest
from northstar_core.derivatives import ExpirationDate
from northstar_core.foundation.value_objects import ExchangeCode, PointInTime, Symbol
from northstar_core.futures import FuturesContract, FuturesProductReference

import northstar_application.ports.futures_daily_bar_finality_policy as port_module
from northstar_application.ports import (
    FuturesDailyBarFinality,
    FuturesDailyBarFinalityOutcome,
    FuturesDailyBarFinalityPolicy,
    FuturesTradingSession,
    InvalidFuturesDailyBarFinalityError,
)

_CONTRACT = FuturesContract(
    FuturesProductReference(Symbol("NIFTY"), ExchangeCode("NSE")), ExpirationDate("2026-10-27")
)
_SESSION = FuturesTradingSession(
    date(2026, 10, 16),
    PointInTime("2026-10-16T03:30:00Z"),
    PointInTime("2026-10-16T10:10:00Z"),
)
_FINAL = FuturesDailyBarFinalityOutcome.FINAL


def _assessment(**overrides: object) -> FuturesDailyBarFinality:
    values: dict[str, object] = {
        "contract": _CONTRACT,
        "session": _SESSION,
        "outcome": _FINAL,
        "reason": "approved",
    }
    values.update(overrides)
    return FuturesDailyBarFinality(**values)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Status vocabulary
# ---------------------------------------------------------------------------


def test_the_status_vocabulary_is_exactly_three_distinct_outcomes() -> None:
    assert [outcome.value for outcome in FuturesDailyBarFinalityOutcome] == [
        "FINAL",
        "NOT_YET_FINAL",
        "UNKNOWN",
    ]
    assert FuturesDailyBarFinalityOutcome.NOT_YET_FINAL != FuturesDailyBarFinalityOutcome.UNKNOWN


# ---------------------------------------------------------------------------
# Assessment value
# ---------------------------------------------------------------------------


def test_an_assessment_carries_its_exact_inputs() -> None:
    assessment = _assessment()

    assert assessment.contract == _CONTRACT
    assert assessment.session == _SESSION
    assert assessment.outcome is _FINAL
    assert assessment.reason == "approved"


@pytest.mark.parametrize(
    ("outcome", "expected"),
    [
        (FuturesDailyBarFinalityOutcome.FINAL, True),
        (FuturesDailyBarFinalityOutcome.NOT_YET_FINAL, False),
        (FuturesDailyBarFinalityOutcome.UNKNOWN, False),
    ],
)
def test_only_final_may_be_acquired(
    outcome: FuturesDailyBarFinalityOutcome, expected: bool
) -> None:
    assert _assessment(outcome=outcome).is_final is expected


def test_an_assessment_is_immutable() -> None:
    assessment = _assessment()

    for field in ("contract", "session", "outcome", "reason"):
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(assessment, field, None)


def test_equal_inputs_give_equal_assessments() -> None:
    assert _assessment() == _assessment()
    assert hash(_assessment()) == hash(_assessment())
    assert _assessment() != _assessment(outcome=FuturesDailyBarFinalityOutcome.UNKNOWN)


def test_the_reason_is_trimmed() -> None:
    assert _assessment(reason="  approved  ").reason == "approved"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("contract", "NIFTY"),
        ("contract", None),
        ("session", date(2026, 10, 16)),
        ("session", None),
        ("outcome", "FINAL"),
        ("outcome", None),
        ("reason", ""),
        ("reason", "   "),
        ("reason", None),
        ("reason", 1),
    ],
)
def test_an_invalid_assessment_is_rejected(field: str, value: object) -> None:
    with pytest.raises(InvalidFuturesDailyBarFinalityError):
        _assessment(**{field: value})


def test_the_assessment_error_is_a_value_error() -> None:
    assert issubclass(InvalidFuturesDailyBarFinalityError, ValueError)


def test_the_assessment_renders_its_session_and_status() -> None:
    assert str(_assessment()) == "NIFTY@NSE 2026-10-27 2026-10-16 FINAL: approved"


# ---------------------------------------------------------------------------
# Port shape
# ---------------------------------------------------------------------------


def test_the_port_assesses_an_exact_contract_and_resolved_session() -> None:
    signature = inspect.signature(FuturesDailyBarFinalityPolicy.assess)

    assert list(signature.parameters) == ["self", "contract", "session"]
    assert signature.parameters["contract"].annotation == "FuturesContract"
    assert signature.parameters["session"].annotation == "FuturesTradingSession"
    assert signature.return_annotation == "FuturesDailyBarFinality"
    assert FuturesDailyBarFinalityPolicy.__abstractmethods__ == frozenset({"assess"})


def test_the_port_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        FuturesDailyBarFinalityPolicy()  # type: ignore[abstract]


def test_the_port_takes_no_clock_or_instant() -> None:
    parameters = inspect.signature(FuturesDailyBarFinalityPolicy.assess).parameters

    for absent in ("now", "clock", "as_of", "assessed_at", "trading_date"):
        assert absent not in parameters


# ---------------------------------------------------------------------------
# Boundaries
# ---------------------------------------------------------------------------


def _tree() -> ast.Module:
    return ast.parse(Path(port_module.__file__).read_text(encoding="utf-8"))


def test_the_port_reads_no_clock_and_depends_on_nothing_provider_shaped() -> None:
    tree = _tree()
    modules = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)} | {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }

    for module in modules:
        assert str(module).split(".")[0] not in {
            "northstar_infrastructure",
            "time",
            "datetime",
            "sqlite3",
            "requests",
            "urllib",
        }
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            assert node.attr not in {"now", "utcnow", "today", "monotonic"}


def test_no_provider_or_venue_concept_leaks_into_the_contract() -> None:
    text = Path(port_module.__file__).read_text(encoding="utf-8")

    assert not re.search(r"\b(upstox|databento|nse|cme|instrument_key|21:00)\b", text, re.I)
