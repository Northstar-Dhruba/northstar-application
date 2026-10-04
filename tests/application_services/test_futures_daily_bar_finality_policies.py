"""Tests for the deterministic Application finality policies.

OperatorApprovedFuturesDailyBarFinalityPolicy: a resolved session is FINAL when
its trading date is on or before the operator's final-through date, and
NOT_YET_FINAL after it. DisabledFuturesDailyBarFinalityPolicy: every session is
UNKNOWN, so unattended acquisition is impossible under it.

Sessions are built directly: a policy receives an already resolved session and
never consults a calendar, a provider or a clock.
"""

from __future__ import annotations

import ast
import re
from datetime import date, datetime
from pathlib import Path

import pytest
from northstar_core.derivatives import ExpirationDate
from northstar_core.foundation.value_objects import ExchangeCode, PointInTime, Symbol
from northstar_core.futures import FuturesContract, FuturesProductReference

import northstar_application.application_services.futures_daily_bar_finality_policies as module
from northstar_application.application_services import (
    DisabledFuturesDailyBarFinalityPolicy,
    InvalidFuturesDailyBarFinalityPolicyError,
    OperatorApprovedFuturesDailyBarFinalityPolicy,
)
from northstar_application.ports import (
    FuturesDailyBarFinality,
    FuturesDailyBarFinalityOutcome,
    FuturesDailyBarFinalityPolicy,
    FuturesTradingSession,
)

_NIFTY = FuturesProductReference(Symbol("NIFTY"), ExchangeCode("NSE"))
_OCT = FuturesContract(_NIFTY, ExpirationDate("2026-10-27"))
_NOV = FuturesContract(_NIFTY, ExpirationDate("2026-11-23"))
_ES = FuturesContract(
    FuturesProductReference(Symbol("ES"), ExchangeCode("CME")), ExpirationDate("2026-12-18")
)

_FINAL = FuturesDailyBarFinalityOutcome.FINAL
_NOT_YET = FuturesDailyBarFinalityOutcome.NOT_YET_FINAL
_UNKNOWN = FuturesDailyBarFinalityOutcome.UNKNOWN


def _session(day: date) -> FuturesTradingSession:
    label = day.isoformat()
    return FuturesTradingSession(
        day, PointInTime(f"{label}T03:30:00Z"), PointInTime(f"{label}T10:10:00Z")
    )


_THU = _session(date(2026, 10, 15))
_FRI = _session(date(2026, 10, 16))
_MON = _session(date(2026, 10, 19))


def _approved(final_through: date = date(2026, 10, 16)):
    return OperatorApprovedFuturesDailyBarFinalityPolicy(final_through)


# ---------------------------------------------------------------------------
# Operator-approved boundaries
# ---------------------------------------------------------------------------


def test_a_session_before_final_through_is_final() -> None:
    assessment = _approved().assess(_OCT, _THU)

    assert assessment.outcome is _FINAL
    assert assessment.is_final


def test_a_session_on_final_through_is_final() -> None:
    assert _approved().assess(_OCT, _FRI).outcome is _FINAL


def test_a_session_after_final_through_is_not_yet_final() -> None:
    assessment = _approved().assess(_OCT, _MON)

    assert assessment.outcome is _NOT_YET
    assert not assessment.is_final


@pytest.mark.parametrize(
    ("final_through", "expected"),
    [
        (date(2026, 10, 14), (_NOT_YET, _NOT_YET, _NOT_YET)),
        (date(2026, 10, 15), (_FINAL, _NOT_YET, _NOT_YET)),
        (date(2026, 10, 16), (_FINAL, _FINAL, _NOT_YET)),
        (date(2026, 10, 17), (_FINAL, _FINAL, _NOT_YET)),  # a non-session approval date
        (date(2026, 10, 19), (_FINAL, _FINAL, _FINAL)),
    ],
)
def test_the_boundary_moves_exactly_with_final_through(final_through: date, expected) -> None:
    policy = _approved(final_through)

    assert tuple(policy.assess(_OCT, s).outcome for s in (_THU, _FRI, _MON)) == expected


def test_the_approval_compares_the_session_label_not_its_instants() -> None:
    """A session whose close falls on the next civil day is still judged by its label."""
    overnight = FuturesTradingSession(
        date(2026, 10, 16),
        PointInTime("2026-10-15T22:00:00Z"),
        PointInTime("2026-10-17T02:00:00Z"),
    )

    assert _approved(date(2026, 10, 16)).assess(_OCT, overnight).outcome is _FINAL


# ---------------------------------------------------------------------------
# Reasons and identity
# ---------------------------------------------------------------------------


def test_reasons_are_stable_and_explain_the_boundary() -> None:
    policy = _approved()

    assert policy.assess(_OCT, _FRI).reason == (
        "Session 2026-10-16 is on or before the operator-approved final-through date 2026-10-16."
    )
    assert policy.assess(_OCT, _MON).reason == (
        "Session 2026-10-19 is after the operator-approved final-through date 2026-10-16; "
        "it has not been approved as final."
    )


def test_the_assessment_preserves_the_exact_contract_and_session() -> None:
    for policy in (_approved(), DisabledFuturesDailyBarFinalityPolicy()):
        for contract in (_OCT, _NOV):
            assessment = policy.assess(contract, _FRI)
            assert assessment.contract == contract
            assert assessment.session == _FRI


def test_the_same_inputs_give_equal_assessments() -> None:
    assert _approved().assess(_OCT, _FRI) == _approved().assess(_OCT, _FRI)
    assert _approved().assess(_OCT, _FRI) == OperatorApprovedFuturesDailyBarFinalityPolicy(
        date(2026, 10, 16)
    ).assess(_OCT, _FRI)
    disabled = DisabledFuturesDailyBarFinalityPolicy()
    assert disabled.assess(_OCT, _FRI) == disabled.assess(_OCT, _FRI)


def test_assessments_are_finality_values_from_a_port_implementation() -> None:
    for policy in (_approved(), DisabledFuturesDailyBarFinalityPolicy()):
        assert isinstance(policy, FuturesDailyBarFinalityPolicy)
        assert isinstance(policy.assess(_OCT, _FRI), FuturesDailyBarFinality)


def test_the_policy_is_venue_neutral() -> None:
    """The session carries no instrument identity; the policy applies the same rule."""
    assert _approved().assess(_ES, _FRI).outcome is _FINAL


def test_the_policy_exposes_its_approval_and_a_readable_repr() -> None:
    policy = _approved()

    assert policy.final_through == date(2026, 10, 16)
    assert (
        repr(policy) == "OperatorApprovedFuturesDailyBarFinalityPolicy(final_through='2026-10-16')"
    )
    assert (
        repr(DisabledFuturesDailyBarFinalityPolicy()) == "DisabledFuturesDailyBarFinalityPolicy()"
    )


# ---------------------------------------------------------------------------
# Disabled policy: fail closed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("session", [_THU, _FRI, _MON, _session(date(2000, 1, 3))])
@pytest.mark.parametrize("contract", [_OCT, _NOV, _ES])
def test_the_disabled_policy_never_establishes_finality(contract, session) -> None:
    assessment = DisabledFuturesDailyBarFinalityPolicy().assess(contract, session)

    assert assessment.outcome is _UNKNOWN
    assert not assessment.is_final
    assert assessment.reason == (
        "Daily-bar finality is not established: automatic finality is disabled "
        "and no operator approval applies."
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value", [None, "2026-10-16", 20261016, datetime(2026, 10, 16, 10, 10), 1.5]
)
def test_an_invalid_final_through_is_rejected(value: object) -> None:
    with pytest.raises(InvalidFuturesDailyBarFinalityPolicyError):
        OperatorApprovedFuturesDailyBarFinalityPolicy(value)  # type: ignore[arg-type]


def test_the_policy_error_is_a_value_error() -> None:
    assert issubclass(InvalidFuturesDailyBarFinalityPolicyError, ValueError)


@pytest.mark.parametrize(
    "policy", [_approved(), DisabledFuturesDailyBarFinalityPolicy()], ids=["approved", "disabled"]
)
def test_assess_rejects_anything_but_a_contract_and_a_resolved_session(policy) -> None:
    with pytest.raises(TypeError, match="FuturesContract"):
        policy.assess(_NIFTY, _FRI)
    with pytest.raises(TypeError, match="FuturesTradingSession"):
        policy.assess(_OCT, date(2026, 10, 16))
    with pytest.raises(TypeError, match="FuturesTradingSession"):
        policy.assess(_OCT, None)


# ---------------------------------------------------------------------------
# No clock, no calendar, no provider
# ---------------------------------------------------------------------------


class _NoNow(datetime):
    @classmethod
    def now(cls, tz=None):
        raise AssertionError("a finality policy must not read the clock")

    @classmethod
    def utcnow(cls):
        raise AssertionError("a finality policy must not read the clock")


def test_no_clock_is_consulted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(module, "datetime", _NoNow)

    assert _approved().assess(_OCT, _FRI).outcome is _FINAL
    assert _approved().assess(_OCT, _MON).outcome is _NOT_YET
    assert DisabledFuturesDailyBarFinalityPolicy().assess(_OCT, _FRI).outcome is _UNKNOWN


def _tree() -> ast.Module:
    return ast.parse(Path(module.__file__).read_text(encoding="utf-8"))


def test_the_policies_read_no_clock_resolve_no_calendar_and_call_no_provider() -> None:
    tree = _tree()
    modules = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    names = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }

    assert not any(str(m).startswith("northstar_infrastructure") for m in modules)
    assert not {"time", "urllib", "requests", "sqlite3"} & {str(m).split(".")[0] for m in modules}
    assert "FuturesTradingSessionResolver" not in names
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            assert node.attr not in {"now", "utcnow", "today", "monotonic"}
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id != "timedelta"


def test_no_provider_concept_or_wall_clock_threshold_appears() -> None:
    text = Path(module.__file__).read_text(encoding="utf-8")

    assert not re.search(r"\b(upstox|databento|nse|cme|instrument_key)\b", text, re.I)
    assert not re.search(r"\b\d{1,2}:\d{2}\b", text)  # no time-of-day rule


def test_the_public_surface_is_exported() -> None:
    import northstar_application.application_services as services
    import northstar_application.ports as ports

    for name in (
        "OperatorApprovedFuturesDailyBarFinalityPolicy",
        "DisabledFuturesDailyBarFinalityPolicy",
        "InvalidFuturesDailyBarFinalityPolicyError",
    ):
        assert name in services.__all__
    for name in (
        "FuturesDailyBarFinality",
        "FuturesDailyBarFinalityPolicy",
        "FuturesDailyBarFinalityOutcome",
        "InvalidFuturesDailyBarFinalityError",
    ):
        assert name in ports.__all__
    assert not hasattr(services, "_validate_inputs")
