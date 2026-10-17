"""Contract tests for the futures trade recommendation lifecycle store port (F-5B-2A).

Synthetic data only: the contract, instants, levels and bars are fixtures.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
from datetime import date, datetime
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
from northstar_core.futures import FuturesContract, FuturesOHLCVBar, FuturesProductReference
from northstar_core.recommendations.futures import (
    CoverageDeficiency,
    EntryZone,
    FuturesResearchDecisionReference,
    FuturesTradeRecommendation,
    FuturesTradeRecommendationLifecycle,
    ObservationCoverage,
    PriceTarget,
    RecommendationPolicyVersion,
    RecommendationStatus,
    RecommendationValidity,
    StopLoss,
    TimeExpiryEvidence,
    TradeDirection,
)
from northstar_core.strategy import ExplanationReason, StrategyIdentity

import northstar_application.ports as ports
import northstar_application.ports.futures_trade_recommendation_lifecycle_store as module
from northstar_application.application_services import (
    FuturesTradeRecommendationGenerationProfile,
    FuturesTradeRecommendationIdentityService,
)
from northstar_application.ports import (
    FuturesTradeRecommendationLifecycleConflictError,
    FuturesTradeRecommendationLifecycleEvidenceError,
    FuturesTradeRecommendationLifecycleStore,
    FuturesTradeRecommendationObservationStep,
    FuturesTradeRecommendationTimeExpiryStep,
    InvalidFuturesTradeRecommendationLifecycleProgressError,
    apply_futures_trade_recommendation_lifecycle_step,
    validate_futures_trade_recommendation_lifecycle_progress,
)

_NIFTY = FuturesProductReference(Symbol("NIFTY"), ExchangeCode("NSE"))
_OCT = FuturesContract(_NIFTY, ExpirationDate("2026-10-27"))
_NOV = FuturesContract(_NIFTY, ExpirationDate("2026-11-24"))
_STRATEGY = StrategyIdentity("synthetic-directional-v1")
_PROFILE = FuturesTradeRecommendationGenerationProfile(
    "f5a-synthetic-profile-v1", 5, RecommendationPolicyVersion("f4-synthetic-fixture-v1")
)


def _q(value: str) -> QuoteValue:
    return QuoteValue(Decimal(value))


def _recommendation(
    contract: FuturesContract = _OCT, issued_at: str = "2026-10-07T12:00:00Z"
) -> FuturesTradeRecommendation:
    source = FuturesResearchDecisionReference(
        contract, Timeframe("1d"), PointInTime("2026-10-07T10:00:00Z"), _STRATEGY
    )
    return FuturesTradeRecommendation(
        identity=FuturesTradeRecommendationIdentityService().identity(source, _PROFILE),
        contract=contract,
        direction=TradeDirection.LONG,
        entry=EntryZone(_q("24378"), _q("24388")),
        stop=StopLoss(_q("24347.75")),
        target=PriceTarget(_q("24448.45")),
        validity=RecommendationValidity(
            PointInTime("2026-10-08T10:00:00Z"),
            PointInTime("2026-10-14T10:00:00Z"),
            PointInTime("2026-10-27T10:00:00Z"),
        ),
        source=source,
        policy_version=_PROFILE.policy_version,
        issued_at=PointInTime(issued_at),
        thesis=(ExplanationReason("Synthetic thesis"),),
    )


_RECOMMENDATION = _recommendation()
_INITIAL = FuturesTradeRecommendationLifecycle(_RECOMMENDATION)


def _bar(day: int, low: str, high: str, contract: FuturesContract = _OCT) -> FuturesOHLCVBar:
    middle = (Decimal(low) + Decimal(high)) / 2
    return FuturesOHLCVBar(
        contract=contract,
        point_in_time=PointInTime(f"2026-10-{day:02d}T10:00:00Z"),
        timeframe=Timeframe("1d"),
        open=QuoteValue(middle),
        high=_q(high),
        low=_q(low),
        close=QuoteValue(middle),
        volume=Quantity(Decimal("1000")),
    )


def _observe(
    day: int, low: str = "24400", high: str = "24420"
) -> FuturesTradeRecommendationObservationStep:
    return FuturesTradeRecommendationObservationStep(_bar(day, low, high), date(2026, 10, day))


_QUIET_8, _QUIET_9 = _observe(8), _observe(9)
_ENTRY_12 = _observe(12, "24380", "24400")
_STOP_13 = _observe(13, "24340", "24390")


def _apply(
    lifecycle: FuturesTradeRecommendationLifecycle, *steps
) -> FuturesTradeRecommendationLifecycle:
    for step in steps:
        lifecycle = apply_futures_trade_recommendation_lifecycle_step(lifecycle, step)
    return lifecycle


def _expiry(
    evaluated_through: str | None,
    coverage: ObservationCoverage = ObservationCoverage.INCOMPLETE_OR_UNVERIFIED_COVERAGE,
    as_of: str = "2026-10-14T12:00:00Z",
) -> FuturesTradeRecommendationTimeExpiryStep:
    complete = coverage is ObservationCoverage.COMPLETE_VERIFIED_COVERAGE
    return FuturesTradeRecommendationTimeExpiryStep(
        TimeExpiryEvidence(
            as_of=PointInTime(as_of),
            evaluated_through=None if evaluated_through is None else PointInTime(evaluated_through),
            coverage=coverage,
            attested_by="synthetic coverage fixture",
            deficiency=None if complete else CoverageDeficiency.MISSING_OBSERVATIONS,
        )
    )


# -- step values ----------------------------------------------------------------------


def test_an_observation_step_names_its_bar_and_session_label() -> None:
    step = _observe(8)

    assert step.bar.point_in_time == PointInTime("2026-10-08T10:00:00Z")
    assert step.trading_date == date(2026, 10, 8)


@pytest.mark.parametrize(
    "bar, trading_date",
    [
        (None, date(2026, 10, 8)),
        ("bar", date(2026, 10, 8)),
        (_bar(8, "24400", "24420"), None),
        (_bar(8, "24400", "24420"), "2026-10-08"),
        (_bar(8, "24400", "24420"), datetime(2026, 10, 8)),
    ],
    ids=["no-bar", "text-bar", "no-date", "text-date", "datetime"],
)
def test_an_invalid_observation_step_is_refused(bar: object, trading_date: object) -> None:
    with pytest.raises(InvalidFuturesTradeRecommendationLifecycleProgressError):
        FuturesTradeRecommendationObservationStep(bar, trading_date)  # type: ignore[arg-type]


@pytest.mark.parametrize("evidence", [None, "evidence", PointInTime("2026-10-14T10:00:00Z")])
def test_a_time_expiry_step_requires_time_evidence(evidence: object) -> None:
    with pytest.raises(InvalidFuturesTradeRecommendationLifecycleProgressError):
        FuturesTradeRecommendationTimeExpiryStep(evidence)  # type: ignore[arg-type]


def test_steps_are_immutable() -> None:
    step = _observe(8)
    expiry = _expiry(None)

    with pytest.raises(dataclasses.FrozenInstanceError):
        step.trading_date = date(2026, 10, 9)  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        expiry.evidence = expiry.evidence  # type: ignore[misc]


# -- applying one step ----------------------------------------------------------------


def test_a_quiet_bar_is_an_evaluation_that_advances_only_the_cursor() -> None:
    after = apply_futures_trade_recommendation_lifecycle_step(_INITIAL, _QUIET_8)

    assert after.events == ()
    assert after.last_observation == _QUIET_8.bar
    assert after.status is RecommendationStatus.WAITING_FOR_ENTRY


@pytest.mark.parametrize(
    "before, step",
    [
        (_INITIAL, _observe(7)),
        (_apply(_INITIAL, _QUIET_8), _QUIET_8),
        (_apply(_INITIAL, _ENTRY_12, _STOP_13), _observe(14)),
        (_INITIAL, _expiry(None, as_of="2026-10-13T12:00:00Z")),
        (_apply(_INITIAL, _ENTRY_12, _STOP_13), _expiry("2026-10-13T10:00:00Z")),
    ],
    ids=[
        "ineligible-bar",
        "repeated-bar",
        "after-terminal",
        "expiry-before-validity-ends",
        "expiry-after-terminal",
    ],
)
def test_a_step_core_leaves_unchanged_is_not_an_evaluation(
    before: FuturesTradeRecommendationLifecycle, step: object
) -> None:
    with pytest.raises(InvalidFuturesTradeRecommendationLifecycleProgressError, match="unchanged"):
        apply_futures_trade_recommendation_lifecycle_step(before, step)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "before, step",
    [
        (_apply(_INITIAL, _QUIET_9), _QUIET_8),
        (
            _apply(_INITIAL, _QUIET_8),
            FuturesTradeRecommendationObservationStep(_bar(8, "24401", "24420"), date(2026, 10, 8)),
        ),
        (
            _INITIAL,
            FuturesTradeRecommendationObservationStep(
                _bar(8, "24400", "24420", _NOV), date(2026, 10, 8)
            ),
        ),
        (_apply(_INITIAL, _QUIET_8), _expiry(None)),
        (_INITIAL, "step"),
    ],
    ids=["out-of-order", "revised-bar", "wrong-contract", "evidence-mismatch", "not-a-step"],
)
def test_a_step_core_refuses_is_a_typed_progress_error(
    before: FuturesTradeRecommendationLifecycle, step: object
) -> None:
    with pytest.raises(InvalidFuturesTradeRecommendationLifecycleProgressError):
        apply_futures_trade_recommendation_lifecycle_step(before, step)  # type: ignore[arg-type]


# -- validating progress --------------------------------------------------------------


@pytest.mark.parametrize(
    "previous_steps, steps",
    [
        ((), (_QUIET_8,)),
        ((), (_QUIET_8, _QUIET_9)),
        ((_QUIET_8,), (_QUIET_9, _ENTRY_12)),
        ((_QUIET_8, _ENTRY_12), (_STOP_13,)),
        ((), (_expiry(None),)),
        (
            (_QUIET_8,),
            (_expiry("2026-10-08T10:00:00Z", ObservationCoverage.COMPLETE_VERIFIED_COVERAGE),),
        ),
    ],
    ids=["one-quiet", "two-quiet", "quiet-then-entry", "stop", "time-expiry", "verified-expiry"],
)
def test_progress_that_core_produces_is_accepted(previous_steps: tuple, steps: tuple) -> None:
    previous = _apply(_INITIAL, *previous_steps)
    current = _apply(previous, *steps)

    validate_futures_trade_recommendation_lifecycle_progress(previous, current, steps)


def test_progress_must_be_for_one_identity() -> None:
    other = FuturesTradeRecommendationLifecycle(_recommendation(_NOV))

    with pytest.raises(InvalidFuturesTradeRecommendationLifecycleProgressError, match="cannot end"):
        validate_futures_trade_recommendation_lifecycle_progress(
            _INITIAL, other.advance(_bar(8, "24400", "24420", _NOV)), (_QUIET_8,)
        )


def test_progress_must_be_for_one_definition_not_only_one_identity() -> None:
    redefined = _recommendation(issued_at="2026-10-07T13:00:00Z")
    assert redefined.identity == _RECOMMENDATION.identity
    current = _apply(FuturesTradeRecommendationLifecycle(redefined), _QUIET_8)

    with pytest.raises(InvalidFuturesTradeRecommendationLifecycleProgressError, match="defin"):
        validate_futures_trade_recommendation_lifecycle_progress(_INITIAL, current, (_QUIET_8,))


@pytest.mark.parametrize(
    "previous, current, steps",
    [
        ("lifecycle", _apply(_INITIAL, _QUIET_8), (_QUIET_8,)),
        (_INITIAL, None, (_QUIET_8,)),
        (_INITIAL, _apply(_INITIAL, _QUIET_8), [_QUIET_8]),
        (_INITIAL, _apply(_INITIAL, _QUIET_8), (_QUIET_8.bar,)),
        (_INITIAL, _INITIAL, ()),
        (_INITIAL, _apply(_INITIAL, _QUIET_8), ()),
    ],
    ids=["previous-type", "current-type", "list", "bare-bar", "empty-no-op", "empty-progress"],
)
def test_malformed_progress_is_refused(previous: object, current: object, steps: object) -> None:
    with pytest.raises(InvalidFuturesTradeRecommendationLifecycleProgressError):
        validate_futures_trade_recommendation_lifecycle_progress(
            previous,  # type: ignore[arg-type]
            current,  # type: ignore[arg-type]
            steps,  # type: ignore[arg-type]
        )


def test_progress_missing_a_quiet_step_is_refused() -> None:
    current = _apply(_INITIAL, _QUIET_8, _QUIET_9)

    with pytest.raises(InvalidFuturesTradeRecommendationLifecycleProgressError, match="not the"):
        validate_futures_trade_recommendation_lifecycle_progress(_INITIAL, current, (_QUIET_8,))


def test_progress_with_an_extra_step_is_refused() -> None:
    current = _apply(_INITIAL, _QUIET_8)

    with pytest.raises(InvalidFuturesTradeRecommendationLifecycleProgressError):
        validate_futures_trade_recommendation_lifecycle_progress(
            _INITIAL, current, (_QUIET_8, _QUIET_9)
        )


def test_regression_is_refused() -> None:
    later = _apply(_INITIAL, _QUIET_8, _ENTRY_12)
    earlier = _apply(_INITIAL, _QUIET_8)

    with pytest.raises(InvalidFuturesTradeRecommendationLifecycleProgressError):
        validate_futures_trade_recommendation_lifecycle_progress(later, earlier, (_QUIET_8,))


def test_a_rewritten_history_is_refused() -> None:
    """``current`` must keep ``previous``'s events; a different history is not progress."""
    previous = _apply(_INITIAL, _ENTRY_12)
    rewritten = _apply(_INITIAL, _observe(12), _observe(13))

    with pytest.raises(InvalidFuturesTradeRecommendationLifecycleProgressError):
        validate_futures_trade_recommendation_lifecycle_progress(
            previous, rewritten, (_observe(13),)
        )


def test_a_status_cannot_be_invented() -> None:
    """A later status reachable only from other evidence is not what the steps produce."""
    previous = _apply(_INITIAL, _ENTRY_12)
    target = _apply(previous, _observe(13, "24400", "24450"))

    with pytest.raises(InvalidFuturesTradeRecommendationLifecycleProgressError):
        validate_futures_trade_recommendation_lifecycle_progress(previous, target, (_STOP_13,))


def test_validation_does_not_mutate_its_inputs() -> None:
    previous = _apply(_INITIAL, _QUIET_8)
    steps = (_QUIET_9, _ENTRY_12)
    current = _apply(previous, *steps)
    snapshot = (repr(previous), repr(current), repr(steps))

    validate_futures_trade_recommendation_lifecycle_progress(previous, current, steps)

    assert (repr(previous), repr(current), repr(steps)) == snapshot


# -- the store contract ---------------------------------------------------------------


def test_the_store_is_abstract() -> None:
    with pytest.raises(TypeError):
        FuturesTradeRecommendationLifecycleStore()  # type: ignore[abstract]


def test_record_progress_takes_previous_current_and_keyword_steps() -> None:
    parameters = inspect.signature(
        FuturesTradeRecommendationLifecycleStore.record_progress
    ).parameters

    assert list(parameters) == ["self", "previous", "current", "steps"]
    assert parameters["steps"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["steps"].default is inspect.Parameter.empty


def test_the_errors_are_distinct_value_errors() -> None:
    errors = (
        InvalidFuturesTradeRecommendationLifecycleProgressError,
        FuturesTradeRecommendationLifecycleConflictError,
        FuturesTradeRecommendationLifecycleEvidenceError,
    )
    for error in errors:
        assert issubclass(error, ValueError)
        for other in errors:
            if other is not error:
                assert not issubclass(error, other)


def test_public_exports() -> None:
    for name in (
        "FuturesTradeRecommendationLifecycleConflictError",
        "FuturesTradeRecommendationLifecycleEvidenceError",
        "FuturesTradeRecommendationLifecycleStep",
        "FuturesTradeRecommendationLifecycleStore",
        "FuturesTradeRecommendationObservationStep",
        "FuturesTradeRecommendationTimeExpiryStep",
        "InvalidFuturesTradeRecommendationLifecycleProgressError",
        "apply_futures_trade_recommendation_lifecycle_step",
        "validate_futures_trade_recommendation_lifecycle_progress",
    ):
        assert name in ports.__all__


def test_the_port_depends_on_no_storage_provider_or_clock() -> None:
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    modules = {
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module
    } | {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }

    for name in modules:
        assert name.split(".")[0] not in {"sqlite3", "time", "requests", "urllib", "socket"}, name
        for fragment in ("infrastructure", "upstox", "sqlite"):
            assert fragment not in name, name
    assert "now" not in {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
