"""Tests for the futures recommendation generation audit value object (F-5B-2B).

Synthetic data only: the contract, sessions, instants, identities and every
configuration value are fixtures.
"""

from __future__ import annotations

import dataclasses
import hashlib
from datetime import date, datetime

import pytest
from northstar_core.derivatives import ExpirationDate
from northstar_core.foundation.value_objects import ExchangeCode, PointInTime, Symbol, Timeframe
from northstar_core.futures import FuturesContract, FuturesProductReference
from northstar_core.recommendations.futures import (
    CandidateRejectionReason,
    FuturesResearchDecisionReference,
    FuturesTradeRecommendationIdentity,
    GenerationRejectionReason,
    RecommendationPolicyVersion,
)
from northstar_core.strategy import RecommendationAction, StrategyIdentity

from northstar_application.application_services import (
    FUTURES_RECOMMENDATION_GENERATION_AUDIT_IDENTITY_NAMESPACE,
    FuturesCandidateRejectionAudit,
    FuturesCandleReviewAssessment,
    FuturesGenerationRejectionAudit,
    FuturesRecommendationGenerationAudit,
    FuturesRecommendationGenerationAuditOutcome,
    FuturesResearchHistoryCoverage,
    FuturesResearchHistoryDeficiency,
    FuturesTradeRecommendationBlockReason,
    FuturesTradeRecommendationGenerationStatus,
    InvalidFuturesRecommendationGenerationAuditError,
)
from northstar_application.ports import (
    FuturesDailyBarFinality,
    FuturesDailyBarFinalityOutcome,
    FuturesTradingSession,
)

Outcome = FuturesRecommendationGenerationAuditOutcome
Block = FuturesTradeRecommendationBlockReason
Status = FuturesTradeRecommendationGenerationStatus
Coverage = FuturesResearchHistoryCoverage

_NIFTY = FuturesProductReference(Symbol("NIFTY"), ExchangeCode("NSE"))
_OCT = FuturesContract(_NIFTY, ExpirationDate("2026-10-27"))
_NOV = FuturesContract(_NIFTY, ExpirationDate("2026-11-24"))
_STRATEGY = StrategyIdentity("synthetic-directional-v1")
_SESSION = FuturesTradingSession(
    date(2026, 10, 7), PointInTime("2026-10-07T03:45:00Z"), PointInTime("2026-10-07T10:00:00Z")
)
_FINAL = FuturesDailyBarFinality(_OCT, _SESSION, FuturesDailyBarFinalityOutcome.FINAL, "synthetic")
_NOT_FINAL = FuturesDailyBarFinality(
    _OCT, _SESSION, FuturesDailyBarFinalityOutcome.NOT_YET_FINAL, "synthetic"
)
_SOURCE = FuturesResearchDecisionReference(_OCT, Timeframe("1d"), _SESSION.closes_at, _STRATEGY)
_A = FuturesTradeRecommendationIdentity("a" * 64)
_B = FuturesTradeRecommendationIdentity("b" * 64)


def _audit(**overrides: object) -> FuturesRecommendationGenerationAudit:
    values: dict[str, object] = {
        "contract": _OCT,
        "strategy": _STRATEGY,
        "policy_version": RecommendationPolicyVersion("f4-synthetic-fixture-v1"),
        "profile_version": "f5a-synthetic-profile-v1",
        "attempted_at": PointInTime("2026-10-07T12:00:00Z"),
        "status": Status.COMPLETED,
        "session": _SESSION,
        "finality": _FINAL,
        "history_coverage": Coverage.VERIFIED,
        "candidates": (_SOURCE,),
        "proposed": (_A,),
        "recorded": (_A,),
    }
    values.update(overrides)
    return FuturesRecommendationGenerationAudit(**values)  # type: ignore[arg-type]


def _blocked(reason: Block, **overrides: object) -> FuturesRecommendationGenerationAudit:
    values: dict[str, object] = {
        "status": Status.BLOCKED,
        "block_reason": reason,
        "history_coverage": None,
        "candidates": (),
        "proposed": (),
        "recorded": (),
    }
    if reason is Block.NO_PERSISTED_SESSION:
        values.update(session=None, finality=None)
    if reason is Block.SESSION_NOT_FINAL:
        values.update(finality=_NOT_FINAL)
    values.update(overrides)
    return _audit(**values)


# -- identity ---------------------------------------------------------------------------


def _expected_identity(*parts: str) -> str:
    encoded = b"".join(f"{len(p.encode())}:".encode() + p.encode() for p in parts)
    return hashlib.sha256(encoded).hexdigest()


def test_the_identity_is_the_documented_length_prefixed_sha256() -> None:
    assert _audit().identity == _expected_identity(
        FUTURES_RECOMMENDATION_GENERATION_AUDIT_IDENTITY_NAMESPACE,
        "NIFTY",
        "NSE",
        "2026-10-27",
        "synthetic-directional-v1",
        "f4-synthetic-fixture-v1",
        "f5a-synthetic-profile-v1",
        "2026-10-07T12:00:00Z",
        "2026-10-07",
        "RECORDED",
        "",
    )
    assert _blocked(Block.NO_PERSISTED_SESSION).identity == _expected_identity(
        FUTURES_RECOMMENDATION_GENERATION_AUDIT_IDENTITY_NAMESPACE,
        "NIFTY",
        "NSE",
        "2026-10-27",
        "synthetic-directional-v1",
        "f4-synthetic-fixture-v1",
        "f5a-synthetic-profile-v1",
        "2026-10-07T12:00:00Z",
        "",
        "BLOCKED",
        "NO_PERSISTED_SESSION",
    )


def test_the_identity_is_deterministic_and_reads_one_instant_one_way() -> None:
    assert _audit().identity == _audit().identity
    assert len(_audit().identity) == 64
    assert (
        _audit(attempted_at=PointInTime("2026-10-07T17:30:00+05:30")).identity == _audit().identity
    )


@pytest.mark.parametrize(
    "variant",
    [
        lambda: _audit(
            contract=_NOV,
            finality=dataclasses.replace(_FINAL, contract=_NOV),
            candidates=(dataclasses.replace(_SOURCE, contract=_NOV),),
        ),
        lambda: _audit(strategy=StrategyIdentity("synthetic-other-v1")),
        lambda: _audit(policy_version=RecommendationPolicyVersion("f4-synthetic-fixture-v2")),
        lambda: _audit(profile_version="f5a-synthetic-profile-v2"),
        lambda: _audit(attempted_at=PointInTime("2026-10-07T12:00:01Z")),
        lambda: _audit(recorded=(), existing=(_A,)),
        lambda: _blocked(Block.SESSION_NOT_OPERATOR_REVIEWED),
        lambda: _blocked(Block.NO_RESEARCH_FOR_SESSION),
        lambda: _blocked(Block.NO_PERSISTED_SESSION),
    ],
    ids=[
        "contract",
        "strategy",
        "policy",
        "profile",
        "attempt-instant",
        "existing-not-recorded",
        "blocked-not-reviewed",
        "blocked-no-research",
        "no-session",
    ],
)
def test_every_identity_input_distinguishes_audits(variant) -> None:
    assert variant().identity != _audit().identity


def test_content_outside_the_identity_does_not_change_it() -> None:
    assert _audit(candidates=()).identity == _audit().identity


# -- derived outcome and review ---------------------------------------------------------


@pytest.mark.parametrize(
    "audit, outcome",
    [
        (lambda: _audit(), Outcome.RECORDED),
        (lambda: _audit(recorded=(), existing=(_A,)), Outcome.EXISTING_RETURNED),
        (lambda: _audit(proposed=(), recorded=()), Outcome.REJECTED),
        (lambda: _blocked(Block.NO_RESEARCH_FOR_SESSION), Outcome.BLOCKED),
    ],
    ids=["recorded", "existing", "rejected", "blocked"],
)
def test_the_outcome_is_derived_from_status_and_identities(audit, outcome: Outcome) -> None:
    assert audit().outcome is outcome


@pytest.mark.parametrize(
    "audit, review",
    [
        (lambda: _blocked(Block.NO_PERSISTED_SESSION), "NOT_ASSESSED"),
        (lambda: _blocked(Block.SESSION_NOT_FINAL), "NOT_ASSESSED"),
        (lambda: _blocked(Block.SESSION_NOT_OPERATOR_REVIEWED), "NOT_REVIEWED"),
        (lambda: _blocked(Block.NO_RESEARCH_FOR_SESSION), "REVIEWED"),
        (lambda: _audit(), "REVIEWED"),
    ],
)
def test_the_operator_review_assessment_is_derived(audit, review: str) -> None:
    assert audit().operator_review is FuturesCandleReviewAssessment(review)


def test_unverified_history_is_carried_with_its_sessions() -> None:
    audit = _blocked(
        Block.RESEARCH_HISTORY_NOT_VERIFIED,
        history_coverage=Coverage.UNVERIFIED,
        history_deficiency=FuturesResearchHistoryDeficiency.MISSING_BAR,
        history_deficient_sessions=(date(2026, 9, 30), date(2026, 10, 1)),
        existing=(_B,),
    )

    assert audit.outcome is Outcome.BLOCKED
    assert audit.history_deficient_sessions == (date(2026, 9, 30), date(2026, 10, 1))


def test_rejection_references_carry_source_action_and_reason() -> None:
    audit = _audit(
        proposed=(),
        recorded=(),
        candidates=(),
        candidate_rejections=(
            FuturesCandidateRejectionAudit(
                _SOURCE, RecommendationAction("HOLD"), CandidateRejectionReason.NO_DIRECTIONAL_VIEW
            ),
        ),
        generation_rejections=(
            FuturesGenerationRejectionAudit(
                _SOURCE, GenerationRejectionReason.DUPLICATE_OPEN_RECOMMENDATION
            ),
        ),
        exclusivity_breaches=(_NIFTY,),
    )

    assert audit.outcome is Outcome.REJECTED


# -- invariants -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "build",
    [
        lambda: _audit(block_reason=Block.NO_RESEARCH_FOR_SESSION),
        lambda: _audit(status=Status.BLOCKED),
        lambda: _audit(session=None, finality=None),
        lambda: _audit(finality=None),
        lambda: _audit(finality=dataclasses.replace(_FINAL, contract=_NOV)),
        lambda: _audit(finality=_NOT_FINAL),
        lambda: _blocked(Block.SESSION_NOT_FINAL, finality=_FINAL),
        lambda: _blocked(Block.NO_RESEARCH_FOR_SESSION, session=None, finality=None),
        lambda: _audit(recorded=(_B,)),
        lambda: _audit(existing=(_A,)),
        lambda: _audit(history_coverage=None),
        lambda: _audit(
            history_coverage=Coverage.UNVERIFIED,
            history_deficiency=FuturesResearchHistoryDeficiency.MISSING_BAR,
        ),
        lambda: _audit(history_deficiency=FuturesResearchHistoryDeficiency.MISSING_BAR),
        lambda: _audit(history_deficient_sessions=(date(2026, 10, 1),)),
        lambda: _blocked(
            Block.RESEARCH_HISTORY_NOT_VERIFIED,
            history_coverage=Coverage.UNVERIFIED,
            history_deficiency=FuturesResearchHistoryDeficiency.MISSING_BAR,
            history_deficient_sessions=(date(2026, 10, 2), date(2026, 10, 1)),
        ),
        lambda: _blocked(
            Block.RESEARCH_HISTORY_NOT_VERIFIED,
            history_coverage=Coverage.UNVERIFIED,
            history_deficiency=FuturesResearchHistoryDeficiency.MISSING_BAR,
            history_deficient_sessions=(datetime(2026, 10, 1),),
        ),
        lambda: _blocked(Block.RESEARCH_HISTORY_NOT_VERIFIED),
        lambda: _blocked(Block.NO_RESEARCH_FOR_SESSION, proposed=(_A,)),
        lambda: _blocked(Block.NO_RESEARCH_FOR_SESSION, exclusivity_breaches=(_NIFTY,)),
        lambda: _audit(candidates=(dataclasses.replace(_SOURCE, contract=_NOV),)),
        lambda: _audit(
            exclusivity_breaches=(
                FuturesProductReference(Symbol("BANKNIFTY"), ExchangeCode("NSE")),
            )
        ),
        lambda: _audit(proposed=(_A, _A)),
        lambda: _audit(proposed=[_A]),
        lambda: _audit(recorded=("a" * 64,)),
        lambda: _audit(version=2),
        lambda: _audit(version=True),
        lambda: _audit(profile_version=" f5a-synthetic-profile-v1"),
        lambda: _audit(profile_version=""),
        lambda: _audit(attempted_at="2026-10-07T12:00:00Z"),
        lambda: _audit(status="COMPLETED"),
    ],
)
def test_a_contradictory_audit_is_refused(build) -> None:
    with pytest.raises(InvalidFuturesRecommendationGenerationAuditError):
        build()


@pytest.mark.parametrize(
    "build",
    [
        lambda: FuturesCandidateRejectionAudit(_SOURCE, "HOLD", "NO_DIRECTIONAL_VIEW"),
        lambda: FuturesGenerationRejectionAudit(
            "source", GenerationRejectionReason.ZERO_PRICE_MOVEMENT_UNIT
        ),
    ],
)
def test_an_invalid_rejection_reference_is_refused(build) -> None:
    with pytest.raises(InvalidFuturesRecommendationGenerationAuditError):
        build()


def test_an_audit_is_immutable() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        _audit().recorded = ()  # type: ignore[misc]


def test_an_audit_is_built_only_from_a_generation_result() -> None:
    with pytest.raises(TypeError):
        FuturesRecommendationGenerationAudit.of(_audit())  # type: ignore[arg-type]


def test_the_audit_errors_are_value_errors() -> None:
    assert issubclass(InvalidFuturesRecommendationGenerationAuditError, ValueError)
