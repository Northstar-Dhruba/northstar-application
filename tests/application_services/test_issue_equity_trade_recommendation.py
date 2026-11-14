"""EI-5D: MODELED recommendation issuance, as Application contract tests (ADR-024).

Scenario identifiers are the EI-5D rows of the EI-5 design acceptance matrix
(IS, ID, BD, HW, NB, AR) at layer C (Application contract test with test
doubles) and A (source scan). The SQLite authority, real concurrency, restart
and separate-process acceptance are EI-6's (layer S) and are not claimed here.

Everything is synthetic and runs in memory:

- the research world, reference doubles and the EI-4C generator are the EI-4C
  test module's own fixtures (``ei4c``), reused unchanged;
- generation records are recorded by the real verified EI-4D recorder into
  the EI-4D-A test store (``ei4d._Store``) and re-verified at levels A to C;
- the issuance authority is ``_Authority`` below: a LABELLED SYNTHETIC
  IN-MEMORY TEST AUTHORITY. It is not durable and not a production adapter.
  It stores records and applies the production invariants
  (``equity_recommendation_issuance_rules``); like an adapter it validates
  every authorization with the public validators before anything changes.

Nothing here reads a clock, opens a connection, places an order or records a
fill.
"""

from __future__ import annotations

import ast
import dataclasses
import fnmatch
import inspect
import json
import os
import pathlib
import subprocess
import sys
from typing import Any

import pytest
import test_equity_recommendation_v2_generation as v2
import test_generate_equity_trade_recommendation as ei4c
import test_record_equity_recommendation_generation as ei4d
from northstar_core.foundation.value_objects import PointInTime
from northstar_core.recommendations.equity import EquityIssuanceKind
from northstar_core.strategy import StrategyIdentity

from northstar_application import application_services
from northstar_application.application_services import (
    EQUITY_RECOMMENDATION_ISSUANCE_AUTHORITY_REFERENCE,
    EQUITY_RECOMMENDATION_ISSUANCE_KEY_NAMESPACE,
    AdvanceEquityIntradayResearchTimelineUseCase,
    AuthoritativeEquitySessionRecommendationHistorySource,
    DeclareEquitySessionIssuanceCompletenessUseCase,
    EquityIssuanceCompletenessDeclaration,
    EquityIssuanceDeclarationResult,
    EquityIssuedRecommendation,
    EquityRecommendationEligibilityAudit,
    EquityRecommendationIssuanceRefusal,
    EquityRecommendationIssuanceRefusedError,
    EquityRecommendationIssuanceResult,
    EquityRecommendationIssuanceUnresolvedError,
    EquityRecordedCompletenessDeclaration,
    EquityRecordedIssuance,
    InvalidEquityIssuanceCompletenessDeclarationError,
    InvalidEquityIssuedRecommendationError,
    IssueEquityTradeRecommendationUseCase,
    VerifiedEquityIssuanceCompletenessDeclaration,
    VerifiedEquityRecommendationIssuance,
    VerifyStoredEquityRecommendationGenerationUseCase,
    equity_declaration_write_decision,
    equity_issuance_is_backdated,
    equity_issuance_session_view,
    equity_issuance_write_decision,
    equity_issued_recommendation,
    equity_recommendation_disposition,
    equity_recommendation_issuance_identity,
    is_equity_recommendation_issuance_identity,
    require_verified_equity_issuance_completeness_declaration,
    require_verified_equity_recommendation_issuance,
)
from northstar_application.application_services import (
    equity_recommendation_issuance_records as records_module,
)
from northstar_application.ports import (
    EquityIssuanceDeclarationOutcome,
    EquityIssuanceDeclarationWrite,
    EquityIssuanceWatermarkBeyondAuthorityError,
    EquityPriorSessionRecommendation,
    EquityRecommendationHistoryUnavailableError,
    EquityRecommendationIssuanceCommitUnconfirmedError,
    EquityRecommendationIssuanceOutcome,
    EquityRecommendationIssuanceReader,
    EquityRecommendationIssuanceUnavailableError,
    EquityRecommendationIssuanceWrite,
    EquityRecommendationIssuanceWriter,
    EquitySessionRecommendationHistory,
    UnverifiedEquityRecommendationIssuanceWriteError,
)

_O = EquityRecommendationIssuanceOutcome
_D = EquityIssuanceDeclarationOutcome
_R = EquityRecommendationIssuanceRefusal
_REF = EQUITY_RECOMMENDATION_ISSUANCE_AUTHORITY_REFERENCE
_V2 = "northstar.equity-recommendation-evaluator/2"
_T = PointInTime(v2._T)  # 09:36 IST: the default scenario's modeled issuance
_B = PointInTime(v2._B)  # b = t - 1 us

# ID-01 golden vectors, computed independently of production code (plain hashlib
# over the length-prefixed parts) and pinned here.
_GOLDEN_SYNTHETIC = "eiris1-sha256:545abd237c87b33a33afb633f3b326006be736859eacb884e0dbd36a6d04d655"
_GOLDEN_FIXTURE = "eiris1-sha256:3954fcc79f33df12abefbb9f3d504815436ff708134bd4ab2adc5cd0ea9dc9cb"
_KEY_1 = "eirk1-sha256:" + "1" * 64
_PROPOSAL_2 = "eirp1-sha256:" + "2" * 64


# ----------------------------------------------------------------------
# The labelled synthetic in-memory test authority
# ----------------------------------------------------------------------


class _Authority(EquityRecommendationIssuanceReader, EquityRecommendationIssuanceWriter):
    """LABELLED SYNTHETIC IN-MEMORY TEST AUTHORITY -- not durable, not an adapter.

    It holds append-only lists of recorded issuances and declarations and one
    recording sequence, and applies the production invariants. ``script``
    injects one behaviour per write, consumed in order: ``unavailable`` (nothing
    written), ``lost_before`` (a lost COMMIT with no effect) and ``lost_after``
    (committed, then the acknowledgment lost). ``unreadable`` makes every read
    fail. ``plant`` is TEST-ONLY corruption setup outside any write path.
    """

    def __init__(self) -> None:
        self.issuances: list[EquityRecordedIssuance] = []
        self.declarations: list[EquityRecordedCompletenessDeclaration] = []
        self.sequence = 0
        self.script: list[str] = []
        self.unreadable = False
        self.writes = 0
        self.issuance_writes = 0
        self.reads = 0

    # -- reads --------------------------------------------------------------

    def _read(self) -> None:
        self.reads += 1
        if self.unreadable:
            raise EquityRecommendationIssuanceUnavailableError("test: authority unreadable")

    def load_issuance(self, issuance_id: str) -> EquityRecordedIssuance | None:
        self._read()
        return next(
            (r for r in self.issuances if r.recommendation.issuance_id == issuance_id), None
        )

    def load_session_issuance(self, strategy, listing, trading_date):  # type: ignore[no-untyped-def]
        self._read()
        return next(
            (
                r
                for r in self.issuances
                if (
                    r.recommendation.strategy,
                    r.recommendation.listing,
                    r.recommendation.trading_date,
                )
                == (strategy, listing, trading_date)
            ),
            None,
        )

    def session_view(self, exchange_code, trading_date):  # type: ignore[no-untyped-def]
        self._read()
        return self._view(exchange_code, trading_date, self.sequence)

    def session_view_at(self, exchange_code, trading_date, watermark):  # type: ignore[no-untyped-def]
        self._read()
        if watermark > self.sequence:
            raise EquityIssuanceWatermarkBeyondAuthorityError(
                f"test: watermark {watermark} is beyond {self.sequence}"
            )
        return self._view(exchange_code, trading_date, watermark)

    def _view(self, exchange_code, trading_date, watermark):  # type: ignore[no-untyped-def]
        return equity_issuance_session_view(
            exchange_code,
            trading_date,
            watermark,
            [r.history_record() for r in self.issuances],
            self.declarations,
        )

    # -- authorized writes ----------------------------------------------------

    def _behaviour(self) -> str:
        self.writes += 1
        mode = self.script.pop(0) if self.script else "ok"
        if mode == "unavailable":
            raise EquityRecommendationIssuanceUnavailableError("test: authority unavailable")
        if mode == "lost_before":
            raise EquityRecommendationIssuanceCommitUnconfirmedError("test: COMMIT lost")
        return mode

    def write_issuance(self, authorization):  # type: ignore[no-untyped-def]
        issuance = require_verified_equity_recommendation_issuance(authorization).issuance
        self.issuance_writes += 1
        mode = self._behaviour()
        outcome, stored = equity_issuance_write_decision(
            issuance, self.issuances, self.declarations
        )
        if outcome is _O.ISSUED:
            self.sequence += 1
            stored = EquityRecordedIssuance(issuance, self.sequence)
            self.issuances.append(stored)
        if mode == "lost_after":
            raise EquityRecommendationIssuanceCommitUnconfirmedError("test: ack lost")
        return EquityRecommendationIssuanceWrite(outcome, stored)

    def write_declaration(self, authorization):  # type: ignore[no-untyped-def]
        declaration = require_verified_equity_issuance_completeness_declaration(
            authorization
        ).declaration
        mode = self._behaviour()
        outcome, recorded = equity_declaration_write_decision(declaration, self.declarations)
        if outcome is _D.DECLARED:
            self.sequence += 1
            recorded = EquityRecordedCompletenessDeclaration(declaration, self.sequence)
            self.declarations.append(recorded)
        if mode == "lost_after":
            raise EquityRecommendationIssuanceCommitUnconfirmedError("test: ack lost")
        assert recorded is not None
        return EquityIssuanceDeclarationWrite(outcome, recorded)

    def plant(self, recommendation: EquityIssuedRecommendation) -> None:
        """TEST-ONLY CORRUPTION SETUP: append a record outside any authorized write."""
        self.sequence += 1
        self.issuances.append(EquityRecordedIssuance(recommendation, self.sequence))


# ----------------------------------------------------------------------
# The synthetic world: EI-4C research, EI-4D records, the test authority
# ----------------------------------------------------------------------


@dataclasses.dataclass
class _World:
    ports: ei4c._Ports
    authority: _Authority
    store: ei4d._Store

    def declare(self, through: PointInTime) -> EquityIssuanceDeclarationResult:
        return DeclareEquitySessionIssuanceCompletenessUseCase(
            self.ports.world.venue, self.authority, self.authority
        ).execute(ei4c._NSE, ei4c._DAY, through)

    def generate(self, **changes: Any) -> EquityRecommendationEligibilityAudit:
        """Version-2 generation against the authoritative history source."""
        self.ports.evaluator_version = _V2
        self.ports.state = AuthoritativeEquitySessionRecommendationHistorySource(self.authority)  # type: ignore[assignment]
        return self.ports.run(**changes)

    def record(self, audit: EquityRecommendationEligibilityAudit) -> Any:
        result = ei4d._recorder(self.ports, self.store).execute(audit)
        assert result.is_stored, (result.outcome, result.detail)
        return result

    def verifier(self) -> VerifyStoredEquityRecommendationGenerationUseCase:
        return VerifyStoredEquityRecommendationGenerationUseCase(
            self.store, ei4d._verifier(self.ports)
        )

    def issuer(self, max_writes: int = 3) -> IssueEquityTradeRecommendationUseCase:
        return IssueEquityTradeRecommendationUseCase(
            self.verifier(), self.authority, self.authority, max_writes
        )

    def issue(self, generation_key: str, **options: Any) -> EquityRecommendationIssuanceResult:
        return self.issuer(**options).execute(generation_key)

    def refusal(self, generation_key: str) -> EquityRecommendationIssuanceRefusedError:
        with pytest.raises(EquityRecommendationIssuanceRefusedError) as raised:
            self.issue(generation_key)
        return raised.value


def _world() -> _World:
    return _World(ei4c._ports(), _Authority(), ei4d._Store())


def _proposed(world: _World | None = None, **changes: Any) -> tuple[_World, Any]:
    """Declare through b, generate a version-2 PROPOSED disposition and record it."""
    world = world or _world()
    if world.authority.declarations == []:
        assert world.declare(_B).outcome is _D.DECLARED
    audit = world.generate(**changes)
    assert audit.status is ei4c._S.PROPOSED, audit.reasons
    world.record(audit)
    disposition = equity_recommendation_disposition(audit)
    assert disposition is not None
    return world, disposition


def _variant(issued: EquityIssuedRecommendation, **changes: Any) -> EquityIssuedRecommendation:
    """TEST-ONLY: another issued record; the identity re-derived from its parts."""
    values = {f.name: getattr(issued, f.name) for f in dataclasses.fields(issued)}
    values.update(changes)
    values["issuance_id"] = equity_recommendation_issuance_identity(
        values["generation_key"], values["proposal_fingerprint"], values["issuance_kind"]
    )
    return EquityIssuedRecommendation(**values)


def _ist(hour: int, minute: int) -> PointInTime:
    return ei4c._at(hour, minute)


# ----------------------------------------------------------------------
# IS: issuance
# ----------------------------------------------------------------------


def test_is01_a_verified_v2_proposed_disposition_is_issued_modeled() -> None:
    world, disposition = _proposed()
    audit_history = disposition.inputs.issued_history
    assert audit_history.reference == _REF  # D5D-3, frozen exactly into the v2 inputs
    assert (audit_history.as_of, audit_history.complete_through) == (_B, _B)
    result = world.issue(disposition.generation_key)
    assert result.outcome is _O.ISSUED and result.is_issued
    assert (result.writes, result.reconciled, result.recording_sequence) == (1, False, 2)
    issued = result.issuance
    assert issued == equity_issued_recommendation(disposition)
    assert issued.issuance_kind is EquityIssuanceKind.MODELED
    assert (issued.generation_key, issued.proposal_fingerprint, issued.assessment_fingerprint) == (
        disposition.generation_key,
        disposition.proposal_fingerprint,
        disposition.assessment_fingerprint,
    )
    assert world.authority.issuances == [EquityRecordedIssuance(issued, 2)]


def test_is02_issuing_again_is_equivalent_and_writes_nothing() -> None:
    world, disposition = _proposed()
    first = world.issue(disposition.generation_key)
    again = world.issue(disposition.generation_key)
    assert again.outcome is _O.EQUIVALENT and again.is_issued
    assert again.authoritative == first.authoritative
    assert len(world.authority.issuances) == 1 and world.authority.sequence == 2
    # Still EQUIVALENT after the session is declared further: identity before backdating.
    assert world.declare(_ist(10, 0)).outcome is _D.DECLARED
    assert world.issue(disposition.generation_key).outcome is _O.EQUIVALENT


def test_is03_another_policy_set_for_the_same_strategy_listing_and_session_is_refused() -> None:
    world, first = _proposed()
    assert world.issue(first.generation_key).outcome is _O.ISSUED
    _, second = _proposed(world, policy_set_version="set/2")
    assert second.generation_key != first.generation_key
    result = world.issue(second.generation_key)
    assert result.outcome is _O.SESSION_STRATEGY_ALREADY_ISSUED and not result.is_issued
    assert result.authoritative is not None
    assert result.authoritative.recommendation.generation_key == first.generation_key
    assert len(world.authority.issuances) == 1  # the first stays authoritative


def test_is04_another_strategy_on_the_same_listing_and_session_is_issued() -> None:
    world, first = _proposed()
    assert world.issue(first.generation_key).outcome is _O.ISSUED
    other = dataclasses.replace(ei4c._CONFIG, strategy=StrategyIdentity("ei5d-second-strategy"))
    # The same synthetic snapshots, re-issued idempotently, for a second strategy.
    ids = [world.ports.world.issue(ei4c._ist(9, m)).snapshot_id for m in (31, 36, 41)]
    AdvanceEquityIntradayResearchTimelineUseCase(
        world.ports.world.timelines, world.ports.world.snapshots, other
    ).execute(ei4c._LISTING, ei4c._DAY, ids)
    _, second = _proposed(world, configuration=other)
    result = world.issue(second.generation_key)
    assert result.outcome is _O.ISSUED and result.recording_sequence == 3
    assert {r.recommendation.strategy for r in world.authority.issuances} == {
        ei4c._CONFIG.strategy,
        other.strategy,
    }


def test_is05_a_not_eligible_disposition_is_refused_with_no_write() -> None:
    world = _world()
    world.declare(_B)
    world.ports.references.hold("generation", ei4c._generation("generation/strict", "100"))
    audit = world.generate(generation_policy_version=ei4c._v("generation/strict"))
    assert audit.status is ei4c._S.NOT_ELIGIBLE
    world.record(audit)
    refused = world.refusal(audit.generation_key)  # type: ignore[arg-type]
    assert refused.reason is _R.DISPOSITION_NOT_PROPOSED and not refused.retryable
    assert world.authority.issuance_writes == 0 and world.authority.issuances == []


def test_is06_an_unresolved_disposition_is_refused_with_no_write() -> None:
    world = _world()
    world.ports.evaluator_version = _V2
    world.ports.state.answer = ei4c._history(ei4c._prior(ei4c._LISTING), through=_B)
    audit = world.ports.run()
    assert audit.status is ei4c._S.EVALUATION_UNRESOLVED
    world.record(audit)
    assert world.refusal(audit.generation_key).reason is _R.DISPOSITION_NOT_PROPOSED  # type: ignore[arg-type]
    assert world.authority.issuance_writes == 0


def test_is07_an_unknown_generation_key_is_refused() -> None:
    world = _world()
    refused = world.refusal("eirk1-sha256:" + "0" * 64)
    assert refused.reason is _R.GENERATION_NOT_FOUND and not refused.retryable
    assert world.authority.writes == 0


@pytest.mark.parametrize(
    ("failure", "reason", "retryable"),
    [
        ("undecodable", _R.STORED_GENERATION_INVALID, False),
        ("incompatible", _R.STORED_GENERATION_INCOMPATIBLE, False),
        ("unavailable", _R.GENERATION_STORE_UNAVAILABLE, True),
    ],
)
def test_is08_a_stored_record_failing_level_a_is_refused_and_preserved(
    failure: str, reason: Any, retryable: bool
) -> None:
    world, disposition = _proposed()
    before = dict(world.store.dispositions)
    world.store.load_failure = failure
    refused = world.refusal(disposition.generation_key)
    assert (refused.reason, refused.retryable) == (reason, retryable)
    assert world.store.dispositions == before  # preserved, never repaired
    assert world.authority.issuance_writes == 0


def test_is09_unavailable_source_evidence_is_a_retryable_refusal() -> None:
    world, disposition = _proposed()
    world.ports.world.timelines.unavailable = True
    refused = world.refusal(disposition.generation_key)
    assert (refused.reason, refused.retryable) == (_R.SOURCE_EVIDENCE_UNAVAILABLE, True)
    world.ports.world.timelines.unavailable = False
    world.ports.world.timelines.records.clear()  # the source is gone: permanent
    assert world.refusal(disposition.generation_key).reason is _R.SOURCE_EVIDENCE_INVALID
    assert world.authority.issuance_writes == 0


def test_is10_a_replay_mismatch_at_level_c_is_a_permanent_refusal() -> None:
    world, disposition = _proposed()
    document = json.loads(disposition.assessment_document)
    for timing in (
        document["entry_window"]["timing"],
        document["proposal"]["entry_window"]["timing"],
    ):
        timing["eligible_from"] = "2026-08-12T04:15:00Z"
    proposal_text, proposal_fingerprint = v2._resealed(document["proposal"])
    document["proposal_fingerprint"] = proposal_fingerprint
    text, fingerprint = v2._reseal_assessment(document)
    world.store.dispositions[disposition.generation_key] = ei4d._rebuilt(
        disposition,
        assessment_document=text,
        assessment_fingerprint=fingerprint,
        proposal_document=proposal_text,
        proposal_fingerprint=proposal_fingerprint,
    )
    refused = world.refusal(disposition.generation_key)
    assert refused.reason is _R.REPLAY_MISMATCH and not refused.retryable
    assert world.authority.issuance_writes == 0


def test_is11_actual_issuance_is_refused_before_any_read() -> None:
    world, disposition = _proposed()
    world.store.before_load.append(lambda: pytest.fail("the generation store was read"))
    reads = world.authority.reads
    with pytest.raises(EquityRecommendationIssuanceRefusedError) as raised:
        world.issuer().execute(disposition.generation_key, EquityIssuanceKind.ACTUAL)
    assert raised.value.reason is _R.ACTUAL_ISSUANCE_NOT_AUTHORIZED
    assert world.authority.reads == reads and world.authority.issuance_writes == 0
    world.store.before_load.clear()


def test_is12_the_issued_record_is_non_actionable_with_its_unassessed_controls() -> None:
    world, disposition = _proposed()
    issued = world.issue(disposition.generation_key).issuance
    assert (issued.is_actionable, issued.is_synthetic) == (False, True)
    assert disposition.proposal is not None
    assert issued.not_assessed == disposition.proposal.not_assessed  # unchanged
    with pytest.raises(dataclasses.FrozenInstanceError):
        issued.is_actionable = True  # type: ignore[misc]


def test_is13_timing_is_the_proposals_and_no_caller_parameter_exists() -> None:
    world, disposition = _proposed()
    issued = world.issue(disposition.generation_key).issuance
    proposal = disposition.proposal
    assert proposal is not None
    timing, window = proposal.entry_window.timing, proposal.entry_window
    assert (issued.decision_as_of, issued.issued_at, issued.eligible_from) == (
        timing.decision_as_of,
        timing.issued_at,
        timing.eligible_from,
    )
    assert (issued.latest_entry_bar_opens_at, issued.modeled_exit_at) == (
        window.latest_entry_bar_opens_at,
        window.modeled_exit_at,
    )
    assert issued.issued_at == _T and issued.levels == proposal.definition.levels
    parameters = list(inspect.signature(IssueEquityTradeRecommendationUseCase.execute).parameters)
    assert parameters == ["self", "generation_key", "issuance_kind"]


def test_is14_a_writer_call_without_a_verified_authorization_is_refused() -> None:
    world, disposition = _proposed()
    issued = equity_issued_recommendation(disposition)
    verified = world.verifier().execute(disposition.generation_key)
    unissued = VerifiedEquityRecommendationIssuance(verified, issued)  # public construction
    for candidate in (None, issued, unissued, "eiris1"):
        with pytest.raises(UnverifiedEquityRecommendationIssuanceWriteError):
            world.authority.write_issuance(candidate)
    assert (world.authority.issuance_writes, world.authority.issuances) == (0, [])


def test_is15_an_authorization_with_swapped_content_is_refused() -> None:
    world, disposition = _proposed()
    captured: list[Any] = []

    class _Capture(_Authority):
        def write_issuance(self, authorization):  # type: ignore[no-untyped-def]
            captured.append(authorization)
            return super().write_issuance(authorization)

    capture = _Capture()
    capture.declarations = world.authority.declarations
    capture.sequence = world.authority.sequence
    IssueEquityTradeRecommendationUseCase(world.verifier(), capture, capture).execute(
        disposition.generation_key
    )
    (authorization,) = captured
    for replacement in (
        dataclasses.replace(authorization),  # a copy is never issued
    ):
        with pytest.raises(UnverifiedEquityRecommendationIssuanceWriteError):
            capture.write_issuance(replacement)
    object.__setattr__(  # TEST-ONLY privileged tampering with an issued authorization
        authorization,
        "issuance",
        _variant(authorization.issuance, modeled_exit_at=PointInTime("2026-08-12T09:00:00Z")),
    )
    with pytest.raises(UnverifiedEquityRecommendationIssuanceWriteError):
        capture.write_issuance(authorization)
    assert len(capture.issuances) == 1


def test_is16_a_version1_disposition_is_refused_and_preserved() -> None:
    world = _world()
    audit = world.ports.run()  # version 1 over the labelled fixture history
    assert audit.status is ei4c._S.PROPOSED and audit.evaluator_version != _V2
    world.record(audit)
    before = dict(world.store.dispositions)
    refused = world.refusal(audit.generation_key)  # type: ignore[arg-type]
    assert refused.reason is _R.EVALUATOR_NOT_ISSUABLE and not refused.retryable
    assert world.store.dispositions == before and world.authority.issuance_writes == 0


def test_is16_v2_history_from_a_fixture_never_names_the_authority() -> None:
    world = _world()
    _, audit = ei4c._proposed_v2(world.ports)  # v2 over a labelled fixture history
    world.record(audit)
    assert audit.inputs is not None and audit.inputs.issued_history.reference != _REF
    refused = world.refusal(audit.generation_key)  # type: ignore[arg-type]
    assert refused.reason is _R.HISTORY_NOT_AUTHORITATIVE
    assert world.authority.issuance_writes == 0


@pytest.mark.parametrize("case", ["another declaration at the watermark", "watermark beyond"])
def test_is16_v2_history_that_does_not_reproduce_at_its_watermark_is_refused(case: str) -> None:
    world, disposition = _proposed()  # frozen against this authority at watermark 1
    other = _Authority()  # the issuer's authority holds another history
    if case == "another declaration at the watermark":
        other_world = _World(world.ports, other, world.store)
        other_world.declare(_ist(9, 0))  # sequence 1 declares only 09:00
        assert other.sequence == 1
    issuer = IssueEquityTradeRecommendationUseCase(world.verifier(), other, other)
    with pytest.raises(EquityRecommendationIssuanceRefusedError) as raised:
        issuer.execute(disposition.generation_key)
    assert raised.value.reason is _R.HISTORY_NOT_REPRODUCED
    assert other.issuances == []


def test_is16_a_session_declared_beyond_b_is_backdated_and_nothing_is_written() -> None:
    world, disposition = _proposed()
    assert world.declare(_T).outcome is _D.DECLARED  # declared through t: beyond b
    result = world.issue(disposition.generation_key)
    assert result.outcome is _O.BACKDATED_ISSUANCE and not result.is_issued
    assert result.authoritative is None and world.authority.issuances == []


def test_is17_issuance_leaves_the_ei4d_records_byte_identical() -> None:
    world, disposition = _proposed()
    before = (
        {
            k: (d.assessment_document, d.proposal_document)
            for k, d in world.store.dispositions.items()
        },
        {k: a.attempt_document for k, a in world.store.attempts.items()},
    )
    writes = world.store.writes
    world.issue(disposition.generation_key)
    after = (
        {
            k: (d.assessment_document, d.proposal_document)
            for k, d in world.store.dispositions.items()
        },
        {k: a.attempt_document for k, a in world.store.attempts.items()},
    )
    assert after == before and world.store.writes == writes
    assert not any(hasattr(d, "state") for d in world.store.dispositions.values())
    assert world.store.dispositions[disposition.generation_key].is_issued is False


# ----------------------------------------------------------------------
# D5D-4: missing authoritative completeness at the issuer
# ----------------------------------------------------------------------


def test_d5d4_an_issuer_without_the_authoritative_declaration_refuses() -> None:
    world, disposition = _proposed()

    class _Undeclared(_Authority):
        """TEST-ONLY inconsistent reader: the current view states no declaration."""

        def session_view(self, exchange_code, trading_date):  # type: ignore[no-untyped-def]
            return None

    undeclared = _Undeclared()
    undeclared.declarations, undeclared.sequence = world.authority.declarations, 1
    issuer = IssueEquityTradeRecommendationUseCase(world.verifier(), undeclared, undeclared)
    with pytest.raises(EquityRecommendationIssuanceRefusedError) as raised:
        issuer.execute(disposition.generation_key)
    assert raised.value.reason is _R.AUTHORITATIVE_COMPLETENESS_NOT_DECLARED
    assert undeclared.issuance_writes == 0


# ----------------------------------------------------------------------
# Writer outcomes, conflicts and commit uncertainty
# ----------------------------------------------------------------------


def test_a_different_stored_record_under_the_identity_is_never_adopted() -> None:
    world, disposition = _proposed()
    issued = equity_issued_recommendation(disposition)
    tampered = dataclasses.replace(issued, modeled_exit_at=PointInTime("2026-08-12T09:00:00Z"))
    assert tampered.issuance_id == issued.issuance_id
    world.authority.plant(tampered)
    result = world.issue(disposition.generation_key)
    assert result.outcome is _O.STORED_ISSUANCE_INVALID and result.authoritative is None
    assert world.authority.issuances[-1].recommendation == tampered  # preserved


def test_another_issuance_of_the_generation_key_is_generation_already_issued() -> None:
    world, disposition = _proposed()
    issued = equity_issued_recommendation(disposition)
    world.authority.plant(_variant(issued, proposal_fingerprint="eirp1-sha256:" + "3" * 64))
    result = world.issue(disposition.generation_key)
    assert result.outcome is _O.GENERATION_ALREADY_ISSUED
    assert result.authoritative is not None
    assert result.authoritative.recommendation.issuance_id != issued.issuance_id


@pytest.mark.parametrize(
    ("script", "outcome", "writes", "reconciled"),
    [
        (["lost_after"], _O.EQUIVALENT, 1, True),
        (["lost_before"], _O.ISSUED, 2, False),
    ],
    ids=["acknowledgment lost", "commit lost before it took effect"],
)
def test_an_unconfirmed_commit_reconciles_by_reload(
    script: list[str], outcome: Any, writes: int, reconciled: bool
) -> None:
    world, disposition = _proposed()
    world.authority.script = list(script)
    result = world.issue(disposition.generation_key)
    assert (result.outcome, result.writes, result.reconciled) == (outcome, writes, reconciled)
    assert len(world.authority.issuances) == 1  # never duplicated


def test_unsettled_commit_uncertainty_claims_nothing() -> None:
    world, disposition = _proposed()
    world.authority.script = ["lost_before"] * 3
    with pytest.raises(EquityRecommendationIssuanceUnresolvedError):
        world.issue(disposition.generation_key)
    assert world.authority.issuances == []

    class _LostThenUnreadable(_Authority):
        def write_issuance(self, authorization):  # type: ignore[no-untyped-def]
            require_verified_equity_recommendation_issuance(authorization)
            self.unreadable = True
            raise EquityRecommendationIssuanceCommitUnconfirmedError("test: COMMIT lost")

    lost = _LostThenUnreadable()
    lost.declarations, lost.sequence = world.authority.declarations, world.authority.sequence
    with pytest.raises(EquityRecommendationIssuanceUnresolvedError):
        IssueEquityTradeRecommendationUseCase(world.verifier(), lost, lost).execute(
            disposition.generation_key
        )


def test_an_unavailable_authority_is_a_retryable_refusal_with_nothing_written() -> None:
    world, disposition = _proposed()
    world.authority.script = ["unavailable"]
    refused = world.refusal(disposition.generation_key)
    assert (refused.reason, refused.retryable) == (_R.AUTHORITY_UNAVAILABLE, True)
    assert world.authority.issuances == []
    world.authority.unreadable = True
    assert world.refusal(disposition.generation_key).reason is _R.AUTHORITY_UNAVAILABLE


def test_a_writer_answer_contradicting_the_offered_record_is_never_adopted() -> None:
    world, disposition = _proposed()

    class _Contradicting(_Authority):
        def write_issuance(self, authorization):  # type: ignore[no-untyped-def]
            issued = require_verified_equity_recommendation_issuance(authorization).issuance
            other = dataclasses.replace(issued, modeled_exit_at=PointInTime("2026-08-12T09:00:00Z"))
            return EquityRecommendationIssuanceWrite(
                _O.EQUIVALENT, EquityRecordedIssuance(other, 9)
            )

    contradicting = _Contradicting()
    contradicting.declarations, contradicting.sequence = world.authority.declarations, 1
    result = IssueEquityTradeRecommendationUseCase(
        world.verifier(), contradicting, contradicting
    ).execute(disposition.generation_key)
    assert result.outcome is _O.STORED_ISSUANCE_INVALID and result.authoritative is None


# ----------------------------------------------------------------------
# ID: identity
# ----------------------------------------------------------------------


def test_id01_the_issuance_identity_matches_its_golden_vectors() -> None:
    assert (
        equity_recommendation_issuance_identity(_KEY_1, _PROPOSAL_2, EquityIssuanceKind.MODELED)
        == _GOLDEN_SYNTHETIC
    )
    assert (
        equity_recommendation_issuance_identity(v2._EIRK1, v2._EIRP1, EquityIssuanceKind.MODELED)
        == _GOLDEN_FIXTURE
    )
    _, disposition = _proposed()
    assert (disposition.generation_key, disposition.proposal_fingerprint) == (v2._EIRK1, v2._EIRP1)
    assert equity_issued_recommendation(disposition).issuance_id == _GOLDEN_FIXTURE


def test_id02_the_identity_is_identical_in_two_fresh_processes() -> None:
    code = (
        "from northstar_application.application_services import "
        "equity_recommendation_issuance_identity as f\n"
        "from northstar_core.recommendations.equity import EquityIssuanceKind as K\n"
        f"print(f({v2._EIRK1!r}, {v2._EIRP1!r}, K.MODELED))\n"
    )
    seen = set()
    for seed in ("1", "4242"):
        completed = subprocess.run(  # the test's own interpreter, running fixed code only
            [sys.executable, "-c", code],
            env=dict(os.environ, PYTHONHASHSEED=seed),
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        seen.add(completed.stdout.strip())
    assert seen == {_GOLDEN_FIXTURE}


def test_id03_the_prefix_is_eiris1_and_distinct_from_every_ei4_prefix() -> None:
    identity = _GOLDEN_SYNTHETIC
    assert is_equity_recommendation_issuance_identity(identity)
    assert fnmatch.fnmatchcase(identity, "eiris1-sha256:?*")  # SQLite GLOB semantics
    assert not fnmatch.fnmatchcase(identity, "eiri1-sha256:?*")
    for prefix in ("eiri1", "eira1", "eirp1", "eirat1", "eirk1", "eirs1", "eipd1"):
        assert not identity.startswith(prefix + "-sha256:")
    assert EQUITY_RECOMMENDATION_ISSUANCE_KEY_NAMESPACE == (
        "northstar.equity-recommendation-issuance-key/1"
    )


def test_id04_changing_any_part_changes_the_identity() -> None:
    modeled = EquityIssuanceKind.MODELED
    base = equity_recommendation_issuance_identity(_KEY_1, _PROPOSAL_2, modeled)
    others = {
        equity_recommendation_issuance_identity("eirk1-sha256:" + "4" * 64, _PROPOSAL_2, modeled),
        equity_recommendation_issuance_identity(_KEY_1, "eirp1-sha256:" + "5" * 64, modeled),
        equity_recommendation_issuance_identity(_KEY_1, _PROPOSAL_2, EquityIssuanceKind.ACTUAL),
    }
    assert base not in others and len(others) == 3
    with pytest.raises(InvalidEquityIssuedRecommendationError):
        equity_recommendation_issuance_identity("eiri1-sha256:" + "1" * 64, _PROPOSAL_2, modeled)


def test_id05_shifting_a_part_boundary_changes_the_identity() -> None:
    parts = (EQUITY_RECOMMENDATION_ISSUANCE_KEY_NAMESPACE, _KEY_1, _PROPOSAL_2, "MODELED")
    shifted = (parts[0], parts[1][:-1], parts[1][-1] + parts[2], parts[3])
    assert "".join(parts) == "".join(shifted)  # plain concatenation could not tell them apart
    assert records_module._identity_from_parts(parts) != records_module._identity_from_parts(
        shifted
    )
    assert records_module._identity_from_parts(parts) == _GOLDEN_SYNTHETIC


# ----------------------------------------------------------------------
# BD: backdating (D5D-4: BD-04 is a writer-contract scenario)
# ----------------------------------------------------------------------


def test_bd01_bd02_bd03_the_no_backdating_rule() -> None:
    assert equity_issuance_is_backdated(_ist(10, 10), _ist(10, 30))  # BD-01
    assert equity_issuance_is_backdated(_ist(10, 30), _ist(10, 30))  # BD-02: equal is backdated
    assert not equity_issuance_is_backdated(_ist(10, 40), _ist(10, 30))  # BD-03
    assert not equity_issuance_is_backdated(_T, _B)  # BD-03, C-2: t against exactly t - 1 us


def test_bd01_at_the_issuer_a_declaration_covering_t_refuses_the_issuance() -> None:
    world, disposition = _proposed()
    world.declare(_ist(10, 30))
    assert world.issue(disposition.generation_key).outcome is _O.BACKDATED_ISSUANCE


def test_bd03_at_the_issuer_t_against_exactly_b_is_issued() -> None:
    world, disposition = _proposed()
    assert world.authority.session_view(ei4c._NSE, ei4c._DAY).complete_through == _B
    assert world.issue(disposition.generation_key).outcome is _O.ISSUED


def test_bd04_with_no_declaration_the_writer_contract_alone_does_not_refuse() -> None:
    _, disposition = _proposed()
    issued = equity_issued_recommendation(disposition)
    assert not equity_issuance_is_backdated(issued.issued_at, None)
    assert equity_issuance_write_decision(issued, [], []) == (_O.ISSUED, None)
    # At the issuer the same request is ineligible: see the D5D-4 and IS-16 tests.


# ----------------------------------------------------------------------
# HW: the authoritative history
# ----------------------------------------------------------------------


def test_hw01_an_undeclared_session_is_unknown_never_empty() -> None:
    source = AuthoritativeEquitySessionRecommendationHistorySource(_Authority())
    assert source.session_history(ei4c._NSE, ei4c._DAY, _B) is None


def test_hw02_a_declaration_with_no_issuance_is_an_explicit_empty_tuple() -> None:
    world = _world()
    world.declare(_B)
    view = AuthoritativeEquitySessionRecommendationHistorySource(world.authority).session_history(
        ei4c._NSE, ei4c._DAY, _B
    )
    assert view is not None and view.recommendations == () and view.complete_through == _B
    assert (view.recording_watermark, view.reference) == (1, _REF)


def _record(key: str, at: PointInTime, sequence: int) -> EquityPriorSessionRecommendation:
    return EquityPriorSessionRecommendation(
        key, StrategyIdentity(f"ei5d-{key}"), ei4c._LISTING, ei4c._DAY,
        EquityIssuanceKind.MODELED, at, sequence,
    )  # fmt: skip


def _declared(at: PointInTime, sequence: int) -> EquityRecordedCompletenessDeclaration:
    return EquityRecordedCompletenessDeclaration(
        EquityIssuanceCompletenessDeclaration(ei4c._NSE, ei4c._DAY, at), sequence
    )


def test_hw03_the_ei5_d4_example_keeps_its_view_after_later_recording() -> None:
    a, b = _record("A", _ist(9, 41), 1), _record("B", _ist(10, 5), 2)
    declarations = [_declared(_ist(10, 30), 3)]
    assert equity_issuance_is_backdated(_ist(10, 10), _ist(10, 30))  # C at 10:10 refused
    before = equity_issuance_session_view(ei4c._NSE, ei4c._DAY, 3, [a, b], declarations)
    c = _record("C", _ist(10, 40), 4)  # C at 10:40 accepted at sequence 4
    after = equity_issuance_session_view(ei4c._NSE, ei4c._DAY, 3, [a, b, c], declarations)
    assert before == after
    assert before is not None and [r.recommendation_key for r in before.recommendations] == [
        "A",
        "B",
    ]
    assert before.complete_through == _ist(10, 30)
    latest = equity_issuance_session_view(ei4c._NSE, ei4c._DAY, 4, [a, b, c], declarations)
    assert latest is not None and [r.recommendation_key for r in latest.recommendations] == [
        "A",
        "B",
    ]  # C is after the declared completeness


def test_hw04_view_content_is_sequence_and_completeness_bounded_in_order() -> None:
    records = [
        _record("z", _ist(9, 50), 1),
        _record("a", _ist(9, 50), 4),  # same instant: ordered by key
        _record("late", _ist(11, 0), 2),  # issued after completeness
        _record("unseen", _ist(9, 40), 6),  # recorded after the watermark
    ]
    declarations = [_declared(_ist(10, 0), 3), _declared(_ist(10, 30), 7)]
    view = equity_issuance_session_view(ei4c._NSE, ei4c._DAY, 5, records, declarations)
    assert view is not None and view.complete_through == _ist(10, 0)
    assert [r.recommendation_key for r in view.recommendations] == ["a", "z"]
    assert equity_issuance_session_view(ei4c._NSE, ei4c._DAY, 2, records, declarations) is None


def test_hw05_declarations_are_monotone_regressions_refused_equals_idempotent() -> None:
    world = _world()
    assert world.declare(_ist(10, 30)).outcome is _D.DECLARED
    lower = world.declare(_ist(10, 0))
    assert lower.outcome is _D.COMPLETENESS_REGRESSION and not lower.is_declared
    assert world.declare(_ist(10, 30)).outcome is _D.DECLARATION_EQUIVALENT
    higher = world.declare(_ist(11, 0))
    assert higher.outcome is _D.DECLARED and higher.recorded.recording_sequence == 2
    assert world.authority.sequence == 2  # only real declarations take a sequence


def test_hw06_no_caller_supplies_a_sequence_or_watermark() -> None:
    signatures = {
        "issue": inspect.signature(IssueEquityTradeRecommendationUseCase.execute),
        "declare": inspect.signature(DeclareEquitySessionIssuanceCompletenessUseCase.execute),
        "write_issuance": inspect.signature(EquityRecommendationIssuanceWriter.write_issuance),
        "write_declaration": inspect.signature(
            EquityRecommendationIssuanceWriter.write_declaration
        ),
    }
    for name, signature in signatures.items():
        for parameter in signature.parameters:
            assert "sequence" not in parameter and "watermark" not in parameter, name
    assert list(signatures["declare"].parameters) == [
        "self",
        "exchange_code",
        "trading_date",
        "complete_through",
    ]


def test_hw07_a_view_beyond_the_current_watermark_is_refused() -> None:
    authority = _Authority()
    with pytest.raises(EquityIssuanceWatermarkBeyondAuthorityError):
        authority.session_view_at(ei4c._NSE, ei4c._DAY, 1)


def test_hw08_an_unreadable_authority_is_history_unavailable() -> None:
    authority = _Authority()
    authority.unreadable = True
    source = AuthoritativeEquitySessionRecommendationHistorySource(authority)
    with pytest.raises(EquityRecommendationHistoryUnavailableError):
        source.session_history(ei4c._NSE, ei4c._DAY, _B)


def test_hw08_a_view_not_naming_the_authority_is_history_unavailable() -> None:
    class _Foreign(_Authority):
        def session_view(self, exchange_code, trading_date):  # type: ignore[no-untyped-def]
            return EquitySessionRecommendationHistory(
                exchange_code, trading_date, (), _B, 0, "test-history/1"
            )

    source = AuthoritativeEquitySessionRecommendationHistorySource(_Foreign())
    with pytest.raises(EquityRecommendationHistoryUnavailableError):
        source.session_history(ei4c._NSE, ei4c._DAY, _B)


def test_hw09_every_produced_view_satisfies_the_existing_history_contract() -> None:
    world, disposition = _proposed()
    world.issue(disposition.generation_key)
    world.declare(_ist(10, 0))
    for watermark in range(1, world.authority.sequence + 1):
        view = world.authority.session_view_at(ei4c._NSE, ei4c._DAY, watermark)
        assert isinstance(view, EquitySessionRecommendationHistory)
        rebuilt = EquitySessionRecommendationHistory(  # its own validation re-runs
            view.exchange_code,
            view.trading_date,
            view.recommendations,
            view.complete_through,
            view.recording_watermark,
            view.reference,
        )
        assert rebuilt == view and view.reference == _REF


# ----------------------------------------------------------------------
# Declarations: venue day, authorization and commit uncertainty
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "case", ["before the local day", "after the local day", "closed venue day", "other date"]
)
def test_a_declaration_outside_its_venue_day_is_refused_with_no_write(case: str) -> None:
    world = _world()
    through = _ist(10, 0)
    venue = world.ports.world.venue
    if case == "before the local day":
        through = PointInTime("2026-08-11T18:29:59.999999Z")  # 23:59:59.999999 IST, 11 Aug
    elif case == "after the local day":
        through = PointInTime("2026-08-12T18:30:00Z")  # 00:00 IST, 13 Aug
    elif case == "closed venue day":
        venue.classification = ei4c.EquityVenueDayClassification.CLOSED
    else:
        venue.answer_date = ei4c.date(2026, 8, 13)
    with pytest.raises(InvalidEquityIssuanceCompletenessDeclarationError):
        world.declare(through)
    assert world.authority.writes == 0 and world.authority.declarations == []


def test_the_local_day_bounds_are_inclusive_of_midnight_and_exclusive_of_the_next() -> None:
    world = _world()
    assert world.declare(PointInTime("2026-08-11T18:30:00Z")).outcome is _D.DECLARED  # 00:00 IST
    assert world.declare(PointInTime("2026-08-12T18:29:59.999999Z")).outcome is _D.DECLARED


def test_declaration_authorizations_are_content_bound() -> None:
    world = _world()
    venue_day = world.ports.world.venue.venue_day(ei4c._NSE, ei4c._DAY)
    declaration = EquityIssuanceCompletenessDeclaration(ei4c._NSE, ei4c._DAY, _B)
    unissued = VerifiedEquityIssuanceCompletenessDeclaration(declaration, venue_day)
    for candidate in (None, declaration, unissued, dataclasses.replace(unissued)):
        with pytest.raises(UnverifiedEquityRecommendationIssuanceWriteError):
            world.authority.write_declaration(candidate)
    assert world.authority.writes == 0


@pytest.mark.parametrize(
    ("script", "outcome", "writes", "reconciled"),
    [
        (["lost_after"], _D.DECLARATION_EQUIVALENT, 1, True),
        (["lost_before"], _D.DECLARED, 2, False),
    ],
)
def test_an_unconfirmed_declaration_commit_reconciles(
    script: list[str], outcome: Any, writes: int, reconciled: bool
) -> None:
    world = _world()
    world.authority.script = list(script)
    result = world.declare(_B)
    assert (result.outcome, result.writes, result.reconciled) == (outcome, writes, reconciled)
    assert len(world.authority.declarations) == 1


# ----------------------------------------------------------------------
# NB, AR: no execution and the write boundary as an architecture rule
# ----------------------------------------------------------------------

_FORBIDDEN_FIELDS = (
    "order",
    "fill",
    "quantity",
    "broker",
    "account",
    "pnl",
    "exit_price",
    "position",
)


def test_nb01_the_issued_record_carries_no_order_fill_quantity_or_exit_price() -> None:
    names = [f.name for f in dataclasses.fields(EquityIssuedRecommendation)]
    assert not [n for n in names for word in _FORBIDDEN_FIELDS if word in n]
    _, disposition = _proposed()
    content = json.loads(equity_issued_recommendation(disposition).content())
    assert not [k for k in content for word in _FORBIDDEN_FIELDS if word in k]
    assert "sizing" not in content  # the modeled Core sizing is not part of the record


def _production_root() -> pathlib.Path:
    source = inspect.getsourcefile(application_services)
    assert source is not None
    return pathlib.Path(source).parents[1]


_ISSUANCE_MODULES = (
    "application_services/issue_equity_trade_recommendation.py",
    "application_services/declare_equity_session_issuance_completeness.py",
    "application_services/equity_recommendation_issuance_records.py",
    "application_services/equity_recommendation_issuance_rules.py",
    "application_services/equity_recommendation_issuance_authorization.py",
    "application_services/authoritative_equity_session_recommendation_history.py",
    "ports/equity_recommendation_issuance.py",
)
_ISSUE_MODULE = "application_services/issue_equity_trade_recommendation.py"
_DECLARE_MODULE = "application_services/declare_equity_session_issuance_completeness.py"
_WRITERS = {"write_issuance": _ISSUE_MODULE, "write_declaration": _DECLARE_MODULE}
_ISSUERS = {"_issue_issuance_write": _ISSUE_MODULE, "_issue_declaration_write": _DECLARE_MODULE}
_AUTHORIZATION_MODULE = "application_services/equity_recommendation_issuance_authorization.py"


def _trees() -> dict[str, ast.Module]:
    root = _production_root()
    return {
        path.relative_to(root).as_posix(): ast.parse(path.read_text(encoding="utf-8"))
        for path in sorted(root.rglob("*.py"))
    }


def test_ar_only_the_two_use_cases_call_the_writer_or_issue_authorizations() -> None:
    violations: list[str] = []
    for path, tree in _trees().items():
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in _WRITERS:
                if (
                    path != _WRITERS[node.attr]
                    and path != "ports/equity_recommendation_issuance.py"
                ):
                    violations.append(f"{path}:{node.lineno}: calls {node.attr}")
            name = node.id if isinstance(node, ast.Name) else getattr(node, "name", None)
            if isinstance(node, ast.alias):
                name = node.name
            if name in _ISSUERS and path not in (_ISSUERS[name], _AUTHORIZATION_MODULE):
                violations.append(f"{path}:{getattr(node, 'lineno', '?')}: uses {name}")
    assert violations == []


def test_nb03_ar_issuance_modules_import_no_execution_infrastructure_or_clock() -> None:
    trees = _trees()
    forbidden = ("infrastructure", "broker", "execution", "paper", "futures", "option", "order")
    violations = []
    for path in _ISSUANCE_MODULES:
        for node in ast.walk(trees[path]):
            if isinstance(node, ast.ImportFrom | ast.Import):
                modules = (
                    [node.module or ""]
                    if isinstance(node, ast.ImportFrom)
                    else [alias.name for alias in node.names]
                )
                for module in modules:
                    if any(word in module for word in forbidden):
                        violations.append(f"{path}: imports {module}")
            if isinstance(node, ast.Attribute) and node.attr in {"now", "utcnow", "today", "time"}:
                if isinstance(node.value, ast.Name) and node.value.id in {
                    "datetime",
                    "time",
                    "date",
                }:
                    violations.append(f"{path}:{node.lineno}: reads a clock")
            if isinstance(node, ast.Name) and node.id in {"random", "uuid4"}:
                violations.append(f"{path}:{node.lineno}: uses randomness")
    assert violations == []


def test_ar_the_writer_port_has_no_raw_write_and_both_ports_are_abstract() -> None:
    assert EquityRecommendationIssuanceWriter.__abstractmethods__ == frozenset(
        {"write_issuance", "write_declaration"}
    )
    assert EquityRecommendationIssuanceReader.__abstractmethods__ == frozenset(
        {"load_issuance", "load_session_issuance", "session_view", "session_view_at"}
    )
    assert not issubclass(
        AuthoritativeEquitySessionRecommendationHistorySource, EquityRecommendationIssuanceWriter
    )
    parameters = inspect.signature(AuthoritativeEquitySessionRecommendationHistorySource).parameters
    assert list(parameters) == ["reader"]  # read-only consumers receive only the reader


def test_the_authority_reference_and_its_export_are_fixed() -> None:
    assert _REF == "northstar.equity-recommendation-issuance-authority/1"
    assert "EQUITY_RECOMMENDATION_ISSUANCE_AUTHORITY_REFERENCE" in application_services.__all__
