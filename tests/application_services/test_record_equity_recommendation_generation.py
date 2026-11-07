"""EI-4D-A: generation-record contracts, verified rehydration and commit reconciliation.

Everything is synthetic. The research world -- EI-2B snapshots issued through
the real use case, EI-3C timelines committed by the real advancement -- and
the EI-4C reference doubles are the EI-4C test module's own fixtures, reused
unchanged. The generation store here is a test-only fake that models committed
state explicitly: every write either commits or does not, and failures --
an unavailable store, a definite rollback, a commit lost before or after it
happens, a competing writer -- are injected one write at a time. It is not a
production adapter and proves nothing about real concurrent SQLite, which is
EI-4D-B's to enforce and EI-4E's to accept. No clock, network or database is
used.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import json
import sys
from collections.abc import Callable
from datetime import time
from typing import Any

import pytest
import test_generate_equity_trade_recommendation as ei4c

from northstar_application import application_services, ports
from northstar_application.application_services import (
    EQUITY_RECOMMENDATION_EVALUATOR_VERSION,
    EquityRecommendationDispositionKind,
    EquityRecommendationEligibilityAudit,
    EquityRecommendationGenerationAttempt,
    EquityRecommendationGenerationDisposition,
    EquityRecommendationGenerationVerifier,
    EquityRecommendationRecordingOutcome,
    EquityRecommendationReplayMismatchError,
    EquityRecommendationSourceUnavailableError,
    EquityRecommendationSourceVerificationError,
    IncompatibleEquityRecommendationRecordError,
    InvalidEquityRecommendationRecordError,
    RecordEquityRecommendationGenerationUseCase,
    VerifyStoredEquityRecommendationGenerationUseCase,
    canonical_equity_recommendation_json,
    decode_equity_core_outcome,
    decode_equity_policy_set,
    decode_equity_recommendation_assessment,
    decode_equity_recommendation_inputs,
    decode_equity_research_configuration,
    decode_equity_trade_recommendation_proposal,
    decode_equity_trade_recommendation_request,
    equity_recommendation_attempt,
    equity_recommendation_disposition,
    equity_recommendation_fingerprint,
)
from northstar_application.application_services import (
    equity_recommendation_generation_decoding as decoding_module,
)
from northstar_application.application_services import (
    equity_recommendation_generation_records as records_module,
)
from northstar_application.application_services import (
    record_equity_recommendation_generation as record_module,
)
from northstar_application.application_services import (
    verify_equity_recommendation_generation as verify_module,
)
from northstar_application.ports import (
    EquityRecommendationGenerationCommitUnconfirmedError,
    EquityRecommendationGenerationNotCommittedError,
    EquityRecommendationGenerationStore,
    EquityRecommendationGenerationStoreUnavailableError,
    EquityRecommendationStoreOutcome,
    EquityRecommendationStoreWrite,
    EquitySessionRecommendationHistorySource,
)
from northstar_application.ports import (
    equity_recommendation_generation_store as store_port_module,
)

_O = EquityRecommendationRecordingOutcome
_K = EquityRecommendationDispositionKind
_S = ei4c._S
_R = ei4c._R
_v, _at, _ist = ei4c._v, ei4c._at, ei4c._ist

Disposition = EquityRecommendationGenerationDisposition
Attempt = EquityRecommendationGenerationAttempt


# ----------------------------------------------------------------------
# The test-only store model
# ----------------------------------------------------------------------


class _Store(EquityRecommendationGenerationStore):
    """Test-only model of committed generation storage.

    ``script`` holds one injected behaviour per write, consumed in order;
    an empty script means the write commits normally:

    - ``unavailable``: the store cannot be opened; nothing is written;
    - ``rollback``: a definite rollback before COMMIT; nothing is written;
    - ``lost_before``: the COMMIT is lost and nothing became durable;
    - ``lost_after``: the write committed, then its acknowledgment was lost;
    - ``undecodable``: the stored row could not be decoded; nothing is written;
    - ``incompatible``: the stored row names an unsupported evaluator or format.

    ``before_write`` and ``before_load`` hooks run once each -- a competing
    writer committing first. ``load_failure`` makes every load fail.
    """

    def __init__(self) -> None:
        self.dispositions: dict[str, Disposition] = {}
        self.attempts: dict[str, Attempt] = {}
        self.script: list[str] = []
        self.before_write: list[Callable[[], object]] = []
        self.before_load: list[Callable[[], object]] = []
        self.load_failure: str | None = None
        self.writes = 0

    def _fail_load(self) -> None:
        if self.load_failure == "unavailable":
            raise EquityRecommendationGenerationStoreUnavailableError("test: store unreadable")
        if self.load_failure == "undecodable":
            raise InvalidEquityRecommendationRecordError("test: a stored row does not decode")
        if self.load_failure == "incompatible":
            raise IncompatibleEquityRecommendationRecordError("test: a future evaluator wrote it")

    def load_disposition(self, generation_key: str) -> Disposition | None:
        while self.before_load:
            self.before_load.pop(0)()
        self._fail_load()
        return self.dispositions.get(generation_key)

    def load_attempt(self, attempt_identity: str) -> Attempt | None:
        self._fail_load()
        return self.attempts.get(attempt_identity)

    def _behaviour(self) -> str:
        self.writes += 1
        while self.before_write:
            self.before_write.pop(0)()
        mode = self.script.pop(0) if self.script else "ok"
        if mode == "unavailable":
            raise EquityRecommendationGenerationStoreUnavailableError("test: store unavailable")
        if mode == "rollback":
            raise EquityRecommendationGenerationNotCommittedError("test: refused before COMMIT")
        if mode == "lost_before":
            raise EquityRecommendationGenerationCommitUnconfirmedError("test: COMMIT lost")
        if mode == "undecodable":
            raise InvalidEquityRecommendationRecordError("test: the stored row does not decode")
        if mode == "incompatible":
            raise IncompatibleEquityRecommendationRecordError("test: a stored row's evaluator")
        return mode

    def _insert_attempt(self, attempt: Attempt) -> bool:
        if attempt.attempt_identity in self.attempts:
            return False
        self.attempts[attempt.attempt_identity] = attempt
        return True

    def commit_generation(
        self, disposition: Disposition, attempt: Attempt
    ) -> EquityRecommendationStoreWrite:
        """The committed effect of one write, with no injected failure."""
        stored = self.dispositions.get(disposition.generation_key)
        inserted = self._insert_attempt(attempt)
        if stored is None:
            self.dispositions[disposition.generation_key] = disposition
            return EquityRecommendationStoreWrite(
                EquityRecommendationStoreOutcome.COMMITTED, None, inserted
            )
        # The port contract: equivalence is equality of the whole record --
        # fingerprints and evaluator version -- never of fingerprints alone.
        same = stored == disposition
        outcome = (
            EquityRecommendationStoreOutcome.EQUIVALENT
            if same
            else EquityRecommendationStoreOutcome.CONFLICT
        )
        return EquityRecommendationStoreWrite(outcome, stored, inserted)

    def record_generation(
        self, disposition: Disposition, attempt: Attempt
    ) -> EquityRecommendationStoreWrite:
        mode = self._behaviour()
        write = self.commit_generation(disposition, attempt)
        if mode == "lost_after":
            raise EquityRecommendationGenerationCommitUnconfirmedError("test: ack lost")
        return write

    def record_attempt(self, attempt: Attempt) -> EquityRecommendationStoreWrite:
        mode = self._behaviour()
        inserted = self._insert_attempt(attempt)
        if mode == "lost_after":
            raise EquityRecommendationGenerationCommitUnconfirmedError("test: ack lost")
        outcome = (
            EquityRecommendationStoreOutcome.COMMITTED
            if inserted
            else EquityRecommendationStoreOutcome.EQUIVALENT
        )
        return EquityRecommendationStoreWrite(outcome, None, inserted)


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


def _verifier(ports_: ei4c._Ports) -> EquityRecommendationGenerationVerifier:
    return EquityRecommendationGenerationVerifier(ports_.world.timelines, ports_.world.snapshots)


def _recorder(
    ports_: ei4c._Ports, store: _Store, max_writes: int = 3
) -> RecordEquityRecommendationGenerationUseCase:
    return RecordEquityRecommendationGenerationUseCase(store, _verifier(ports_), max_writes)


def _unresolved(ports_: ei4c._Ports) -> EquityRecommendationEligibilityAudit:
    ports_.state.answer = ei4c._history(ei4c._prior(ei4c._LISTING))
    audit = ports_.run()
    ports_.state.answer = ei4c._history()
    assert audit.status is _S.EVALUATION_UNRESOLVED
    return audit


def _not_eligible(ports_: ei4c._Ports) -> EquityRecommendationEligibilityAudit:
    ports_.references.hold("generation", ei4c._generation("generation/strict", minimum="100"))
    audit = ports_.run(generation_policy_version=_v("generation/strict"))
    assert audit.status is _S.NOT_ELIGIBLE and audit.generation_key is not None
    return audit


def _rebuilt(disposition: Disposition, **changes: Any) -> Disposition:
    """Rebuild a disposition from its stored fields, as an adapter would."""
    stored = {
        "generation_key": disposition.generation_key,
        "kind": disposition.kind,
        "evaluator_version": disposition.evaluator_version,
        "assessment_fingerprint": disposition.assessment_fingerprint,
        "assessment_document": disposition.assessment_document,
        "proposal_fingerprint": disposition.proposal_fingerprint,
        "proposal_document": disposition.proposal_document,
    }
    stored.update(changes)
    return Disposition(**stored)


def _tampered(disposition: Disposition, change: Callable[[dict[str, Any]], None]) -> Disposition:
    """Test-only: a disposition whose assessment was altered, then re-fingerprinted."""
    document = json.loads(disposition.assessment_document)
    change(document)
    if document["inputs"] is not None:
        document["inputs_fingerprint"] = equity_recommendation_fingerprint(document["inputs"])
    return _rebuilt(
        disposition,
        assessment_document=canonical_equity_recommendation_json(document),
        assessment_fingerprint=equity_recommendation_fingerprint(document),
    )


def _manifest_relinked(disposition: Disposition) -> Disposition:
    def change(document: dict[str, Any]) -> None:
        fake = "ecm1-sha256:" + "f" * 64
        document["candidate"]["manifest_fingerprint"] = fake
        document["inputs"]["candidate"]["manifest_fingerprint"] = fake

    return _tampered(disposition, change)


# ----------------------------------------------------------------------
# A-I, AJ -- decoding
# ----------------------------------------------------------------------


def test_a_a_stored_disposition_decodes_into_its_verified_parts() -> None:
    _, audit = ei4c._proposed()
    disposition = equity_recommendation_disposition(audit)
    assert disposition is not None
    rebuilt = _rebuilt(disposition)
    assert rebuilt == disposition
    assert rebuilt.kind is _K.PROPOSED
    assert rebuilt.evaluator_version == EQUITY_RECOMMENDATION_EVALUATOR_VERSION
    assert audit.inputs is not None and audit.proposal is not None
    assert rebuilt.request == audit.request
    assert rebuilt.inputs == audit.inputs
    assert rebuilt.candidate == audit.inputs.candidate
    assert rebuilt.policy_set_fingerprint == audit.inputs.policy_set.fingerprint
    assert rebuilt.inputs_fingerprint == audit.inputs.fingerprint
    assert rebuilt.inputs_document == canonical_equity_recommendation_json(audit.inputs.document())
    assert rebuilt.reasons == ()
    assert rebuilt.proposal == audit.proposal
    assert not rebuilt.is_issued


def test_b_a_stored_attempt_decodes() -> None:
    _, audit = ei4c._proposed()
    attempt = equity_recommendation_attempt(audit)
    rebuilt = Attempt(attempt.attempt_identity, attempt.attempt_document)
    assert rebuilt == attempt
    assert (rebuilt.status, rebuilt.generation_key, rebuilt.disposition_kind) == (
        _S.PROPOSED,
        audit.generation_key,
        _K.PROPOSED,
    )
    assert rebuilt.assessment_fingerprint == audit.assessment_fingerprint
    outage = ei4c._ports()
    outage.references.unavailable = {"costs"}
    refused = equity_recommendation_attempt(outage.run())
    assert refused.reasons == (_R.COST_SCHEDULE_UNAVAILABLE,)
    assert refused.generation_key is None and refused.assessment_fingerprint is None


def test_c_every_document_round_trips_canonically() -> None:
    _, audit = ei4c._proposed()
    document = json.loads(canonical_equity_recommendation_json(audit.assessment_document()))
    assert decode_equity_trade_recommendation_request(document["request"]) == audit.request
    assert (
        decode_equity_research_configuration(document["request"]["research_configuration"])
        == ei4c._CONFIG
    )
    inputs = decode_equity_recommendation_inputs(document["inputs"])
    assert inputs == audit.inputs and inputs.document() == document["inputs"]
    assert decode_equity_policy_set(document["inputs"]["policy_set"]) == inputs.policy_set
    assert decode_equity_core_outcome(document["core_outcome"]) == audit.core_outcome
    proposal = decode_equity_trade_recommendation_proposal(document["proposal"])
    assert proposal == audit.proposal and proposal.fingerprint == document["proposal_fingerprint"]
    decoded = decode_equity_recommendation_assessment(
        canonical_equity_recommendation_json(document), audit.assessment_fingerprint
    )
    assert (decoded.status, decoded.generation_key, decoded.inputs) == (
        _S.PROPOSED,
        audit.generation_key,
        audit.inputs,
    )


def test_d_aj_unsupported_formats_and_evaluators_are_incompatible_not_migrated() -> None:
    _, audit = ei4c._proposed()
    disposition = equity_recommendation_disposition(audit)
    assert disposition is not None
    future = json.loads(disposition.assessment_document)
    future["format"] = "northstar.equity-recommendation-assessment/2"
    text = canonical_equity_recommendation_json(future)
    with pytest.raises(IncompatibleEquityRecommendationRecordError, match="Nothing is migrated"):
        decode_equity_recommendation_assessment(text, disposition.assessment_fingerprint)
    stranger = dict(future, format="northstar.something-else/1")
    with pytest.raises(InvalidEquityRecommendationRecordError) as raised:
        decode_equity_recommendation_assessment(
            canonical_equity_recommendation_json(stranger), disposition.assessment_fingerprint
        )
    assert not isinstance(raised.value, IncompatibleEquityRecommendationRecordError)
    nested = json.loads(disposition.assessment_document)["inputs"]
    nested["format"] = "northstar.equity-recommendation-generation-inputs/9"
    with pytest.raises(IncompatibleEquityRecommendationRecordError):
        decode_equity_recommendation_inputs(nested)
    with pytest.raises(IncompatibleEquityRecommendationRecordError, match="Evaluator"):
        _rebuilt(disposition, evaluator_version="northstar.equity-recommendation-evaluator/2")
    with pytest.raises(InvalidEquityRecommendationRecordError):
        _rebuilt(disposition, evaluator_version="")


def test_e_a_fingerprint_that_does_not_re_derive_is_refused() -> None:
    _, audit = ei4c._proposed()
    disposition = equity_recommendation_disposition(audit)
    assert disposition is not None
    with pytest.raises(InvalidEquityRecommendationRecordError, match="fingerprint"):
        _rebuilt(disposition, assessment_fingerprint="eira1-sha256:" + "0" * 64)
    with pytest.raises(InvalidEquityRecommendationRecordError, match="another proposal"):
        _rebuilt(disposition, proposal_fingerprint="eirp1-sha256:" + "0" * 64)
    with pytest.raises(InvalidEquityRecommendationRecordError):
        _rebuilt(
            disposition,
            assessment_document=disposition.assessment_document.replace("102.5", "102.6", 1),
        )
    attempt = equity_recommendation_attempt(audit)
    with pytest.raises(InvalidEquityRecommendationRecordError, match="fingerprint"):
        Attempt("eirat1-sha256:" + "0" * 64, attempt.attempt_document)


def test_f_a_policy_set_component_must_be_the_version_its_reference_names() -> None:
    _, audit = ei4c._proposed()
    document = json.loads(canonical_equity_recommendation_json(audit.assessment_document()))
    policy_set = document["inputs"]["policy_set"]
    policy_set["risk_policy"]["version"] = _v("risk/other").value
    with pytest.raises(InvalidEquityRecommendationRecordError, match="version the policy set"):
        decode_equity_policy_set(policy_set)


def test_g_strict_decoding_refuses_every_malformation() -> None:
    _, audit = ei4c._proposed()
    base = json.loads(canonical_equity_recommendation_json(audit.assessment_document()))

    def refused(change: Callable[[dict[str, Any]], None], decode: Callable[[Any], Any]) -> None:
        document = json.loads(json.dumps(base))
        change(document)
        with pytest.raises(InvalidEquityRecommendationRecordError):
            decode(document)

    def inputs(document: dict[str, Any]) -> Any:
        return decode_equity_recommendation_inputs(document["inputs"])

    # Non-canonical decimals, floats, missing and extra fields, invalid instants.
    refused(
        lambda d: d["inputs"]["policy_set"]["risk_policy"].update(risk_fraction="0.010"), inputs
    )
    refused(lambda d: d["inputs"]["candidate"].update(signal_close=102.5), inputs)
    refused(lambda d: d["inputs"]["venue_day"].pop("reference"), inputs)
    refused(lambda d: d["inputs"].update(extra=True), inputs)
    refused(
        lambda d: d["inputs"]["issued_history"].update(as_of="2026-08-12T09:36:00+05:30"), inputs
    )
    refused(lambda d: d["inputs"]["issued_history"].update(as_of="not an instant"), inputs)
    refused(lambda d: d["inputs"]["candidate"].update(candidate_sequence="2"), inputs)
    # A frozen input inconsistent with the rest of the envelope.
    refused(lambda d: d["inputs"].update(modeled_issued_at=_at(9, 37).value), inputs)
    refused(lambda d: d["inputs"].update(policy_set_fingerprint="eirs1-sha256:" + "0" * 64), inputs)
    refused(
        lambda d: d["inputs"]["listing_eligibility"]["listing"].update(symbol="RELIANCE"), inputs
    )


def test_g_contradictory_source_identities_are_refused_at_assessment_level() -> None:
    # Core's outcome alone is self-consistent; only the assessment can see that it
    # names another candidate than the frozen inputs (a NOT_ELIGIBLE Core rejection).
    audit = _not_eligible(ei4c._ports())
    document = json.loads(canonical_equity_recommendation_json(audit.assessment_document()))
    document["core_outcome"]["outcome"]["source"]["decision_as_of"] = _at(9, 37).value
    assert decode_equity_core_outcome(document["core_outcome"]).source.decision_as_of == _at(9, 37)
    with pytest.raises(InvalidEquityRecommendationRecordError, match="another candidate"):
        decode_equity_recommendation_assessment(
            canonical_equity_recommendation_json(document),
            equity_recommendation_fingerprint(document),
        )


def test_g_malformed_history_ordering_is_refused() -> None:
    ports_ = ei4c._ports()
    ports_.state.answer = ei4c._history(
        ei4c._issued("a", ei4c._OTHER, _at(9, 20), sequence=1),
        ei4c._issued("b", ei4c._OTHER, _at(9, 30), sequence=2),
    )
    audit = ports_.run()
    document = json.loads(canonical_equity_recommendation_json(audit.assessment_document()))
    history = document["inputs"]["issued_history"]
    assert [r["recommendation_key"] for r in history["recommendations"]] == ["a", "b"]
    assert decode_equity_recommendation_inputs(document["inputs"]) == audit.inputs
    history["recommendations"].reverse()
    with pytest.raises(InvalidEquityRecommendationRecordError, match="order"):
        decode_equity_recommendation_inputs(document["inputs"])


def test_g_a_core_rejection_decodes_through_core() -> None:
    audit = _not_eligible(ei4c._ports())
    document = json.loads(canonical_equity_recommendation_json(audit.assessment_document()))
    rejection = decode_equity_core_outcome(document["core_outcome"])
    assert rejection == audit.core_outcome
    assert document["core_outcome"]["kind"] == "REJECTION"


def test_h_a_proposed_disposition_must_carry_its_proposal() -> None:
    _, audit = ei4c._proposed()
    disposition = equity_recommendation_disposition(audit)
    assert disposition is not None
    with pytest.raises(InvalidEquityRecommendationRecordError, match="PROPOSED"):
        _rebuilt(disposition, proposal_document=None, proposal_fingerprint=None)


def test_i_no_other_disposition_may_carry_a_proposal() -> None:
    _, proposed = ei4c._proposed()
    with_proposal = equity_recommendation_disposition(proposed)
    unresolved = equity_recommendation_disposition(_unresolved(ei4c._ports()))
    assert with_proposal is not None and unresolved is not None
    assert unresolved.kind is _K.UNRESOLVED and unresolved.proposal is None
    with pytest.raises(InvalidEquityRecommendationRecordError):
        _rebuilt(
            unresolved,
            proposal_document=with_proposal.proposal_document,
            proposal_fingerprint=with_proposal.proposal_fingerprint,
        )
    with pytest.raises(InvalidEquityRecommendationRecordError, match="kind"):
        _rebuilt(unresolved, kind=_K.PROPOSED)

    def smuggle(document: dict[str, Any]) -> None:
        proposal_assessment = json.loads(with_proposal.assessment_document)
        document["proposal"] = proposal_assessment["proposal"]
        document["proposal_fingerprint"] = proposal_assessment["proposal_fingerprint"]

    with pytest.raises(InvalidEquityRecommendationRecordError, match="PROPOSED"):
        _tampered(unresolved, smuggle)


# ----------------------------------------------------------------------
# J-N -- source verification through EI-3C and EI-2B
# ----------------------------------------------------------------------


def test_j_the_original_candidate_is_re_established_and_matches() -> None:
    ports_, audit = ei4c._proposed()
    disposition = equity_recommendation_disposition(audit)
    assert disposition is not None
    verified = _verifier(ports_).verify(disposition)
    assert verified.disposition == disposition
    assert verified.replayed.assessment_fingerprint == disposition.assessment_fingerprint
    assert verified.observed_timeline_head == ports_.world.timelines.only().head_fingerprint
    assert (
        verified.document_integrity_verified,
        verified.source_integrity_verified,
        verified.evaluation_reproduced,
        verified.issuance_established,
    ) == (True, True, True, False)


def test_k_a_missing_timeline_fails_source_verification() -> None:
    ports_, audit = ei4c._proposed()
    disposition = equity_recommendation_disposition(audit)
    assert disposition is not None
    ports_.world.timelines.records.clear()  # test-only: the history is gone
    with pytest.raises(EquityRecommendationSourceVerificationError) as raised:
        _verifier(ports_).verify(disposition)
    assert raised.value.reason is _R.NO_AUTHORITATIVE_CANDIDATE
    ports_.world.timelines.unavailable = True
    with pytest.raises(EquityRecommendationSourceUnavailableError):
        _verifier(ports_).verify(disposition)


def test_l_a_record_bound_to_another_candidate_is_refused() -> None:
    _, audit = ei4c._proposed()
    disposition = equity_recommendation_disposition(audit)
    assert disposition is not None
    # Another authoritative history for the same key: its CANDIDATE was decided at 09:35:30.
    other = ei4c._ports(_ist(9, 31), _ist(9, 35, 30))
    with pytest.raises(EquityRecommendationSourceVerificationError, match="source"):
        _verifier(other).verify(disposition)


def test_m_a_missing_or_unavailable_snapshot_fails_closed() -> None:
    ports_, audit = ei4c._proposed()
    disposition = equity_recommendation_disposition(audit)
    assert disposition is not None
    snapshots = ports_.world.snapshots
    snapshot_id = disposition.candidate.source.evidence_snapshot_id
    saved = snapshots.documents.pop(snapshot_id)  # test-only
    with pytest.raises(EquityRecommendationSourceVerificationError) as raised:
        _verifier(ports_).verify(disposition)
    assert raised.value.reason is _R.RESEARCH_TIMELINE_INVALID
    snapshots.documents[snapshot_id] = saved
    snapshots.unavailable = True
    with pytest.raises(EquityRecommendationSourceUnavailableError):
        _verifier(ports_).verify(disposition)


def test_n_a_stored_manifest_that_is_not_the_snapshots_is_refused() -> None:
    ports_, audit = ei4c._proposed()
    disposition = equity_recommendation_disposition(audit)
    assert disposition is not None
    relinked = _manifest_relinked(disposition)
    assert relinked.generation_key == disposition.generation_key
    with pytest.raises(EquityRecommendationSourceVerificationError, match="manifest_fingerprint"):
        _verifier(ports_).verify(relinked)


# ----------------------------------------------------------------------
# O-S -- historical replay from frozen inputs
# ----------------------------------------------------------------------


def _recorded_proposal() -> tuple[ei4c._Ports, Disposition]:
    ports_, audit = ei4c._proposed()
    disposition = equity_recommendation_disposition(audit)
    assert disposition is not None
    return ports_, disposition


def test_o_replay_uses_the_frozen_policy_documents() -> None:
    ports_, disposition = _recorded_proposal()
    # The mutable source now answers the same version with other content.
    ports_.references.answers["risk"] = dataclasses.replace(
        ei4c._risk(), model_capital=ei4c.Money(ei4c.Decimal("50000"), ei4c._INR)
    )
    assert ports_.run().assessment_fingerprint != disposition.assessment_fingerprint
    assert _verifier(ports_).verify(disposition).disposition == disposition


def test_p_replay_uses_the_frozen_venue_day_evidence() -> None:
    ports_, disposition = _recorded_proposal()
    ports_.world.venue.closes = time(15, 15)
    assert ports_.run().assessment_fingerprint != disposition.assessment_fingerprint
    _verifier(ports_).verify(disposition)


def test_q_replay_uses_the_frozen_as_of_issued_history() -> None:
    ports_, disposition = _recorded_proposal()
    ports_.state.answer = ei4c._history(ei4c._prior(ei4c._LISTING))
    assert ports_.run().status is _S.EVALUATION_UNRESOLVED
    verified = _verifier(ports_).verify(disposition)
    history = verified.disposition.inputs.issued_history
    assert (history.recording_watermark, history.recommendations) == (0, ())


def test_r_replay_needs_no_current_reference_source() -> None:
    ports_, disposition = _recorded_proposal()
    ports_.references.unavailable = {"timing", "generation", "risk", "costs", "increment"}
    ports_.eligibility.unavailable = True
    ports_.state.unavailable = True
    ports_.world.venue.classification = ei4c.EquityVenueDayClassification.NOT_STATED
    assert ports_.run().status is _S.DEPENDENCY_UNAVAILABLE
    verified = _verifier(ports_).verify(disposition)
    assert verified.evaluation_reproduced


def test_r_a_replay_that_does_not_reproduce_fails_closed() -> None:
    ports_ = ei4c._ports()
    disposition = equity_recommendation_disposition(_not_eligible(ports_))
    assert disposition is not None and disposition.kind is _K.NOT_ELIGIBLE

    def misstate(document: dict[str, Any]) -> None:
        document["core_outcome"]["outcome"]["reasons"] = ["INSUFFICIENT_RISK_BUDGET"]

    altered = _tampered(disposition, misstate)  # structurally sound, historically false
    with pytest.raises(EquityRecommendationReplayMismatchError):
        _verifier(ports_).verify(altered)


def test_s_a_later_correction_never_changes_the_stored_generation() -> None:
    ports_, disposition = _recorded_proposal()
    store = _Store()
    assert (
        _recorder(ports_, store)
        .record(disposition, equity_recommendation_attempt(ports_.run()))
        .outcome
        is _O.COMMITTED
    )
    ports_.world.add({4: ("101", "103", "101", "101.8", "2000")}, revision="r2")
    ports_.world.commit(_ist(9, 46), _ist(9, 51))
    entries = ports_.world.timelines.only().entries
    assert entries[1].role is ei4c._ROLE.CANDIDATE and entries[-1].role is ei4c._ROLE.REJECTED
    verified = VerifyStoredEquityRecommendationGenerationUseCase(store, _verifier(ports_)).execute(
        disposition.generation_key
    )
    assert verified is not None and verified.disposition == disposition
    assert verified.disposition.candidate.source.evidence_snapshot_id == entries[1].snapshot_id
    assert ports_.world.snapshots.get_snapshot(entries[1].snapshot_id) is not None
    # Generating again finds the same candidate: an equivalent record, never a new key.
    again = _recorder(ports_, store).execute(ports_.run())
    assert again.outcome is _O.EQUIVALENT_COMMITTED
    assert list(store.dispositions) == [disposition.generation_key]


# ----------------------------------------------------------------------
# T-AI -- recording, idempotency, conflicts and reconciliation
# ----------------------------------------------------------------------


def test_t_a_verified_disposition_is_recorded_atomically() -> None:
    ports_, audit = ei4c._proposed()
    store = _Store()
    result = _recorder(ports_, store).execute(audit)
    assert result.outcome is _O.COMMITTED and result.is_stored
    assert result.authoritative == result.disposition
    assert result.attempt_recorded is True and not result.reconciled and result.writes == 1
    assert store.dispositions == {audit.generation_key: result.disposition}
    assert list(store.attempts) == [result.attempt.attempt_identity]


def test_u_an_equivalent_duplicate_is_idempotent() -> None:
    ports_, audit = ei4c._proposed()
    store = _Store()
    first = _recorder(ports_, store).execute(audit)
    second = _recorder(ports_, store).execute(ports_.run())
    assert (first.outcome, second.outcome) == (_O.COMMITTED, _O.EQUIVALENT_COMMITTED)
    assert second.authoritative == first.disposition
    assert len(store.dispositions) == 1 and len(store.attempts) == 1


def test_v_same_key_different_content_is_a_conflict_never_an_overwrite() -> None:
    ports_, audit = ei4c._proposed()
    store = _Store()
    first = _recorder(ports_, store).execute(audit)
    other = ports_.run(listing_eligibility_version=_v("eligibility/2"))
    assert other.generation_key == audit.generation_key
    assert other.assessment_fingerprint != audit.assessment_fingerprint
    conflict = _recorder(ports_, store).execute(other)
    assert conflict.outcome is _O.CONTENT_CONFLICT and not conflict.is_stored
    assert conflict.authoritative == first.disposition
    assert conflict.attempt_recorded is True
    assert store.dispositions[audit.generation_key] == first.disposition
    assert len(store.attempts) == 2  # the conflicting attempt is audited too


def test_w_an_unresolved_disposition_never_becomes_proposed() -> None:
    ports_ = ei4c._ports()
    store = _Store()
    unresolved = _recorder(ports_, store).execute(_unresolved(ports_))
    assert unresolved.outcome is _O.COMMITTED
    assert unresolved.disposition is not None and unresolved.disposition.kind is _K.UNRESOLVED
    proposed = ports_.run()  # the same key, now with an empty as-of history
    assert (
        proposed.status is _S.PROPOSED
        and proposed.generation_key == unresolved.disposition.generation_key
    )
    result = _recorder(ports_, store).execute(proposed)
    assert result.outcome is _O.CONTENT_CONFLICT
    stored = store.dispositions[proposed.generation_key]
    assert stored.kind is _K.UNRESOLVED and stored.proposal is None


def test_x_y_attempt_only_results_are_recorded_and_deduplicated() -> None:
    store = _Store()
    never = ei4c._Ports(ei4c._World())
    audit = never.run()
    assert audit.status is _S.NO_CANDIDATE
    recorder = _recorder(never, store)
    first, second = recorder.execute(audit), recorder.execute(never.run())
    assert (first.outcome, second.outcome) == (_O.COMMITTED, _O.EQUIVALENT_COMMITTED)
    assert first.disposition is None and first.authoritative is None
    assert store.dispositions == {} and len(store.attempts) == 1
    # Two outages that differ only in their messages are one logical attempt.
    a, b = ei4c._ports(), ei4c._ports()
    a.references.unavailable = {"risk"}
    b.references.unavailable = {"risk"}
    b_audit = b.run()
    reworded = dataclasses.replace(
        b_audit,
        checks=tuple(dataclasses.replace(c, detail="another message") for c in b_audit.checks),
    )
    assert recorder.execute(a.run()).outcome is _O.COMMITTED
    assert recorder.execute(reworded).outcome is _O.EQUIVALENT_COMMITTED
    assert len(store.attempts) == 2


@pytest.mark.parametrize(
    ("mode", "outcome"),
    [("unavailable", _O.PERSISTENCE_UNAVAILABLE), ("rollback", _O.NOT_COMMITTED)],
)
def test_z_aa_failures_before_commit_are_reported_never_audited(mode, outcome) -> None:  # type: ignore[no-untyped-def]
    ports_, audit = ei4c._proposed()
    store = _Store()
    store.script = [mode]
    result = _recorder(ports_, store).execute(audit)
    assert result.outcome is outcome and not result.is_stored
    assert result.attempt_recorded is False and result.authoritative is None
    assert store.dispositions == {} and store.attempts == {}
    never = ei4c._Ports(ei4c._World())
    store.script = [mode]
    attempt_only = _recorder(never, store).execute(never.run())
    assert attempt_only.outcome is outcome and attempt_only.attempt_recorded is False
    assert store.attempts == {}


def test_ab_an_unconfirmed_commit_that_happened_reconciles_as_committed() -> None:
    ports_, audit = ei4c._proposed()
    store = _Store()
    store.script = ["lost_after"]
    result = _recorder(ports_, store).execute(audit)
    assert result.outcome is _O.EQUIVALENT_COMMITTED and result.reconciled
    assert result.attempt_recorded is True and result.writes == 1
    assert store.dispositions[audit.generation_key] == result.disposition


def test_ac_an_unconfirmed_commit_that_lost_to_other_content_is_a_conflict() -> None:
    ports_, audit = ei4c._proposed()
    store = _Store()
    competitor = equity_recommendation_disposition(_unresolved(ports_))
    competitor_attempt = equity_recommendation_attempt(_unresolved(ports_))
    assert competitor is not None
    store.before_write.append(lambda: store.commit_generation(competitor, competitor_attempt))
    store.script = ["lost_before"]
    result = _recorder(ports_, store).execute(audit)
    assert result.outcome is _O.CONTENT_CONFLICT and result.reconciled
    assert result.authoritative == competitor
    assert result.attempt_recorded is False  # our write never committed
    assert store.dispositions[audit.generation_key] == competitor


def test_ad_an_unconfirmed_commit_that_did_not_happen_is_retried_with_the_same_records() -> None:
    ports_, audit = ei4c._proposed()
    store = _Store()
    store.script = ["lost_before"]
    result = _recorder(ports_, store).execute(audit)
    assert result.outcome is _O.COMMITTED and result.writes == 2
    assert store.dispositions[audit.generation_key] == result.disposition
    assert store.writes == 2


@pytest.mark.parametrize("failure", ["unavailable", "undecodable", "incompatible"])
def test_ae_an_unconfirmed_commit_with_a_failed_reload_is_unresolved(failure) -> None:  # type: ignore[no-untyped-def]
    ports_, audit = ei4c._proposed()
    store = _Store()
    store.script = ["lost_after"]
    store.load_failure = failure
    result = _recorder(ports_, store).execute(audit)
    assert result.outcome is _O.COMMIT_RECONCILIATION_UNRESOLVED
    assert result.attempt_recorded is None and result.authoritative is None
    assert not result.is_stored


def test_af_attempt_only_commit_uncertainty_reconciles_by_attempt_identity() -> None:
    never = ei4c._Ports(ei4c._World())
    audit = never.run()
    store = _Store()
    store.script = ["lost_after"]
    happened = _recorder(never, store).execute(audit)
    assert happened.outcome is _O.EQUIVALENT_COMMITTED and happened.reconciled
    store = _Store()
    store.script = ["lost_before"]
    retried = _recorder(never, store).execute(audit)
    assert retried.outcome is _O.COMMITTED and retried.writes == 2
    store = _Store()
    store.script = ["lost_after"]
    store.load_failure = "unavailable"
    unknown = _recorder(never, store).execute(audit)
    assert unknown.outcome is _O.COMMIT_RECONCILIATION_UNRESOLVED
    assert unknown.attempt_recorded is None


def test_ag_retries_are_bounded() -> None:
    ports_, audit = ei4c._proposed()
    store = _Store()
    store.script = ["lost_before"] * 5
    result = _recorder(ports_, store).execute(audit)
    assert result.outcome is _O.RETRY_LIMIT_REACHED and result.writes == 3
    assert store.writes == 3 and store.dispositions == {}
    single = _Store()
    single.script = ["lost_before"]
    assert _recorder(ports_, single, max_writes=1).execute(audit).outcome is _O.RETRY_LIMIT_REACHED
    with pytest.raises(ValueError):
        _recorder(ports_, single, max_writes=0)


def test_ah_a_competing_equivalent_writer_during_reconciliation() -> None:
    ports_, audit = ei4c._proposed()
    store = _Store()
    disposition = equity_recommendation_disposition(audit)
    attempt = equity_recommendation_attempt(audit)
    assert disposition is not None
    store.script = ["lost_before"]
    # Another process commits the same content while this one reconciles.
    store.before_load.append(lambda: store.commit_generation(disposition, attempt))
    result = _recorder(ports_, store).execute(audit)
    assert result.outcome is _O.EQUIVALENT_COMMITTED and result.reconciled
    assert result.attempt_recorded is True  # the identical logical attempt
    assert store.writes == 1


def test_ai_a_corrupt_or_unverifiable_stored_record_is_never_authoritative() -> None:
    ports_, audit = ei4c._proposed()
    disposition = equity_recommendation_disposition(audit)
    assert disposition is not None
    store = _Store()
    store.script = ["undecodable"]
    assert _recorder(ports_, store).execute(audit).outcome is _O.STORED_RECORD_INVALID
    # A stored record that decodes but is not the snapshot's evidence.
    store = _Store()
    store.dispositions[disposition.generation_key] = _manifest_relinked(disposition)
    result = _recorder(ports_, store).execute(audit)
    assert result.outcome is _O.STORED_RECORD_INVALID and result.authoritative is None
    assert "manifest_fingerprint" in result.detail
    with pytest.raises(EquityRecommendationSourceVerificationError):
        VerifyStoredEquityRecommendationGenerationUseCase(store, _verifier(ports_)).execute(
            disposition.generation_key
        )


def test_ai_a_disposition_whose_source_no_longer_verifies_is_never_written() -> None:
    ports_, audit = ei4c._proposed()
    store = _Store()
    ports_.world.timelines.records.clear()  # test-only
    result = _recorder(ports_, store).execute(audit)
    assert result.outcome is _O.SOURCE_EVIDENCE_INVALID and store.writes == 0
    ports_ = ei4c._ports()
    audit = ports_.run()
    ports_.world.snapshots.unavailable = True
    assert _recorder(ports_, store).execute(audit).outcome is _O.SOURCE_EVIDENCE_UNAVAILABLE
    assert store.writes == 0


# ----------------------------------------------------------------------
# Concurrency, modelled
# ----------------------------------------------------------------------


def test_concurrency_writers_resolve_through_the_authoritative_store() -> None:
    ports_, audit = ei4c._proposed()
    store = _Store()
    a, b = _recorder(ports_, store), _recorder(ports_, store)
    assert (a.execute(audit).outcome, b.execute(audit).outcome) == (
        _O.COMMITTED,
        _O.EQUIVALENT_COMMITTED,
    )
    different = ports_.run(listing_eligibility_version=_v("eligibility/2"))
    assert b.execute(different).outcome is _O.CONTENT_CONFLICT
    # A distinct approved policy set is a distinct identity: both are recorded.
    ports_.references.hold("risk", ei4c._risk("risk/2", capital="90000"))
    distinct = ports_.run(risk_policy_version=_v("risk/2"))
    assert distinct.generation_key != audit.generation_key
    assert a.execute(distinct).outcome is _O.COMMITTED
    assert len(store.dispositions) == 2


def test_concurrency_a_retry_after_an_interrupted_process_finds_the_commit() -> None:
    ports_, audit = ei4c._proposed()
    store = _Store()
    store.script = ["lost_after"]
    store.load_failure = "unavailable"
    interrupted = _recorder(ports_, store).execute(audit)
    assert interrupted.outcome is _O.COMMIT_RECONCILIATION_UNRESOLVED
    store.load_failure = None
    restarted = _recorder(ports_, store).execute(ports_.run())
    assert restarted.outcome is _O.EQUIVALENT_COMMITTED
    assert len(store.dispositions) == 1


# ----------------------------------------------------------------------
# AK-AO -- boundaries
# ----------------------------------------------------------------------


def test_ak_recording_never_writes_or_fabricates_issued_history() -> None:
    ports_, audit = ei4c._proposed()
    before = ports_.state.answer
    store = _Store()
    assert _recorder(ports_, store).execute(audit).outcome is _O.COMMITTED
    assert ports_.state.answer == before
    # The recorded proposal does not block its own retry through the history port.
    again = ports_.run()
    assert (
        again.status is _S.PROPOSED and again.assessment_fingerprint == audit.assessment_fingerprint
    )
    for module in (
        record_module,
        verify_module,
        records_module,
        decoding_module,
        store_port_module,
    ):
        assert "session_history" not in inspect.getsource(module)
    implementations = [
        obj
        for name, module in list(sys.modules.items())
        if name.startswith("northstar_application")
        for obj in vars(module).values()
        if isinstance(obj, type)
        and issubclass(obj, EquitySessionRecommendationHistorySource)
        and obj is not EquitySessionRecommendationHistorySource
    ]
    assert implementations == []


def test_al_no_issuance_or_lifecycle_state_exists_here() -> None:
    assert {o.value for o in EquityRecommendationRecordingOutcome}.isdisjoint(
        {"ISSUED", "RECORDED", "ACTIVE", "WAITING_FOR_ENTRY", "CLOSED_TARGET", "CLOSED_STOP"}
    )
    assert {k.value for k in _K} == {"PROPOSED", "NOT_ELIGIBLE", "UNRESOLVED"}
    ports_, audit = ei4c._proposed()
    disposition = equity_recommendation_disposition(audit)
    assert disposition is not None and not disposition.is_issued
    assert disposition.proposal is not None and not disposition.proposal.is_actionable
    assert not _verifier(ports_).verify(disposition).issuance_established


_MODULES = [decoding_module, records_module, verify_module, record_module, store_port_module]


@pytest.mark.parametrize("module", _MODULES, ids=lambda m: m.__name__.rsplit(".", 1)[-1])
def test_am_no_sqlite_infrastructure_broker_clock_or_sleep(module) -> None:  # type: ignore[no-untyped-def]
    tree = ast.parse(inspect.getsource(module))
    imported = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    imported |= {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    for name in imported:
        assert name is not None
        assert name.split(".")[0] not in {
            "sqlite3",
            "sqlalchemy",
            "time",
            "random",
            "uuid",
            "threading",
            "asyncio",
            "socket",
            "os",
        }, name
        for forbidden in (
            "northstar_infrastructure",
            "http",
            "urllib",
            "broker",
            "futures",
            "paper",
        ):
            assert forbidden not in name, name
    called = {
        node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    }
    assert not called & {
        "now",
        "today",
        "utcnow",
        "sleep",
        "random",
        "uuid4",
        "connect",
        "cursor",
        "execute_sql",
        "float",
        "hash",
        "Thread",
    }


def test_an_no_production_persistence_adapter_is_shipped() -> None:
    adapters = [
        obj
        for name, module in list(sys.modules.items())
        if name.startswith("northstar_application")
        for obj in vars(module).values()
        if isinstance(obj, type)
        and issubclass(obj, EquityRecommendationGenerationStore)
        and obj is not EquityRecommendationGenerationStore
    ]
    assert adapters == []
    assert not hasattr(application_services, "InMemoryEquityRecommendationGenerationStore")


def test_ao_generation_is_unchanged_and_never_touches_storage() -> None:
    ports_, audit = ei4c._proposed()
    store = _Store()
    for _ in range(2):
        assert ports_.run() == audit
    assert store.writes == 0 and store.dispositions == {}
    assert "EquityRecommendationGenerationStore" not in inspect.getsource(
        application_services.generate_equity_trade_recommendation
    )


def test_contracts_are_validated_and_exported() -> None:
    with pytest.raises(TypeError):
        RecordEquityRecommendationGenerationUseCase(None, None)  # type: ignore[arg-type]
    ports_, audit = ei4c._proposed()
    disposition = equity_recommendation_disposition(audit)
    assert disposition is not None
    other = equity_recommendation_attempt(_unresolved(ports_))
    with pytest.raises(ValueError, match="attempt of this disposition"):
        _recorder(ports_, _Store()).record(disposition, other)
    with pytest.raises(ValueError, match="recorded with it"):
        _recorder(ports_, _Store()).record_attempt(equity_recommendation_attempt(audit))
    with pytest.raises(ValueError):
        EquityRecommendationStoreWrite(
            EquityRecommendationStoreOutcome.COMMITTED, disposition, True
        )
    for name in (
        "RecordEquityRecommendationGenerationUseCase",
        "VerifyStoredEquityRecommendationGenerationUseCase",
        "EquityRecommendationGenerationVerifier",
        "EquityRecommendationRecordingOutcome",
        "decode_equity_recommendation_assessment",
        "establish_equity_recommendation_candidate",
        "EQUITY_RECOMMENDATION_EVALUATOR_VERSION",
    ):
        assert name in application_services.__all__
    for name in (
        "EquityRecommendationGenerationStore",
        "EquityRecommendationStoreWrite",
        "EquityRecommendationGenerationCommitUnconfirmedError",
        "EquityRecommendationGenerationNotCommittedError",
        "EquityRecommendationGenerationStoreUnavailableError",
    ):
        assert name in ports.__all__


# ----------------------------------------------------------------------
# Evaluator-version integrity
#
# No fingerprint covers ``evaluator_version``: eira1 and eirp1 are over the
# assessment and proposal documents only, unchanged. The evaluator is bound by
# (1) construction, which accepts only a supported version; (2) whole-record
# equality, which compares it, in every equivalence decision; and (3) the
# verifier, which replays only the evaluator a record names. Only one
# evaluator exists, so a second *supported* label is simulated below by
# widening the supported set, test-only. No second evaluator is executed.
# ----------------------------------------------------------------------

_FUTURE = "northstar.equity-recommendation-evaluator/2"


@pytest.fixture
def _second_label(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test-only: admit a second evaluator label to construction, never to replay."""
    monkeypatch.setattr(
        decoding_module,
        "SUPPORTED_EQUITY_RECOMMENDATION_EVALUATOR_VERSIONS",
        frozenset({EQUITY_RECOMMENDATION_EVALUATOR_VERSION, _FUTURE}),
    )


def _current_and_relabelled() -> tuple[
    ei4c._Ports, EquityRecommendationEligibilityAudit, Disposition, Disposition
]:
    ports_, audit = ei4c._proposed()
    current = equity_recommendation_disposition(audit)
    assert current is not None
    return ports_, audit, current, _rebuilt(current, evaluator_version=_FUTURE)


def test_evaluator_1_same_key_content_and_evaluator_are_equivalent() -> None:
    ports_, audit = ei4c._proposed()
    store = _Store()
    first = _recorder(ports_, store).execute(audit)
    again = _recorder(ports_, store).execute(ports_.run())
    assert (first.outcome, again.outcome) == (_O.COMMITTED, _O.EQUIVALENT_COMMITTED)
    stored = store.dispositions[audit.generation_key]
    assert (
        stored.evaluator_version
        == EQUITY_RECOMMENDATION_EVALUATOR_VERSION
        == (
            again.authoritative.evaluator_version  # type: ignore[union-attr]
        )
    )


@pytest.mark.usefixtures("_second_label")
def test_evaluator_2_equal_fingerprints_under_another_evaluator_are_not_equivalent() -> None:
    ports_, audit, current, relabelled = _current_and_relabelled()
    # The fingerprints do not cover the evaluator, and the records are still unequal.
    assert relabelled.assessment_fingerprint == current.assessment_fingerprint
    assert relabelled.proposal_fingerprint == current.proposal_fingerprint
    assert relabelled.assessment_document == current.assessment_document
    assert relabelled != current
    store = _Store()
    store.dispositions[current.generation_key] = relabelled
    result = _recorder(ports_, store).execute(audit)
    assert result.outcome is _O.EVALUATOR_VERSION_CONFLICT and not result.is_stored
    assert result.authoritative is None  # never adopted, never replayed
    assert "identical" in result.detail
    # No second disposition: the stored record is untouched; the attempt is audited.
    assert store.dispositions == {current.generation_key: relabelled}
    assert result.attempt_recorded is True


def test_evaluator_3_an_unsupported_evaluator_is_incompatible_and_fails_closed() -> None:
    ports_, audit = ei4c._proposed()
    current = equity_recommendation_disposition(audit)
    assert current is not None
    with pytest.raises(IncompatibleEquityRecommendationRecordError, match="not supported"):
        _rebuilt(current, evaluator_version=_FUTURE)
    store = _Store()
    store.script = ["incompatible"]
    result = _recorder(ports_, store).execute(audit)
    assert result.outcome is _O.STORED_RECORD_INCOMPATIBLE and not result.is_stored
    assert store.dispositions == {}


@pytest.mark.parametrize("label", ["", None, 7, b"northstar.equity-recommendation-evaluator/1"])
def test_evaluator_4_corrupt_evaluator_metadata_is_invalid_not_incompatible(label) -> None:  # type: ignore[no-untyped-def]
    _, audit = ei4c._proposed()
    current = equity_recommendation_disposition(audit)
    assert current is not None
    with pytest.raises(InvalidEquityRecommendationRecordError) as raised:
        _rebuilt(current, evaluator_version=label)
    assert not isinstance(raised.value, IncompatibleEquityRecommendationRecordError)


def test_evaluator_5_unconfirmed_commit_with_the_same_evaluator_reconciles() -> None:
    ports_, audit = ei4c._proposed()
    store = _Store()
    store.script = ["lost_after"]
    result = _recorder(ports_, store).execute(audit)
    assert result.outcome is _O.EQUIVALENT_COMMITTED and result.reconciled
    assert store.dispositions[audit.generation_key] == result.disposition


@pytest.mark.usefixtures("_second_label")
def test_evaluator_6_unconfirmed_commit_finding_another_evaluator_is_not_committed() -> None:
    ports_, audit, current, relabelled = _current_and_relabelled()
    store = _Store()
    store.script = ["lost_before"]
    store.before_load.append(
        lambda: store.dispositions.__setitem__(current.generation_key, relabelled)
    )
    result = _recorder(ports_, store).execute(audit)
    assert result.outcome is _O.EVALUATOR_VERSION_CONFLICT and result.reconciled
    assert not result.is_stored and result.authoritative is None
    # An unsupported stored evaluator on reload leaves the outcome unresolved.
    store = _Store()
    store.script = ["lost_after"]
    store.load_failure = "incompatible"
    unresolved = _recorder(ports_, store).execute(audit)
    assert unresolved.outcome is _O.COMMIT_RECONCILIATION_UNRESOLVED


@pytest.mark.usefixtures("_second_label")
def test_evaluator_7_replay_requires_the_evaluator_the_record_names() -> None:
    ports_, _, current, relabelled = _current_and_relabelled()
    assert _verifier(ports_).verify(current).evaluation_reproduced
    with pytest.raises(IncompatibleEquityRecommendationRecordError, match="replays only"):
        _verifier(ports_).verify(relabelled)
    store = _Store()
    store.dispositions[current.generation_key] = relabelled
    with pytest.raises(IncompatibleEquityRecommendationRecordError):
        VerifyStoredEquityRecommendationGenerationUseCase(store, _verifier(ports_)).execute(
            current.generation_key
        )


def test_evaluator_9_fingerprint_formats_are_unchanged() -> None:
    _, audit = ei4c._proposed()
    current = equity_recommendation_disposition(audit)
    assert current is not None
    assessment = json.loads(current.assessment_document)
    assert assessment["format"] == "northstar.equity-recommendation-assessment/1"
    assert "evaluator_version" not in current.assessment_document
    assert "evaluator_version" not in (current.proposal_document or "")
    assert current.assessment_fingerprint == audit.assessment_fingerprint
    assert audit.proposal is not None
    assert current.proposal_fingerprint == audit.proposal.fingerprint
