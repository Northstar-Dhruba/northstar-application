"""EI-4C slices 3a + 3b: two-version verification and version-2 generation (ADR-025).

Version 2 differs from version 1 only in the issued-history step. With ``t``
the modeled issuance and ``b = t - 1 us``, version 2 reads the history as of
``b`` and requires it complete *exactly* through ``b`` (N8, N10); its inputs,
assessment and attempt use the ``/2`` formats under the unchanged ``eiri1``,
``eira1`` and ``eirat1`` families (N3, N4). Every record pairs its evaluator
label with exactly that evaluator's formats (N6), and verification dispatches
on the label (N1). Version-1 behaviour is pinned separately by the slice-1
golden vectors and must not move.

The scenarios reuse the EI-4C synthetic world (test_generate_equity_trade_
recommendation) and the EI-4D test store, recorder and verifier (test_record_
equity_recommendation_generation). Nothing here issues, persists to a
database, reads a clock or opens a connection.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import test_evaluate_synthetic_equity_recommendation_lifecycle as lifecycle
import test_generate_equity_trade_recommendation as ei4c
import test_record_equity_recommendation_generation as ei4d
from northstar_core.foundation.value_objects import PointInTime

from northstar_application.application_services import (
    EQUITY_RECOMMENDATION_DOCUMENT_PREFIXES,
    EQUITY_RECOMMENDATION_EVALUATORS,
    EQUITY_RECOMMENDATION_STRICTLY_BEFORE_ISSUANCE,
    EquityRecommendationDispositionKind,
    EquityRecommendationEligibilityAudit,
    EquityRecommendationGenerationAttempt,
    EquityRecommendationGenerationDisposition,
    EquityRecommendationReplayMismatchError,
    GenerateEquityTradeRecommendationUseCase,
    IncompatibleEquityRecommendationRecordError,
    InvalidEquityRecommendationEvidenceError,
    InvalidEquityRecommendationRecordError,
    VerifyStoredEquityRecommendationGenerationUseCase,
    canonical_equity_recommendation_json,
    decode_equity_recommendation_assessment,
    decode_equity_recommendation_inputs,
    equity_history_boundary,
    equity_recommendation_attempt,
    equity_recommendation_disposition,
    equity_recommendation_evaluator,
    equity_recommendation_fingerprint,
)

_V1 = "northstar.equity-recommendation-evaluator/1"
_V2 = "northstar.equity-recommendation-evaluator/2"
_R, _S, _C, _CS = ei4c._R, ei4c._S, ei4c._C, ei4c._CS
_K = EquityRecommendationDispositionKind
_O = ei4d._O
Disposition = EquityRecommendationGenerationDisposition
Attempt = EquityRecommendationGenerationAttempt

_T = "2026-08-12T04:06:00Z"  # 09:36 IST, the modeled issuance of the default scenario
_B = "2026-08-12T04:05:59.999999Z"  # b = t - 1 us
_ONE_US_BEFORE_B = "2026-08-12T04:05:59.999998Z"

# Version-2 reference identities of the default scenario, pinned when version 2 was
# introduced (two processes, two hash seeds). The key and proposal are the version-1
# goldens: neither depends on the evaluator.
_V2_EIRI1 = "eiri1-sha256:a5c982499c78033cd1c32f75b58ca50afedf94de6c4adeb5c363b7534f400009"
_V2_EIRA1 = "eira1-sha256:462950adc3b31624a781359f5db2a4187079b3c3197b291014ba09b1de820f56"
_V2_KEYED = "eirat1-sha256:dd629a4148992a9fb76213124a728c8d73196efb421f78ccb59f305106d00fab"
_V2_CASE_C = "eirat1-sha256:181da75191349ba862b527eda14ff16099256fade76022164fba399aeaab9a3c"
_EIRK1 = "eirk1-sha256:4161d4d3fa27fcf1ceffb413a2428e04cfa9bb8154aad02935778cc217a1ece6"
_EIRP1 = "eirp1-sha256:678242e1c71544d99abba7882eb68c3645eda2e656cc06e08b5b6fe38eb2d5fd"


def _v2(through: str | None = _B, *records: Any) -> tuple[ei4c._Ports, Any]:
    """Generate under version 2 with the issued history complete through ``through``."""
    ports_ = ei4c._ports()
    ports_.evaluator_version = _V2
    ports_.state.answer = (
        None if through is None else ei4c._history(*records, through=PointInTime(through))
    )
    return ports_, ports_.run()


def _keyless(audit: EquityRecommendationEligibilityAudit, reason: Any) -> Attempt:
    """A version-2 refusal: DEPENDENCY_UNAVAILABLE, keyless, attempt-only, format /2."""
    assert audit.status is _S.DEPENDENCY_UNAVAILABLE and audit.reasons == (reason,)
    assert audit.evaluator_version == _V2
    assert audit.inputs is None and audit.generation_key is None and audit.proposal is None
    assert equity_recommendation_disposition(audit) is None  # no disposition, no key consumed
    assert ei4c._check(audit, _C.SYNTHETIC_REFERENCES).status is _CS.FAILED
    attempt = equity_recommendation_attempt(audit)
    document = json.loads(attempt.attempt_document)
    assert document["format"] == "northstar.equity-recommendation-generation-attempt/2"
    assert (
        document["generation_key"],
        document["disposition_kind"],
        document["assessment_fingerprint"],
    ) == (None, None, None)
    assert attempt.evaluator_version == _V2 and attempt.attempt_identity.startswith("eirat1-")
    return attempt


def _resealed(document: dict[str, Any]) -> tuple[str, str]:
    """TEST-ONLY tampering: canonical text and a fingerprint that matches it."""
    return canonical_equity_recommendation_json(document), equity_recommendation_fingerprint(
        document
    )


def _reseal_assessment(document: dict[str, Any]) -> tuple[str, str]:
    if document["inputs"] is not None:
        document["inputs_fingerprint"] = equity_recommendation_fingerprint(document["inputs"])
    return _resealed(document)


# ======================================================================
# The closed table and evaluator selection (N1, N7)
# ======================================================================


def test_the_closed_table_holds_exactly_the_two_governed_evaluators() -> None:
    assert list(EQUITY_RECOMMENDATION_EVALUATORS) == [_V1, _V2]
    v1, v2 = EQUITY_RECOMMENDATION_EVALUATORS[_V1], EQUITY_RECOMMENDATION_EVALUATORS[_V2]
    assert (v1.inputs_format, v1.assessment_format, v1.attempt_format) == (
        "northstar.equity-recommendation-generation-inputs/1",
        "northstar.equity-recommendation-assessment/1",
        "northstar.equity-recommendation-generation-attempt/1",
    )
    assert (v2.inputs_format, v2.assessment_format, v2.attempt_format) == (
        "northstar.equity-recommendation-generation-inputs/2",
        "northstar.equity-recommendation-assessment/2",
        "northstar.equity-recommendation-generation-attempt/2",
    )
    assert (v1.strictly_before_issuance, v2.strictly_before_issuance) == (False, True)
    for row in (v1, v2):  # the families are retained (N3)
        assert EQUITY_RECOMMENDATION_DOCUMENT_PREFIXES[row.inputs_format] == "eiri1-sha256:"
        assert EQUITY_RECOMMENDATION_DOCUMENT_PREFIXES[row.assessment_format] == "eira1-sha256:"
        assert EQUITY_RECOMMENDATION_DOCUMENT_PREFIXES[row.attempt_format] == "eirat1-sha256:"
    with pytest.raises(TypeError):
        EQUITY_RECOMMENDATION_EVALUATORS["x"] = v1  # type: ignore[index]  # not extensible
    assert EQUITY_RECOMMENDATION_STRICTLY_BEFORE_ISSUANCE == "STRICTLY_BEFORE_MODELED_ISSUANCE"


def test_selection_is_keyword_only_closed_and_defaults_to_version_1() -> None:
    ports_ = ei4c._ports()
    default = ports_.use_case()
    assert default.evaluator_version == _V1  # every existing construction site
    ports_.evaluator_version = _V2
    assert ports_.use_case().evaluator_version == _V2
    arguments = (
        ports_.world.timelines,
        ports_.world.snapshots,
        ports_.world.venue,
        ports_.references,
        ports_.eligibility,
        ports_.state,
        ports_.catalog,
    )
    for refused in ("northstar.equity-recommendation-evaluator/999", "", _V1 + " "):
        with pytest.raises(ValueError, match="governed evaluator"):
            GenerateEquityTradeRecommendationUseCase(*arguments, evaluator_version=refused)
    with pytest.raises(TypeError):
        GenerateEquityTradeRecommendationUseCase(*arguments, evaluator_version=None)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        GenerateEquityTradeRecommendationUseCase(*arguments, _V2)  # type: ignore[misc]
    with pytest.raises(ValueError):
        equity_recommendation_evaluator("northstar.equity-recommendation-evaluator/3")


def test_an_explicit_version_1_selection_is_the_default_byte_for_byte() -> None:
    _, default = ei4c._proposed()
    explicit = ei4c._ports()
    explicit.evaluator_version = _V1
    audit = explicit.run()
    assert audit.assessment_document() == default.assessment_document()
    assert audit.assessment_fingerprint == default.assessment_fingerprint
    assert explicit.state.asked == [PointInTime(_T)]  # version 1 reads as of t


# ======================================================================
# The history boundary b = t - 1 us (N8; HB-01 to HB-06)
# ======================================================================


@pytest.mark.parametrize(
    ("t", "b"),
    [
        ("2026-08-12T09:36:00+05:30", "2026-08-12T04:05:59.999999Z"),  # second and minute
        ("2026-08-12T04:06:00.000001Z", "2026-08-12T04:06:00Z"),  # the fraction drops
        ("2026-08-12T04:06:00.5Z", "2026-08-12T04:06:00.499999Z"),  # within the second
        ("2026-08-12T05:00:00Z", "2026-08-12T04:59:59.999999Z"),  # hour
        ("2026-08-13T00:00:00+05:30", "2026-08-12T18:29:59.999999Z"),  # IST date
        ("2026-08-13T00:00:00Z", "2026-08-12T23:59:59.999999Z"),  # UTC date
        ("2027-01-01T00:00:00Z", "2026-12-31T23:59:59.999999Z"),  # year
    ],
)
def test_the_boundary_is_the_last_representable_instant_before_t(t: str, b: str) -> None:
    boundary = equity_history_boundary(PointInTime(t))
    assert boundary.value == b
    assert boundary.compare(PointInTime(t)) < 0
    assert equity_history_boundary(PointInTime(t)) == boundary  # pure, no clock


def test_no_boundary_precedes_the_lowest_representable_instant() -> None:
    with pytest.raises(InvalidEquityRecommendationEvidenceError, match="precedes"):
        equity_history_boundary(PointInTime("0001-01-01T00:00:00Z"))
    with pytest.raises(TypeError):
        equity_history_boundary("2026-08-12T04:06:00Z")  # type: ignore[arg-type]


# ======================================================================
# N10 case A -- complete exactly through b
# ======================================================================


def test_case_a_version_2_proposes_on_history_complete_exactly_through_b() -> None:
    ports_, audit = _v2(_B)
    assert audit.status is _S.PROPOSED and audit.evaluator_version == _V2
    assert ports_.state.asked == [PointInTime(_B)]  # read strictly before issuance
    inputs = audit.inputs
    assert inputs is not None and inputs.evaluator_version == _V2
    document = inputs.document()
    assert document["format"] == "northstar.equity-recommendation-generation-inputs/2"
    assert document["modeled_issued_at"] == _T  # t is unchanged in both versions
    history = document["issued_history"]
    assert history["history_boundary"] == "STRICTLY_BEFORE_MODELED_ISSUANCE"
    assert history["as_of"] == history["complete_through"] == _B
    assessment = audit.assessment_document()
    assert assessment["format"] == "northstar.equity-recommendation-assessment/2"
    assert assessment["inputs"] == document and assessment["inputs_fingerprint"] == _V2_EIRI1
    # The proposal, key and Core outcome are /1 and evaluator-independent.
    assert audit.proposal is not None
    assert audit.proposal.document()["format"] == "northstar.equity-recommendation-proposal/1"
    assert audit.proposal.entry_window.timing.issued_at.value == _T
    assert (audit.generation_key, audit.proposal.fingerprint) == (_EIRK1, _EIRP1)
    assert (inputs.fingerprint, audit.assessment_fingerprint) == (_V2_EIRI1, _V2_EIRA1)
    attempt = equity_recommendation_attempt(audit)
    assert attempt.attempt_identity == _V2_KEYED and attempt.evaluator_version == _V2
    assert json.loads(attempt.attempt_document)["assessment_fingerprint"] == _V2_EIRA1


def test_case_a_every_version_2_identity_differs_from_its_version_1_counterpart() -> None:
    _, v1 = ei4c._proposed()
    _, v2 = _v2(_B)
    assert v1.inputs is not None and v2.inputs is not None
    assert v1.inputs.fingerprint != v2.inputs.fingerprint
    assert v1.assessment_fingerprint != v2.assessment_fingerprint
    assert (
        equity_recommendation_attempt(v1).attempt_identity
        != equity_recommendation_attempt(v2).attempt_identity
    )
    assert v1.generation_key == v2.generation_key  # the key names no evaluator (N5)


def test_case_a_the_strict_boundary_includes_b_and_never_t() -> None:
    at_b = ei4c._issued("at-b", ei4c._LISTING, PointInTime(_B))
    _, audit = _v2(_B, at_b)
    # Issued at b is before t: open under EI4-D8, so exclusivity is unresolved (keyed, v2).
    assert audit.status is _S.EVALUATION_UNRESOLVED
    assert audit.reasons == (_R.EXCLUSIVITY_POLICY_UNRESOLVED,)
    assert audit.inputs is not None
    assert [r.recommendation_key for r in audit.inputs.issued_history.recommendations] == ["at-b"]
    disposition = equity_recommendation_disposition(audit)
    assert disposition is not None and disposition.kind is _K.UNRESOLVED
    assert disposition.evaluator_version == _V2
    # A record at t cannot sit in a history complete only through b: it is never seen.
    with pytest.raises(ValueError):
        ei4c._history(ei4c._issued("at-t", ei4c._LISTING, PointInTime(_T)), through=PointInTime(_B))
    # Version 1 sees an issuance at t (inclusive), as it always has.
    ports_ = ei4c._ports()
    ports_.state.answer = ei4c._history(ei4c._issued("at-t", ei4c._LISTING, PointInTime(_T)))
    v1 = ports_.run()
    assert v1.inputs is not None
    assert [r.recommendation_key for r in v1.inputs.issued_history.recommendations] == ["at-t"]


def test_repeated_version_2_generation_is_identical() -> None:
    _, first = _v2(_B)
    _, again = _v2(_B)
    assert first.assessment_document() == again.assessment_document()
    assert first.assessment_fingerprint == again.assessment_fingerprint == _V2_EIRA1
    assert (
        equity_recommendation_attempt(first).attempt_identity
        == equity_recommendation_attempt(again).attempt_identity
    )


# ======================================================================
# N10 cases B, C and D -- keyless, attempt-only refusals
# ======================================================================


def test_case_b_history_complete_only_before_b_is_incomplete() -> None:
    _, audit = _v2(_ONE_US_BEFORE_B)
    _keyless(audit, _R.RECOMMENDATION_HISTORY_INCOMPLETE)
    assert "complete only through" in ei4c._check(audit, _C.SYNTHETIC_REFERENCES).detail


@pytest.mark.parametrize(
    "through",
    [_T, "2026-08-12T04:06:00.000001Z", "2026-08-12T10:00:00Z"],  # b + 1 us, t + 1 us, 15:30 IST
)
def test_case_c_history_declared_beyond_b_is_the_approved_refusal(through: str) -> None:
    _, audit = _v2(through)
    _keyless(audit, _R.RECOMMENDATION_HISTORY_BOUNDARY_EXCEEDED)
    assert _R.RECOMMENDATION_HISTORY_INCOMPLETE not in audit.reasons  # never case B


def test_case_c_is_recorded_as_one_idempotent_keyless_attempt_and_consumes_no_key() -> None:
    ports_, audit = _v2("2026-08-12T10:00:00Z")
    attempt = _keyless(audit, _R.RECOMMENDATION_HISTORY_BOUNDARY_EXCEEDED)
    assert attempt.attempt_identity == _V2_CASE_C
    store = ei4d._Store()
    first = ei4d._recorder(ports_, store).execute(audit)
    again = ei4d._recorder(ports_, store).execute(ports_.run())
    assert (first.outcome, again.outcome) == (_O.COMMITTED, _O.EQUIVALENT_COMMITTED)
    assert first.disposition is None and not first.authoritative
    assert store.dispositions == {}  # no disposition, so the key is free
    assert list(store.attempts) == [attempt.attempt_identity]  # one logical attempt
    # The key is not consumed: a version-1 generation over the same history still records,
    # and version 1 never reports the version-2 reason.
    ports_.evaluator_version = None
    v1 = ports_.run()
    assert v1.status is _S.PROPOSED
    assert _R.RECOMMENDATION_HISTORY_BOUNDARY_EXCEEDED not in v1.reasons
    assert ei4d._recorder(ports_, store).execute(v1).outcome is _O.COMMITTED
    assert list(store.dispositions) == [v1.generation_key]


@pytest.mark.parametrize("answer", ["missing", "unavailable", "another session"])
def test_case_d_missing_unreadable_or_foreign_history_is_unavailable(answer: str) -> None:
    ports_ = ei4c._ports()
    ports_.evaluator_version = _V2
    if answer == "missing":
        ports_.state.answer = None
    elif answer == "unavailable":
        ports_.state.unavailable = True
    else:
        ports_.state.answer = ei4c._history(
            day=ei4c.date(2026, 8, 13), through=PointInTime("2026-08-13T04:05:59.999999Z")
        )
    audit = ports_.run()
    _keyless(audit, _R.OPEN_RECOMMENDATION_STATE_UNAVAILABLE)
    assert ports_.state.asked == [PointInTime(_B)]


def test_keyless_attempts_are_distinct_across_versions_and_stable_within_one() -> None:
    v1_ports = ei4c._ports()
    v1_ports.state.answer = ei4c._history(through=PointInTime(_ONE_US_BEFORE_B))
    v1 = equity_recommendation_attempt(v1_ports.run())
    _, v2_audit = _v2(_ONE_US_BEFORE_B)
    v2 = equity_recommendation_attempt(v2_audit)
    assert json.loads(v1.attempt_document)["reasons"] == json.loads(v2.attempt_document)["reasons"]
    assert v1.attempt_identity != v2.attempt_identity  # N4
    assert (v1.evaluator_version, v2.evaluator_version) == (_V1, _V2)
    assert (
        v2.attempt_identity
        == equity_recommendation_attempt(_v2(_ONE_US_BEFORE_B)[1]).attempt_identity
    )


def test_version_1_never_reports_the_version_2_reason() -> None:
    for through in (_T, "2026-08-12T10:00:00Z"):  # complete exactly through t, and beyond
        ports_ = ei4c._ports()
        ports_.state.answer = ei4c._history(through=PointInTime(through))
        audit = ports_.run()
        assert audit.status is _S.PROPOSED and audit.evaluator_version == _V1
    ports_ = ei4c._ports()
    ports_.state.answer = ei4c._history(through=PointInTime(_B))  # before t: version-1 case B
    audit = ports_.run()
    assert audit.reasons == (_R.RECOMMENDATION_HISTORY_INCOMPLETE,)
    # A version-1 audit, attempt or assessment stating the reason is refused.
    _, v2_audit = _v2(_T)
    with pytest.raises(ValueError, match="version-2 reason"):
        EquityRecommendationEligibilityAudit(
            request=v2_audit.request,
            checks=v2_audit.checks,
            candidate=v2_audit.candidate,
            inputs=None,
            entry_window=None,
            core_outcome=None,
            proposal=None,
            evaluator_version=_V1,
        )
    document = json.loads(equity_recommendation_attempt(v2_audit).attempt_document)
    document["format"] = "northstar.equity-recommendation-generation-attempt/1"
    text, identity = _resealed(document)
    with pytest.raises(InvalidEquityRecommendationRecordError, match="version-2 reason"):
        Attempt(identity, text)


# ======================================================================
# Verification: levels A to C under each record's own evaluator
# ======================================================================


def _v2_record() -> tuple[ei4c._Ports, Any, Disposition]:
    ports_, audit = _v2(_B)
    disposition = equity_recommendation_disposition(audit)
    assert disposition is not None and disposition.evaluator_version == _V2
    return ports_, audit, disposition


def test_level_a_a_version_2_record_decodes_re_encodes_and_pairs() -> None:
    _, audit, disposition = _v2_record()
    rebuilt = ei4d._rebuilt(disposition)
    assert rebuilt == disposition and rebuilt.assessment.evaluator_version == _V2
    assert rebuilt.inputs == audit.inputs and rebuilt.inputs_fingerprint == _V2_EIRI1
    assert rebuilt.inputs_document == canonical_equity_recommendation_json(
        audit.inputs.document()  # type: ignore[union-attr]
    )
    decoded = decode_equity_recommendation_inputs(json.loads(rebuilt.inputs_document))
    assert decoded.evaluator_version == _V2 and decoded.fingerprint == _V2_EIRI1
    attempt = equity_recommendation_attempt(audit)
    assert Attempt(attempt.attempt_identity, attempt.attempt_document) == attempt


def test_levels_b_and_c_a_version_2_record_verifies_under_version_2() -> None:
    ports_, _, disposition = _v2_record()
    store = ei4d._Store()
    store.dispositions[disposition.generation_key] = disposition
    verified = VerifyStoredEquityRecommendationGenerationUseCase(
        store, ei4d._verifier(ports_)
    ).execute(disposition.generation_key)
    assert verified is not None
    assert (
        verified.document_integrity_verified,
        verified.source_integrity_verified,
        verified.evaluation_reproduced,
        verified.issuance_established,
    ) == (True, True, True, False)
    assert verified.replayed.evaluator_version == _V2
    assert verified.replayed.assessment_fingerprint == _V2_EIRA1
    assert verified.replayed.inputs is not None
    assert verified.replayed.inputs.issued_history.as_of.value == _B
    # Replay uses the frozen history, never the current source.
    ports_.state.unavailable = True
    assert ei4d._verifier(ports_).verify(disposition).evaluation_reproduced
    # Level B is unchanged: a missing source fails closed for version 2 as for version 1.
    ports_.world.timelines.records.clear()
    with pytest.raises(ei4d.EquityRecommendationSourceVerificationError):
        ei4d._verifier(ports_).verify(disposition)


def _tampered_v2(change: Any) -> tuple[str, str]:
    _, _, disposition = _v2_record()
    document = json.loads(disposition.assessment_document)
    change(document)
    return _reseal_assessment(document)


@pytest.mark.parametrize(
    ("label", "change"),
    [
        (
            "history as of t instead of b",
            lambda d: d["inputs"]["issued_history"].update(as_of=_T, complete_through=_T),
        ),
        (
            "history complete beyond b",
            lambda d: d["inputs"]["issued_history"].update(complete_through=_T),
        ),
        (
            "boundary field missing",
            lambda d: d["inputs"]["issued_history"].pop("history_boundary"),
        ),
        (
            "boundary field changed",
            lambda d: d["inputs"]["issued_history"].update(history_boundary="AT_ISSUANCE"),
        ),
        (
            "assessment /2 embedding inputs /1",
            lambda d: d["inputs"].update(
                format="northstar.equity-recommendation-generation-inputs/1"
            ),
        ),
    ],
)
def test_a_resealed_version_2_record_with_version_1_semantics_is_invalid(
    label: str, change: Any
) -> None:
    text, fingerprint = _tampered_v2(change)
    with pytest.raises(InvalidEquityRecommendationRecordError) as raised:
        decode_equity_recommendation_assessment(text, fingerprint)
    assert not isinstance(raised.value, IncompatibleEquityRecommendationRecordError), label


def test_version_1_inputs_never_state_a_boundary_and_never_nest_under_version_2() -> None:
    _, audit = ei4c._proposed()
    disposition = equity_recommendation_disposition(audit)
    assert disposition is not None
    with_boundary = json.loads(disposition.assessment_document)
    with_boundary["inputs"]["issued_history"]["history_boundary"] = (
        "STRICTLY_BEFORE_MODELED_ISSUANCE"
    )
    nested = json.loads(disposition.assessment_document)
    nested["inputs"]["format"] = "northstar.equity-recommendation-generation-inputs/2"
    relabelled = json.loads(disposition.assessment_document)
    relabelled["format"] = "northstar.equity-recommendation-assessment/2"  # v1 inputs inside
    for document in (with_boundary, nested, relabelled):
        text, fingerprint = _reseal_assessment(document)
        with pytest.raises(InvalidEquityRecommendationRecordError) as raised:
            decode_equity_recommendation_assessment(text, fingerprint)
        assert not isinstance(raised.value, IncompatibleEquityRecommendationRecordError)


def test_unknown_versions_within_a_family_are_incompatible() -> None:
    _, audit, disposition = _v2_record()
    document = json.loads(disposition.assessment_document)
    document["format"] = "northstar.equity-recommendation-assessment/3"
    with pytest.raises(IncompatibleEquityRecommendationRecordError):
        decode_equity_recommendation_assessment(
            canonical_equity_recommendation_json(document), disposition.assessment_fingerprint
        )
    attempt = json.loads(equity_recommendation_attempt(audit).attempt_document)
    attempt["format"] = "northstar.equity-recommendation-generation-attempt/9"
    text = canonical_equity_recommendation_json(attempt)
    with pytest.raises(IncompatibleEquityRecommendationRecordError):
        Attempt(equity_recommendation_attempt(audit).attempt_identity, text)
    with pytest.raises(IncompatibleEquityRecommendationRecordError, match="not supported"):
        ei4d._rebuilt(disposition, evaluator_version="northstar.equity-recommendation-evaluator/3")
    with pytest.raises(InvalidEquityRecommendationRecordError) as raised:
        ei4d._rebuilt(disposition, evaluator_version="")
    assert not isinstance(raised.value, IncompatibleEquityRecommendationRecordError)


def test_a_version_2_replay_that_does_not_reproduce_fails_closed() -> None:
    ports_, _, disposition = _v2_record()

    def misstate(document: dict[str, Any]) -> None:
        document["entry_window"]["timing"]["eligible_from"] = "2026-08-12T04:15:00Z"
        document["proposal"]["entry_window"]["timing"]["eligible_from"] = "2026-08-12T04:15:00Z"

    document = json.loads(disposition.assessment_document)
    misstate(document)
    proposal_text, proposal_fingerprint = _resealed(document["proposal"])
    document["proposal_fingerprint"] = proposal_fingerprint
    text, fingerprint = _reseal_assessment(document)
    altered = ei4d._rebuilt(
        disposition,
        assessment_document=text,
        assessment_fingerprint=fingerprint,
        proposal_document=proposal_text,
        proposal_fingerprint=proposal_fingerprint,
    )
    with pytest.raises(EquityRecommendationReplayMismatchError):
        ei4d._verifier(ports_).verify(altered)


# ======================================================================
# Recording: authorized writes, idempotence and cross-version conflicts (N5)
# ======================================================================


def test_a_version_2_record_is_recorded_through_the_authorized_path_idempotently() -> None:
    ports_, audit, disposition = _v2_record()
    store = ei4d._Store()
    first = ei4d._recorder(ports_, store).execute(audit)
    assert first.outcome is _O.COMMITTED and first.authoritative == disposition
    again = ei4d._recorder(ports_, store).execute(ports_.run())
    assert again.outcome is _O.EQUIVALENT_COMMITTED
    assert store.dispositions == {disposition.generation_key: disposition}
    assert store.dispositions[disposition.generation_key].evaluator_version == _V2
    assert list(store.attempts) == [_V2_KEYED]
    assert store.writes == 2  # every write went through write_generation(authorization)


def test_the_recorder_refuses_an_attempt_of_another_evaluator() -> None:
    ports_, _, v1, v2_audit, v2 = ei4d._both_versions()
    v1_attempt = equity_recommendation_attempt(ei4c._proposed()[1])
    store = ei4d._Store()
    with pytest.raises(ValueError, match="not the attempt of this disposition"):
        ei4d._recorder(ports_, store).record(v2, v1_attempt)
    assert store.writes == 0 and store.dispositions == {}
    assert v1.generation_key == v2.generation_key and v2_audit.evaluator_version == _V2


def test_a_verified_version_2_proposal_is_lifecycle_evaluable() -> None:
    # EI-5C consumes the verified-stored use case and never reads the label (design 4).
    world = ei4c._World({index: ei4c._MORNING[index] for index in range(1, 6)})
    world.commit(ei4c._ist(9, 31), ei4c._ist(9, 36), ei4c._ist(9, 41))
    ports_ = ei4c._Ports(world)
    ports_.evaluator_version = _V2
    ports_.state.answer = ei4c._history(through=PointInTime(_B))
    disposition = equity_recommendation_disposition(ports_.run())
    assert disposition is not None and disposition.evaluator_version == _V2
    store = ei4d._Store()
    store.dispositions[disposition.generation_key] = disposition
    verify = VerifyStoredEquityRecommendationGenerationUseCase(store, lifecycle._verifier(world))
    scenario = lifecycle._Scenario(
        world,
        ports_,
        disposition,
        store,
        lifecycle.EvaluateSyntheticEquityRecommendationLifecycleUseCase(verify, world.snapshots),
    )
    assert lifecycle._active(scenario).state is lifecycle.S.ACTIVE


# ======================================================================
# Determinism in a fresh process (HB-08)
# ======================================================================


def test_a_fresh_process_reproduces_the_version_2_identities() -> None:
    here = Path(__file__).parent
    code = (
        "import json, test_equity_recommendation_v2_generation as v2\n"
        "from northstar_application.application_services import equity_recommendation_attempt\n"
        "_, a = v2._v2(v2._B)\n"
        "_, c = v2._v2('2026-08-12T10:00:00Z')\n"
        "print(json.dumps([a.inputs.fingerprint, a.assessment_fingerprint,\n"
        "    equity_recommendation_attempt(a).attempt_identity,\n"
        "    equity_recommendation_attempt(c).attempt_identity]))\n"
    )
    completed = subprocess.run(  # the test's own interpreter, running fixed code only
        [sys.executable, "-c", code],
        cwd=here,
        env=dict(os.environ, PYTHONHASHSEED="4242", PYTHONPATH=str(here)),
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout.strip().splitlines()[-1]) == [
        _V2_EIRI1,
        _V2_EIRA1,
        _V2_KEYED,
        _V2_CASE_C,
    ]
