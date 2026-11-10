"""EI4-D10-S3: policy-catalog enforcement at generation and before recording (ADR-022).

Contract tests for the two independent Application checks of ADR-022 section
7: generation validation (layer D) in GenerateEquityTradeRecommendationUseCase
and the recording gate (layer E) in RecordEquityRecommendationGenerationUseCase,
both through verify_equity_synthetic_policy_set.

Everything is synthetic and reuses the existing helpers: the EI-4C research
world and reference doubles, the EI-4D-A generation-store double, and the
in-memory catalog double (``ei4c._Catalog``), which stores canonical text and
rebuilds every binding with ``from_stored`` on load. Policies are published
only explicitly -- through the real S1 PublishEquitySyntheticPoliciesUseCase,
or by test-only seeding that says so. A "forged" disposition is a genuine one
produced by a generator wired to another catalog: it verifies as a record but
does not match the authoritative catalog. Nothing here tests a direct
Infrastructure write, which EI4-D11 governs, or SQLite, which S2 and S4 do.
No clock, network, broker or database is used.
"""

from __future__ import annotations

import ast
import inspect
from collections.abc import Callable
from decimal import Decimal
from typing import Any

import pytest
import test_equity_synthetic_policy_catalog as s1
import test_generate_equity_trade_recommendation as ei4c
import test_record_equity_recommendation_generation as ei4d
from northstar_core.recommendations.equity import EquityPriceIncrement

from northstar_application import application_services
from northstar_application.application_services import (
    AuditEquityRecommendationPolicyBindingsUseCase,
    EquityRecommendationEligibilityAudit,
    EquityRecommendationGenerationAttempt,
    EquityRecommendationGenerationDisposition,
    EquityRecommendationGenerationVerifier,
    EquityRecommendationPolicyBindingStatus,
    EquityRecommendationRecordingOutcome,
    EquityResolvedSyntheticPolicySet,
    EquitySyntheticPolicyBindingIdentity,
    EquitySyntheticPolicyBindingVerdict,
    EquitySyntheticPolicyBundle,
    EquitySyntheticPolicyDefinitionBinding,
    EquitySyntheticPolicySetVerification,
    EquitySyntheticPolicyType,
    GenerateEquityTradeRecommendationUseCase,
    RecordEquityRecommendationGenerationUseCase,
    assess_equity_recommendation_inputs,
    equity_recommendation_attempt,
    equity_recommendation_disposition,
    verify_equity_synthetic_policy_set,
)
from northstar_application.ports import EquitySyntheticEligibilityVerdict

_S = ei4c._S
_R = ei4c._R
_O = EquityRecommendationRecordingOutcome
_V = EquitySyntheticPolicyBindingVerdict
_T = EquitySyntheticPolicyType
_NS = application_services.EQUITY_SYNTHETIC_POLICY_NAMESPACE
_v = ei4c._v
Disposition = EquityRecommendationGenerationDisposition
Attempt = EquityRecommendationGenerationAttempt
_ALL_ABSENT = (
    _R.TIMING_POLICY_UNAVAILABLE,
    _R.GENERATION_POLICY_UNAVAILABLE,
    _R.RISK_POLICY_UNAVAILABLE,
    _R.COST_SCHEDULE_UNAVAILABLE,
    _R.PRICE_INCREMENT_UNAVAILABLE,
    _R.POLICY_SET_UNAVAILABLE,
)


def _published() -> tuple[ei4c._Ports, EquityRecommendationEligibilityAudit]:
    """The default world: set/1 and its five components published, then PROPOSED."""
    ports_, audit = ei4c._proposed()
    assert ports_.catalog.publishes == 1 and len(ports_.catalog.sets) == 1
    return ports_, audit


def _only(ports_: ei4c._Ports, **request: Any) -> EquityRecommendationEligibilityAudit:
    """Generate with the catalog as it is: the harness publishes nothing first."""
    ports_.publish = False
    try:
        return ports_.run(**request)
    finally:
        ports_.publish = True


def _identity(policy_type: EquitySyntheticPolicyType, label: str) -> Any:
    return EquitySyntheticPolicyBindingIdentity(_NS, policy_type, _v(label))


def _corrupt(catalog: ei4c._Catalog, case: str) -> None:
    """Test-only: replace the published increment/1 row with an S1 corrupted variant."""
    (text, fingerprint), _ = s1._stored_variants()[case]
    catalog.definitions[_identity(_T.PRICE_INCREMENT, "increment/1")] = (text, fingerprint)


def _forged(
    ports_: ei4c._Ports, change: Callable[[ei4c._References], None], **request: Any
) -> tuple[Disposition, Attempt]:
    """A genuine disposition generated against ANOTHER catalog: a recorder caller's bypass."""
    authority = ports_.catalog
    ports_.catalog = ei4c._Catalog()
    try:
        change(ports_.references)
        audit = ports_.run(**request)
    finally:
        ports_.catalog = authority
    disposition = equity_recommendation_disposition(audit)
    assert disposition is not None, audit.reasons
    return disposition, equity_recommendation_attempt(audit)


def _other_costs(references: ei4c._References) -> None:
    references.hold("costs", ei4c._costs(slippage="0.0003"))  # costs/1, other content


def _snapshot(catalog: ei4c._Catalog) -> tuple[Any, ...]:
    return (*catalog.rows(), catalog.publishes)


# ----------------------------------------------------------------------
# The shared verification
# ----------------------------------------------------------------------


def test_the_verification_checks_six_identities_by_content_never_by_label() -> None:
    ports_, audit = _published()
    assert audit.inputs is not None
    loads = ports_.catalog.loads
    verified = verify_equity_synthetic_policy_set(ports_.catalog, audit.inputs.policy_set)
    assert verified.is_verified and verified.verdict is _V.VERIFIED and verified.failed == ()
    assert ports_.catalog.loads == loads + 6  # each identity read exactly once
    assert [c.identity.policy_type for c in verified.checks] == [
        *application_services.EQUITY_SYNTHETIC_POLICY_COMPONENT_TYPES,
        _T.POLICY_SET,
    ]
    bundle = EquitySyntheticPolicyBundle.of(audit.inputs.policy_set)
    for check, binding in zip(
        verified.checks, (*bundle.components, bundle.policy_set), strict=True
    ):
        assert check.identity == binding.identity
        assert check.offered_fingerprint == check.catalog_fingerprint
    assert verified.checks[-1].offered_fingerprint == s1._GOLDEN_EIRS1
    # Same labels, other content: never accepted on the label.
    other = EquityResolvedSyntheticPolicySet(
        reference=audit.inputs.policy_set.reference,
        generation_policy=audit.inputs.policy_set.generation_policy,
        risk_policy=audit.inputs.policy_set.risk_policy,
        cost_schedule=ei4c._costs(slippage="0.0003"),
        price_increment=audit.inputs.policy_set.price_increment,
        timing_policy=audit.inputs.policy_set.timing_policy,
    )
    refused = verify_equity_synthetic_policy_set(ports_.catalog, other)
    assert refused.verdict is _V.CONFLICT
    assert {c.identity.policy_type for c in refused.failed} == {_T.COST_SCHEDULE, _T.POLICY_SET}
    with pytest.raises(TypeError):
        verify_equity_synthetic_policy_set(object(), other)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        verify_equity_synthetic_policy_set(ports_.catalog, "set/1")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        EquitySyntheticPolicySetVerification(verified.checks[:5])


def test_a_set_alone_with_a_valid_eirs1_is_not_established() -> None:
    ports_, audit = _published()
    assert audit.inputs is not None
    ports_.catalog.definitions.clear()  # test-only: the set row without its components
    result = verify_equity_synthetic_policy_set(ports_.catalog, audit.inputs.policy_set)
    assert result.verdict is _V.NOT_PUBLISHED
    assert result.checks[-1].verdict is _V.VERIFIED  # the eirs1 itself matches
    assert len(result.failed) == 5


def test_an_offered_set_that_cannot_be_bound_fails_closed() -> None:
    class _Subclassed(EquityPriceIncrement):
        """Test-only: not exactly a Core price increment, so it cannot be bound."""

    ports_, audit = _published()
    assert audit.inputs is not None
    policy = audit.inputs.policy_set
    odd = EquityResolvedSyntheticPolicySet(
        reference=policy.reference,
        generation_policy=policy.generation_policy,
        risk_policy=policy.risk_policy,
        cost_schedule=policy.cost_schedule,
        price_increment=_Subclassed(Decimal("0.05"), _v("increment/1")),
        timing_policy=policy.timing_policy,
    )
    loads = ports_.catalog.loads
    result = verify_equity_synthetic_policy_set(ports_.catalog, odd)
    assert {c.verdict for c in result.checks} == {_V.INVALID}
    assert ports_.catalog.loads == loads  # nothing to compare, so nothing is read


# ----------------------------------------------------------------------
# Generation validation (layer D)
# ----------------------------------------------------------------------


def test_g1_g2_a_published_bundle_generates_with_unchanged_identities() -> None:
    ports_, audit = _published()
    assert audit.generation_key == s1._GOLDEN_EIRK1
    assert audit.inputs is not None and audit.inputs.policy_set.fingerprint == s1._GOLDEN_EIRS1
    # The resolver over the same catalog gives the same assessment.
    ports_.references = application_services.CatalogEquitySyntheticRecommendationReferences(  # type: ignore[assignment]
        ports_.catalog
    )
    again = ports_.run()
    assert again.status is _S.PROPOSED
    assert again.assessment_fingerprint == audit.assessment_fingerprint


def test_g3_an_unpublished_component_fails_closed_with_its_existing_reason() -> None:
    ports_, _ = _published()
    ports_.references.hold("timing", ei4c._timing_policy("timing/new"))
    audit = _only(ports_, timing_policy_version=_v("timing/new"))
    ei4c._refused(
        audit, _S.DEPENDENCY_UNAVAILABLE, _R.TIMING_POLICY_UNAVAILABLE, _R.POLICY_SET_UNAVAILABLE
    )
    assert audit.generation_key is None and audit.inputs is None


def test_g4_an_unpublished_policy_set_fails_closed() -> None:
    ports_, _ = _published()
    audit = _only(ports_, policy_set_version="set/2")  # the same five published components
    ei4c._refused(audit, _S.DEPENDENCY_UNAVAILABLE, _R.POLICY_SET_UNAVAILABLE)
    assert audit.generation_key is None


def test_g5_a_component_label_with_other_content_is_a_policy_reference_conflict() -> None:
    ports_, _ = _published()
    before = _snapshot(ports_.catalog)
    _other_costs(ports_.references)
    audit = ports_.run()  # the harness tries to publish it: refused, nothing written
    ei4c._refused(audit, _S.DEPENDENCY_UNAVAILABLE, _R.POLICY_REFERENCE_CONFLICT)
    assert audit.generation_key is None and audit.inputs is None
    assert ports_.catalog.rows() == before[:2]  # the first binding stays


def test_g6_a_set_label_naming_other_components_is_a_policy_reference_conflict() -> None:
    ports_, _ = _published()
    ports_.references.hold("timing", ei4c._timing_policy("timing/2"))
    other = ports_.run(policy_set_version="set/other", timing_policy_version=_v("timing/2"))
    assert other.status is _S.PROPOSED  # timing/2 is published, under another set
    audit = _only(ports_, policy_set_version="set/1", timing_policy_version=_v("timing/2"))
    ei4c._refused(audit, _S.DEPENDENCY_UNAVAILABLE, _R.POLICY_REFERENCE_CONFLICT)
    detail = ei4c._check(audit, ei4c._C.SYNTHETIC_REFERENCES).detail
    assert "POLICY_SET" in detail and "TIMING_POLICY" not in detail


def test_g7_a_catalog_answer_for_another_identity_is_a_conflict() -> None:
    ports_, _ = _published()
    stranger = EquitySyntheticPolicyDefinitionBinding.of(ei4c._timing_policy("timing/other"))
    ports_.catalog.definitions[_identity(_T.TIMING_POLICY, "timing/1")] = (
        stranger.canonical_json(),
        stranger.fingerprint,
    )
    audit = _only(ports_)
    ei4c._refused(audit, _S.DEPENDENCY_UNAVAILABLE, _R.POLICY_REFERENCE_CONFLICT)


@pytest.mark.parametrize(
    ("case", "verdict"),
    [("fingerprint mismatch", _V.INVALID), ("unsupported version", _V.INCOMPATIBLE)],
)
def test_g8_g9_corrupt_and_unsupported_bindings_fail_closed_and_stay_distinct(
    case: str, verdict: EquitySyntheticPolicyBindingVerdict
) -> None:
    ports_, original = _published()
    assert original.inputs is not None
    _corrupt(ports_.catalog, case)
    audit = _only(ports_)
    # ADR-022 section 8, case D: the generation reason is the permanent conflict ...
    ei4c._refused(audit, _S.DEPENDENCY_UNAVAILABLE, _R.POLICY_REFERENCE_CONFLICT)
    detail = ei4c._check(audit, ei4c._C.SYNTHETIC_REFERENCES).detail
    assert detail.startswith(f"{verdict.value}:")
    # ... and the typed verdict keeps invalid apart from unsupported.
    result = verify_equity_synthetic_policy_set(ports_.catalog, original.inputs.policy_set)
    (failed,) = result.failed
    assert failed.identity.policy_type is _T.PRICE_INCREMENT and failed.verdict is verdict
    assert result.verdict is verdict and failed.catalog_fingerprint is None


def test_g10_g11_a_catalog_outage_and_a_conflict_have_distinct_reasons() -> None:
    ports_, _ = _published()
    ports_.catalog.load_failure = "unavailable"
    outage = _only(ports_)
    ei4c._refused(outage, _S.DEPENDENCY_UNAVAILABLE, *_ALL_ABSENT)
    assert _R.POLICY_REFERENCE_CONFLICT not in outage.reasons
    ports_.catalog.load_failure = None
    _other_costs(ports_.references)
    conflict = ports_.run()
    ei4c._refused(conflict, _S.DEPENDENCY_UNAVAILABLE, _R.POLICY_REFERENCE_CONFLICT)
    # The same status; told apart by reason alone, and so by attempt identity.
    assert outage.status is conflict.status
    assert (
        equity_recommendation_attempt(outage).attempt_identity
        != equity_recommendation_attempt(conflict).attempt_identity
    )
    for audit in (outage, conflict):
        assert audit.status not in (_S.NOT_ELIGIBLE, _S.EVALUATION_UNRESOLVED)


def test_g12_g13_a_conflict_is_read_once_never_retried_and_never_published() -> None:
    ports_, _ = _published()
    _other_costs(ports_.references)
    before = _snapshot(ports_.catalog)
    loads = ports_.catalog.loads
    first = _only(ports_)
    assert ports_.catalog.loads == loads + 6  # each identity once; no retry
    second = _only(ports_)
    assert second.reasons == first.reasons == (_R.POLICY_REFERENCE_CONFLICT,)
    assert ports_.catalog.loads == loads + 12
    assert _snapshot(ports_.catalog) == before  # generation never publishes


def test_g13_an_empty_catalog_is_never_an_empty_authorized_registry() -> None:
    ports_ = ei4c._ports()
    audit = _only(ports_)
    ei4c._refused(audit, _S.DEPENDENCY_UNAVAILABLE, *_ALL_ABSENT)
    assert ports_.catalog.rows() == ({}, {}) and ports_.catalog.publishes == 0


def test_g14_an_eligibility_refusal_under_published_policies_stays_not_eligible() -> None:
    ports_, _ = _published()
    criterion = ei4c._CRITERIA[0]
    ports_.eligibility.answer = ei4c._eligibility(
        statuses={criterion: EquitySyntheticEligibilityVerdict.INELIGIBLE}
    )
    audit = ports_.run()
    ei4c._refused(audit, _S.NOT_ELIGIBLE, _R.LISTING_NOT_ELIGIBLE)
    assert audit.generation_key is not None  # assessed, with a disposition to record


def test_g15_no_candidate_never_consults_the_catalog() -> None:
    ports_ = ei4c._Ports(ei4c._World())
    audit = ports_.run()
    ei4c._refused(audit, _S.NO_CANDIDATE, _R.NO_AUTHORITATIVE_CANDIDATE)
    assert ports_.catalog.loads == 0


def test_g16_research_and_formation_are_unchanged_by_the_check() -> None:
    _, audit = _published()
    assert audit.inputs is not None
    replayed = assess_equity_recommendation_inputs(audit.request, audit.inputs)
    assert replayed.assessment_fingerprint == audit.assessment_fingerprint
    assert replayed.proposal == audit.proposal


def test_g_the_generator_requires_a_catalog() -> None:
    ports_ = ei4c._ports()
    arguments = (
        ports_.world.timelines,
        ports_.world.snapshots,
        ports_.world.venue,
        ports_.references,
        ports_.eligibility,
        ports_.state,
    )
    with pytest.raises(TypeError):
        GenerateEquityTradeRecommendationUseCase(*arguments)  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="policy_catalog"):
        GenerateEquityTradeRecommendationUseCase(*arguments, None)  # type: ignore[arg-type]


# ----------------------------------------------------------------------
# The recording gate (layer E)
# ----------------------------------------------------------------------


def _refused_write(result: Any, outcome: EquityRecommendationRecordingOutcome) -> None:
    assert result.outcome is outcome, (result.outcome, result.detail)
    assert result.writes == 0 and result.attempt_recorded is False
    assert result.authoritative is None and not result.is_stored
    assert result.policy_verification is not None and not result.policy_verification.is_verified


def test_r1_r2_a_published_disposition_records_atomically_through_the_gate() -> None:
    ports_, audit = _published()
    store = ei4d._Store()
    loads = ports_.catalog.loads
    result = ei4d._recorder(ports_, store).execute(audit)
    assert result.outcome is _O.COMMITTED and result.attempt_recorded is True
    assert result.policy_verification is not None and result.policy_verification.is_verified
    assert ports_.catalog.loads == loads + 6
    assert len(store.dispositions) == len(store.attempts) == 1 and store.writes == 1


def test_r3_r4_r7_a_forged_policy_definition_is_refused_before_any_write() -> None:
    ports_, _ = _published()
    disposition, attempt = _forged(ports_, _other_costs)
    store, before = ei4d._Store(), _snapshot(ports_.catalog)
    result = ei4d._recorder(ports_, store).record(disposition, attempt)
    _refused_write(result, _O.POLICY_REFERENCE_CONFLICT)
    assert result.policy_verification is not None
    failed = {c.identity.policy_type: c.verdict for c in result.policy_verification.failed}
    assert failed == {_T.COST_SCHEDULE: _V.CONFLICT, _T.POLICY_SET: _V.CONFLICT}
    assert store.writes == 0 and store.dispositions == {} and store.attempts == {}
    assert _snapshot(ports_.catalog) == before  # no implicit publication


def test_r5_component_labels_published_but_the_set_bound_to_others_is_a_conflict() -> None:
    ports_, _ = _published()
    ports_.references.hold("timing", ei4c._timing_policy("timing/2"))
    assert (
        ports_.run(policy_set_version="set/other", timing_policy_version=_v("timing/2")).status
        is _S.PROPOSED
    )
    disposition, attempt = _forged(
        ports_,
        lambda _: None,
        policy_set_version="set/1",
        timing_policy_version=_v("timing/2"),
    )
    result = ei4d._recorder(ports_, ei4d._Store()).record(disposition, attempt)
    _refused_write(result, _O.POLICY_REFERENCE_CONFLICT)
    assert result.policy_verification is not None
    (failed,) = result.policy_verification.failed
    assert failed.identity.policy_type is _T.POLICY_SET and failed.verdict is _V.CONFLICT


def test_r6_r11_an_unpublished_bundle_is_not_established_and_never_published() -> None:
    ports_, audit = _published()
    ports_.catalog = ei4c._Catalog()  # the authority holds nothing
    store = ei4d._Store()
    result = ei4d._recorder(ports_, store).execute(audit)
    _refused_write(result, _O.POLICY_REFERENCE_NOT_ESTABLISHED)
    assert ports_.catalog.rows() == ({}, {}) and ports_.catalog.publishes == 0
    assert store.writes == 0


def test_r8_a_catalog_outage_is_policy_reference_unavailable() -> None:
    ports_, audit = _published()
    ports_.catalog.load_failure = "unavailable"
    store = ei4d._Store()
    result = ei4d._recorder(ports_, store).execute(audit)
    _refused_write(result, _O.POLICY_REFERENCE_UNAVAILABLE)
    assert store.writes == 0


@pytest.mark.parametrize(
    ("case", "verdict"),
    [("fingerprint mismatch", _V.INVALID), ("unsupported version", _V.INCOMPATIBLE)],
)
def test_r9_r10_invalid_and_incompatible_bindings_fail_closed_and_stay_distinct(
    case: str, verdict: EquitySyntheticPolicyBindingVerdict
) -> None:
    ports_, audit = _published()
    _corrupt(ports_.catalog, case)
    store = ei4d._Store()
    result = ei4d._recorder(ports_, store).execute(audit)
    _refused_write(result, _O.POLICY_REFERENCE_CONFLICT)  # ADR-022 section 8, case D
    assert result.policy_verification is not None
    assert result.policy_verification.verdict is verdict
    assert store.writes == 0


def test_r12_r13_a_failed_gate_writes_nothing_and_claims_no_attempt() -> None:
    ports_, audit = _published()
    ports_.catalog.load_failure = "unavailable"
    store = ei4d._Store()
    store.script = ["lost_after"]  # would commit, if the gate let it write
    result = ei4d._recorder(ports_, store).execute(audit)
    assert result.attempt_recorded is False and store.writes == 0
    assert store.script == ["lost_after"] and store.attempts == {}
    with pytest.raises(ValueError, match="policy-reference refusal"):
        application_services.EquityRecommendationRecordingResult(
            _O.POLICY_REFERENCE_CONFLICT,
            result.attempt,
            result.disposition,
            None,
            False,
            False,
            1,
            "a refusal that wrote",
            result.policy_verification,
        )


def test_r14_an_equivalent_record_stays_immutable_and_is_never_certified() -> None:
    ports_, audit = _published()
    store = ei4d._Store()
    recorder = ei4d._recorder(ports_, store)
    assert recorder.execute(audit).outcome is _O.COMMITTED
    rows = (dict(store.dispositions), dict(store.attempts))
    again = recorder.execute(audit)
    assert again.outcome is _O.EQUIVALENT_COMMITTED and again.policy_verification is not None
    assert (dict(store.dispositions), dict(store.attempts)) == rows
    # A record stored before enforcement, whose policies are not published:
    # re-recording it never adopts or certifies it, and writes nothing.
    ports_.catalog = ei4c._Catalog()
    writes = store.writes
    refused = ei4d._recorder(ports_, store).execute(audit)
    _refused_write(refused, _O.POLICY_REFERENCE_NOT_ESTABLISHED)
    assert store.writes == writes and (dict(store.dispositions), dict(store.attempts)) == rows


def test_r15_a_content_conflict_under_published_policies_is_unchanged() -> None:
    ports_, audit = _published()
    store = ei4d._Store()
    unresolved = ei4d._unresolved(ports_)  # the same key and policies, other content
    assert ei4d._recorder(ports_, store).execute(unresolved).outcome is _O.COMMITTED
    result = ei4d._recorder(ports_, store).execute(audit)
    assert result.outcome is _O.CONTENT_CONFLICT
    assert result.policy_verification is not None and result.policy_verification.is_verified


@pytest.mark.parametrize(
    ("mode", "outcome"),
    [("undecodable", _O.STORED_RECORD_INVALID), ("incompatible", _O.STORED_RECORD_INCOMPATIBLE)],
)
def test_r16_w1_stored_record_mappings_are_unchanged_after_the_gate(
    mode: str, outcome: EquityRecommendationRecordingOutcome
) -> None:
    ports_, audit = _published()
    store = ei4d._Store()
    store.script = [mode]
    result = ei4d._recorder(ports_, store).execute(audit)
    assert result.outcome is outcome and result.attempt_recorded is None and result.writes == 1


def test_r17_commit_uncertainty_stays_bounded_and_the_gate_runs_once() -> None:
    ports_, audit = _published()
    store = ei4d._Store()
    store.script = ["lost_before"] * 3
    loads = ports_.catalog.loads
    result = ei4d._recorder(ports_, store).execute(audit)
    assert (result.outcome, result.writes) == (_O.RETRY_LIMIT_REACHED, 3)
    assert ports_.catalog.loads == loads + 6
    store.script = ["lost_after"]
    reconciled = ei4d._recorder(ports_, store).execute(audit)
    assert (reconciled.outcome, reconciled.reconciled) == (_O.EQUIVALENT_COMMITTED, True)


def test_r18_attempt_only_results_record_as_before_without_the_gate() -> None:
    ports_, _ = _published()
    _other_costs(ports_.references)
    conflict = ports_.run()  # POLICY_REFERENCE_CONFLICT at generation: attempt-only
    assert equity_recommendation_disposition(conflict) is None
    store, loads = ei4d._Store(), ports_.catalog.loads
    result = ei4d._recorder(ports_, store).execute(conflict)
    assert result.outcome is _O.COMMITTED and result.attempt_recorded is True
    assert result.disposition is None and result.policy_verification is None
    assert ports_.catalog.loads == loads and store.dispositions == {}
    attempt = store.attempts[result.attempt.attempt_identity]
    assert '"POLICY_REFERENCE_CONFLICT"' in attempt.attempt_document
    # A failed attempt write is reported as such, never as recorded.
    failing = ei4d._Store()
    failing.script = ["unavailable"]
    down = ei4d._recorder(ports_, failing).execute(conflict)
    assert down.outcome is _O.PERSISTENCE_UNAVAILABLE and down.attempt_recorded is False


# ----------------------------------------------------------------------
# Bypass attempts at the recorder
# ----------------------------------------------------------------------


def test_b_a_valid_eirs1_without_published_components_is_refused() -> None:
    ports_, audit = _published()
    ports_.catalog.definitions.clear()  # test-only: only the set row remains
    result = ei4d._recorder(ports_, ei4d._Store()).execute(audit)
    _refused_write(result, _O.POLICY_REFERENCE_NOT_ESTABLISHED)
    assert result.policy_verification is not None
    assert result.policy_verification.checks[-1].verdict is _V.VERIFIED


def test_b_a_published_definition_under_another_label_is_refused() -> None:
    ports_, _ = _published()

    def relabel(references: ei4c._References) -> None:
        references.hold("timing", ei4c._timing_policy("timing/copy"))  # timing/1's values

    disposition, attempt = _forged(ports_, relabel, timing_policy_version=_v("timing/copy"))
    result = ei4d._recorder(ports_, ei4d._Store()).record(disposition, attempt)
    _refused_write(result, _O.POLICY_REFERENCE_NOT_ESTABLISHED)
    assert result.policy_verification is not None
    failed = {c.identity for c in result.policy_verification.failed}
    assert _identity(_T.TIMING_POLICY, "timing/copy") in failed


def test_b_the_recorder_requires_a_catalog() -> None:
    ports_ = ei4c._ports()
    verifier = ei4d._verifier(ports_)
    with pytest.raises(TypeError):
        RecordEquityRecommendationGenerationUseCase(ei4d._Store(), verifier)  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="policy_catalog"):
        RecordEquityRecommendationGenerationUseCase(ei4d._Store(), verifier, None)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="policy_catalog"):
        RecordEquityRecommendationGenerationUseCase(ei4d._Store(), verifier, 3)  # type: ignore[arg-type]


# ----------------------------------------------------------------------
# Historical replay stays independent of the catalog
# ----------------------------------------------------------------------


def test_h_frozen_records_replay_and_verify_with_the_catalog_down() -> None:
    ports_, audit = _published()
    disposition = equity_recommendation_disposition(audit)
    assert disposition is not None and audit.inputs is not None
    ports_.catalog.load_failure = "unavailable"
    loads, before = ports_.catalog.loads, _snapshot(ports_.catalog)
    EquityRecommendationGenerationVerifier(ports_.world.timelines, ports_.world.snapshots).verify(
        disposition
    )
    replayed = assess_equity_recommendation_inputs(audit.request, disposition.inputs)
    assert replayed.assessment_fingerprint == disposition.assessment_fingerprint
    assert ports_.catalog.loads == loads and _snapshot(ports_.catalog) == before


def test_h_a_pre_enforcement_record_stays_verifiable_and_is_never_backfilled() -> None:
    ports_, _ = _published()
    disposition, _ = _forged(ports_, _other_costs)  # its costs/1 conflicts with the catalog
    before = _snapshot(ports_.catalog)
    EquityRecommendationGenerationVerifier(ports_.world.timelines, ports_.world.snapshots).verify(
        disposition
    )  # replay from its own frozen evidence still verifies
    audit = AuditEquityRecommendationPolicyBindingsUseCase(
        s1._Records(disposition), ports_.catalog
    ).execute([disposition.generation_key])
    (record,) = audit.records
    assert (
        record.status is EquityRecommendationPolicyBindingStatus.HISTORICAL_POLICY_VERSION_CONFLICT
    )
    assert _snapshot(ports_.catalog) == before  # nothing backfilled or repaired


def test_h_a_gated_recording_is_still_not_historical_publication_proof() -> None:
    ports_, audit = _published()
    store = ei4d._Store()
    assert ei4d._recorder(ports_, store).execute(audit).outcome is _O.COMMITTED
    (stored,) = store.dispositions.values()
    report = AuditEquityRecommendationPolicyBindingsUseCase(
        s1._Records(stored), ports_.catalog
    ).execute([stored.generation_key])
    (record,) = report.records
    assert record.status is EquityRecommendationPolicyBindingStatus.BINDING_NOT_ESTABLISHED


# ----------------------------------------------------------------------
# Boundaries
# ----------------------------------------------------------------------


def test_the_enforcement_reads_only_through_the_port() -> None:
    for module in (
        application_services.verify_equity_synthetic_policy_bindings,
        application_services.generate_equity_trade_recommendation,
        application_services.record_equity_recommendation_generation,
    ):
        tree = ast.parse(inspect.getsource(module))
        imported = {
            node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
        } | {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        assert not any("sqlite" in m or "infrastructure" in m for m in imported), module
        assert not any(isinstance(node, ast.While) for node in ast.walk(tree)), module
        called = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        assert not called & {"publish", "sleep", "now", "today"}, (module, called)
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        assert "PublishEquitySyntheticPoliciesUseCase" not in names
    replay = inspect.getsource(application_services.verify_equity_recommendation_generation)
    assert "verify_equity_synthetic_policy_set" not in replay


def test_the_new_contracts_are_exported() -> None:
    for name in (
        "EquitySyntheticPolicyBindingCheck",
        "EquitySyntheticPolicyBindingVerdict",
        "EquitySyntheticPolicySetVerification",
        "verify_equity_synthetic_policy_set",
    ):
        assert name in application_services.__all__, name
        assert getattr(application_services, name) is not None
    assert {_R.POLICY_REFERENCE_CONFLICT.value, _R.POLICY_SET_UNAVAILABLE.value} <= {
        r.value for r in _R
    }
    assert {
        "POLICY_REFERENCE_CONFLICT",
        "POLICY_REFERENCE_NOT_ESTABLISHED",
        "POLICY_REFERENCE_UNAVAILABLE",
    } <= {o.value for o in _O}
    assert (
        application_services.equity_recommendation_status((_R.POLICY_REFERENCE_CONFLICT,))
        is _S.DEPENDENCY_UNAVAILABLE
    )
    assert (
        application_services.equity_recommendation_status((_R.POLICY_SET_UNAVAILABLE,))
        is _S.DEPENDENCY_UNAVAILABLE
    )
