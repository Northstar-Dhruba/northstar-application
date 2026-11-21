"""EI-7C Application: verified recommendation detail, lineage, evidence panels, summaries.

Application contract tests for the EI-7 Decision Register, sections 13.4 and
13.6 (EI7-CF4, EI7-EC1, EI7-P5) and design section 23.4. The world is the
accepted EI-5E/EI-6B synthetic chain, reused unchanged (``ei6b._recording``):
a real verified evaluator-/2 disposition, a real MODELED issuance in the
labelled in-memory EI-5D authority, the labelled in-memory EI-6B lifecycle
store and the EI-2B test snapshot store.

Lineage mismatches are injected through TEST-ONLY read wrappers that return a
changed *copy* of an issuance or disposition; the stored records of the world
are never changed, so the primary authority keeps verifying as before.

Nothing reads a clock, opens a connection or writes anything.
"""

from __future__ import annotations

import copy
import dataclasses
import inspect
from collections.abc import Callable
from datetime import date
from decimal import Decimal
from typing import Any

import pytest
import test_evaluate_synthetic_equity_recommendation_lifecycle as lifecycle
import test_record_equity_recommendation_lifecycle as ei6b
from northstar_core.domain.value_objects import ListingReference
from northstar_core.foundation.value_objects import ExchangeCode, Price, Symbol
from northstar_core.recommendations.equity import (
    EquityIssuanceKind,
    EquitySyntheticVersion,
)
from northstar_core.recommendations.equity.lifecycle import (
    EquityLifecycleEventKind,
    EquityLifecycleOutcomeClass,
    EquityLifecycleState,
)
from northstar_core.strategy import StrategyIdentity

from northstar_application.application_services import (
    EQUITY_RECOMMENDATION_EVALUATOR_V2,
    EquityEvidencePanel,
    EquityEvidencePanelReason,
    EquityEvidencePanelStatus,
    EquityEvidenceRole,
    EquityLifecycleSummary,
    EquityLifecycleSummaryStatus,
    EquityReadQueryRefusal,
    EquityReadQueryRefusedError,
    EquityRecommendationDetailView,
    EquityRecommendationDispositionKind,
    EquityRecordedIssuance,
    EquityVerifiedGenerationLineage,
    GetEquityLifecycleSummaryUseCase,
    GetEquityRecommendationDetailUseCase,
    IncompatibleEquityRecommendationRecordError,
    InvalidEquityRecommendationDetailViewError,
    InvalidEquityRecommendationRecordError,
    equity_recommendation_issuance_identity,
)
from northstar_application.ports import (
    EquityCandleEvidenceUnavailableError,
    EquityRecommendationGenerationRecordReader,
    EquityRecommendationGenerationStoreUnavailableError,
    EquityRecommendationIssuanceUnavailableError,
    EquityRecommendationLifecycleStoreUnavailableError,
    EquityRecordedLifecycle,
    InvalidStoredEquityCandleEvidenceSnapshotError,
    InvalidStoredEquityRecommendationIssuanceAuthorityError,
    InvalidStoredEquityRecommendationLifecycleError,
)

_R = EquityReadQueryRefusal
S = EquityLifecycleState
_STATUS = EquityEvidencePanelStatus
_REASON = EquityEvidencePanelReason
_UNKNOWN_ID = "eiris1-sha256:" + "0" * 64


# ----------------------------------------------------------------------
# TEST-ONLY read wrappers: read operations only, no write method
# ----------------------------------------------------------------------


class _Lifecycles:
    """A read-only view of a lifecycle source; ``change`` maps the recorded result."""

    def __init__(self, inner: Any, change: Callable[[Any], Any] | None = None) -> None:
        self.inner, self.change, self.error = inner, change, None
        self.calls: list[str] = []

    def load_lifecycle(self, issuance_id: str):  # type: ignore[no-untyped-def]
        self.calls.append(issuance_id)
        if self.error is not None:
            raise self.error
        recorded = self.inner.load_lifecycle(issuance_id)
        return recorded if self.change is None or recorded is None else self.change(recorded)


class _Generations(EquityRecommendationGenerationRecordReader):
    """A read-only view of generation records; ``change`` maps the disposition (a copy)."""

    def __init__(self, inner: Any, change: Callable[[Any], Any] | None = None) -> None:
        self.inner, self.change, self.error = inner, change, None
        self.keys: list[str] = []

    def load_disposition(self, generation_key: str):  # type: ignore[no-untyped-def]
        self.keys.append(generation_key)
        if self.error is not None:
            raise self.error
        found = self.inner.load_disposition(generation_key)
        return found if self.change is None else self.change(found)

    def load_attempt(self, attempt_identity: str):  # type: ignore[no-untyped-def]
        raise AssertionError("an EI-7 read never searches generation attempts")


class _Snapshots:
    """A read-only view of a snapshot store; ``overrides`` answer or raise per identity."""

    def __init__(self, inner: Any, overrides: dict[str, Any] | None = None) -> None:
        self.inner, self.overrides = inner, dict(overrides or {})
        self.reads: list[str] = []

    def get_snapshot(self, snapshot_id: str):  # type: ignore[no-untyped-def]
        self.reads.append(snapshot_id)
        if snapshot_id in self.overrides:
            answer = self.overrides[snapshot_id]
            if isinstance(answer, Exception):
                raise answer
            return answer
        return self.inner.get_snapshot(snapshot_id)


def _detail_use_case(
    recording: ei6b._Recording,
    lifecycles: Any = None,
    generations: Any = None,
    snapshots: Any = None,
) -> GetEquityRecommendationDetailUseCase:
    return GetEquityRecommendationDetailUseCase(
        lifecycles or _Lifecycles(recording.store),
        generations or _Generations(recording.world.store),
        snapshots or _Snapshots(recording.scenario.world.snapshots),
    )


def _detail(recording: ei6b._Recording, **readers: Any) -> EquityRecommendationDetailView:
    return _detail_use_case(recording, **readers).execute(recording.issuance_id)


def _refused(recording: ei6b._Recording, **readers: Any) -> EquityReadQueryRefusedError:
    with pytest.raises(EquityReadQueryRefusedError) as raised:
        _detail(recording, **readers)
    return raised.value


def _issuance_changed(**changes: Any) -> Callable[[EquityRecordedLifecycle], Any]:
    def change(recorded: EquityRecordedLifecycle) -> EquityRecordedLifecycle:
        issued = dataclasses.replace(recorded.issuance.recommendation, **changes)
        return EquityRecordedLifecycle(
            EquityRecordedIssuance(issued, recorded.issuance.recording_sequence),
            recorded.lifecycle,
        )

    return change


def _clone(obj: Any, **changes: Any) -> Any:
    """TEST-ONLY: a shallow copy of a frozen record with fields replaced (no validation)."""
    clone = copy.copy(obj)
    for name, value in changes.items():
        object.__setattr__(clone, name, value)
    return clone


def _with_proposal(disposition: Any, **definition_changes: Any) -> Any:
    proposal = disposition.proposal
    definition = dataclasses.replace(proposal.definition, **definition_changes)
    changed = dataclasses.replace(proposal, definition=definition)
    return _clone(disposition, assessment=_clone(disposition.assessment, proposal=changed))


def _production(recording: ei6b._Recording) -> Any:
    """A genuine PRODUCTION-class snapshot, as the EI-6B acceptance tests build one."""
    real = recording.scenario.world.issue(lifecycle._ist("09:51"))
    emptied = tuple(
        lifecycle._INTERVAL(
            i.opens_at,
            i.closes_at,
            i.outcome if i.outcome is lifecycle._EI2A.NOT_ENDED else lifecycle._EI2A.MISSING,
            (),
            (),
        )
        for i in real.intervals
    )
    accepted = lifecycle.snapshot_module.EquityReferenceProvenanceKind.ACCEPTED_PRODUCTION
    production = lifecycle._rebuilt(
        real,
        emptied,
        observations=(),
        finalities=(),
        unmatched_sequences=(),
        venue_provenance=accepted,
        applicability_provenance=accepted,
        finality_policy=lifecycle.NO_APPROVED_EQUITY_CANDLE_FINALITY,
    )
    assert production.evidence_class.value == "PRODUCTION"
    return production


def _closed(recording: ei6b._Recording) -> None:
    ei6b._active(recording)
    recording.scenario.fill("09:50", "10:05", {"10:00": ("110", "111")})
    assert recording.record(recording.snap("10:06")).state is S.CLOSED_TARGET


def _candidate_id(recording: ei6b._Recording) -> str:
    return recording.scenario.disposition.candidate.source.evidence_snapshot_id


def _lifecycle_ids(view: EquityRecommendationDetailView) -> list[str]:
    return [
        p.snapshot_id for p in view.evidence if p.roles == (EquityEvidenceRole.LIFECYCLE_EVENT,)
    ]


# ----------------------------------------------------------------------
# Identity, primary authority and lifecycle presentation
# ----------------------------------------------------------------------


def test_d01_an_unknown_issuance_is_not_found_and_a_malformed_one_is_unsupported() -> None:
    recording = ei6b._recording()
    lifecycles = _Lifecycles(recording.store)
    use_case = _detail_use_case(recording, lifecycles=lifecycles)
    with pytest.raises(EquityReadQueryRefusedError) as missing:
        use_case.execute(_UNKNOWN_ID)
    assert (missing.value.reason, missing.value.retryable) == (_R.ISSUANCE_NOT_FOUND, False)
    for malformed in ("eiris1-sha256:xyz", "", 42):
        with pytest.raises(EquityReadQueryRefusedError) as raised:
            use_case.execute(malformed)  # type: ignore[arg-type]
        assert raised.value.reason is _R.UNSUPPORTED_SCOPE
    assert lifecycles.calls == [_UNKNOWN_ID]  # malformed identities never reach a reader


def test_d02_an_issuance_with_no_lifecycle_events_is_recorded_with_verified_facts() -> None:
    recording = ei6b._recording()
    view = _detail(recording)
    disposition = recording.world.store.load_disposition(recording.issued.generation_key)
    assert view.issuance_id == recording.issuance_id
    assert view.issuance.recommendation == recording.issued
    assert (view.state, view.is_terminal, view.outcome_class) == (S.RECORDED, False, None)
    assert (view.events, view.position) == ((), 0)
    assert view.issuance.recommendation.issuance_kind is EquityIssuanceKind.MODELED
    assert (view.is_synthetic, view.is_actionable) == (True, False)
    lineage = view.lineage
    assert lineage.generation_key == recording.issued.generation_key
    assert lineage.evaluator_version == EQUITY_RECOMMENDATION_EVALUATOR_V2
    assert lineage.policy_set == disposition.inputs.policy_set.reference
    assert lineage.policy_set_fingerprint == disposition.policy_set_fingerprint
    assert lineage.research_reference == disposition.candidate.source
    assert lineage.research_configuration == disposition.request.configuration
    assert (lineage.policy_publication_established, lineage.research_candidate_rehydrated) == (
        False,
        False,
    )
    # Sizing and modeled estimates are the stored values, exactly (int and Decimal kept).
    sizing = lineage.sizing
    assert sizing == disposition.proposal.definition.sizing
    assert type(sizing.quantity) is int and isinstance(sizing.net_risk, Decimal)
    assert [(p.snapshot_id, p.roles, p.status) for p in view.evidence] == [
        (_candidate_id(recording), (EquityEvidenceRole.CANDIDATE,), _STATUS.AVAILABLE)
    ]


def test_d03_an_active_modeled_lifecycle_lists_every_event_and_its_cited_evidence() -> None:
    recording = ei6b._recording()
    ei6b._active(recording)
    view = _detail(recording)
    stored = recording.store.load_lifecycle(recording.issuance_id)
    assert view.state is S.ACTIVE and not view.is_terminal
    assert view.events == stored.lifecycle.events and view.position == stored.position
    assert [e.sequence for e in view.events] == list(range(1, view.position + 1))
    cited = {e.evidence.reference for e in view.events}
    assert cited <= {p.snapshot_id for p in view.evidence}
    assert all(p.status is _STATUS.AVAILABLE for p in view.evidence)
    assert view.evidence[0].roles[0] is EquityEvidenceRole.CANDIDATE


def test_d04_a_terminal_modeled_lifecycle_states_its_outcome_class() -> None:
    recording = ei6b._recording()
    _closed(recording)
    view = _detail(recording)
    assert (view.state, view.is_terminal) == (S.CLOSED_TARGET, True)
    assert view.outcome_class is EquityLifecycleOutcomeClass.RESOLVED
    assert view.events[-1].kind is EquityLifecycleEventKind.TARGET_REACHED


def test_d05_terminal_incomplete_evidence_is_unresolved_with_its_deficiencies() -> None:
    recording = ei6b._recording()
    ei6b._active(recording)
    recording.scenario.fill("09:50", "14:00", {"12:00": None})
    assert recording.record(recording.snap("14:01")).state is S.INCOMPLETE_EVIDENCE
    view = _detail(recording)
    assert view.is_terminal and view.outcome_class is EquityLifecycleOutcomeClass.UNRESOLVED
    last = view.events[-1]
    assert last.kind is EquityLifecycleEventKind.EVIDENCE_INCOMPLETE and last.deficiencies


def test_d06_superseded_evidence_of_a_source_revision_is_cited_too() -> None:
    recording = ei6b._recording()
    _closed(recording)
    recording.scenario.put("10:00", ("105", "108"), revision="r2")
    assert recording.record(recording.snap("16:00")).new_events
    view = _detail(recording)
    revision = view.events[-1]
    assert revision.kind is EquityLifecycleEventKind.SOURCE_REVISION_RECORDED
    assert revision.superseded_evidence is not None
    panels = {p.snapshot_id for p in view.evidence}
    assert {revision.evidence.reference, revision.superseded_evidence.reference} <= panels


# ----------------------------------------------------------------------
# Exact generation lookup and the ten lineage comparisons (EI7-CF4)
# ----------------------------------------------------------------------


def test_d07_the_generation_is_read_once_under_the_exact_issued_key_only() -> None:
    recording = ei6b._recording()
    generations = _Generations(recording.world.store)
    lifecycles = _Lifecycles(recording.store)
    _detail(recording, generations=generations, lifecycles=lifecycles)
    assert generations.keys == [recording.issued.generation_key]
    assert lifecycles.calls == [recording.issuance_id]


@pytest.mark.parametrize(
    "changes, check",
    [
        ({"strategy": StrategyIdentity("ei7c-other-strategy")}, "2: strategy"),
        ({"listing": ListingReference(Symbol("RELIANCE"), ExchangeCode("NSE"))}, "3: listing"),
        ({"trading_date": date(2026, 8, 13)}, "4: trading date"),
        (
            {"timing_policy_version": EquitySyntheticVersion("synthetic-ei7c-other/1")},
            "7: modeled timing",
        ),
        ({"assessment_fingerprint": "eira1-sha256:" + "e" * 64}, "8: proposal and assessment"),
    ],
)
def test_d08_an_issuance_that_disagrees_with_its_generation_fails_the_whole_detail(
    changes: dict[str, Any], check: str
) -> None:
    recording = ei6b._recording()
    refused = _refused(
        recording, lifecycles=_Lifecycles(recording.store, _issuance_changed(**changes))
    )
    assert (refused.reason, refused.retryable) == (_R.PERSISTED_AUTHORITY_INVALID, False)
    assert check in str(refused)


def test_d08_reconstructed_levels_or_unassessed_controls_that_differ_fail_closed() -> None:
    recording = ei6b._recording()
    levels = recording.issued.levels
    target = Price(levels.target.amount + Decimal("1"), levels.target.currency)
    moved = dataclasses.replace(levels, target=target)
    refused = _refused(
        recording, lifecycles=_Lifecycles(recording.store, _issuance_changed(levels=moved))
    )
    assert "7: levels and direction" in str(refused)
    fewer = recording.issued.not_assessed[:1]
    refused = _refused(
        recording, lifecycles=_Lifecycles(recording.store, _issuance_changed(not_assessed=fewer))
    )
    assert "10: reconstructed issuance equivalence" in str(refused)


def test_d09_a_missing_or_non_proposed_or_foreign_disposition_fails_closed() -> None:
    recording = ei6b._recording()
    for change, check in (
        (lambda d: None, "1: the issuance's generation disposition is missing"),
        (lambda d: _clone(d, generation_key="eirk1-sha256:" + "f" * 64), "1: generation key"),
        (
            lambda d: _clone(d, kind=EquityRecommendationDispositionKind.NOT_ELIGIBLE),
            "1: the disposition is not PROPOSED",
        ),
        (
            lambda d: _clone(d, evaluator_version="northstar.equity-recommendation-evaluator/1"),
            "9: evaluator compatibility",
        ),
    ):
        refused = _refused(recording, generations=_Generations(recording.world.store, change))
        assert refused.reason is _R.PERSISTED_AUTHORITY_INVALID and check in str(refused)


def test_d09_policy_version_and_candidate_reference_mismatches_fail_closed() -> None:
    recording = ei6b._recording()
    other = EquitySyntheticVersion("synthetic-ei7c-risk/9")
    refused = _refused(
        recording,
        generations=_Generations(
            recording.world.store, lambda d: _with_proposal(d, risk_policy_version=other)
        ),
    )
    assert "5: policy component versions" in str(refused)
    source = recording.scenario.disposition.candidate.source
    elsewhere = dataclasses.replace(source, candidate_entry_fingerprint="eirt1-sha256:" + "c" * 64)
    refused = _refused(
        recording,
        generations=_Generations(
            recording.world.store, lambda d: _with_proposal(d, source=elsewhere)
        ),
    )
    assert "6: research candidate reference" in str(refused)


def _with_request(disposition: Any, **request_changes: Any) -> Any:
    changed = _clone(disposition.assessment.request, **request_changes)
    return _clone(disposition, assessment=_clone(disposition.assessment, request=changed))


def _with_proposal_fields(disposition: Any, **proposal_changes: Any) -> Any:
    changed = dataclasses.replace(disposition.proposal, **proposal_changes)
    return _clone(disposition, assessment=_clone(disposition.assessment, proposal=changed))


@pytest.mark.parametrize(
    "change, check",
    [
        (
            lambda d: _with_request(
                d,
                policy_set=dataclasses.replace(
                    d.request.policy_set, version=EquitySyntheticVersion("synthetic-ei7c-set/9")
                ),
            ),
            "5: applied policy set is the requested one",
        ),
        (
            lambda d: _with_proposal_fields(
                d, listing_eligibility_version=EquitySyntheticVersion("synthetic-ei7c-elig/9")
            ),
            "5: listing eligibility version",
        ),
        (
            lambda d: _with_proposal_fields(
                d, universe_version=EquitySyntheticVersion("synthetic-ei7c-universe/9")
            ),
            "5: universe version",
        ),
        (
            lambda d: _with_request(
                d,
                configuration=dataclasses.replace(
                    d.request.configuration, version="synthetic-ei7c-research/9"
                ),
            ),
            "5: research configuration version",
        ),
    ],
)
def test_d09_every_policy_and_version_identity_sub_check_fails_closed(
    change: Callable[[Any], Any], check: str
) -> None:
    recording = ei6b._recording()
    refused = _refused(recording, generations=_Generations(recording.world.store, change))
    assert (refused.reason, refused.retryable) == (_R.PERSISTED_AUTHORITY_INVALID, False)
    assert check in str(refused)


def test_d10_a_malformed_or_incompatible_generation_record_is_never_unknown() -> None:
    recording = ei6b._recording()
    for error in (
        InvalidEquityRecommendationRecordError("test: undecodable"),
        IncompatibleEquityRecommendationRecordError("test: evaluator /9"),
    ):
        generations = _Generations(recording.world.store)
        generations.error = error
        refused = _refused(recording, generations=generations)
        assert (refused.reason, refused.retryable) == (_R.PERSISTED_AUTHORITY_INVALID, False)
    wrong = _Generations(recording.world.store, lambda d: "not a disposition")
    assert _refused(recording, generations=wrong).reason is _R.PERSISTED_AUTHORITY_INVALID


def test_d10_a_generation_outage_is_a_retryable_dependency_failure() -> None:
    recording = ei6b._recording()
    generations = _Generations(recording.world.store)
    generations.error = EquityRecommendationGenerationStoreUnavailableError("test: down")
    refused = _refused(recording, generations=generations)
    assert (refused.reason, refused.retryable) == (_R.DEPENDENCY_UNAVAILABLE, True)


# ----------------------------------------------------------------------
# Lifecycle and issuance authority failures
# ----------------------------------------------------------------------


def test_d11_a_corrupt_lifecycle_history_fails_the_detail_distinctly() -> None:
    recording = ei6b._recording()
    ei6b._active(recording)
    recording.store.rows[recording.issuance_id][0] += " "  # non-canonical stored document
    refused = _refused(recording)
    assert (refused.reason, refused.retryable) == (_R.LIFECYCLE_HISTORY_INVALID, False)
    assert isinstance(refused.__cause__, InvalidStoredEquityRecommendationLifecycleError)


@pytest.mark.parametrize(
    "error, reason, retryable",
    [
        (
            InvalidStoredEquityRecommendationIssuanceAuthorityError("LEDGER", "test: gap"),
            _R.PERSISTED_AUTHORITY_INVALID,
            False,
        ),
        (
            EquityRecommendationLifecycleStoreUnavailableError("test: down"),
            _R.DEPENDENCY_UNAVAILABLE,
            True,
        ),
        (
            EquityRecommendationIssuanceUnavailableError("test: down"),
            _R.DEPENDENCY_UNAVAILABLE,
            True,
        ),
    ],
)
def test_d11_primary_authority_failures_are_classified(
    error: Exception, reason: EquityReadQueryRefusal, retryable: bool
) -> None:
    recording = ei6b._recording()
    lifecycles = _Lifecycles(recording.store)
    lifecycles.error = error
    refused = _refused(recording, lifecycles=lifecycles)
    assert (refused.reason, refused.retryable) == (reason, retryable)


def test_d11_a_lifecycle_reader_answering_another_issuance_is_invalid_authority() -> None:
    recording = ei6b._recording()
    key, proposal = "eirk1-sha256:" + "1" * 64, "eirp1-sha256:" + "2" * 64
    other = equity_recommendation_issuance_identity(key, proposal, EquityIssuanceKind.MODELED)
    change = _issuance_changed(generation_key=key, proposal_fingerprint=proposal, issuance_id=other)
    refused = _refused(recording, lifecycles=_Lifecycles(recording.store, change))
    assert refused.reason is _R.PERSISTED_AUTHORITY_INVALID


# ----------------------------------------------------------------------
# Optional evidence panels and the SYNTHETIC gate (EI7-EC1)
# ----------------------------------------------------------------------


def test_d12_a_missing_lifecycle_cited_snapshot_is_an_unavailable_panel() -> None:
    recording = ei6b._recording()
    ei6b._active(recording)
    cited = _lifecycle_ids(_detail(recording))[0]
    view = _detail(
        recording, snapshots=_Snapshots(recording.scenario.world.snapshots, {cited: None})
    )
    panel = next(p for p in view.evidence if p.snapshot_id == cited)
    assert (panel.status, panel.reason, panel.snapshot) == (
        _STATUS.UNAVAILABLE,
        _REASON.EVIDENCE_NOT_FOUND,
        None,
    )
    assert view.lineage.generation_key == recording.issued.generation_key  # required facts kept


def test_d12_a_missing_candidate_snapshot_is_a_broken_reference_panel() -> None:
    recording = ei6b._recording()
    candidate = _candidate_id(recording)
    view = _detail(
        recording, snapshots=_Snapshots(recording.scenario.world.snapshots, {candidate: None})
    )
    assert (view.evidence[0].status, view.evidence[0].reason) == (
        _STATUS.INVALID,
        _REASON.EVIDENCE_REFERENCE_BROKEN,
    )


def test_d12_corrupt_foreign_or_unreadable_snapshots_degrade_only_their_panels() -> None:
    recording = ei6b._recording()
    ei6b._active(recording)
    candidate = _candidate_id(recording)
    cited = _lifecycle_ids(_detail(recording))[0]
    store = recording.scenario.world.snapshots
    for answer, status, reason in (
        (
            InvalidStoredEquityCandleEvidenceSnapshotError("test: bad"),
            _STATUS.INVALID,
            _REASON.EVIDENCE_INVALID,
        ),
        (
            EquityCandleEvidenceUnavailableError("test: down"),
            _STATUS.UNAVAILABLE,
            _REASON.DEPENDENCY_UNAVAILABLE,
        ),
        (store.get_snapshot(candidate), _STATUS.INVALID, _REASON.EVIDENCE_INVALID),
    ):
        view = _detail(recording, snapshots=_Snapshots(store, {cited: answer}))
        panel = next(p for p in view.evidence if p.snapshot_id == cited)
        assert (panel.status, panel.reason, panel.snapshot) == (status, reason, None)
        assert view.state is S.ACTIVE


def test_d13_a_non_synthetic_snapshot_refuses_the_whole_detail_and_exposes_nothing() -> None:
    recording = ei6b._recording()
    ei6b._active(recording)
    production = _production(recording)
    store = recording.scenario.world.snapshots
    for snapshot_id in (_candidate_id(recording), _lifecycle_ids(_detail(recording))[0]):
        refused = _refused(recording, snapshots=_Snapshots(store, {snapshot_id: production}))
        assert (refused.reason, refused.retryable) == (_R.EVIDENCE_CLASS_UNSUPPORTED, False)
        text = str(refused)
        assert production.snapshot_id not in text and production.manifest_fingerprint not in text
        assert "PRODUCTION" not in text


# ----------------------------------------------------------------------
# Read-only boundary, consistency, vocabulary and immutability
# ----------------------------------------------------------------------


def test_d14_the_detail_needs_only_reader_operations_and_holds_independent_reads() -> None:
    recording = ei6b._recording()
    ei6b._active(recording)
    appends_before = recording.store.appends
    first = _detail(recording)
    # A later lifecycle event is visible to a later read; the earlier view is unchanged.
    recording.scenario.fill("09:50", "10:05", {"10:00": ("110", "111")})
    recording.record(recording.snap("10:06"))
    second = _detail(recording)
    assert first.state is S.ACTIVE and second.state is S.CLOSED_TARGET
    assert second.events[: first.position] == first.events
    assert recording.store.appends == appends_before + 1  # only the test's own recording
    for reader_type in (_Lifecycles, _Snapshots):
        assert not {"append", "issue", "write_generation"} & set(dir(reader_type))
    with pytest.raises(TypeError):
        GetEquityRecommendationDetailUseCase(object(), recording.world.store, _Snapshots(None))  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        GetEquityRecommendationDetailUseCase(_Lifecycles(None), object(), _Snapshots(None))  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        GetEquityRecommendationDetailUseCase(_Lifecycles(None), recording.world.store, object())  # type: ignore[arg-type]


_FORBIDDEN = (
    "fill",
    "pnl",
    "profit",
    "execut",
    "win_rate",
    "return",
    "realized",
    "recorded_at",
    "monitor",
    "performance",
)


def test_d15_no_view_claims_an_execution_fill_profit_or_recording_time() -> None:
    for view_type in (
        EquityRecommendationDetailView,
        EquityVerifiedGenerationLineage,
        EquityEvidencePanel,
        EquityLifecycleSummary,
    ):
        names = {f.name for f in dataclasses.fields(view_type)} | {
            name for name, _ in inspect.getmembers(view_type) if not name.startswith("_")
        }
        assert not [n for n in names for word in _FORBIDDEN if word in n.lower()], view_type


def test_d15_the_detail_view_is_immutable_and_refuses_inconsistent_construction() -> None:
    recording = ei6b._recording()
    ei6b._active(recording)
    view = _detail(recording)
    with pytest.raises(dataclasses.FrozenInstanceError):
        view.evidence = ()  # type: ignore[misc]
    assert isinstance(view.events, tuple) and isinstance(view.evidence, tuple)
    with pytest.raises(InvalidEquityRecommendationDetailViewError):
        dataclasses.replace(view, evidence=view.evidence[:1])
    with pytest.raises(InvalidEquityRecommendationDetailViewError):
        dataclasses.replace(
            view.evidence[0], status=_STATUS.AVAILABLE, reason=_REASON.EVIDENCE_NOT_FOUND
        )
    with pytest.raises(InvalidEquityRecommendationDetailViewError):
        EquityEvidencePanel(
            "ecs1-sha256:" + "a" * 64,
            (EquityEvidenceRole.CANDIDATE,),
            _STATUS.INVALID,
            _REASON.EVIDENCE_NOT_FOUND,
            None,
        )
    # StrEnum members equal their text: a reason given as text is refused, not coerced.
    with pytest.raises(InvalidEquityRecommendationDetailViewError):
        EquityEvidencePanel(
            "ecs1-sha256:" + "a" * 64,
            (EquityEvidenceRole.CANDIDATE,),
            _STATUS.INVALID,
            "EVIDENCE_INVALID",  # type: ignore[arg-type]
            None,
        )


# ----------------------------------------------------------------------
# The independent per-item lifecycle summary (register section 13.2, item 7)
# ----------------------------------------------------------------------


def test_d16_a_lifecycle_summary_is_an_independent_read_with_its_own_status() -> None:
    recording = ei6b._recording()
    ei6b._active(recording)
    summaries = GetEquityLifecycleSummaryUseCase(_Lifecycles(recording.store))
    summary = summaries.execute(recording.issuance_id)
    assert (summary.status, summary.state, summary.is_terminal, summary.outcome_class) == (
        EquityLifecycleSummaryStatus.AVAILABLE,
        S.ACTIVE,
        False,
        None,
    )
    assert summary.position == recording.store.load_lifecycle(recording.issuance_id).position
    recording.store.rows[recording.issuance_id][0] += " "
    invalid = summaries.execute(recording.issuance_id)
    assert (invalid.status, invalid.state, invalid.position) == (
        EquityLifecycleSummaryStatus.INVALID,
        None,
        None,
    )
    recording.store.unreadable = True
    assert (
        summaries.execute(recording.issuance_id).status is EquityLifecycleSummaryStatus.UNAVAILABLE
    )


def test_d16_a_summary_refuses_unknown_or_corrupt_issuance_authority_and_bad_input() -> None:
    recording = ei6b._recording()
    lifecycles = _Lifecycles(recording.store)
    summaries = GetEquityLifecycleSummaryUseCase(lifecycles)
    with pytest.raises(EquityReadQueryRefusedError) as missing:
        summaries.execute(_UNKNOWN_ID)
    assert missing.value.reason is _R.ISSUANCE_NOT_FOUND
    with pytest.raises(EquityReadQueryRefusedError) as malformed:
        summaries.execute("nonsense")
    assert malformed.value.reason is _R.UNSUPPORTED_SCOPE
    lifecycles.error = InvalidStoredEquityRecommendationIssuanceAuthorityError("LEDGER", "test")
    with pytest.raises(EquityReadQueryRefusedError) as corrupt:
        summaries.execute(recording.issuance_id)
    assert corrupt.value.reason is _R.PERSISTED_AUTHORITY_INVALID
    with pytest.raises(InvalidEquityRecommendationDetailViewError):
        EquityLifecycleSummary(
            recording.issuance_id, EquityLifecycleSummaryStatus.INVALID, S.ACTIVE, False, None, 1
        )
