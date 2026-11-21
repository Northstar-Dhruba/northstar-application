"""EI-7C Application: persisted research inspection (EI7-P3, EI7-P5, EI7-FC2, EI7-FC3).

Application contract tests for the EI-7 Decision Register, sections 13.4 and
13.12, and design section 23.5. The worlds are the accepted synthetic EI-4C
and EI-5E chains, reused unchanged: real committed EI-3C timelines, real EI-2B
snapshots, a real evaluator-/2 disposition and, where issued, a real MODELED
issuance. Read wrappers are TEST-ONLY and expose read operations only.

Nothing reads a clock, opens a connection or writes anything.
"""

from __future__ import annotations

import copy
import dataclasses
import inspect
from datetime import datetime
from decimal import Decimal
from typing import Any

import pytest
import test_equity_recommendation_issuance_end_to_end as e2e
import test_generate_equity_trade_recommendation as ei4c
import test_issue_equity_trade_recommendation as ei5d
import test_read_equity_recommendation_detail as detail
import test_record_equity_recommendation_generation as ei4d
import test_record_equity_recommendation_lifecycle as ei6b
from northstar_core.domain.value_objects import ListingReference
from northstar_core.foundation.value_objects import ExchangeCode
from northstar_core.strategy import StrategyIdentity

from northstar_application.application_services import (
    EquityGenerationNotQueryableReason,
    EquityIntradayResearchOutcome,
    EquityIntradayResearchRejection,
    EquityIntradayResearchTimelineIdentity,
    EquityIntradayResearchTimelineRole,
    EquityOpeningRangeBreakoutEvaluator,
    EquityReadQueryRefusal,
    EquityReadQueryRefusedError,
    EquityResearchGenerationStatus,
    EquityResearchIssuanceStatus,
    EquityResearchTimelineStatus,
    EquityResearchTimelineView,
    GetEquityResearchTimelineUseCase,
    InvalidEquityResearchTimelineViewError,
    canonical_equity_candle_json,
    rehydrate_equity_intraday_research_timeline,
)
from northstar_application.ports import (
    EquityCandleEvidenceUnavailableError,
    EquityIntradayResearchTimelineUnavailableError,
    EquityRecommendationGenerationStoreUnavailableError,
    EquityRecommendationIssuanceUnavailableError,
    InvalidStoredEquityCandleEvidenceSnapshotError,
    InvalidStoredEquityIntradayResearchTimelineError,
    InvalidStoredEquityRecommendationIssuanceAuthorityError,
)

_R = EquityReadQueryRefusal
_ROLE = EquityIntradayResearchTimelineRole
_G = EquityResearchGenerationStatus
_Q = EquityGenerationNotQueryableReason
_STRATEGY = ei4c._CONFIG.strategy
_LISTING = ei4c._LISTING
_DAY = ei4c._DAY


class _Timelines:
    """A read-only view of a timeline source; ``change`` maps the stored record."""

    def __init__(self, inner: Any, change: Any = None) -> None:
        self.inner, self.change, self.error = inner, change, None
        self.loads = 0

    def load(self, strategy, listing, trading_date):  # type: ignore[no-untyped-def]
        self.loads += 1
        if self.error is not None:
            raise self.error
        stored = self.inner.load(strategy, listing, trading_date)
        return stored if self.change is None else self.change(stored)


class _Issuances(ei5d._Authority):
    """The labelled EI-5D test authority, read through; ``error`` makes reads fail."""

    def __init__(self, inner: ei5d._Authority) -> None:
        super().__init__()
        self.issuances, self.declarations, self.sequence = (
            inner.issuances,
            inner.declarations,
            inner.sequence,
        )
        self.error: Exception | None = None

    def load_session_issuance(self, strategy, listing, trading_date):  # type: ignore[no-untyped-def]
        if self.error is not None:
            raise self.error
        return super().load_session_issuance(strategy, listing, trading_date)


@dataclasses.dataclass
class _Research:
    timelines: Any
    snapshots: Any
    issuances: Any
    generations: Any
    world: Any

    def use_case(self, **changes: Any) -> GetEquityResearchTimelineUseCase:
        readers = {
            "timelines": self.timelines,
            "snapshots": self.snapshots,
            "issuances": self.issuances,
            "generations": self.generations,
            **changes,
        }
        return GetEquityResearchTimelineUseCase(**readers)

    def view(
        self, listing: ListingReference = _LISTING, **changes: Any
    ) -> EquityResearchTimelineView:
        return self.use_case(**changes).execute(_STRATEGY, listing, _DAY)

    def refused(self, **changes: Any) -> EquityReadQueryRefusedError:
        with pytest.raises(EquityReadQueryRefusedError) as raised:
            self.view(**changes)
        return raised.value


def _research(world: Any, authority: Any, store: Any) -> _Research:
    return _Research(
        _Timelines(world.timelines),
        detail._Snapshots(world.snapshots),
        _Issuances(authority),
        detail._Generations(store),
        world,
    )


def _unissued() -> _Research:
    """The EI-5E chain: a recorded PROPOSED disposition, nothing issued."""
    issuance, scenario = e2e._chain()
    return _research(scenario.world, issuance.authority, issuance.store)


def _issued() -> tuple[_Research, ei6b._Recording]:
    recording = ei6b._recording()
    research = _research(recording.scenario.world, recording.world.authority, recording.world.store)
    return research, recording


def _stored(research: _Research) -> Any:
    return research.world.timelines.load(_STRATEGY, _LISTING, _DAY)


# ----------------------------------------------------------------------
# Timeline status and stored research outcomes
# ----------------------------------------------------------------------


def test_r01_no_persisted_timeline_is_unknown_and_generation_not_queryable() -> None:
    research = _unissued()
    view = research.view(listing=ei4c._OTHER)
    assert view.timeline_status is EquityResearchTimelineStatus.UNKNOWN
    assert (view.configuration, view.entries, view.candidate_entry) == (None, (), None)
    assert (view.generation_status, view.not_queryable_reason) == (
        _G.NOT_QUERYABLE,
        _Q.NO_RESEARCH_TIMELINE,
    )
    assert (view.issuance_status, view.issuance, view.lineage) == (
        EquityResearchIssuanceStatus.NOT_RECORDED,
        None,
        None,
    )
    assert research.generations.keys == []


def test_r02_a_stored_unissued_candidate_is_not_queryable_without_any_generation_search() -> None:
    research = _unissued()
    view = research.view()
    assert view.timeline_status is EquityResearchTimelineStatus.STORED
    assert [e.role for e in view.entries] == [
        _ROLE.NO_SIGNAL,
        _ROLE.CANDIDATE,
        _ROLE.REPEAT_QUALIFYING,
    ]
    assert view.candidate_entry is view.entries[1]
    assert (view.generation_status, view.not_queryable_reason) == (
        _G.NOT_QUERYABLE,
        _Q.NO_RECORDED_ISSUANCE,
    )
    assert view.issuance_status is EquityResearchIssuanceStatus.NOT_RECORDED
    # A PROPOSED disposition is stored, yet nothing looks it up: no key is guessed.
    assert research.generations.keys == []


def test_r03_stored_outcomes_and_reasons_are_shown_exactly_as_persisted() -> None:
    research = _unissued()
    view = research.view()
    stored = _stored(research)
    assert [
        (e.sequence, e.role, e.decision.outcome, e.decision.fingerprint, e.fingerprint)
        for e in view.entries
    ] == [
        (s.sequence, s.role, s.decision_outcome, s.decision_fingerprint, s.entry_fingerprint)
        for s in stored.entries
    ]
    no_signal = view.entries[0]
    assert no_signal.decision.outcome is EquityIntradayResearchOutcome.NO_SIGNAL
    assert no_signal.decision.reasons  # the authoritative reasons, covered by the fingerprint
    assert (
        canonical_equity_candle_json(no_signal.decision.document())
        == stored.entries[0].decision_document
    )


def test_r04_a_rejected_timeline_shows_its_rejections_and_has_no_candidate() -> None:
    gap = ei4c._World({k: v for k, v in ei4c._MORNING.items() if k != 2})  # bar 2 missing
    gap.commit(ei4c._ist(9, 31), ei4c._ist(9, 36), ei4c._ist(9, 41))
    research = _research(gap, ei5d._Authority(), ei4d._Store())
    view = research.view()
    assert {e.role for e in view.entries} == {_ROLE.REJECTED}
    assert all(e.decision.outcome is EquityIntradayResearchOutcome.REJECTED for e in view.entries)
    assert EquityIntradayResearchRejection.MISSING_CANDLE in view.entries[0].decision.reasons
    assert (view.generation_status, view.not_queryable_reason) == (
        _G.NOT_QUERYABLE,
        _Q.NO_CANDIDATE_ENTRY,
    )


# ----------------------------------------------------------------------
# Issuance-linked generation (EI7-P3)
# ----------------------------------------------------------------------


def test_r05_an_issued_candidate_is_proposed_and_issued_with_verified_lineage() -> None:
    research, recording = _issued()
    view = research.view()
    assert (view.generation_status, view.not_queryable_reason) == (_G.PROPOSED, None)
    assert view.issuance_status is EquityResearchIssuanceStatus.ISSUED
    assert view.issuance is not None
    assert view.issuance.recommendation == recording.issued
    assert view.lineage is not None and view.lineage.issuance == view.issuance
    candidate = view.candidate_entry
    assert candidate is not None
    assert view.lineage.research_reference.candidate_entry_fingerprint == candidate.fingerprint
    assert view.lineage.research_configuration.document() == view.configuration.document()
    assert research.generations.keys == [recording.issued.generation_key]  # the exact key only


def test_r06_an_issuance_without_its_persisted_timeline_fails_closed() -> None:
    research, _ = _issued()
    refused = research.refused(timelines=_Timelines(research.world.timelines, lambda s: None))
    assert (refused.reason, refused.retryable) == (_R.PERSISTED_AUTHORITY_INVALID, False)


def test_r07_an_issuance_whose_timeline_holds_no_candidate_fails_closed() -> None:
    research, _ = _issued()
    stored = _stored(research)
    assert stored.entries[0].role is _ROLE.NO_SIGNAL
    first_only = _Timelines(
        research.world.timelines, lambda s: dataclasses.replace(s, entries=s.entries[:1])
    )
    refused = research.refused(timelines=first_only)
    assert refused.reason is _R.PERSISTED_AUTHORITY_INVALID and "CANDIDATE" in str(refused)


def test_r08_an_issuance_whose_generation_cites_another_candidate_entry_fails_closed() -> None:
    # A self-consistent lineage (all ten comparisons hold) against a verified timeline whose
    # CANDIDATE entry is a different one: the breakout bar closes at 102.6, not 102.5.
    research, _ = _issued()
    moved = dict(ei4c._MORNING)
    moved[4] = ("101", "103", "101", "102.6", "2000")
    other = ei4c._World({index: moved[index] for index in range(1, 6)})
    other.commit(ei4c._ist(9, 31), ei4c._ist(9, 36), ei4c._ist(9, 41))
    other_candidate = next(e for e in other.timelines.only().entries if e.role is _ROLE.CANDIDATE)
    assert (
        other_candidate.entry_fingerprint
        != next(e for e in _stored(research).entries if e.role is _ROLE.CANDIDATE).entry_fingerprint
    )
    refused = research.refused(
        timelines=_Timelines(other.timelines), snapshots=detail._Snapshots(other.snapshots)
    )
    assert refused.reason is _R.PERSISTED_AUTHORITY_INVALID
    assert "does not cite the timeline's CANDIDATE entry" in str(refused)


def test_r09_a_generation_under_another_research_configuration_fails_closed() -> None:
    research, _ = _issued()

    def other_configuration(disposition: Any) -> Any:
        request = disposition.assessment.request
        configuration = dataclasses.replace(request.configuration, volume_multiplier=Decimal("2"))
        changed = detail._clone(request, configuration=configuration)
        return detail._clone(
            disposition, assessment=detail._clone(disposition.assessment, request=changed)
        )

    generations = detail._Generations(research.generations.inner, other_configuration)
    refused = research.refused(generations=generations)
    assert "used another research configuration" in str(refused)


def test_r09_a_missing_or_unreadable_generation_of_an_issuance_is_never_unknown() -> None:
    research, _ = _issued()
    refused = research.refused(
        generations=detail._Generations(research.generations.inner, lambda d: None)
    )
    assert refused.reason is _R.PERSISTED_AUTHORITY_INVALID
    down = detail._Generations(research.generations.inner)
    down.error = EquityRecommendationGenerationStoreUnavailableError("test: down")
    assert research.refused(generations=down).reason is _R.DEPENDENCY_UNAVAILABLE


# ----------------------------------------------------------------------
# Stored configuration (EI7-FC3)
# ----------------------------------------------------------------------


def test_r10_rehydration_uses_only_the_stored_configuration() -> None:
    research = _unissued()
    view = research.view()
    assert view.configuration is not None
    assert (
        canonical_equity_candle_json(view.configuration.document())
        == _stored(research).configuration_document
    )
    signature = inspect.signature(GetEquityResearchTimelineUseCase.execute)
    assert list(signature.parameters) == ["self", "strategy", "listing", "trading_date"]
    constructor = inspect.signature(GetEquityResearchTimelineUseCase.__init__)
    assert list(constructor.parameters) == [
        "self",
        "timelines",
        "snapshots",
        "issuances",
        "generations",
    ]


@pytest.mark.parametrize(
    "document",
    [
        "{}",
        "not json",
        '{"volume_multiplier": 1.5}',
        "[]",
    ],
)
def test_r11_a_missing_or_undecodable_stored_configuration_fails_the_response(
    document: str,
) -> None:
    research = _unissued()
    changed = _Timelines(
        research.world.timelines, lambda s: dataclasses.replace(s, configuration_document=document)
    )
    refused = research.refused(timelines=changed)
    assert (refused.reason, refused.retryable) == (_R.PERSISTED_AUTHORITY_INVALID, False)


def test_r11_a_non_canonical_or_inconsistent_stored_configuration_fails_the_response() -> None:
    research = _unissued()
    text = _stored(research).configuration_document
    spaced = text.replace(",", ", ")
    assert spaced != text
    another_version = canonical_equity_candle_json(
        dataclasses.replace(ei4c._CONFIG, version="synthetic-ei4c-research/2").document()
    )
    another_strategy = canonical_equity_candle_json(
        dataclasses.replace(ei4c._CONFIG, strategy=StrategyIdentity("ei7c-other")).document()
    )
    for document in (spaced, another_version, another_strategy):
        changed = _Timelines(
            research.world.timelines,
            lambda s, d=document: dataclasses.replace(s, configuration_document=d),
        )
        assert research.refused(timelines=changed).reason is _R.PERSISTED_AUTHORITY_INVALID


# ----------------------------------------------------------------------
# The SYNTHETIC gate before any outcome presentation (EI7-EC1, EI7-FC3)
# ----------------------------------------------------------------------


def test_r12_a_non_synthetic_snapshot_is_refused_before_it_could_become_a_rejected_outcome() -> (
    None
):
    issuance, scenario = e2e._chain()
    research = _research(scenario.world, issuance.authority, issuance.store)
    production = detail._production(SimpleRecording(scenario))
    # Without the gate, the current evaluator would present it as an ordinary REJECTED decision.
    decision = EquityOpeningRangeBreakoutEvaluator(ei4c._CONFIG).evaluate(
        _LISTING, _DAY, production, production.snapshot_id
    )
    assert decision.outcome is EquityIntradayResearchOutcome.REJECTED
    assert EquityIntradayResearchRejection.EVIDENCE_NOT_SYNTHETIC in decision.reasons
    first = _stored(research).entries[0].snapshot_id
    snapshots = detail._Snapshots(scenario.world.snapshots, {first: production})
    refused = research.refused(snapshots=snapshots)
    assert (refused.reason, refused.retryable) == (_R.EVIDENCE_CLASS_UNSUPPORTED, False)
    assert snapshots.reads == [first]  # refused at the first gated snapshot, before rehydration
    assert production.snapshot_id not in str(refused) and "REJECTED" not in str(refused)


@dataclasses.dataclass
class SimpleRecording:
    """Adapts an EI-5E scenario to the detail tests' production-snapshot builder."""

    scenario: Any


@pytest.mark.parametrize(
    "answer, reason",
    [
        (None, _R.PERSISTED_AUTHORITY_INVALID),
        (
            InvalidStoredEquityCandleEvidenceSnapshotError("test: bad"),
            _R.PERSISTED_AUTHORITY_INVALID,
        ),
        (EquityCandleEvidenceUnavailableError("test: down"), _R.DEPENDENCY_UNAVAILABLE),
        ("not a snapshot", _R.PERSISTED_AUTHORITY_INVALID),
    ],
)
def test_r13_a_missing_corrupt_or_unreadable_research_snapshot_is_classified(
    answer: Any, reason: EquityReadQueryRefusal
) -> None:
    research = _unissued()
    first = _stored(research).entries[0].snapshot_id
    refused = research.refused(
        snapshots=detail._Snapshots(research.world.snapshots, {first: answer})
    )
    assert refused.reason is reason


def test_r13_a_snapshot_answering_another_identity_is_invalid_research_provenance() -> None:
    research = _unissued()
    entries = _stored(research).entries
    other = research.world.snapshots.get_snapshot(entries[1].snapshot_id)
    snapshots = detail._Snapshots(research.world.snapshots, {entries[0].snapshot_id: other})
    assert research.refused(snapshots=snapshots).reason is _R.PERSISTED_AUTHORITY_INVALID


# ----------------------------------------------------------------------
# Reader failures, scope, read-only boundary
# ----------------------------------------------------------------------


def test_r14_timeline_and_issuance_reader_failures_are_classified() -> None:
    research = _unissued()
    timelines = _Timelines(research.world.timelines)
    timelines.error = EquityIntradayResearchTimelineUnavailableError("test: down")
    assert research.refused(timelines=timelines).reason is _R.DEPENDENCY_UNAVAILABLE
    timelines.error = InvalidStoredEquityIntradayResearchTimelineError("test: bad")
    assert research.refused(timelines=timelines).reason is _R.PERSISTED_AUTHORITY_INVALID
    elsewhere = _Timelines(
        research.world.timelines,
        lambda s: dataclasses.replace(s, trading_date=ei4c._DAY.replace(day=13)),
    )
    assert research.refused(timelines=elsewhere).reason is _R.PERSISTED_AUTHORITY_INVALID
    research.issuances.error = InvalidStoredEquityRecommendationIssuanceAuthorityError(
        "LEDGER", "t"
    )
    assert research.refused().reason is _R.PERSISTED_AUTHORITY_INVALID
    research.issuances.error = EquityRecommendationIssuanceUnavailableError("test: down")
    refused = research.refused()
    assert (refused.reason, refused.retryable) == (_R.DEPENDENCY_UNAVAILABLE, True)


@pytest.mark.parametrize(
    "strategy, listing, day",
    [
        ("ei4c-orb-vwap-volume", _LISTING, _DAY),
        (_STRATEGY, ListingReference(_LISTING.symbol, ExchangeCode("BSE")), _DAY),
        (_STRATEGY, _LISTING, datetime(2026, 8, 12, 9, 15)),
    ],
)
def test_r15_an_unsupported_or_malformed_scope_is_refused_before_any_read(
    strategy: Any, listing: Any, day: Any
) -> None:
    research = _unissued()
    with pytest.raises(EquityReadQueryRefusedError) as raised:
        research.use_case().execute(strategy, listing, day)
    assert raised.value.reason is _R.UNSUPPORTED_SCOPE
    assert research.timelines.loads == 0


def test_r16_research_inspection_writes_nothing_and_never_searches_generations() -> None:
    research, recording = _issued()
    records_before = dict(research.world.timelines.records)
    documents_before = dict(research.world.snapshots.documents)
    issuances_before = list(recording.world.authority.issuances)
    research.view()
    research.view(listing=ei4c._OTHER)
    assert research.world.timelines.records == records_before
    assert research.world.snapshots.documents == documents_before
    assert recording.world.authority.issuances == issuances_before
    assert research.generations.keys == [recording.issued.generation_key]  # load_attempt never


def test_r17_the_view_never_states_an_unsupported_generation_status() -> None:
    research = _unissued()
    view = research.view()
    for status in (_G.UNKNOWN, _G.NOT_ELIGIBLE, _G.UNRESOLVED):
        with pytest.raises(InvalidEquityResearchTimelineViewError):
            dataclasses.replace(view, generation_status=status, not_queryable_reason=None)
    with pytest.raises(InvalidEquityResearchTimelineViewError):
        dataclasses.replace(view, not_queryable_reason=_Q.NO_CANDIDATE_ENTRY)
    with pytest.raises(InvalidEquityResearchTimelineViewError):
        dataclasses.replace(view, timeline_status=EquityResearchTimelineStatus.UNKNOWN)
    with pytest.raises(dataclasses.FrozenInstanceError):
        view.entries = ()  # type: ignore[misc]
    assert len({s.value for s in _G}) == 5  # distinct vocabulary, nothing merged


def test_r17_timeline_and_generation_statuses_cannot_be_confused_or_given_as_text() -> None:
    # StrEnum members equal their text, so the types are enforced, not only the values.
    research = _unissued()
    view = research.view()
    for changes in (
        {"generation_status": "NOT_QUERYABLE"},
        {"generation_status": EquityResearchTimelineStatus.STORED},
        {"timeline_status": "STORED"},
        {"timeline_status": _G.UNKNOWN},
        {"issuance_status": "NOT_RECORDED"},
        {"not_queryable_reason": "NO_RECORDED_ISSUANCE"},
    ):
        with pytest.raises(InvalidEquityResearchTimelineViewError):
            dataclasses.replace(view, **changes)


def test_r18_the_rehydrated_view_equals_direct_ei3c_rehydration() -> None:
    research = _unissued()
    view = research.view()
    stored = _stored(research)
    identity = EquityIntradayResearchTimelineIdentity(_LISTING, _DAY, ei4c._CONFIG)
    direct = rehydrate_equity_intraday_research_timeline(identity, stored, research.world.snapshots)
    assert view.entries == direct.entries
    assert copy.copy(view) == view
