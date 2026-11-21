"""EI-7C: the EI7-P4 reader-only compatibility change to EI-3C research rehydration.

``rehydrate_equity_intraday_research_timeline`` now accepts any
``EquityCandleEvidenceSnapshotReader`` instead of only the write-capable
``EquityCandleEvidenceSnapshotStore`` (EI-7 Decision Register, sections 13.5
and 13.12: EI7-P4, EI7-FC6). These tests prove what EI7-FC6 requires:
compliant and non-compliant implementations, structural acceptance, preserved
rehydration semantics, no write surface, and an unchanged write side.
``runtime_checkable`` validation alone proves nothing about behaviour, so
behaviour is exercised directly.

The world is the accepted EI-4C synthetic world (real committed timeline,
real EI-2B snapshots), reused unchanged.
"""

from __future__ import annotations

import pytest
import test_generate_equity_trade_recommendation as ei4c

from northstar_application.application_services import (
    AdvanceEquityIntradayResearchTimelineUseCase,
    EquityIntradayResearchOutcome,
    EquityIntradayResearchTimeline,
    EquityIntradayResearchTimelineIdentity,
    EquityIntradayResearchTimelineRole,
    rehydrate_equity_intraday_research_timeline,
)
from northstar_application.ports import (
    EquityCandleEvidenceSnapshotReader,
    EquityCandleEvidenceSnapshotStore,
    EquityCandleEvidenceUnavailableError,
    InvalidStoredEquityIntradayResearchTimelineError,
)

_IDENTITY = EquityIntradayResearchTimelineIdentity(ei4c._LISTING, ei4c._DAY, ei4c._CONFIG)


class _ReadOnlySnapshots:
    """A reader-only object: ``get_snapshot`` and nothing else, no write method at all."""

    def __init__(self, inner: object, answer: object = "delegate") -> None:
        self.inner, self.answer = inner, answer
        self.reads: list[str] = []

    def get_snapshot(self, snapshot_id: str):  # type: ignore[no-untyped-def]
        self.reads.append(snapshot_id)
        if self.answer == "delegate":
            return self.inner.get_snapshot(snapshot_id)  # type: ignore[attr-defined]
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


class _CountingStore(ei4c._Snapshots):
    """The EI-4C writer-capable test store, counting any issue during rehydration."""

    def __init__(self, inner: ei4c._Snapshots) -> None:
        super().__init__()
        self.documents = inner.documents
        self.issues = 0

    def issue(self, snapshot):  # type: ignore[no-untyped-def]
        self.issues += 1
        return super().issue(snapshot)


def _world() -> ei4c._World:
    return ei4c._committed()


def test_p4_01_an_existing_writer_capable_store_still_rehydrates_unchanged() -> None:
    world = _world()
    store = _CountingStore(world.snapshots)
    assert isinstance(store, EquityCandleEvidenceSnapshotStore)
    timeline = rehydrate_equity_intraday_research_timeline(_IDENTITY, world.timelines.only(), store)
    assert isinstance(timeline, EquityIntradayResearchTimeline) and timeline.entries
    assert store.issues == 0  # rehydration never issues


def test_p4_02_a_reader_only_object_rehydrates_identically_through_structural_acceptance() -> None:
    world = _world()
    stored = world.timelines.only()
    reader = _ReadOnlySnapshots(world.snapshots)
    assert isinstance(reader, EquityCandleEvidenceSnapshotReader)
    assert not isinstance(reader, EquityCandleEvidenceSnapshotStore)
    via_store = rehydrate_equity_intraday_research_timeline(_IDENTITY, stored, world.snapshots)
    via_reader = rehydrate_equity_intraday_research_timeline(_IDENTITY, stored, reader)
    assert via_reader == via_store
    assert [e.fingerprint for e in via_reader.entries] == [
        s.entry_fingerprint for s in stored.entries
    ]
    assert reader.reads == [s.snapshot_id for s in stored.entries]


@pytest.mark.parametrize("candidate", [object(), "snapshots", None, 42])
def test_p4_03_a_non_compliant_object_is_still_refused_with_type_error(candidate: object) -> None:
    world = _world()
    with pytest.raises(TypeError, match="EquityCandleEvidenceSnapshotReader"):
        rehydrate_equity_intraday_research_timeline(_IDENTITY, world.timelines.only(), candidate)  # type: ignore[arg-type]


def test_p4_03_a_reader_that_answers_none_or_fails_keeps_the_ei3d13_semantics() -> None:
    world = _world()
    stored = world.timelines.only()
    with pytest.raises(InvalidStoredEquityIntradayResearchTimelineError, match="unavailable"):
        rehydrate_equity_intraday_research_timeline(
            _IDENTITY, stored, _ReadOnlySnapshots(world.snapshots, None)
        )
    outage = EquityCandleEvidenceUnavailableError("test: down")
    with pytest.raises(EquityCandleEvidenceUnavailableError):
        rehydrate_equity_intraday_research_timeline(
            _IDENTITY, stored, _ReadOnlySnapshots(world.snapshots, outage)
        )


def test_p4_04_no_write_surface_and_no_fake_issue_anywhere() -> None:
    assert "issue" not in dir(EquityCandleEvidenceSnapshotReader)
    assert not hasattr(_ReadOnlySnapshots, "issue")


def test_p4_05_research_vocabulary_is_unchanged() -> None:
    assert {r.value for r in EquityIntradayResearchTimelineRole} == {
        "CANDIDATE",
        "REPEAT_QUALIFYING",
        "NO_SIGNAL",
        "REJECTED",
    }
    assert {o.value for o in EquityIntradayResearchOutcome} == {
        "LONG_CANDIDATE",
        "NO_SIGNAL",
        "REJECTED",
    }


def test_p4_06_the_write_side_still_requires_the_writer_capable_store() -> None:
    world = _world()
    with pytest.raises(TypeError):
        AdvanceEquityIntradayResearchTimelineUseCase(
            world.timelines,
            _ReadOnlySnapshots(world.snapshots),  # type: ignore[arg-type]
            ei4c._CONFIG,
        )
