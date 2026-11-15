"""EI-6C Application PR-1: lifecycle monitoring and snapshot selection.

Application-layer evidence for the EI-6C Lifecycle Monitoring and Snapshot
Selection Design v1.0 (MS-01 to MS-40, all PLANNED in the governance). Test
names carry the scenario identifiers they contribute to. Real-SQLite catalog
behaviour, separate OS processes and Linux CI belong to EI-6C Infrastructure
PR-2 and final acceptance, and are not claimed here.

Everything is synthetic and in memory:

- the issued recommendation, the EI-2B snapshots and the EI-6B recorder are
  the real chain of the EI-6B Application tests (``ei6b._recording()``);
- ``_Catalog`` is a LABELLED SYNTHETIC IN-MEMORY TEST CATALOG -- not durable,
  not an adapter. It projects the verified snapshots of the in-memory
  snapshot store, applies the port's scope, eligibility and limit contract,
  and can inject unavailability, incompleteness, tampering and extra entries;
- ``_SpyRecorder`` is the real recording use case, recording the snapshot
  identities of each call and optionally raising an injected failure.

Nothing reads a clock, opens a connection or calls out.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import test_evaluate_synthetic_equity_recommendation_lifecycle as lifecycle
import test_generate_equity_trade_recommendation as ei4c
import test_record_equity_recommendation_lifecycle as ei6b
from northstar_core.foundation.value_objects import PointInTime, Timeframe
from northstar_core.recommendations.equity.lifecycle import (
    EquityLifecycleEventKind,
    EquityLifecycleState,
    InvalidEquityRecommendationLifecycleError,
)

from northstar_application import application_services, ports
from northstar_application.application_services import (
    EQUITY_LIFECYCLE_MONITORING_CATALOG_LIMIT,
    EQUITY_LIFECYCLE_MONITORING_POLICY,
    NSE_EQUITY_INTRADAY_V1_RESEARCH_WINDOW,
    EquityCandleEvidenceClass,
    EquityLifecycleMonitoringDeclaredSessionScope,
    EquityLifecycleMonitoringDiagnosticKind,
    EquityLifecycleMonitoringExplicitScope,
    EquityLifecycleMonitoringOutcome,
    EquityLifecycleMonitoringRefusal,
    EquityLifecycleMonitoringReport,
    EquityLifecycleMonitoringRequest,
    EquityLifecycleMonitoringResult,
    EquityLifecycleMonitoringScopeError,
    EquityLifecycleMonitoringStrategyListingScope,
    EquityLifecycleRecordingOutcome,
    EquityLifecycleRecordingRefusal,
    EquityLifecycleRecordingRefusedError,
    EquityLifecycleSelectionStop,
    EquityLifecycleSnapshotAnchor,
    EvaluateEquityResearchWindowUseCase,
    MonitorEquityRecommendationLifecyclesUseCase,
    RecordEquityRecommendationLifecycleUseCase,
    equity_lifecycle_event_document,
    equity_snapshot_eligible,
    select_equity_lifecycle_snapshots,
)
from northstar_application.application_services import (
    equity_lifecycle_monitoring_selection as selection_module,
)
from northstar_application.application_services import (
    monitor_equity_recommendation_lifecycles as monitor_module,
)
from northstar_application.ports import (
    EquityCandleEvidenceSnapshotCatalog,
    EquityCandleEvidenceSnapshotCatalogEntry,
    EquityCandleEvidenceSnapshotCatalogLimitExceededError,
    EquityCandleEvidenceSnapshotCatalogScope,
    EquityCandleEvidenceSnapshotCatalogUnavailableError,
    EquityRecommendationIssuanceReader,
    EquityRecommendationIssuanceUnavailableError,
    EquityRecommendationLifecycleReader,
    InvalidEquityCandleEvidenceSnapshotCatalogError,
    InvalidStoredEquityRecommendationIssuanceAuthorityError,
    UnverifiedEquityRecommendationLifecycleAppendError,
)

K = EquityLifecycleEventKind
S = EquityLifecycleState
M = EquityLifecycleMonitoringOutcome
RO = EquityLifecycleRecordingOutcome
R = EquityLifecycleRecordingRefusal
MR = EquityLifecycleMonitoringRefusal
_Catalogue = InvalidEquityCandleEvidenceSnapshotCatalogError
_FIVE = Timeframe("5m")
_SRC = Path(application_services.__file__).parent
_PORTS = Path(ports.__file__).parent


def _pit(clock: str) -> PointInTime:
    return lifecycle._pit(clock)


# ----------------------------------------------------------------------
# The labelled synthetic in-memory test catalog and the spying recorder
# ----------------------------------------------------------------------


def _project(snapshot: Any) -> EquityCandleEvidenceSnapshotCatalogEntry:
    return EquityCandleEvidenceSnapshotCatalogEntry(
        snapshot.snapshot_id,
        snapshot.listing,
        snapshot.trading_date,
        snapshot.timeframe,
        snapshot.window_policy_version,
        snapshot.evidence_class,
        snapshot.as_of,
        snapshot.issued_at,
        snapshot.watermark,
        snapshot.manifest_fingerprint,
    )


class _Catalog(EquityCandleEvidenceSnapshotCatalog):
    """LABELLED SYNTHETIC IN-MEMORY TEST CATALOG -- not durable, not an adapter.

    It projects every verified snapshot of the in-memory snapshot store and
    applies the port contract: scope, ``as_of <= T`` and ``issued_at <= T``
    on instants, and the limit (raised, never truncated). ``unavailable``,
    ``incomplete``, ``tamper`` and ``extra`` inject faults.
    """

    def __init__(self, snapshots: Any) -> None:
        self.snapshots = snapshots
        self.unavailable = False
        self.incomplete = False
        self.tamper: Callable[[Any], Any] | None = None
        self.extra: tuple[Any, ...] = ()
        self.answer_list = False
        self.calls: list[tuple[Any, PointInTime, int]] = []

    def list_snapshots(self, scope, monitoring_instant, limit):  # type: ignore[no-untyped-def]
        self.calls.append((scope, monitoring_instant, limit))
        if self.unavailable:
            raise EquityCandleEvidenceSnapshotCatalogUnavailableError("test: catalog unavailable")
        if self.incomplete:
            raise _Catalogue(_Catalogue.INCOMPLETE, "test: no consistent view")
        entries = []
        for snapshot_id in sorted(self.snapshots.documents):
            entry = _project(self.snapshots.get_snapshot(snapshot_id))
            if selection_module._in_scope(entry, scope) and equity_snapshot_eligible(
                entry, monitoring_instant
            ):
                entries.append(entry)
        entries += list(self.extra)
        if len(entries) > limit:
            raise EquityCandleEvidenceSnapshotCatalogLimitExceededError(
                limit, f"test: {len(entries)} snapshots exceed {limit}"
            )
        if self.tamper is not None:
            entries = [self.tamper(entry) for entry in entries]
        return entries if self.answer_list else tuple(entries)


class _SpyRecorder(RecordEquityRecommendationLifecycleUseCase):
    """The real recording use case, keeping each call's snapshot identities."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.calls: list[tuple[str, ...]] = []
        self.fail_at: int | None = None
        self.failure: BaseException | None = None

    def execute(self, issuance_id, snapshot_ids):  # type: ignore[no-untyped-def]
        self.calls.append(tuple(snapshot_ids))
        if self.fail_at is not None and len(self.calls) == self.fail_at:
            assert self.failure is not None
            raise self.failure
        return super().execute(issuance_id, snapshot_ids)


@dataclasses.dataclass
class _Monitoring:
    recording: ei6b._Recording
    catalog: _Catalog
    recorder: _SpyRecorder

    @property
    def issuance_id(self) -> str:
        return self.recording.issuance_id

    @property
    def scenario(self) -> Any:
        return self.recording.scenario

    def use_case(self, issuances: Any = None) -> MonitorEquityRecommendationLifecyclesUseCase:
        return MonitorEquityRecommendationLifecyclesUseCase(
            issuances or self.recording.world.authority,
            self.recording.world.store,
            self.scenario.world.snapshots,
            self.catalog,
            self.recording.store,
            self.recorder,
        )

    def run(self, clock: str, scope: Any = None, **options: Any) -> EquityLifecycleMonitoringReport:
        request = EquityLifecycleMonitoringRequest(
            scope or EquityLifecycleMonitoringExplicitScope((self.issuance_id,)), _pit(clock)
        )
        return self.use_case(**options).execute(request)

    def result(self, clock: str, **options: Any) -> EquityLifecycleMonitoringResult:
        (result,) = self.run(clock, **options).results
        return result

    def snap(self, clock: str) -> str:
        return self.scenario.snap(clock)

    def issue(self, as_of: str, issued: str) -> str:
        """Issue a snapshot at ``as_of`` that declares ``issued`` as its issuance instant."""
        window = EvaluateEquityResearchWindowUseCase(
            self.scenario.world.venue, ei4c._Applicability(), NSE_EQUITY_INTRADAY_V1_RESEARCH_WINDOW
        ).execute(ei4c._LISTING, ei4c._DAY, _pit(as_of))
        return self.scenario.world.issuer.execute(window, _pit(as_of), _pit(issued)).snapshot_id

    def documents(self) -> list[str]:
        return self.recording.documents()

    def kinds(self) -> list[EquityLifecycleEventKind]:
        loaded = self.recording.store.load_lifecycle(self.issuance_id)
        assert loaded is not None
        return [event.kind for event in loaded.lifecycle.events]

    def events(self) -> tuple[Any, ...]:
        loaded = self.recording.store.load_lifecycle(self.issuance_id)
        assert loaded is not None
        return loaded.lifecycle.events


def _monitoring() -> _Monitoring:
    recording = ei6b._recording()
    recorder = _SpyRecorder(
        recording.world.authority,
        recording.world.verifier(),
        recording.scenario.world.snapshots,
        recording.store,
    )
    return _Monitoring(recording, _Catalog(recording.scenario.world.snapshots), recorder)


def _target_day(monitoring: _Monitoring) -> tuple[str, str]:
    """Bars 09:40-10:05: entry on 09:45, target on 10:00; snapshots 09:51 and 10:06."""
    monitoring.scenario.fill("09:40", "10:05", {"09:45": lifecycle._ENTRY, "10:00": ("110", "111")})
    return monitoring.snap("09:51"), monitoring.snap("10:06")


# ----------------------------------------------------------------------
# Pure selection: eligibility, order, groups, anchor, watermark (no IO)
# ----------------------------------------------------------------------

_SCOPE = EquityCandleEvidenceSnapshotCatalogScope(
    ei4c._LISTING, ei4c._DAY, _FIVE, "nse-window/1", EquityCandleEvidenceClass.SYNTHETIC
)
_ISSUED = _pit("09:36")


def _entry(
    name: str,
    as_of: str,
    issued: str | None = None,
    watermark: int = 10,
    manifest: str | None = None,
    **changes: Any,
) -> EquityCandleEvidenceSnapshotCatalogEntry:
    values: dict[str, Any] = {
        "snapshot_id": f"ecs1-sha256:{name}",
        "listing": _SCOPE.listing,
        "trading_date": _SCOPE.trading_date,
        "timeframe": _FIVE,
        "window_policy_version": _SCOPE.window_policy_version,
        "evidence_class": EquityCandleEvidenceClass.SYNTHETIC,
        "as_of": _pit(as_of),
        "issued_at": _pit(issued or as_of),
        "watermark": watermark,
        "manifest_fingerprint": manifest or f"ecm1-sha256:{name}",
    }
    values.update(changes)
    return EquityCandleEvidenceSnapshotCatalogEntry(**values)


def _select(entries: Any, at: str, anchor: Any = None) -> Any:
    return select_equity_lifecycle_snapshots(entries, _SCOPE, _pit(at), _ISSUED, anchor)


def _ids(entries: Any) -> list[str]:
    return [e.snapshot_id.split(":")[1] for e in entries]


@pytest.mark.parametrize(
    ("as_of", "issued", "at", "eligible"),
    [
        ("09:50", "09:55", "10:00", True),  # both before T
        ("10:00", "10:00", "10:00", True),  # both exactly T
        ("10:00:00.000001", "10:00:00.000001", "10:00", False),  # as_of after T
        ("09:50", "10:00:00.000001", "10:00", False),  # issued after T, as_of before
        ("09:50", "10:00", "10:00", True),  # issued exactly T
        # Outside the EI-2B contract (issued before as_of), so the as_of guard alone refuses it.
        ("10:00:00.000001", "09:59", "10:00", False),
    ],
)
def test_ms02_ms40_eligibility_requires_both_instants_at_or_before_t(
    as_of: str, issued: str, at: str, eligible: bool
) -> None:
    assert equity_snapshot_eligible(_entry("a", as_of, issued), _pit(at)) is eligible


def test_ms02_ms29_instants_compare_parsed_never_as_text() -> None:
    whole = _entry("whole", "09:50", watermark=1)
    half = dataclasses.replace(
        whole,
        snapshot_id="ecs1-sha256:half",
        manifest_fingerprint="ecm1-sha256:half",
        as_of=PointInTime("2026-08-12T04:20:00.5Z"),
        issued_at=PointInTime("2026-08-12T04:20:00.5Z"),
    )
    assert "2026-08-12T04:20:00.5Z" < whole.as_of.value  # text order is the wrong order
    assert _ids(_select([half, whole], "10:00").selected) == ["whole", "half"]
    offset = PointInTime("2026-08-12T09:50:00+05:30")  # the same instant, another offset
    assert offset == whole.as_of
    assert equity_snapshot_eligible(whole, offset) is True
    with pytest.raises(ValueError):
        PointInTime("2026-08-12T04:20:00")  # no designated offset: not an instant


def test_ms03_equivalent_manifests_resolve_to_the_lowest_snapshot_id() -> None:
    entries = [
        _entry("c", "09:50", manifest="m"),
        _entry("a", "09:50", "09:59", manifest="m"),
        _entry("b", "09:50", manifest="m"),
    ]
    chosen = _select(entries, "10:00")
    assert (_ids(chosen.selected), chosen.duplicates) == (
        ["a"],
        ("ecs1-sha256:b", "ecs1-sha256:c"),
    )
    assert chosen.stop is None


def test_ms03_a_same_instant_conflict_stops_after_the_earlier_prefix() -> None:
    entries = [
        _entry("early", "09:45", watermark=5),
        _entry("x", "09:50", watermark=6, manifest="m1"),
        _entry("y", "09:50", watermark=7, manifest="m2"),
        _entry("later", "09:55", watermark=8),
    ]
    chosen = _select(entries, "10:00")
    assert _ids(chosen.selected) == ["early"]  # the prefix stays; nothing past the conflict
    assert (chosen.stop, chosen.stop_at) == (EquityLifecycleSelectionStop.CONFLICT, _pit("09:50"))
    assert chosen.stop_snapshot_ids == ("ecs1-sha256:x", "ecs1-sha256:y")


def test_ms27_a_watermark_regression_stops_selection() -> None:
    entries = [_entry("a", "09:45", watermark=9), _entry("b", "09:50", watermark=8)]
    chosen = _select(entries, "10:00")
    assert _ids(chosen.selected) == ["a"]
    assert chosen.stop is EquityLifecycleSelectionStop.REGRESSION
    anchor = EquityLifecycleSnapshotAnchor("ecs1-sha256:z", _pit("09:40"), _pit("09:40"), 9, "m")
    regressing = _select([_entry("c", "09:50", watermark=8)], "10:00", anchor)
    assert (regressing.selected, regressing.stop) == ((), EquityLifecycleSelectionStop.REGRESSION)


def test_ms36_ms37_ms28_the_anchor_bounds_history_and_is_never_reinterpreted() -> None:
    anchor = EquityLifecycleSnapshotAnchor(
        "ecs1-sha256:anchor", _pit("09:50"), _pit("09:50"), 10, "anchor-m"
    )
    older = _entry("older", "09:45", "09:46")  # issued before the anchor: covered history
    late = _entry("late", "09:45", "09:58")  # declared issued after the anchor: late
    twin = _entry("twin", "09:50", "09:55", manifest="anchor-m")  # equivalent, later issued
    after = _entry("after", "09:55", watermark=11)
    chosen = _select([older, late, twin, after], "10:00", anchor)
    assert _ids(chosen.selected) == ["after"]
    assert _ids(chosen.late_historical) == ["late"]
    rival = _entry("rival", "09:50", "09:55", manifest="other-m")
    conflict = _select([rival, after], "10:00", anchor)
    assert (conflict.selected, conflict.stop, conflict.stop_at) == (
        (),
        EquityLifecycleSelectionStop.CONFLICT,
        _pit("09:50"),
    )


def test_snapshots_before_the_modeled_issuance_are_not_candidates() -> None:
    chosen = _select([_entry("pre", "09:35"), _entry("post", "09:40")], "10:00")
    assert (_ids(chosen.selected), chosen.late_historical) == (["post"], ())


@pytest.mark.parametrize(
    "bad",
    [
        {"listing": ei4c._OTHER},
        {"trading_date": ei4c._DAY.replace(day=13)},
        {"timeframe": Timeframe("15m")},
        {"window_policy_version": "another-window/1"},
        {"evidence_class": EquityCandleEvidenceClass.PRODUCTION},
        {"issued_at": _pit("10:30")},
        {"as_of": _pit("10:30"), "issued_at": _pit("10:30")},
    ],
)
def test_ms21_ms25_ms35_a_catalog_answer_out_of_scope_or_ineligible_is_refused(
    bad: dict[str, Any],
) -> None:
    with pytest.raises(_Catalogue) as raised:
        _select([dataclasses.replace(_entry("x", "09:50"), **bad)], "10:00")
    assert raised.value.category == _Catalogue.SCOPE


def test_a_catalog_answer_listing_one_snapshot_twice_is_refused() -> None:
    with pytest.raises(_Catalogue) as raised:
        _select([_entry("x", "09:50"), _entry("x", "09:50")], "10:00")
    assert raised.value.category == _Catalogue.INTEGRITY


def test_catalog_entries_with_malformed_metadata_cannot_be_built() -> None:
    with pytest.raises(_Catalogue) as raised:
        _entry("x", "09:50", watermark=-1)
    assert raised.value.category == _Catalogue.METADATA
    with pytest.raises(_Catalogue):
        dataclasses.replace(_entry("x", "09:50"), as_of="2026-08-12T04:20:00Z")


# ----------------------------------------------------------------------
# Monitoring through the real chain and the real EI-6B recorder
# ----------------------------------------------------------------------


def test_ms01_no_eligible_snapshot_writes_nothing() -> None:
    m = _monitoring()
    result = m.result("09:35")  # before the modeled issuance at 09:36
    assert (result.outcome, result.steps, result.appended) == (M.NO_ELIGIBLE_SNAPSHOT, (), ())
    assert result.outcome.is_success and m.recorder.calls == [] and m.documents() == []


def test_ms08_ms38_progression_to_terminal_one_snapshot_per_recorder_call() -> None:
    m = _monitoring()
    activated, targeted = _target_day(m)
    result = m.result("10:10")
    assert result.outcome is M.APPENDED and result.state is S.CLOSED_TARGET
    assert all(len(call) == 1 for call in m.recorder.calls)  # never a batch
    assert m.recorder.calls == [(sid,) for sid in result.recorded_snapshot_ids]
    assert result.recorded_snapshot_ids[-2:] == (activated, targeted)
    assert [step.outcome for step in result.steps] == [
        RO.NO_NEW_EVENTS,  # 09:36: before eligible_from
        RO.APPENDED,  # 09:41: the entry window opens
        RO.APPENDED,  # 09:51: activation
        RO.APPENDED,  # 10:06: target
    ]
    assert [e.kind for e in result.appended] == [
        K.ENTRY_WINDOW_OPENED,
        K.ENTRY_ACTIVATED,
        K.TARGET_REACHED,
    ]
    assert m.documents() == [equity_lifecycle_event_document(e) for e in result.appended]


def test_ms05_ms06_reruns_and_an_earlier_instant_write_nothing() -> None:
    m = _monitoring()
    _target_day(m)
    m.result("10:10")
    before, appends = m.documents(), m.recording.store.appends
    again = m.result("10:10")
    earlier = m.result("09:55")
    assert again.outcome is M.NO_ELIGIBLE_SNAPSHOT and earlier.outcome is M.NO_ELIGIBLE_SNAPSHOT
    assert (m.documents(), m.recording.store.appends) == (before, appends)
    assert earlier.diagnostics == ()  # older snapshots issued before the anchor: covered history


def test_ms07_a_quiet_evaluation_writes_nothing_and_keeps_no_cursor() -> None:
    m = _monitoring()
    m.result("09:42")  # opens the entry window at 09:41
    before, appends = m.documents(), m.recording.store.appends
    m.scenario.fill("09:40", "10:15", {"10:00": None})  # a gap: a temporary block
    m.snap("10:16")
    quiet = m.result("10:17")
    assert quiet.outcome is M.NO_NEW_EVENTS and quiet.state is S.WAITING_FOR_ENTRY
    assert (m.documents(), m.recording.store.appends) == (before, appends)
    again = m.result("10:17")  # deterministic, and re-reads the quiet snapshot
    assert again.outcome is M.NO_NEW_EVENTS and m.documents() == before


def _two_days(issue: Callable[[_Monitoring], None]) -> tuple[_Monitoring, _Monitoring]:
    first, second = _monitoring(), _monitoring()
    for m in (first, second):
        issue(m)
    return first, second


def test_ms04_conditional_equivalence_holds_for_stable_ordered_prefixes() -> None:
    split, single = _two_days(_target_day)  # issued_at equals as_of: stable prefixes
    split.result("09:55")
    split.result("10:10")
    single.result("10:10")
    assert split.documents() == single.documents()


def _unstable(m: _Monitoring) -> None:
    m.scenario.fill("09:40", "09:55", {"09:45": lifecycle._ENTRY})
    m.issue("09:56", "09:56")  # A
    m.issue("09:51", "10:00")  # B: an earlier as_of, declared issued later


def test_ms34_ms36_a_fixed_catalog_with_unstable_eligibility_is_path_dependent() -> None:
    split, single = _two_days(_unstable)
    split.result("09:57")  # only A is eligible: activation evaluated at 09:56
    later = split.result("10:01")  # B now eligible but earlier than the recorded event
    assert later.outcome is M.NO_ELIGIBLE_SNAPSHOT
    (diagnostic,) = later.diagnostics
    assert (
        diagnostic.kind is EquityLifecycleMonitoringDiagnosticKind.LATE_HISTORICAL_SNAPSHOT_IGNORED
    )
    assert diagnostic.as_of == _pit("09:51")
    single.result("10:01")  # B then A: activation evaluated at 09:51
    assert split.kinds() == single.kinds() == [K.ENTRY_WINDOW_OPENED, K.ENTRY_ACTIVATED]
    assert split.events()[-1].evaluated_at == _pit("09:56")
    assert single.events()[-1].evaluated_at == _pit("09:51")
    assert split.documents() != single.documents()  # accepted path dependence (EI6C-P5)


def test_ms40_a_snapshot_declaring_issuance_after_t_is_excluded_until_t_reaches_it() -> None:
    m = _monitoring()
    m.scenario.fill("09:40", "09:50", {"09:45": lifecycle._ENTRY})
    future = m.issue("09:51", "10:30")
    before = m.result("10:29")  # retrospective call: as_of 09:51 <= T, issued 10:30 > T
    assert future not in before.recorded_snapshot_ids and S.ACTIVE is not before.state
    exact = m.result("10:30")  # issued exactly at T: eligible
    assert exact.recorded_snapshot_ids == (future,) and exact.state is S.ACTIVE


def _changing(m: _Monitoring) -> tuple[str, str]:
    m.scenario.fill("09:40", "09:50", {"09:45": lifecycle._ENTRY})
    first = m.snap("09:51")
    return first, ""


def _rival(m: _Monitoring) -> str:
    """Another manifest at 09:51: a further observation of a bar already ended by then."""
    m.scenario.put("09:40", lifecycle._QUIET, suffix="-again")
    return m.issue("09:51", "09:53")


def test_ms28_ms13_a_later_same_instant_manifest_never_rewrites_recorded_history() -> None:
    m = _monitoring()
    first, _ = _changing(m)
    m.result("09:52")
    recorded = m.documents()
    rival = _rival(m)
    stopped = m.result("09:54")
    assert stopped.outcome is M.STOPPED_AT_CONFLICT and stopped.stopped_at == _pit("09:51")
    assert set(stopped.stop_snapshot_ids) == {first, rival} and stopped.steps == ()
    assert m.documents() == recorded  # the earlier event stays authoritative
    fresh = _monitoring()  # both present from the start: no convergence is required
    _changing(fresh)
    _rival(fresh)
    single = fresh.result("09:54")
    assert single.outcome is M.STOPPED_AT_CONFLICT
    assert fresh.kinds() == [K.ENTRY_WINDOW_OPENED] != m.kinds()


def test_ms37_a_later_equivalent_snapshot_never_rewrites_the_cited_reference() -> None:
    m = _monitoring()
    first, _ = _changing(m)
    m.result("09:52")
    cited = m.events()[-1].evidence.reference
    assert cited == first
    twin = m.issue("09:51", "09:52")  # the same manifest, another issuance instant and identity
    assert twin != first
    result = m.result("09:53")
    assert result.outcome is M.NO_ELIGIBLE_SNAPSHOT and m.recorder.calls[-1:] != [(twin,)]
    assert m.events()[-1].evidence.reference == cited


def test_ms13_a_later_instant_correction_is_recorded_once_as_a_revision() -> None:
    m = _monitoring()
    _changing(m)
    m.result("09:52")
    m.scenario.put("09:45", ("101.7", "102.2"), revision="r2")  # a correction of the cited bar
    m.snap("09:56")
    result = m.result("09:57")
    assert result.outcome is M.APPENDED
    assert [e.kind for e in result.appended] == [K.SOURCE_REVISION_RECORDED]
    again = m.result("09:57")
    assert (
        again.outcome is M.NO_ELIGIBLE_SNAPSHOT and m.kinds().count(K.SOURCE_REVISION_RECORDED) == 1
    )


def test_ms21_ms38_a_real_snapshot_refusal_keeps_the_committed_prefix() -> None:
    m = _monitoring()
    m.scenario.fill("09:40", "09:50", {"09:45": lifecycle._ENTRY})
    m.result("09:42")  # the entry window opened at 09:41
    committed = m.documents()
    m.scenario.world.venue.reference = "test-venue/2"  # another venue-day reference
    foreign = m.snap("09:51")
    result = m.result("09:52")
    assert result.outcome is M.STOPPED_AT_SNAPSHOT_REFUSAL
    assert (result.refusal, result.stopped_at, result.stop_snapshot_ids) == (
        R.SNAPSHOT_SESSION_MISMATCH,
        _pit("09:51"),
        (foreign,),
    )
    assert m.documents() == committed


@pytest.mark.parametrize(
    "failure",
    [
        InvalidEquityRecommendationLifecycleError("test: Core rejection"),
        UnverifiedEquityRecommendationLifecycleAppendError("test: append verification"),
        RuntimeError("test: unexpected"),
    ],
)
def test_ms39_an_unexpected_failure_fails_the_issuance_closed_and_keeps_the_prefix(
    failure: BaseException,
) -> None:
    m = _monitoring()
    _target_day(m)
    m.recorder.fail_at, m.recorder.failure = 3, failure  # the third call: the 09:51 snapshot
    result = m.result("10:10")
    assert result.outcome is M.ISSUANCE_EVALUATION_FAILED and not result.outcome.is_success
    assert result.cause is not None and type(failure).__name__ in result.cause
    assert [e.kind for e in result.appended] == [K.ENTRY_WINDOW_OPENED]
    assert m.kinds() == [K.ENTRY_WINDOW_OPENED]  # the committed prefix stays
    assert len(m.recorder.calls) == 3  # nothing after the failure


def test_ms39_a_process_signal_is_never_swallowed() -> None:
    m = _monitoring()
    m.recorder.fail_at, m.recorder.failure = 1, KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        m.result("09:42")


# ----------------------------------------------------------------------
# O-2 option A, EI5-D7, O-3 and terminal histories
# ----------------------------------------------------------------------


def test_ms09_ms10_exit_boundary_no_allowance_and_the_exact_boundary_rule() -> None:
    m = _monitoring()
    m.scenario.fill("09:40", "13:55", {"09:45": lifecycle._ENTRY})  # 13:55-14:00 missing
    m.snap("09:51")
    m.result("09:52")
    assert m.kinds()[-1] is K.ENTRY_ACTIVATED
    m.snap("14:00")  # exactly at modeled_exit_at, the boundary bar not yet received
    at_boundary = m.result("14:00")
    assert at_boundary.outcome is M.NO_NEW_EVENTS and at_boundary.state is S.ACTIVE
    m.snap("14:01")  # one minute later: evaluated at once, no settlement allowance
    after = m.result("14:01")
    assert after.state is S.INCOMPLETE_EVIDENCE
    assert [e.kind for e in after.appended] == [K.EVIDENCE_INCOMPLETE]


def test_ms09_entry_boundary_without_activation() -> None:
    m = _monitoring()
    m.scenario.fill("09:40", "10:45")  # quiet; the last entry bar 10:45-10:50 is missing
    m.snap("10:50")
    at_boundary = m.result("10:50")
    assert at_boundary.state is S.WAITING_FOR_ENTRY
    m.snap("10:51")
    after = m.result("10:51")
    assert after.state is S.INCOMPLETE_EVIDENCE


def test_ms15_ms10_late_gap_filling_evidence_after_incomplete_evidence_records_nothing() -> None:
    m = _monitoring()
    m.scenario.fill("09:40", "14:00", {"09:45": lifecycle._ENTRY, "12:00": None})
    m.snap("09:51")
    m.snap("14:01")
    assert m.result("14:02").state is S.INCOMPLETE_EVIDENCE
    before = m.documents()
    m.scenario.put("12:00", lifecycle._QUIET)  # the late bar that fills the gap
    m.snap("16:00")
    late = m.result("16:01")
    assert (late.outcome, late.state) == (M.NO_NEW_EVENTS, S.INCOMPLETE_EVIDENCE)
    assert m.documents() == before


def test_ms14_a_terminal_lifecycle_records_only_core_permitted_revisions() -> None:
    m = _monitoring()
    _target_day(m)
    m.result("10:10")
    m.snap("10:30")
    quiet = m.result("10:31")
    assert (quiet.outcome, quiet.state) == (M.NO_NEW_EVENTS, S.CLOSED_TARGET)
    m.scenario.put("10:00", ("110", "111.5"), revision="r2")  # a correction of the target bar
    m.snap("10:40")
    revised = m.result("10:41")
    assert [e.kind for e in revised.appended] == [K.SOURCE_REVISION_RECORDED]
    assert revised.state is S.CLOSED_TARGET


def test_ms33_a_history_recorded_before_monitoring_stays_authoritative() -> None:
    m = _monitoring()
    m.scenario.fill("09:40", "09:50", {"09:45": lifecycle._ENTRY})
    direct = m.recording.record(m.snap("09:51"))  # recorded directly through EI-6B, no monitor
    assert direct.state is S.ACTIVE
    before = m.documents()
    result = m.result("09:55")
    assert result.outcome is M.NO_ELIGIBLE_SNAPSHOT and m.documents() == before


# ----------------------------------------------------------------------
# Concurrency and commit uncertainty, through the EI-6B recorder
# ----------------------------------------------------------------------


def test_ms16_an_identical_competing_append_is_equivalent() -> None:
    m = _monitoring()
    first = m.snap("09:41")

    def competitor() -> None:
        m.recording.recorder().execute(m.issuance_id, (first,))

    m.recording.store.before_append = competitor
    result = m.result("09:42")
    assert [step.outcome for step in result.steps][-1] is RO.EQUIVALENT
    assert result.outcome is M.EQUIVALENT and m.kinds() == [K.ENTRY_WINDOW_OPENED]


def test_ms17_a_divergent_competing_append_is_a_conflict_and_the_winner_stays() -> None:
    m = _monitoring()
    m.scenario.fill("09:40", "09:50", {"09:45": lifecycle._ENTRY})
    competing = m.snap("09:51")

    def competitor() -> None:
        m.recording.recorder().execute(m.issuance_id, (competing,))

    m.recording.store.before_append = competitor
    result = m.result("09:42")  # our 09:41 opening meets the 09:51 history committed first
    assert result.outcome is M.CONFLICT
    assert m.events()[0].evaluated_at == _pit("09:51")  # the first commit won
    assert result.steps[-1].outcome is RO.CONFLICT and result.steps[-1].as_of == _pit("09:41")
    assert len(m.recorder.calls) == len(result.steps) == 2  # one call each; no blind retry


def test_store_unavailable_and_unresolved_commits_stay_typed_with_no_blind_retry() -> None:
    m = _monitoring()
    m.recording.store.script = ["unavailable"]
    unavailable = m.result("09:42")
    assert (unavailable.outcome, unavailable.refusal, unavailable.retryable) == (
        M.REFUSED,
        R.STORE_UNAVAILABLE,
        True,
    )
    m.recording.store.script = ["lost_before"] * 3
    calls = len(m.recorder.calls)
    unresolved = m.result("09:42")
    assert unresolved.outcome is M.UNRESOLVED and not unresolved.outcome.is_success
    assert len(m.recorder.calls) - calls == 2  # 09:36 (quiet) and 09:41, each once
    assert m.documents() == []


def test_a_lost_acknowledgement_is_reconciled_by_the_recorder() -> None:
    m = _monitoring()
    m.recording.store.script = ["lost_after"]
    result = m.result("09:42")
    assert result.outcome is M.EQUIVALENT and result.steps[-1].reconciled
    assert m.kinds() == [K.ENTRY_WINDOW_OPENED]


# ----------------------------------------------------------------------
# Catalog failures and the resource limit
# ----------------------------------------------------------------------


def test_ms30_candidate_overflow_is_a_typed_refusal_and_nothing_is_recorded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    m = _monitoring()
    _target_day(m)
    monkeypatch.setattr(monitor_module, "EQUITY_LIFECYCLE_MONITORING_CATALOG_LIMIT", 2)
    result = m.result("10:10")
    assert result.outcome is M.CATALOG_LIMIT_EXCEEDED
    assert m.catalog.calls[-1][2] == 2  # the governed limit, never a caller's
    assert m.recorder.calls == [] and m.documents() == []


def test_ms30_a_catalog_returning_more_than_the_limit_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    m = _monitoring()
    _target_day(m)
    monkeypatch.setattr(monitor_module, "EQUITY_LIFECYCLE_MONITORING_CATALOG_LIMIT", 2)

    class _Overflowing(_Catalog):
        def list_snapshots(self, scope, monitoring_instant, limit):  # type: ignore[no-untyped-def]
            return super().list_snapshots(scope, monitoring_instant, limit + 10)

    m.catalog = _Overflowing(m.scenario.world.snapshots)
    assert m.result("10:10").outcome is M.CATALOG_LIMIT_EXCEEDED and m.recorder.calls == []


@pytest.mark.parametrize(
    ("fault", "refusal", "retryable"),
    [
        ("unavailable", MR.CATALOG_UNAVAILABLE, True),
        ("incomplete", MR.CATALOG_INVALID, False),
        ("tampered", MR.CATALOG_INVALID, False),
        ("foreign", MR.CATALOG_INVALID, False),
        ("list", MR.CATALOG_INVALID, False),
    ],
)
def test_catalog_faults_are_typed_and_record_nothing(
    fault: str, refusal: EquityLifecycleMonitoringRefusal, retryable: bool
) -> None:
    m = _monitoring()
    _target_day(m)
    if fault == "unavailable":
        m.catalog.unavailable = True
    elif fault == "incomplete":
        m.catalog.incomplete = True
    elif fault == "tampered":
        m.catalog.tamper = lambda e: dataclasses.replace(e, watermark=e.watermark + 1)
    elif fault == "foreign":
        m.catalog.extra = (dataclasses.replace(_entry("foreign", "09:50"), listing=ei4c._OTHER),)
    else:
        m.catalog.answer_list = True
    result = m.result("10:10")
    assert (result.outcome, result.refusal, result.retryable) == (M.REFUSED, refusal, retryable)
    assert m.recorder.calls == [] and m.documents() == []


def test_snapshots_outside_the_scope_are_never_candidates() -> None:
    m = _monitoring()
    m.scenario.fill("09:40", "09:50", {"09:45": lifecycle._ENTRY})
    foreign = lifecycle._foreign_snapshot(m.scenario, listing=ei4c._OTHER)  # another listing
    own = m.snap("09:51")
    result = m.result("09:52")
    assert own in result.recorded_snapshot_ids
    assert foreign not in result.recorded_snapshot_ids and result.state is S.ACTIVE


# ----------------------------------------------------------------------
# Issuance scope modes and per-issuance isolation
# ----------------------------------------------------------------------


def test_ms32_explicit_ids_are_normalized_and_an_unknown_one_is_isolated() -> None:
    m = _monitoring()
    unknown = "eiris1-sha256:" + "0" * 64
    scope = EquityLifecycleMonitoringExplicitScope((m.issuance_id, unknown, m.issuance_id))
    assert scope.issuance_ids == tuple(sorted({m.issuance_id, unknown}))
    report = m.run("09:42", scope)
    by_id = {r.issuance_id: r for r in report.results}
    assert [r.issuance_id for r in report.results] == sorted(by_id)
    assert (by_id[unknown].outcome, by_id[unknown].refusal) == (M.REFUSED, R.ISSUANCE_NOT_FOUND)
    assert by_id[m.issuance_id].outcome is M.APPENDED  # unaffected by the other's failure


def test_ms32_strategy_listing_finds_an_issuance_not_yet_declared() -> None:
    m = _monitoring()
    issued = m.recording.issued
    scope = EquityLifecycleMonitoringStrategyListingScope(
        issued.strategy, issued.listing, issued.trading_date
    )
    assert [r.issuance_id for r in m.run("09:42", scope).results] == [m.issuance_id]
    declared = EquityLifecycleMonitoringDeclaredSessionScope(ei4c._NSE, ei4c._DAY)
    assert m.run("09:42", declared).results == ()  # declared only through 09:35:59.999999


def test_ms32_a_declared_session_lists_an_issuance_once_a_declaration_covers_it() -> None:
    m = _monitoring()
    assert m.recording.world.declare(_pit("09:37")).outcome.value == "DECLARED"
    current = EquityLifecycleMonitoringDeclaredSessionScope(ei4c._NSE, ei4c._DAY)
    assert [r.issuance_id for r in m.run("09:42", current).results] == [m.issuance_id]
    older = EquityLifecycleMonitoringDeclaredSessionScope(ei4c._NSE, ei4c._DAY, watermark=2)
    assert m.run("09:42", older).results == ()  # at watermark 2 only the first declaration


class _Reader(EquityRecommendationIssuanceReader):
    """TEST-ONLY: the in-memory authority, with an injected failure."""

    def __init__(self, inner: Any, failure: BaseException) -> None:
        self.inner, self.failure = inner, failure

    def load_issuance(self, issuance_id):  # type: ignore[no-untyped-def]
        raise self.failure

    def load_session_issuance(self, strategy, listing, trading_date):  # type: ignore[no-untyped-def]
        raise self.failure

    def session_view(self, exchange_code, trading_date):  # type: ignore[no-untyped-def]
        raise self.failure

    def session_view_at(self, exchange_code, trading_date, watermark):  # type: ignore[no-untyped-def]
        raise self.failure


_CORRUPT = InvalidStoredEquityRecommendationIssuanceAuthorityError("ISSUANCE", "test: corrupt")


@pytest.mark.parametrize(
    ("failure", "refusal", "retryable"),
    [
        (_CORRUPT, R.STORED_AUTHORITY_INVALID, False),
        (EquityRecommendationIssuanceUnavailableError("test"), R.AUTHORITY_UNAVAILABLE, True),
    ],
)
def test_corrupt_or_unavailable_issuance_authority_is_typed(
    failure: BaseException, refusal: EquityLifecycleRecordingRefusal, retryable: bool
) -> None:
    m = _monitoring()
    reader = _Reader(m.recording.world.authority, failure)
    result = m.result("09:42", issuances=reader)
    assert (result.outcome, result.refusal, result.retryable) == (M.REFUSED, refusal, retryable)
    declared = EquityLifecycleMonitoringDeclaredSessionScope(ei4c._NSE, ei4c._DAY)
    with pytest.raises(EquityLifecycleMonitoringScopeError) as raised:
        m.run("09:42", declared, issuances=reader)
    assert raised.value.retryable is retryable


def test_ms20_corrupt_lifecycle_history_is_a_permanent_typed_refusal() -> None:
    m = _monitoring()
    m.result("09:42")
    m.recording.store.rows[m.issuance_id][0] += " "  # TEST-ONLY corruption: not canonical
    result = m.result("09:42")
    assert (result.outcome, result.refusal, result.retryable) == (
        M.REFUSED,
        R.STORED_LIFECYCLE_INVALID,
        False,
    )


def test_a_missing_anchor_snapshot_fails_the_issuance_closed() -> None:
    m = _monitoring()
    m.result("09:42")
    cited = m.events()[-1].evidence.reference
    del m.scenario.world.snapshots.documents[cited]  # TEST-ONLY: the cited evidence is gone
    calls = len(m.recorder.calls)
    result = m.result("09:42")
    assert (result.outcome, result.refusal) == (M.REFUSED, MR.ANCHOR_SNAPSHOT_INVALID)
    assert len(m.recorder.calls) == calls  # nothing recorded without its anchor


@pytest.mark.parametrize(
    "build",
    [
        lambda: EquityLifecycleMonitoringExplicitScope(()),
        lambda: EquityLifecycleMonitoringExplicitScope(("",)),
        lambda: EquityLifecycleMonitoringExplicitScope(["x"]),  # type: ignore[arg-type]
        lambda: EquityLifecycleMonitoringDeclaredSessionScope(ei4c._NSE, ei4c._DAY, watermark=-1),
        lambda: EquityLifecycleMonitoringDeclaredSessionScope("NSE", ei4c._DAY),  # type: ignore[arg-type]
        lambda: EquityLifecycleMonitoringRequest(object(), _pit("09:42")),  # type: ignore[arg-type]
        lambda: EquityLifecycleMonitoringRequest(
            EquityLifecycleMonitoringExplicitScope(("x",)),
            "09:42",  # type: ignore[arg-type]
        ),
    ],
)
def test_malformed_scopes_and_requests_are_refused_before_any_dependency(build: Any) -> None:
    with pytest.raises(TypeError):
        build()


# ----------------------------------------------------------------------
# Determinism, policy and architecture
# ----------------------------------------------------------------------


def _shape(report: EquityLifecycleMonitoringReport) -> list[Any]:
    return [
        (r.issuance_id, r.outcome, [(s.snapshot_id, s.outcome) for s in r.steps], r.state)
        for r in report.results
    ]


def test_ms22_identical_inputs_give_identical_selection_and_outcomes() -> None:
    first, second = _two_days(_target_day)
    assert _shape(first.run("10:10")) == _shape(second.run("10:10"))
    assert first.documents() == second.documents()


def test_ms31_one_fixed_policy_with_no_selectable_allowance_or_limit() -> None:
    assert re.fullmatch(r"northstar\.equity-[a-z-]+/\d+", EQUITY_LIFECYCLE_MONITORING_POLICY)
    assert EQUITY_LIFECYCLE_MONITORING_POLICY == "northstar.equity-lifecycle-monitoring-policy/1"
    assert EQUITY_LIFECYCLE_MONITORING_CATALOG_LIMIT == 300
    use_case = MonitorEquityRecommendationLifecyclesUseCase
    assert list(inspect.signature(use_case).parameters) == [
        "issuances",
        "generations",
        "snapshots",
        "catalog",
        "lifecycles",
        "recorder",
    ]
    assert list(inspect.signature(use_case.execute).parameters) == ["self", "request"]
    fields = [f.name for f in dataclasses.fields(EquityLifecycleMonitoringRequest)]
    assert fields == ["scope", "monitoring_instant"]
    m = _monitoring()
    assert m.run("09:42").policy == EQUITY_LIFECYCLE_MONITORING_POLICY


_NEW_MODULES = {
    "monitor": _SRC / "monitor_equity_recommendation_lifecycles.py",
    "selection": _SRC / "equity_lifecycle_monitoring_selection.py",
    "catalog": _PORTS / "equity_candle_evidence_snapshot_catalog.py",
    "reader": _PORTS / "equity_recommendation_lifecycle_reader.py",
}


def _tree(name: str) -> ast.Module:
    return ast.parse(_NEW_MODULES[name].read_text(encoding="utf-8"))


def test_ms24_the_monitoring_modules_read_no_clock_and_reach_no_infrastructure() -> None:
    forbidden = ("infrastructure", "sqlite", "socket", "http", "requests", "broker", "random")
    for name in _NEW_MODULES:
        for node in ast.walk(_tree(name)):
            if isinstance(node, ast.ImportFrom | ast.Import):
                modules = (
                    [node.module or ""]
                    if isinstance(node, ast.ImportFrom)
                    else [a.name for a in node.names]
                )
                assert not [m for m in modules if any(w in m for w in forbidden)], (name, modules)
            if isinstance(node, ast.Attribute):
                assert node.attr not in {"now", "utcnow", "today", "time", "sleep"}, name


def test_ms24_the_monitor_never_writes_lifecycles_itself() -> None:
    tree = _tree("monitor")
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    names |= {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names}
    assert "EquityRecommendationLifecycleStore" not in names  # read-only protocol only
    assert "_issue_lifecycle_append" not in names
    appends_on_dependencies = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Attribute)
        and n.attr in {"append", "issue"}
        and isinstance(n.value, ast.Attribute)
        and n.value.attr in {"_lifecycles", "_snapshots", "_catalog", "_issuances"}
    ]
    assert appends_on_dependencies == []
    assert "load_lifecycle" in EquityRecommendationLifecycleReader.__dict__
    assert set(EquityCandleEvidenceSnapshotCatalog.__abstractmethods__) == {"list_snapshots"}


def test_ms24_selection_is_pure() -> None:
    imports = {n.module for n in ast.walk(_tree("selection")) if isinstance(n, ast.ImportFrom)}
    assert imports == {
        "__future__",
        "collections.abc",
        "dataclasses",
        "enum",
        "functools",
        "northstar_core.foundation.value_objects",
        "northstar_application.ports.equity_candle_evidence_snapshot_catalog",
    }


def test_the_new_contracts_are_exported() -> None:
    for name in (
        "MonitorEquityRecommendationLifecyclesUseCase",
        "EquityLifecycleMonitoringOutcome",
        "select_equity_lifecycle_snapshots",
        "EQUITY_LIFECYCLE_MONITORING_POLICY",
    ):
        assert name in application_services.__all__
    for name in (
        "EquityCandleEvidenceSnapshotCatalog",
        "EquityCandleEvidenceSnapshotCatalogEntry",
        "EquityRecommendationLifecycleReader",
    ):
        assert name in ports.__all__
    assert isinstance(ei6b._recording().store, EquityRecommendationLifecycleReader)
    assert {o.value for o in EquityLifecycleMonitoringOutcome} == {
        "APPENDED",
        "EQUIVALENT",
        "CONFLICT",
        "NO_NEW_EVENTS",
        "NO_ELIGIBLE_SNAPSHOT",
        "STOPPED_AT_CONFLICT",
        "STOPPED_AT_REGRESSION",
        "STOPPED_AT_SNAPSHOT_REFUSAL",
        "CATALOG_LIMIT_EXCEEDED",
        "ISSUANCE_EVALUATION_FAILED",
        "REFUSED",
        "UNRESOLVED",
    }
    assert EquityLifecycleRecordingRefusedError is not None  # typed refusals stay EI-6B's
