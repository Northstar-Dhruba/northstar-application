"""Deterministic snapshot selection for lifecycle monitoring (EI-6C, EI6C-P4, P5, P8).

The EI-6C Lifecycle Monitoring and Snapshot Selection Design v1.0, section 5,
as one pure function: no clock, no input or output, no state. Given catalog
entries the monitor has already re-verified, the issuance's modeled
issuance instant, the anchor of its authoritative lifecycle history and the
monitoring instant ``T``, it returns the snapshots to record, in order, and
where selection must stop.

Rules
-----
1. **Eligibility.** ``as_of <= T`` and ``issued_at <= T``, compared as
   instants (``PointInTime.compare``), never as text; equality is eligible. An
   entry that is not eligible, or not in scope, makes the catalog answer
   untrustworthy (``SCOPE``).
2. **Candidates** start at the issuance's ``issued_at``. With recorded
   history they start after the **anchor** -- the snapshot cited by the last
   persisted lifecycle event:

   - at the anchor's instant, an entry with the anchor's manifest is the same
     evidence and is not recorded again (no rewrite of the event's
     reference); an entry with another manifest is a **same-instant conflict**
     and stops selection there;
   - earlier than the anchor, an entry is never evaluated. One **declared**
     to be issued after the anchor snapshot is reported as late historical
     evidence (``late_historical``); declared, not proven (EI6C-P4).
3. **Order and groups.** Remaining entries are ordered by instant, then by
   ``snapshot_id``. Entries sharing an instant form one group, resolved whole:
   one manifest -- the lowest ``snapshot_id`` represents it, the others are
   duplicates; several manifests -- a same-instant conflict, which stops
   selection at that instant. No conflicting winner is ever chosen.
4. **Watermark.** Along the selected sequence the journal watermark never
   decreases, starting from the anchor's. A decrease stops selection at that
   instant (``REGRESSION``).

Selection stops at the first conflict or regression; what precedes it is
still selected, so earlier evidence is recorded and later evidence is not
skipped over. No snapshot is thinned and no settlement allowance applies
(O-2 option A, Δ = 0).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from functools import cmp_to_key

from northstar_core.foundation.value_objects import PointInTime

from northstar_application.ports.equity_candle_evidence_snapshot_catalog import (
    EquityCandleEvidenceSnapshotCatalogEntry,
    EquityCandleEvidenceSnapshotCatalogScope,
    InvalidEquityCandleEvidenceSnapshotCatalogError,
)

EQUITY_LIFECYCLE_MONITORING_POLICY = "northstar.equity-lifecycle-monitoring-policy/1"
"""The one fixed monitoring policy of synthetic EI-6C v1 (EI6C-P12): not configurable."""

EQUITY_LIFECYCLE_MONITORING_CATALOG_LIMIT = 300
"""The governed candidate limit per issuance per run (EI6C-P6).

Engineering basis (no provider-derived figure): the approved research window,
09:15 to 15:00 IST, has 69 five-minute bars, and a trading date holds at most
288; synthetic fixtures issue one snapshot per evaluated bar or fewer. 300
allows about four snapshots per research-window bar, or one per bar for the
whole calendar date, while bounding a run to 300 recorder calls. Exceeding it
is a typed refusal, never truncation.
"""

_Catalog = InvalidEquityCandleEvidenceSnapshotCatalogError
Entry = EquityCandleEvidenceSnapshotCatalogEntry


class EquityLifecycleSelectionStop(StrEnum):
    """Why selection stopped before the end of the candidates."""

    CONFLICT = "CONFLICT"
    REGRESSION = "REGRESSION"


@dataclass(frozen=True, slots=True)
class EquityLifecycleSnapshotAnchor:
    """The snapshot cited by the last persisted lifecycle event, as authenticated."""

    snapshot_id: str
    as_of: PointInTime
    issued_at: PointInTime
    watermark: int
    manifest_fingerprint: str


@dataclass(frozen=True, slots=True)
class EquityLifecycleSnapshotSelection:
    """The snapshots to record, in order, and where selection stopped (if it did)."""

    selected: tuple[Entry, ...]
    stop: EquityLifecycleSelectionStop | None = None
    stop_at: PointInTime | None = None
    stop_snapshot_ids: tuple[str, ...] = ()
    late_historical: tuple[Entry, ...] = ()
    duplicates: tuple[str, ...] = ()


def equity_snapshot_eligible(entry: Entry, monitoring_instant: PointInTime) -> bool:
    """Whether ``entry`` is eligible at ``T``: ``as_of <= T`` and ``issued_at <= T``."""
    return (
        entry.as_of.compare(monitoring_instant) <= 0
        and entry.issued_at.compare(monitoring_instant) <= 0
    )


def _by_instant(first: Entry, second: Entry) -> int:
    return first.as_of.compare(second.as_of) or (first.snapshot_id > second.snapshot_id) - (
        first.snapshot_id < second.snapshot_id
    )


def _in_scope(entry: Entry, scope: EquityCandleEvidenceSnapshotCatalogScope) -> bool:
    return (
        entry.listing == scope.listing
        and entry.trading_date == scope.trading_date
        and entry.timeframe == scope.timeframe
        and entry.window_policy_version == scope.window_policy_version
        and entry.evidence_class is scope.evidence_class
    )


def _groups(entries: Sequence[Entry]) -> Iterable[tuple[Entry, ...]]:
    group: list[Entry] = []
    for entry in entries:
        if group and group[0].as_of.compare(entry.as_of) != 0:
            yield tuple(group)
            group = []
        group.append(entry)
    if group:
        yield tuple(group)


def select_equity_lifecycle_snapshots(
    entries: Iterable[Entry],
    scope: EquityCandleEvidenceSnapshotCatalogScope,
    monitoring_instant: PointInTime,
    issued_at: PointInTime,
    anchor: EquityLifecycleSnapshotAnchor | None,
) -> EquityLifecycleSnapshotSelection:
    """Select the snapshots to record for one issuance; see the module docstring."""
    if not isinstance(monitoring_instant, PointInTime) or not isinstance(issued_at, PointInTime):
        raise TypeError("monitoring_instant and issued_at must be PointInTime values.")
    if anchor is not None and not isinstance(anchor, EquityLifecycleSnapshotAnchor):
        raise TypeError("anchor must be an EquityLifecycleSnapshotAnchor or None.")
    listed = tuple(entries)
    seen: set[str] = set()
    for entry in listed:
        if not isinstance(entry, Entry):
            raise _Catalog(_Catalog.METADATA, "A catalog answer holds a non-entry value.")
        if entry.snapshot_id in seen:
            raise _Catalog(_Catalog.INTEGRITY, f"Snapshot {entry.snapshot_id} is listed twice.")
        seen.add(entry.snapshot_id)
        if not _in_scope(entry, scope):
            raise _Catalog(_Catalog.SCOPE, f"Snapshot {entry.snapshot_id} is outside the scope.")
        if not equity_snapshot_eligible(entry, monitoring_instant):
            raise _Catalog(
                _Catalog.SCOPE,
                f"Snapshot {entry.snapshot_id} is not eligible at {monitoring_instant}.",
            )
    ordered = sorted(
        (entry for entry in listed if entry.as_of.compare(issued_at) >= 0),
        key=cmp_to_key(_by_instant),
    )
    late: list[Entry] = []
    watermark: int | None = None
    if anchor is not None:
        at_anchor = [e for e in ordered if e.as_of.compare(anchor.as_of) == 0]
        late = [
            e
            for e in ordered
            if e.as_of.compare(anchor.as_of) < 0 and e.issued_at.compare(anchor.issued_at) > 0
        ]
        ordered = [e for e in ordered if e.as_of.compare(anchor.as_of) > 0]
        differing = [e for e in at_anchor if e.manifest_fingerprint != anchor.manifest_fingerprint]
        if differing:
            return EquityLifecycleSnapshotSelection(
                (),
                EquityLifecycleSelectionStop.CONFLICT,
                anchor.as_of,
                tuple(sorted({anchor.snapshot_id, *(e.snapshot_id for e in differing)})),
                tuple(late),
            )
        watermark = anchor.watermark
    selected: list[Entry] = []
    duplicates: list[str] = []
    for group in _groups(ordered):
        instant = group[0].as_of
        ids = tuple(e.snapshot_id for e in group)
        if len({e.manifest_fingerprint for e in group}) > 1:
            return EquityLifecycleSnapshotSelection(
                tuple(selected),
                EquityLifecycleSelectionStop.CONFLICT,
                instant,
                ids,
                tuple(late),
                tuple(duplicates),
            )
        representative, *others = group
        if watermark is not None and representative.watermark < watermark:
            return EquityLifecycleSnapshotSelection(
                tuple(selected),
                EquityLifecycleSelectionStop.REGRESSION,
                instant,
                ids,
                tuple(late),
                tuple(duplicates),
            )
        selected.append(representative)
        duplicates.extend(e.snapshot_id for e in others)
        watermark = representative.watermark
    return EquityLifecycleSnapshotSelection(
        tuple(selected), None, None, (), tuple(late), tuple(duplicates)
    )
