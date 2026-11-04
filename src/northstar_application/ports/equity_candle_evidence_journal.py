"""Application ports for the equity candle evidence journal and its snapshots (EI-2B).

``EquityCandleEvidenceJournal`` is append-only. Every candle observation and
every finality statement Northstar receives becomes one immutable entry,
numbered by a strictly increasing journal sequence that the implementation
assigns when it appends. Nothing is updated, replaced or deleted, and no
source, receipt or confirmation timestamp ever orders the journal: the
sequence alone records insertion order, so a record appended later carries a
later sequence whatever times it claims. ``watermark()`` is the highest
sequence committed (0 when empty).

``append(record)`` applies, atomically and in this order, the rules in
equity_candle_evidence_journal (Application):

1. the same ``ingestion_id`` already journaled with the identical record is a
   retry: nothing is written and the original entry is returned
   (ALREADY_APPENDED); different content under it raises
   EquityCandleIngestionConflictError and nothing is written;
2. earlier entries of the same source identity -- source, source event and
   revision -- that contradict the record's bound facts raise
   EquityCandleSourceIdentityConflictError and nothing is written;
3. otherwise the record is appended (APPENDED). Earlier observations of its
   identity with other OHLCV values are named in the result; both are kept.

``read_window(...)`` returns, for one listing, timeframe and research window
and one watermark, every observation entry in that window at or below the
watermark, and every finality entry at or below it for those observations'
source identities, in sequence order. Entries above the watermark are never
returned, so a reader holding a watermark sees one fixed evidence set however
much is appended meanwhile. A watermark above the journal's raises
InvalidEquityCandleJournalQueryError.

``EquityCandleEvidenceSnapshotStore`` records immutable as-issued snapshots.
``issue(snapshot)`` requires, in the one write transaction that inserts it:

1. that the journal's ``read_window`` at the snapshot's watermark holds
   exactly the entries the snapshot lists, record for record; anything else
   raises EquityCandleSnapshotEvidenceError;
2. that the window's evidence at the journal's current watermark, when the
   transaction commits, is still exactly those entries. Evidence for the
   window committed after the snapshot's watermark -- a correction, a
   finality statement, a late candle -- raises EquityCandleSnapshotStaleError.
   Entries outside the window do not.

Either way nothing is written. So every issued snapshot is the window's
latest committed evidence at the moment it was recorded, never a cutoff that
silently omits evidence already committed; its watermark is the point that
evidence is reproduced from. The same snapshot
issued again is an idempotent no-op (0); different content under its identity
raises EquityCandleSnapshotConflictError. ``get_snapshot`` returns a snapshot
exactly as issued, after verifying it against its fingerprint and the journal;
it never recalculates one. Storing anything approves nothing: finality is
judged only by the EI-2A policy recorded in the snapshot.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from northstar_core.domain.value_objects import ListingReference
    from northstar_core.foundation.value_objects import PointInTime, Timeframe

    from northstar_application.application_services.equity_candle_evidence_journal import (
        EquityCandleAppendResult,
        EquityCandleJournalRecord,
        EquityCandleJournalWindow,
    )
    from northstar_application.application_services.equity_candle_evidence_snapshot import (
        EquityCandleEvidenceSnapshot,
    )


class EquityCandleIngestionConflictError(ValueError):
    """Raised when an ingestion identity is reused for a different record."""


class EquityCandleSourceIdentityConflictError(ValueError):
    """Raised when a record contradicts what its source identity is already bound to."""


class InvalidEquityCandleJournalQueryError(ValueError):
    """Raised when a journal query names a watermark the journal has not reached."""


class EquityCandleSnapshotEvidenceError(ValueError):
    """Raised when a snapshot's evidence is not exactly the journal's at its watermark."""


class EquityCandleSnapshotStaleError(ValueError):
    """Raised when the window gained evidence after a snapshot's watermark, before it committed."""


class EquityCandleSnapshotConflictError(ValueError):
    """Raised when a snapshot identity is already issued with different content."""


class EquityCandleEvidenceJournal(ABC):
    """Append-only journal of equity candle observations and finality statements."""

    @abstractmethod
    def append(self, record: EquityCandleJournalRecord) -> EquityCandleAppendResult:
        """Append ``record`` under the journal's rules, or return the entry a retry repeats."""

    @abstractmethod
    def watermark(self) -> int:
        """Return the highest committed journal sequence, 0 when the journal is empty."""

    @abstractmethod
    def read_window(
        self,
        listing: ListingReference,
        timeframe: Timeframe,
        opens_after: PointInTime,
        closes_at: PointInTime,
        watermark: int,
    ) -> EquityCandleJournalWindow:
        """Return the window's evidence journaled at or below ``watermark``."""


class EquityCandleEvidenceSnapshotStore(ABC):
    """Immutable store of as-issued equity candle evidence snapshots."""

    @abstractmethod
    def issue(self, snapshot: EquityCandleEvidenceSnapshot) -> int:
        """Record ``snapshot``: 1 when recorded, 0 when already recorded identically."""

    @abstractmethod
    def get_snapshot(self, snapshot_id: str) -> EquityCandleEvidenceSnapshot | None:
        """Return the snapshot issued under ``snapshot_id``, verified, if any."""
