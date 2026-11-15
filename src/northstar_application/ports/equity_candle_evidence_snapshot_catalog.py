"""Read-only catalog of EI-2B evidence snapshots for lifecycle monitoring (EI-6C, EI6C-P4).

The EI-6C Lifecycle Monitoring and Snapshot Selection Design v1.0, sections 4
and 5. The existing snapshot store answers by identity only
(``EquityCandleEvidenceSnapshotStore.get_snapshot``); monitoring needs to find
the snapshots of one session. This port lists them. It never creates, issues,
changes or deletes a snapshot, and it is not a second evidence authority: the
snapshot store stays authoritative, and the monitor re-verifies every entry
it is given against the authenticated snapshot.

``list_snapshots(scope, monitoring_instant, limit)``
----------------------------------------------------
Returns every stored snapshot that matches ``scope`` -- listing, trading
date, timeframe, research-window policy version and evidence class -- and is
**eligible at the monitoring instant ``T``**: ``as_of <= T`` **and**
``issued_at <= T``, compared as instants (``PointInTime.compare``), never as
stored text. Equality is eligible.

- **Complete and consistent.** One call returns the complete set, read from
  one consistent view of the store. A partial or internally inconsistent view
  is never returned as complete: it raises
  InvalidEquityCandleEvidenceSnapshotCatalogError (``INCOMPLETE`` or
  ``INTEGRITY``).
- **Bounded.** When more than ``limit`` snapshots match, the call raises
  EquityCandleEvidenceSnapshotCatalogLimitExceededError. It never returns the
  first ``limit`` entries as though they were all, and never thins or pages
  evidence away.
- **Order is not part of the contract.** The monitor orders entries itself,
  by instant.

Declared, not proven
--------------------
``issued_at`` is supplied by the caller under the EI-2B contract (it must only
be at or after ``as_of``), and nothing persisted records when a snapshot was
actually created. The ``issued_at <= T`` guard therefore refuses snapshots
that **declare** issuance after ``T``. It does not prove that a snapshot
existed at ``T``, and it cannot detect privileged backdating.

Venue
-----
The persisted snapshot table has no venue column, so the venue reference is
not a catalog filter: it is verified from the authenticated snapshot when the
lifecycle recorder evaluates it.

Nothing here reads a clock, writes anything or calls out.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, datetime
from typing import TYPE_CHECKING

from northstar_core.domain.value_objects import ListingReference
from northstar_core.foundation.value_objects import PointInTime, Timeframe

if TYPE_CHECKING:
    from northstar_application.application_services.equity_candle_evidence_snapshot import (
        EquityCandleEvidenceClass,
    )


def _evidence_class() -> type:
    # Imported lazily: the Application services import this port.
    from northstar_application.application_services.equity_candle_evidence_snapshot import (
        EquityCandleEvidenceClass,
    )

    return EquityCandleEvidenceClass


class EquityCandleEvidenceSnapshotCatalogUnavailableError(RuntimeError):
    """Raised when the catalog cannot be read; nothing is claimed about its contents."""


class EquityCandleEvidenceSnapshotCatalogLimitExceededError(RuntimeError):
    """Raised when more snapshots match than the governed limit; no entry is returned."""

    def __init__(self, limit: int, message: str) -> None:
        super().__init__(message)
        self.limit = limit


class InvalidEquityCandleEvidenceSnapshotCatalogError(ValueError):
    """Raised when a catalog answer cannot be trusted as the complete, valid candidate set.

    ``category`` is one of:

    - ``INTEGRITY``: an entry does not match the authenticated snapshot it names;
    - ``SCOPE``: an entry is outside the requested scope or not eligible at ``T``;
    - ``METADATA``: an entry's stored metadata is malformed;
    - ``INCOMPLETE``: the catalog could not establish a complete, consistent view.

    Permanent for the answer given; never treated as an empty catalog.
    """

    INTEGRITY = "INTEGRITY"
    SCOPE = "SCOPE"
    METADATA = "METADATA"
    INCOMPLETE = "INCOMPLETE"
    CATEGORIES = frozenset({INTEGRITY, SCOPE, METADATA, INCOMPLETE})

    def __init__(self, category: str, message: str) -> None:
        if category not in self.CATEGORIES:
            raise ValueError(f"category must be one of {sorted(self.CATEGORIES)}.")
        super().__init__(f"{category}: {message}")
        self.category = category


def _metadata(message: str) -> InvalidEquityCandleEvidenceSnapshotCatalogError:
    return InvalidEquityCandleEvidenceSnapshotCatalogError(
        InvalidEquityCandleEvidenceSnapshotCatalogError.METADATA, message
    )


@dataclass(frozen=True, slots=True)
class EquityCandleEvidenceSnapshotCatalogScope:
    """The session and evidence identity a catalog lists snapshots for."""

    listing: ListingReference
    trading_date: date
    timeframe: Timeframe
    window_policy_version: str
    evidence_class: EquityCandleEvidenceClass

    def __post_init__(self) -> None:
        if not isinstance(self.listing, ListingReference):
            raise TypeError("listing must be a ListingReference.")
        if isinstance(self.trading_date, datetime) or not isinstance(self.trading_date, date):
            raise TypeError("trading_date must be a plain date.")
        if not isinstance(self.timeframe, Timeframe):
            raise TypeError("timeframe must be a Timeframe.")
        if not isinstance(self.window_policy_version, str) or not self.window_policy_version:
            raise TypeError("window_policy_version must be non-empty text.")
        if not isinstance(self.evidence_class, _evidence_class()):
            raise TypeError("evidence_class must be an EquityCandleEvidenceClass.")


@dataclass(frozen=True, slots=True)
class EquityCandleEvidenceSnapshotCatalogEntry:
    """One catalog entry: a projection of a stored snapshot's metadata.

    Every field is a claim the monitor re-verifies against the authenticated
    snapshot (``get_snapshot``) before relying on it.
    """

    snapshot_id: str
    listing: ListingReference
    trading_date: date
    timeframe: Timeframe
    window_policy_version: str
    evidence_class: EquityCandleEvidenceClass
    as_of: PointInTime
    issued_at: PointInTime
    watermark: int
    manifest_fingerprint: str

    def __post_init__(self) -> None:
        for name in ("snapshot_id", "window_policy_version", "manifest_fingerprint"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise _metadata(f"{name} must be non-empty text.")
        if not isinstance(self.listing, ListingReference):
            raise _metadata("listing must be a ListingReference.")
        if isinstance(self.trading_date, datetime) or not isinstance(self.trading_date, date):
            raise _metadata("trading_date must be a plain date.")
        if not isinstance(self.timeframe, Timeframe):
            raise _metadata("timeframe must be a Timeframe.")
        if not isinstance(self.evidence_class, _evidence_class()):
            raise _metadata("evidence_class must be an EquityCandleEvidenceClass.")
        for name in ("as_of", "issued_at"):
            if not isinstance(getattr(self, name), PointInTime):
                raise _metadata(f"{name} must be a PointInTime.")
        if isinstance(self.watermark, bool) or not isinstance(self.watermark, int):
            raise _metadata("watermark must be an integer.")
        if self.watermark < 0:
            raise _metadata("watermark must not be negative.")


class EquityCandleEvidenceSnapshotCatalog(ABC):
    """Read-only listing of a session's eligible snapshots at a monitoring instant."""

    @abstractmethod
    def list_snapshots(
        self,
        scope: EquityCandleEvidenceSnapshotCatalogScope,
        monitoring_instant: PointInTime,
        limit: int,
    ) -> tuple[EquityCandleEvidenceSnapshotCatalogEntry, ...]:
        """Return every snapshot in ``scope`` with ``as_of`` and ``issued_at`` at or before ``T``.

        Raises EquityCandleEvidenceSnapshotCatalogLimitExceededError beyond
        ``limit``, EquityCandleEvidenceSnapshotCatalogUnavailableError when the
        catalog cannot be read, and InvalidEquityCandleEvidenceSnapshotCatalogError
        when no complete, consistent answer can be given. See the module docstring.
        """
