"""Read-only view of issued equity candle evidence snapshots (EI-7C, EI7-P4, EI7-FC6).

EI-7 read queries and research rehydration only ever *read* EI-2B snapshots.
This protocol names exactly the read operation of the EI-2B snapshot store,
``get_snapshot``, so a holder of it cannot issue a snapshot. It adds no
operation and changes no EI-2B contract: every existing
``EquityCandleEvidenceSnapshotStore`` satisfies it structurally, and a
read-only adapter (EI-7D) satisfies it without any write method. No
implementation needs, or may add, a refusing ``issue()`` to satisfy it.

``get_snapshot(snapshot_id)`` returns the snapshot issued under
``snapshot_id``, exactly as issued and verified, or None when none was issued.

Failures
--------
- EquityCandleEvidenceUnavailableError (EI-2B): the store could not be read.
  Existing EI-2B stores also raise it for a stored snapshot that fails
  verification; the port contract of EI-2B is unchanged.
- InvalidStoredEquityCandleEvidenceSnapshotError (EI-7C): a stored snapshot
  exists but fails verification -- it does not decode, re-derive its
  fingerprints or agree with its journal. It is a subclass of the EI-2B
  unavailability error, so every existing handler keeps its behaviour, while
  an EI-7 read query can tell a corrupt snapshot (an ``invalid`` evidence
  panel, EI7-EC1) from an outage (an ``unavailable`` panel). A read-only
  adapter should raise it for verification failures; nothing is repaired.

``runtime_checkable`` structural checks only that ``get_snapshot`` exists, not
how it behaves (EI7-FC6); behaviour is proven by contract tests.

Nothing here imports a database, reads a clock or calls out.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from northstar_application.ports.equity_candle_evidence_journal import (
    EquityCandleEvidenceUnavailableError,
)

if TYPE_CHECKING:
    from northstar_application.application_services.equity_candle_evidence_snapshot import (
        EquityCandleEvidenceSnapshot,
    )


class InvalidStoredEquityCandleEvidenceSnapshotError(EquityCandleEvidenceUnavailableError):
    """Raised when a stored snapshot exists but fails verification (EI-7C, EI7-EC1).

    A **permanent** integrity failure of stored evidence: never retryable and
    never a statement that the snapshot is missing. It subclasses
    EquityCandleEvidenceUnavailableError so existing EI-2B handlers are
    unchanged. Messages carry no stored evidence content.
    """

    @property
    def retryable(self) -> bool:
        """Always False: corrupt stored evidence never clears by retrying."""
        return False


@runtime_checkable
class EquityCandleEvidenceSnapshotReader(Protocol):
    """Read verified EI-2B snapshots by identity; None when none was issued."""

    def get_snapshot(self, snapshot_id: str) -> EquityCandleEvidenceSnapshot | None:
        """Return the snapshot issued under ``snapshot_id``, verified, or None."""
