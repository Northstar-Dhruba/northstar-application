"""Verified recommendation detail and independent lifecycle summaries (EI-7C, EI7-CF4).

``GetEquityRecommendationDetailUseCase`` answers one authoritative MODELED
issuance in full (EI-7 Decision Register, sections 13.4 and 13.6; design
section 23.4):

1. the identity must have the ``eiris1`` shape, otherwise UNSUPPORTED_SCOPE;
2. ``EquityRecommendationLifecycleReader.load_lifecycle`` is the primary
   authority: the re-derived issuance and its Core-replayed lifecycle from one
   read transaction (EI-6A, EI-6B). None is ISSUANCE_NOT_FOUND;
3. the disposition is read under the issuance's **exact** generation key
   through ``EquityRecommendationGenerationRecordReader`` -- an independent
   read of insert-only records, guarded by the lineage comparisons, never a
   cross-connection atomic snapshot;
4. the ten lineage comparisons (equity_recommendation_read_lineage); any
   failure is PERSISTED_AUTHORITY_INVALID for the whole detail;
5. optional evidence panels for every snapshot the lifecycle cites and for
   the candidate's evidence snapshot (EI7-EC1).

Evidence panels never hide or downgrade a required fact:

- a lifecycle-cited snapshot that is not found (its reference has no foreign
  key) is ``UNAVAILABLE`` / ``EVIDENCE_NOT_FOUND``;
- the candidate's evidence snapshot not found is ``INVALID`` /
  ``EVIDENCE_REFERENCE_BROKEN`` (its reference is foreign-key bound);
- a stored snapshot that fails verification, answers another identity, or
  disagrees with the candidate's manifest is ``INVALID`` / ``EVIDENCE_INVALID``;
- a snapshot reader outage is ``UNAVAILABLE`` / ``DEPENDENCY_UNAVAILABLE``;
- a retrieved snapshot whose class is not SYNTHETIC refuses the **whole**
  detail with EVIDENCE_CLASS_UNSUPPORTED: its content is never exposed,
  relabelled or reported as a missing panel.

Corruption of a panel's snapshot never compromises the required lineage here,
because the lineage is proven from the disposition's frozen candidate
reference, never from a snapshot (EI-7 R-6).

``GetEquityLifecycleSummaryUseCase`` is the optional per-item lifecycle
summary of a discovery list (EI-7 register, section 13.2, item 7). It is an
**independent** read, made after any session read: never pinned to a session
watermark, never atomic with a list or with another summary. A corrupt or
unreadable lifecycle is that item's explicit ``INVALID`` or ``UNAVAILABLE``
status, never an omitted or successful-looking summary; corrupt issuance
authority refuses with PERSISTED_AUTHORITY_INVALID.

No fill, execution price, realized P&L, win rate, return, recording time or
monitoring time exists in any view: none is stored, so none is shown
(EI-7 R-9, R-11). Nothing here writes, reads a clock or calls out.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from northstar_core.recommendations.equity.lifecycle import (
    EquityLifecycleEvent,
    EquityLifecycleOutcomeClass,
    EquityLifecycleState,
    EquityRecommendationLifecycle,
)

from northstar_application.application_services.equity_candle_evidence_snapshot import (
    EquityCandleEvidenceClass,
    EquityCandleEvidenceSnapshot,
)
from northstar_application.application_services.equity_read_query_errors import (
    EquityReadQueryRefusal,
    refuse_equity_read_query,
)
from northstar_application.application_services.equity_recommendation_generation_decoding import (
    InvalidEquityRecommendationRecordError,
)
from northstar_application.application_services.equity_recommendation_issuance_records import (
    EquityRecordedIssuance,
    is_equity_recommendation_issuance_identity,
)
from northstar_application.application_services.equity_recommendation_read_lineage import (
    EquityVerifiedGenerationLineage,
    verify_equity_issued_generation_lineage,
)
from northstar_application.ports.equity_candle_evidence_journal import (
    EquityCandleEvidenceUnavailableError,
)
from northstar_application.ports.equity_candle_evidence_snapshot_reader import (
    EquityCandleEvidenceSnapshotReader,
    InvalidStoredEquityCandleEvidenceSnapshotError,
)
from northstar_application.ports.equity_recommendation_generation_store import (
    EquityRecommendationGenerationRecordReader,
    EquityRecommendationGenerationStoreUnavailableError,
)
from northstar_application.ports.equity_recommendation_issuance import (
    EquityRecommendationIssuanceUnavailableError,
    InvalidStoredEquityRecommendationIssuanceAuthorityError,
)
from northstar_application.ports.equity_recommendation_lifecycle_reader import (
    EquityRecommendationLifecycleReader,
)
from northstar_application.ports.equity_recommendation_lifecycle_store import (
    EquityRecommendationLifecycleStoreUnavailableError,
    EquityRecordedLifecycle,
    InvalidStoredEquityRecommendationLifecycleError,
)

_R = EquityReadQueryRefusal


class InvalidEquityRecommendationDetailViewError(ValueError):
    """Raised when a detail view, panel or summary is constructed inconsistently."""


def _fail(message: str) -> None:
    raise InvalidEquityRecommendationDetailViewError(message)


# ---------------------------------------------------------------------------
# Evidence panels (EI7-EC1)
# ---------------------------------------------------------------------------


class EquityEvidencePanelStatus(StrEnum):
    """Whether an optional evidence panel could be shown."""

    AVAILABLE = "AVAILABLE"
    UNAVAILABLE = "UNAVAILABLE"
    INVALID = "INVALID"


class EquityEvidencePanelReason(StrEnum):
    """Why an optional evidence panel is not available."""

    EVIDENCE_NOT_FOUND = "EVIDENCE_NOT_FOUND"
    EVIDENCE_REFERENCE_BROKEN = "EVIDENCE_REFERENCE_BROKEN"
    EVIDENCE_INVALID = "EVIDENCE_INVALID"
    DEPENDENCY_UNAVAILABLE = "DEPENDENCY_UNAVAILABLE"


class EquityEvidenceRole(StrEnum):
    """Why a snapshot is cited by a recommendation detail."""

    CANDIDATE = "CANDIDATE"
    LIFECYCLE_EVENT = "LIFECYCLE_EVENT"


_PANEL_REASONS = {
    EquityEvidencePanelStatus.UNAVAILABLE: frozenset(
        {
            EquityEvidencePanelReason.EVIDENCE_NOT_FOUND,
            EquityEvidencePanelReason.DEPENDENCY_UNAVAILABLE,
        }
    ),
    EquityEvidencePanelStatus.INVALID: frozenset(
        {
            EquityEvidencePanelReason.EVIDENCE_REFERENCE_BROKEN,
            EquityEvidencePanelReason.EVIDENCE_INVALID,
        }
    ),
}


@dataclass(frozen=True, slots=True)
class EquityEvidencePanel:
    """One cited EI-2B snapshot, shown only when available and SYNTHETIC.

    ``snapshot`` is present exactly when ``status`` is AVAILABLE, and is then
    a verified SYNTHETIC snapshot under ``snapshot_id``. Otherwise ``reason``
    says why it is not shown; no snapshot content is kept.
    """

    snapshot_id: str
    roles: tuple[EquityEvidenceRole, ...]
    status: EquityEvidencePanelStatus
    reason: EquityEvidencePanelReason | None
    snapshot: EquityCandleEvidenceSnapshot | None

    def __post_init__(self) -> None:
        subject = "EquityEvidencePanel"
        if not isinstance(self.snapshot_id, str) or not self.snapshot_id:
            _fail(f"{subject} snapshot_id must be non-empty text.")
        if (
            not isinstance(self.roles, tuple)
            or not self.roles
            or not all(isinstance(r, EquityEvidenceRole) for r in self.roles)
            or len(set(self.roles)) != len(self.roles)
            or list(self.roles) != sorted(self.roles)
        ):
            _fail(f"{subject} roles must be a sorted tuple of distinct roles.")
        if not isinstance(self.status, EquityEvidencePanelStatus):
            _fail(f"{subject} status must be an EquityEvidencePanelStatus.")
        if self.status is EquityEvidencePanelStatus.AVAILABLE:
            snapshot = self.snapshot
            if (
                self.reason is not None
                or not isinstance(snapshot, EquityCandleEvidenceSnapshot)
                or snapshot.snapshot_id != self.snapshot_id
                or snapshot.evidence_class is not EquityCandleEvidenceClass.SYNTHETIC
            ):
                _fail(f"{subject} AVAILABLE holds exactly its verified SYNTHETIC snapshot.")
        elif (
            self.snapshot is not None
            or not isinstance(self.reason, EquityEvidencePanelReason)
            or self.reason not in _PANEL_REASONS[self.status]
        ):
            _fail(f"{subject} {self.status.value} holds no snapshot and a matching reason.")


def _cited(
    lineage: EquityVerifiedGenerationLineage, events: tuple[EquityLifecycleEvent, ...]
) -> dict[str, set[EquityEvidenceRole]]:
    cited: dict[str, set[EquityEvidenceRole]] = {}
    cited.setdefault(lineage.research_reference.evidence_snapshot_id, set()).add(
        EquityEvidenceRole.CANDIDATE
    )
    for event in events:
        superseded = event.superseded_evidence
        references = (event.evidence.reference,) + (
            () if superseded is None else (superseded.reference,)
        )
        for reference in references:
            cited.setdefault(reference, set()).add(EquityEvidenceRole.LIFECYCLE_EVENT)
    return cited


def _panel(
    snapshots: EquityCandleEvidenceSnapshotReader,
    snapshot_id: str,
    roles: set[EquityEvidenceRole],
    lineage: EquityVerifiedGenerationLineage,
) -> EquityEvidencePanel:
    ordered = tuple(sorted(roles))
    candidate = EquityEvidenceRole.CANDIDATE in roles

    def unavailable(reason: EquityEvidencePanelReason) -> EquityEvidencePanel:
        return EquityEvidencePanel(
            snapshot_id, ordered, EquityEvidencePanelStatus.UNAVAILABLE, reason, None
        )

    def invalid(reason: EquityEvidencePanelReason) -> EquityEvidencePanel:
        return EquityEvidencePanel(
            snapshot_id, ordered, EquityEvidencePanelStatus.INVALID, reason, None
        )

    try:
        snapshot = snapshots.get_snapshot(snapshot_id)
    except InvalidStoredEquityCandleEvidenceSnapshotError:
        return invalid(EquityEvidencePanelReason.EVIDENCE_INVALID)
    except EquityCandleEvidenceUnavailableError:
        return unavailable(EquityEvidencePanelReason.DEPENDENCY_UNAVAILABLE)
    if snapshot is None:
        if candidate:
            return invalid(EquityEvidencePanelReason.EVIDENCE_REFERENCE_BROKEN)
        return unavailable(EquityEvidencePanelReason.EVIDENCE_NOT_FOUND)
    if not isinstance(snapshot, EquityCandleEvidenceSnapshot):
        return invalid(EquityEvidencePanelReason.EVIDENCE_INVALID)
    # EI7-EC1: any retrieved non-SYNTHETIC snapshot refuses the response, before its
    # identity or anything else about it is judged or presented.
    if snapshot.evidence_class is not EquityCandleEvidenceClass.SYNTHETIC:
        raise refuse_equity_read_query(
            _R.EVIDENCE_CLASS_UNSUPPORTED, "a cited snapshot is not SYNTHETIC evidence"
        )
    if snapshot.snapshot_id != snapshot_id:
        return invalid(EquityEvidencePanelReason.EVIDENCE_INVALID)
    if candidate and snapshot.manifest_fingerprint != lineage.candidate.manifest_fingerprint:
        return invalid(EquityEvidencePanelReason.EVIDENCE_INVALID)
    return EquityEvidencePanel(
        snapshot_id, ordered, EquityEvidencePanelStatus.AVAILABLE, None, snapshot
    )


# ---------------------------------------------------------------------------
# The detail view
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EquityRecommendationDetailView:
    """One verified, MODELED, synthetic issued recommendation in full.

    ``issuance`` and ``lifecycle`` come from one lifecycle read; ``lineage``
    holds the verified generation facts (policy and version identities, the
    frozen research reference, sizing and modeled cost, risk and reward);
    ``evidence`` holds one panel per cited snapshot, candidate first, then in
    order of first citation. ``lifecycle.events`` is the complete
    authoritative event sequence, in event-sequence order.
    """

    issuance: EquityRecordedIssuance
    lifecycle: EquityRecommendationLifecycle
    lineage: EquityVerifiedGenerationLineage
    evidence: tuple[EquityEvidencePanel, ...]

    def __post_init__(self) -> None:
        subject = "EquityRecommendationDetailView"
        if not isinstance(self.issuance, EquityRecordedIssuance):
            _fail(f"{subject} issuance must be an EquityRecordedIssuance.")
        if not isinstance(self.lifecycle, EquityRecommendationLifecycle):
            _fail(f"{subject} lifecycle must be an EquityRecommendationLifecycle.")
        if not isinstance(self.lineage, EquityVerifiedGenerationLineage):
            _fail(f"{subject} lineage must be an EquityVerifiedGenerationLineage.")
        if self.lineage.issuance != self.issuance:
            _fail(f"{subject} lineage is for another issuance.")
        if not isinstance(self.evidence, tuple) or not all(
            isinstance(p, EquityEvidencePanel) for p in self.evidence
        ):
            _fail(f"{subject} evidence must be a tuple of EquityEvidencePanel.")
        identities = [p.snapshot_id for p in self.evidence]
        if len(set(identities)) != len(identities):
            _fail(f"{subject} has one panel per cited snapshot.")
        expected = _cited(self.lineage, self.lifecycle.events)
        if {p.snapshot_id: set(p.roles) for p in self.evidence} != expected:
            _fail(f"{subject} evidence panels must be exactly the cited snapshots.")

    @property
    def issuance_id(self) -> str:
        """Return the authoritative issuance identity."""
        return self.issuance.recommendation.issuance_id

    @property
    def state(self) -> EquityLifecycleState:
        """Return the modeled lifecycle state."""
        return self.lifecycle.state

    @property
    def is_terminal(self) -> bool:
        """Return whether the modeled lifecycle is terminal."""
        return self.lifecycle.is_terminal

    @property
    def outcome_class(self) -> EquityLifecycleOutcomeClass | None:
        """Return the outcome class of a terminal state, or None while open."""
        return self.lifecycle.state.outcome_class

    @property
    def position(self) -> int:
        """Return the number of recorded lifecycle events (not a ledger position)."""
        return len(self.lifecycle.events)

    @property
    def events(self) -> tuple[EquityLifecycleEvent, ...]:
        """Return the complete authoritative lifecycle events, by event sequence."""
        return self.lifecycle.events

    @property
    def is_synthetic(self) -> bool:
        """Return True: every input is synthetic."""
        return True

    @property
    def is_actionable(self) -> bool:
        """Return False: a MODELED issued recommendation is never actionable."""
        return False


def _require_identity(issuance_id: object) -> None:
    if not is_equity_recommendation_issuance_identity(issuance_id):
        raise refuse_equity_read_query(
            _R.UNSUPPORTED_SCOPE, "the issuance identity is not an eiris1 identity"
        )


def _load_lifecycle(
    lifecycles: EquityRecommendationLifecycleReader, issuance_id: str
) -> EquityRecordedLifecycle | None:
    try:
        recorded = lifecycles.load_lifecycle(issuance_id)
    except InvalidStoredEquityRecommendationIssuanceAuthorityError as exc:
        raise refuse_equity_read_query(
            _R.PERSISTED_AUTHORITY_INVALID, "the issuance authority is invalid"
        ) from exc
    if recorded is not None and (
        not isinstance(recorded, EquityRecordedLifecycle) or recorded.issuance_id != issuance_id
    ):
        raise refuse_equity_read_query(
            _R.PERSISTED_AUTHORITY_INVALID, "the lifecycle reader answered another issuance"
        )
    return recorded


class GetEquityRecommendationDetailUseCase:
    """Return the verified detail of one authoritative issuance (EI7-CF4, EI7-EC1)."""

    def __init__(
        self,
        lifecycles: EquityRecommendationLifecycleReader,
        generations: EquityRecommendationGenerationRecordReader,
        snapshots: EquityCandleEvidenceSnapshotReader,
    ) -> None:
        if not isinstance(lifecycles, EquityRecommendationLifecycleReader):
            raise TypeError("lifecycles must be an EquityRecommendationLifecycleReader.")
        if not isinstance(generations, EquityRecommendationGenerationRecordReader):
            raise TypeError("generations must be an EquityRecommendationGenerationRecordReader.")
        if not isinstance(snapshots, EquityCandleEvidenceSnapshotReader):
            raise TypeError("snapshots must be an EquityCandleEvidenceSnapshotReader.")
        self._lifecycles = lifecycles
        self._generations = generations
        self._snapshots = snapshots

    def execute(self, issuance_id: str) -> EquityRecommendationDetailView:
        """Refuse with EquityReadQueryRefusedError as the module docstring states."""
        _require_identity(issuance_id)
        try:
            recorded = _load_lifecycle(self._lifecycles, issuance_id)
        except InvalidStoredEquityRecommendationLifecycleError as exc:
            raise refuse_equity_read_query(
                _R.LIFECYCLE_HISTORY_INVALID, "the lifecycle history is invalid"
            ) from exc
        except (
            EquityRecommendationLifecycleStoreUnavailableError,
            EquityRecommendationIssuanceUnavailableError,
        ) as exc:
            raise refuse_equity_read_query(
                _R.DEPENDENCY_UNAVAILABLE, "the lifecycle authority is unavailable"
            ) from exc
        if recorded is None:
            raise refuse_equity_read_query(_R.ISSUANCE_NOT_FOUND, "no such issuance is recorded")
        issuance = recorded.issuance
        try:
            disposition = self._generations.load_disposition(issuance.recommendation.generation_key)
        except InvalidEquityRecommendationRecordError as exc:
            raise refuse_equity_read_query(
                _R.PERSISTED_AUTHORITY_INVALID, "the issuance's generation record is invalid"
            ) from exc
        except EquityRecommendationGenerationStoreUnavailableError as exc:
            raise refuse_equity_read_query(
                _R.DEPENDENCY_UNAVAILABLE, "the generation records are unavailable"
            ) from exc
        lineage = verify_equity_issued_generation_lineage(issuance, disposition)
        cited = _cited(lineage, recorded.lifecycle.events)
        panels = tuple(
            _panel(self._snapshots, snapshot_id, roles, lineage)
            for snapshot_id, roles in cited.items()
        )
        return EquityRecommendationDetailView(issuance, recorded.lifecycle, lineage, panels)


# ---------------------------------------------------------------------------
# Independent per-item lifecycle summaries (EI-7 register, section 13.2, item 7)
# ---------------------------------------------------------------------------


class EquityLifecycleSummaryStatus(StrEnum):
    """Whether one item's independently read lifecycle summary could be observed."""

    AVAILABLE = "AVAILABLE"
    INVALID = "INVALID"
    UNAVAILABLE = "UNAVAILABLE"


@dataclass(frozen=True, slots=True)
class EquityLifecycleSummary:
    """One issuance's lifecycle summary, read independently of any session read.

    It is never pinned to a session recording watermark and never atomic with
    a list or another summary. ``state``, ``is_terminal`` and ``position`` are
    present exactly when AVAILABLE; ``outcome_class`` only for a terminal
    state. ``position`` counts lifecycle events; it is not a ledger position.
    """

    issuance_id: str
    status: EquityLifecycleSummaryStatus
    state: EquityLifecycleState | None
    is_terminal: bool | None
    outcome_class: EquityLifecycleOutcomeClass | None
    position: int | None

    def __post_init__(self) -> None:
        subject = "EquityLifecycleSummary"
        if not is_equity_recommendation_issuance_identity(self.issuance_id):
            _fail(f"{subject} issuance_id must be an eiris1 identity.")
        if not isinstance(self.status, EquityLifecycleSummaryStatus):
            _fail(f"{subject} status must be an EquityLifecycleSummaryStatus.")
        if self.status is EquityLifecycleSummaryStatus.AVAILABLE:
            if not isinstance(self.state, EquityLifecycleState):
                _fail(f"{subject} AVAILABLE states the lifecycle state.")
            assert isinstance(self.state, EquityLifecycleState)
            position = self.position
            if (
                self.is_terminal is not self.state.is_terminal
                or self.outcome_class is not self.state.outcome_class
                or isinstance(position, bool)
                or not isinstance(position, int)
                or position < 0
            ):
                _fail(f"{subject} AVAILABLE facts must agree with the state.")
        elif (self.state, self.is_terminal, self.outcome_class, self.position) != (
            None,
            None,
            None,
            None,
        ):
            _fail(f"{subject} {self.status.value} states no lifecycle facts.")


class GetEquityLifecycleSummaryUseCase:
    """Return one issuance's lifecycle summary by an independent read (optional field)."""

    def __init__(self, lifecycles: EquityRecommendationLifecycleReader) -> None:
        if not isinstance(lifecycles, EquityRecommendationLifecycleReader):
            raise TypeError("lifecycles must be an EquityRecommendationLifecycleReader.")
        self._lifecycles = lifecycles

    def execute(self, issuance_id: str) -> EquityLifecycleSummary:
        """Return the summary, or refuse as the module docstring states."""
        _require_identity(issuance_id)
        status = EquityLifecycleSummaryStatus
        try:
            recorded = _load_lifecycle(self._lifecycles, issuance_id)
        except InvalidStoredEquityRecommendationLifecycleError:
            return EquityLifecycleSummary(issuance_id, status.INVALID, None, None, None, None)
        except (
            EquityRecommendationLifecycleStoreUnavailableError,
            EquityRecommendationIssuanceUnavailableError,
        ):
            return EquityLifecycleSummary(issuance_id, status.UNAVAILABLE, None, None, None, None)
        if recorded is None:
            raise refuse_equity_read_query(_R.ISSUANCE_NOT_FOUND, "no such issuance is recorded")
        state = recorded.lifecycle.state
        return EquityLifecycleSummary(
            issuance_id,
            status.AVAILABLE,
            state,
            state.is_terminal,
            state.outcome_class,
            recorded.position,
        )
