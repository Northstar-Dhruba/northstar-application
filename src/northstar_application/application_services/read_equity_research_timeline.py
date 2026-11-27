"""Persisted research inspection for one strategy, listing and date (EI-7C, EI7-P3, EI7-P5).

``GetEquityResearchTimelineUseCase`` answers only what persisted research and
generation records establish (EI-7 Decision Register, sections 13.4 and 13.12;
design section 23.5). It never evaluates fresh market data, never writes and
never guesses a policy set.

1. **Timeline.** The stored EI-3C timeline is read through
   ``EquityIntradayResearchTimelineReader``. None means the research timeline
   is UNKNOWN -- an authorized exact lookup found no record.
2. **Configuration (EI7-FC3).** The stored, authoritative configuration
   document is decoded exactly (no binary floats) and must re-encode to the
   identical canonical text, with the timeline's strategy. No current,
   default or client-supplied configuration is ever substituted; there is no
   configuration fingerprint to invent. A missing or inconsistent
   configuration refuses the whole response with PERSISTED_AUTHORITY_INVALID.
3. **Evidence-class gate (EI7-EC1, EI7-FC3).** Every snapshot the stored
   entries cite is retrieved through ``EquityCandleEvidenceSnapshotReader``
   and gated **before** any decision is re-derived or presented: a
   non-SYNTHETIC snapshot refuses the whole response with
   EVIDENCE_CLASS_UNSUPPORTED. The current evaluator would otherwise turn such
   evidence into an ordinary ``REJECTED`` decision (``EVIDENCE_NOT_SYNTHETIC``),
   which is never presented. A missing or corrupt cited snapshot is required
   provenance of the primary timeline: PERSISTED_AUTHORITY_INVALID.
4. **Rehydration (EI3-D13, EI7-P5).** The accepted
   ``rehydrate_equity_intraday_research_timeline`` verifies the whole stored
   history -- every decision re-derived, every decision and entry fingerprint
   re-computed -- over exactly the gated snapshots, through a reader-only
   view (EI7-P4). Any failure refuses the whole response with
   PERSISTED_AUTHORITY_INVALID; it is never downgraded to UNKNOWN.
5. **Issuance-linked generation (EI7-P3, EI7-FC2).** The authority's issuance
   for the strategy, listing and date, if any, supplies the exact generation
   key. Its disposition must pass the ten lineage comparisons, cite exactly
   the timeline's CANDIDATE entry, and carry exactly the stored research
   configuration; otherwise PERSISTED_AUTHORITY_INVALID. The generation status
   is then PROPOSED and the issuance status ISSUED.
6. **Otherwise NOT_QUERYABLE.** Without an issuance there is no exact
   generation identity in the approved scope: the generation status is
   NOT_QUERYABLE with a typed reason -- NO_RESEARCH_TIMELINE,
   NO_CANDIDATE_ENTRY or NO_RECORDED_ISSUANCE. It never says that no
   generation was attempted, and NOT_ELIGIBLE or UNRESOLVED are never
   inferred. No policy-set parameter, catalog read, disposition or attempt
   scan, or latest-version guess exists here.

Persisted NO_SIGNAL and REJECTED entries are shown with their stored outcomes
and reasons, in stored sequence order (EI7-FC2). Explanation prose is never
stored; only what EI3-D13 re-derives is held.

**Limitation (EI7-FC2):** full unissued-generation history is not observable
here. The view refuses to state NOT_ELIGIBLE, UNRESOLVED or a generation
UNKNOWN: each needs a separately governed exact lookup that supplies the
record.

A permanent read-storage failure (``EquityReadStorageIntegrityError``,
EI7-SI1) from any reader -- timeline, snapshot, issuance or generation
record -- refuses the whole response with PERSISTED_AUTHORITY_INVALID, not
retryable (SI-A). It establishes nothing about any record, so it never
becomes an UNKNOWN timeline, NOT_QUERYABLE or any of its reasons, a
re-derived decision, EVIDENCE_CLASS_UNSUPPORTED or DEPENDENCY_UNAVAILABLE.

Nothing here writes, reads a clock or calls out.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum
from typing import Any, NoReturn

from northstar_core.domain.value_objects import ListingReference
from northstar_core.strategy import StrategyIdentity

from northstar_application.application_services.advance_equity_intraday_research_timeline import (
    StoredEquityIntradayResearchTimeline,
    rehydrate_equity_intraday_research_timeline,
)
from northstar_application.application_services.equity_candle_evidence_journal import (
    canonical_equity_candle_json,
)
from northstar_application.application_services.equity_candle_evidence_snapshot import (
    EquityCandleEvidenceClass,
    EquityCandleEvidenceSnapshot,
)
from northstar_application.application_services.equity_read_query_errors import (
    EquityReadQueryRefusal,
    refuse_equity_read_query,
    refusing_equity_read_storage_failure,
)
from northstar_application.application_services.equity_recommendation_generation_decoding import (
    InvalidEquityRecommendationRecordError,
    decode_equity_research_configuration,
)
from northstar_application.application_services.equity_recommendation_issuance_records import (
    EquityRecordedIssuance,
)
from northstar_application.application_services.equity_recommendation_read_lineage import (
    EquityVerifiedGenerationLineage,
    verify_equity_issued_generation_lineage,
)
from northstar_application.application_services.evaluate_equity_intraday_research import (
    EquityOpeningRangeBreakoutConfiguration,
)
from northstar_application.application_services.read_equity_session_issuances import (
    EQUITY_READ_SUPPORTED_EXCHANGES,
)
from northstar_application.application_services.schedule_equity_intraday_research import (
    EquityIntradayResearchTimeline,
    EquityIntradayResearchTimelineEntry,
    EquityIntradayResearchTimelineIdentity,
    EquityIntradayResearchTimelineRole,
)
from northstar_application.ports.equity_candle_evidence_journal import (
    EquityCandleEvidenceUnavailableError,
)
from northstar_application.ports.equity_candle_evidence_snapshot_reader import (
    EquityCandleEvidenceSnapshotReader,
    InvalidStoredEquityCandleEvidenceSnapshotError,
)
from northstar_application.ports.equity_intraday_research_timeline_reader import (
    EquityIntradayResearchTimelineReader,
)
from northstar_application.ports.equity_intraday_research_timeline_store import (
    EquityIntradayResearchTimelineConfigurationConflictError,
    EquityIntradayResearchTimelineUnavailableError,
    InvalidStoredEquityIntradayResearchTimelineError,
)
from northstar_application.ports.equity_recommendation_generation_store import (
    EquityRecommendationGenerationRecordReader,
    EquityRecommendationGenerationStoreUnavailableError,
)
from northstar_application.ports.equity_recommendation_issuance import (
    EquityRecommendationIssuanceReader,
    EquityRecommendationIssuanceUnavailableError,
    InvalidStoredEquityRecommendationIssuanceAuthorityError,
)

_R = EquityReadQueryRefusal
_INVALID = EquityReadQueryRefusal.PERSISTED_AUTHORITY_INVALID


class InvalidEquityResearchTimelineViewError(ValueError):
    """Raised when a research view is constructed inconsistently."""


def _fail(message: str) -> NoReturn:
    raise InvalidEquityResearchTimelineViewError(message)


class EquityResearchTimelineStatus(StrEnum):
    """Whether a persisted research timeline exists for the strategy, listing and date."""

    STORED = "STORED"
    UNKNOWN = "UNKNOWN"


class EquityResearchGenerationStatus(StrEnum):
    """The generation vocabulary; distinct values never stand in for each other.

    EI-7C can state only NOT_QUERYABLE or PROPOSED (issuance-linked). UNKNOWN,
    NOT_ELIGIBLE and UNRESOLVED need a separately governed exact lookup that
    supplies the record, so the EI-7C view refuses to state them.
    """

    UNKNOWN = "UNKNOWN"
    NOT_QUERYABLE = "NOT_QUERYABLE"
    PROPOSED = "PROPOSED"
    NOT_ELIGIBLE = "NOT_ELIGIBLE"
    UNRESOLVED = "UNRESOLVED"


class EquityGenerationNotQueryableReason(StrEnum):
    """Why no exact generation identity is available in the approved scope."""

    NO_RESEARCH_TIMELINE = "NO_RESEARCH_TIMELINE"
    NO_CANDIDATE_ENTRY = "NO_CANDIDATE_ENTRY"
    NO_RECORDED_ISSUANCE = "NO_RECORDED_ISSUANCE"


class EquityResearchIssuanceStatus(StrEnum):
    """Whether the authority records an issuance for the strategy, listing and date."""

    ISSUED = "ISSUED"
    NOT_RECORDED = "NOT_RECORDED"


_EI7C_GENERATION_STATUSES = frozenset(
    {EquityResearchGenerationStatus.NOT_QUERYABLE, EquityResearchGenerationStatus.PROPOSED}
)


def _candidate(
    entries: tuple[EquityIntradayResearchTimelineEntry, ...],
) -> EquityIntradayResearchTimelineEntry | None:
    return next(
        (e for e in entries if e.role is EquityIntradayResearchTimelineRole.CANDIDATE), None
    )


@dataclass(frozen=True, slots=True)
class EquityResearchTimelineView:
    """Persisted research and its issuance-linked generation, as far as stored facts go.

    ``entries`` are the rehydrated, verified EI-3C entries in stored sequence
    order, each with its stored decision outcome and reasons. ``lineage`` is
    present exactly when the generation status is PROPOSED, and is then the
    verified lineage of ``issuance``.
    """

    strategy: StrategyIdentity
    listing: ListingReference
    trading_date: date
    timeline_status: EquityResearchTimelineStatus
    configuration: EquityOpeningRangeBreakoutConfiguration | None
    entries: tuple[EquityIntradayResearchTimelineEntry, ...]
    generation_status: EquityResearchGenerationStatus
    not_queryable_reason: EquityGenerationNotQueryableReason | None
    lineage: EquityVerifiedGenerationLineage | None
    issuance_status: EquityResearchIssuanceStatus
    issuance: EquityRecordedIssuance | None

    def __post_init__(self) -> None:
        subject = "EquityResearchTimelineView"
        if not isinstance(self.strategy, StrategyIdentity):
            _fail(f"{subject} strategy must be a StrategyIdentity.")
        if not isinstance(self.listing, ListingReference):
            _fail(f"{subject} listing must be a ListingReference.")
        if isinstance(self.trading_date, datetime) or not isinstance(self.trading_date, date):
            _fail(f"{subject} trading_date must be a plain date.")
        if not isinstance(self.entries, tuple) or not all(
            isinstance(e, EquityIntradayResearchTimelineEntry) for e in self.entries
        ):
            _fail(f"{subject} entries must be a tuple of timeline entries.")
        stored = self.timeline_status is EquityResearchTimelineStatus.STORED
        if not isinstance(self.timeline_status, EquityResearchTimelineStatus):
            _fail(f"{subject} timeline_status must be an EquityResearchTimelineStatus.")
        if stored:
            configuration = self.configuration
            if not isinstance(configuration, EquityOpeningRangeBreakoutConfiguration):
                _fail(f"{subject} a STORED timeline states its configuration.")
            if not self.entries:
                _fail(f"{subject} a STORED timeline has at least one entry (EI3-D11).")
            identity = EquityIntradayResearchTimelineIdentity(
                self.listing, self.trading_date, configuration
            ).document()
            if configuration.strategy != self.strategy or any(
                e.identity.document() != identity for e in self.entries
            ):
                _fail(f"{subject} entries belong to another strategy, listing or date.")
            if [e.sequence for e in self.entries] != list(range(1, len(self.entries) + 1)):
                _fail(f"{subject} entries must be in stored sequence order.")
        elif self.configuration is not None or self.entries:
            _fail(f"{subject} an UNKNOWN timeline states no configuration and no entries.")
        if not isinstance(self.generation_status, EquityResearchGenerationStatus):
            _fail(f"{subject} generation_status must be an EquityResearchGenerationStatus.")
        if self.generation_status not in _EI7C_GENERATION_STATUSES:
            _fail(
                f"{subject} cannot state {self.generation_status!r}: it needs a separately "
                "governed exact generation lookup."
            )
        candidate = _candidate(self.entries)
        if self.issuance_status is EquityResearchIssuanceStatus.ISSUED:
            if not isinstance(self.issuance, EquityRecordedIssuance):
                _fail(f"{subject} ISSUED names the recorded issuance.")
        elif self.issuance_status is EquityResearchIssuanceStatus.NOT_RECORDED:
            if self.issuance is not None:
                _fail(f"{subject} NOT_RECORDED names no issuance.")
        else:
            _fail(f"{subject} issuance_status must be an EquityResearchIssuanceStatus.")
        if self.generation_status is EquityResearchGenerationStatus.PROPOSED:
            if (
                not isinstance(self.lineage, EquityVerifiedGenerationLineage)
                or self.issuance is None
                or self.lineage.issuance != self.issuance
                or self.not_queryable_reason is not None
                or candidate is None
                or self.lineage.research_reference.candidate_entry_fingerprint
                != candidate.fingerprint
            ):
                _fail(f"{subject} PROPOSED is the verified lineage of the issued candidate.")
        else:
            reasons = EquityGenerationNotQueryableReason
            expected = (
                reasons.NO_RESEARCH_TIMELINE
                if not stored
                else reasons.NO_CANDIDATE_ENTRY
                if candidate is None
                else reasons.NO_RECORDED_ISSUANCE
            )
            if (
                self.lineage is not None
                or self.issuance is not None
                or self.not_queryable_reason is not expected
            ):
                _fail(f"{subject} NOT_QUERYABLE states exactly why, and holds no generation.")

    @property
    def candidate_entry(self) -> EquityIntradayResearchTimelineEntry | None:
        """Return the timeline's CANDIDATE entry, or None when it has none."""
        return _candidate(self.entries)


class _GatedSnapshots:
    """A reader-only view of snapshots already retrieved and class-gated (EI7-P4).

    It has no write method; rehydration sees exactly the gated objects.
    """

    def __init__(self, snapshots: dict[str, EquityCandleEvidenceSnapshot]) -> None:
        self._snapshots = dict(snapshots)

    def get_snapshot(self, snapshot_id: str) -> EquityCandleEvidenceSnapshot | None:
        return self._snapshots.get(snapshot_id)


def _refuse_float(value: str) -> NoReturn:
    raise ValueError(f"A stored configuration holds no floating-point number ({value}).")


def _configuration(
    stored: StoredEquityIntradayResearchTimeline,
) -> EquityOpeningRangeBreakoutConfiguration:
    text = stored.configuration_document
    try:
        document: Any = json.loads(text, parse_float=_refuse_float, parse_constant=_refuse_float)
        if not isinstance(document, dict):
            raise ValueError("A stored configuration is a JSON object.")
        configuration = decode_equity_research_configuration(document)
        canonical = canonical_equity_candle_json(configuration.document())
    except (ValueError, TypeError, InvalidEquityRecommendationRecordError) as exc:
        raise refuse_equity_read_query(
            _INVALID, "the stored research configuration is invalid"
        ) from exc
    if canonical != text or configuration.strategy != stored.strategy:
        raise refuse_equity_read_query(
            _INVALID, "the stored research configuration is not the timeline's canonical one"
        )
    return configuration


def _gated(
    snapshots: EquityCandleEvidenceSnapshotReader, stored: StoredEquityIntradayResearchTimeline
) -> dict[str, EquityCandleEvidenceSnapshot]:
    retrieved: dict[str, EquityCandleEvidenceSnapshot] = {}
    for entry in stored.entries:
        snapshot_id = entry.snapshot_id
        if snapshot_id in retrieved:
            continue
        try:
            with refusing_equity_read_storage_failure():  # EI7-SI1
                snapshot = snapshots.get_snapshot(snapshot_id)
        except InvalidStoredEquityCandleEvidenceSnapshotError as exc:
            raise refuse_equity_read_query(
                _INVALID, "a research entry's snapshot is corrupt"
            ) from exc
        except EquityCandleEvidenceUnavailableError as exc:
            raise refuse_equity_read_query(
                _R.DEPENDENCY_UNAVAILABLE, "the evidence snapshots are unavailable"
            ) from exc
        if not isinstance(snapshot, EquityCandleEvidenceSnapshot):
            raise refuse_equity_read_query(_INVALID, "a research entry's snapshot is missing")
        # EI7-EC1, EI7-FC3: any retrieved non-SYNTHETIC snapshot refuses the response,
        # before its identity is judged and before any decision is re-derived or presented.
        if snapshot.evidence_class is not EquityCandleEvidenceClass.SYNTHETIC:
            raise refuse_equity_read_query(
                _R.EVIDENCE_CLASS_UNSUPPORTED, "a research entry's snapshot is not SYNTHETIC"
            )
        if snapshot.snapshot_id != snapshot_id:
            raise refuse_equity_read_query(
                _INVALID, "a research entry's snapshot answers another identity"
            )
        retrieved[snapshot_id] = snapshot
    return retrieved


def _rehydrated(
    identity: EquityIntradayResearchTimelineIdentity,
    stored: StoredEquityIntradayResearchTimeline,
    gated: dict[str, EquityCandleEvidenceSnapshot],
) -> EquityIntradayResearchTimeline:
    try:
        return rehydrate_equity_intraday_research_timeline(identity, stored, _GatedSnapshots(gated))
    except (
        EquityIntradayResearchTimelineConfigurationConflictError,
        InvalidStoredEquityIntradayResearchTimelineError,
    ) as exc:
        raise refuse_equity_read_query(
            _INVALID, "the stored research timeline does not verify"
        ) from exc
    except EquityCandleEvidenceUnavailableError as exc:
        raise refuse_equity_read_query(
            _R.DEPENDENCY_UNAVAILABLE, "the evidence snapshots are unavailable"
        ) from exc


def _require_linkage(
    lineage: EquityVerifiedGenerationLineage,
    candidate: EquityIntradayResearchTimelineEntry | None,
    configuration: EquityOpeningRangeBreakoutConfiguration,
) -> None:
    if candidate is None:
        raise refuse_equity_read_query(
            _INVALID, "an issuance exists but the timeline holds no CANDIDATE entry"
        )
    source = lineage.research_reference
    linked = (
        source.candidate_entry_fingerprint == candidate.fingerprint
        and source.research_decision_fingerprint == candidate.decision.fingerprint
        and source.decision_as_of == candidate.decision_instant
        and source.evidence_snapshot_id == candidate.snapshot_id
        and lineage.candidate.manifest_fingerprint == candidate.manifest_fingerprint
        and lineage.candidate.candidate_sequence == candidate.sequence
    )
    if not linked:
        raise refuse_equity_read_query(
            _INVALID, "the issuance's generation does not cite the timeline's CANDIDATE entry"
        )
    if (
        lineage.research_configuration.document() != configuration.document()
        or source.research_configuration_version.value != configuration.version
    ):
        raise refuse_equity_read_query(
            _INVALID, "the issuance's generation used another research configuration"
        )


class GetEquityResearchTimelineUseCase:
    """Return persisted research and issuance-linked generation status (EI7-P3, EI7-P5)."""

    def __init__(
        self,
        timelines: EquityIntradayResearchTimelineReader,
        snapshots: EquityCandleEvidenceSnapshotReader,
        issuances: EquityRecommendationIssuanceReader,
        generations: EquityRecommendationGenerationRecordReader,
    ) -> None:
        if not isinstance(timelines, EquityIntradayResearchTimelineReader):
            raise TypeError("timelines must be an EquityIntradayResearchTimelineReader.")
        if not isinstance(snapshots, EquityCandleEvidenceSnapshotReader):
            raise TypeError("snapshots must be an EquityCandleEvidenceSnapshotReader.")
        if not isinstance(issuances, EquityRecommendationIssuanceReader):
            raise TypeError("issuances must be an EquityRecommendationIssuanceReader.")
        if not isinstance(generations, EquityRecommendationGenerationRecordReader):
            raise TypeError("generations must be an EquityRecommendationGenerationRecordReader.")
        self._timelines = timelines
        self._snapshots = snapshots
        self._issuances = issuances
        self._generations = generations

    def execute(
        self, strategy: StrategyIdentity, listing: ListingReference, trading_date: date
    ) -> EquityResearchTimelineView:
        """Refuse with EquityReadQueryRefusedError as the module docstring states."""
        _require_scope(strategy, listing, trading_date)
        stored = self._load(strategy, listing, trading_date)
        timeline: EquityIntradayResearchTimeline | None = None
        configuration: EquityOpeningRangeBreakoutConfiguration | None = None
        if stored is not None:
            configuration = _configuration(stored)
            identity = EquityIntradayResearchTimelineIdentity(listing, trading_date, configuration)
            timeline = _rehydrated(identity, stored, _gated(self._snapshots, stored))
        issuance = self._issuance(strategy, listing, trading_date)
        entries = () if timeline is None else timeline.entries
        candidate = _candidate(entries)
        reasons = EquityGenerationNotQueryableReason
        if issuance is None:
            reason = (
                reasons.NO_RESEARCH_TIMELINE
                if timeline is None
                else reasons.NO_CANDIDATE_ENTRY
                if candidate is None
                else reasons.NO_RECORDED_ISSUANCE
            )
            return EquityResearchTimelineView(
                strategy=strategy,
                listing=listing,
                trading_date=trading_date,
                timeline_status=EquityResearchTimelineStatus.UNKNOWN
                if timeline is None
                else EquityResearchTimelineStatus.STORED,
                configuration=configuration,
                entries=entries,
                generation_status=EquityResearchGenerationStatus.NOT_QUERYABLE,
                not_queryable_reason=reason,
                lineage=None,
                issuance_status=EquityResearchIssuanceStatus.NOT_RECORDED,
                issuance=None,
            )
        if configuration is None:
            raise refuse_equity_read_query(
                _INVALID, "an issuance exists without its persisted research timeline"
            )
        lineage = verify_equity_issued_generation_lineage(issuance, self._disposition(issuance))
        _require_linkage(lineage, candidate, configuration)
        return EquityResearchTimelineView(
            strategy=strategy,
            listing=listing,
            trading_date=trading_date,
            timeline_status=EquityResearchTimelineStatus.STORED,
            configuration=configuration,
            entries=entries,
            generation_status=EquityResearchGenerationStatus.PROPOSED,
            not_queryable_reason=None,
            lineage=lineage,
            issuance_status=EquityResearchIssuanceStatus.ISSUED,
            issuance=issuance,
        )

    def _load(
        self, strategy: StrategyIdentity, listing: ListingReference, trading_date: date
    ) -> StoredEquityIntradayResearchTimeline | None:
        try:
            with refusing_equity_read_storage_failure():  # EI7-SI1: never UNKNOWN
                stored = self._timelines.load(strategy, listing, trading_date)
        except InvalidStoredEquityIntradayResearchTimelineError as exc:
            raise refuse_equity_read_query(
                _INVALID, "the stored research timeline is invalid"
            ) from exc
        except EquityIntradayResearchTimelineUnavailableError as exc:
            raise refuse_equity_read_query(
                _R.DEPENDENCY_UNAVAILABLE, "the research timelines are unavailable"
            ) from exc
        if stored is not None and (
            not isinstance(stored, StoredEquityIntradayResearchTimeline)
            or (stored.strategy, stored.listing, stored.trading_date)
            != (strategy, listing, trading_date)
        ):
            raise refuse_equity_read_query(
                _INVALID, "the timeline reader answered another strategy, listing or date"
            )
        return stored

    def _issuance(
        self, strategy: StrategyIdentity, listing: ListingReference, trading_date: date
    ) -> EquityRecordedIssuance | None:
        try:
            with refusing_equity_read_storage_failure():  # EI7-SI1: never NOT_QUERYABLE
                recorded = self._issuances.load_session_issuance(strategy, listing, trading_date)
        except InvalidStoredEquityRecommendationIssuanceAuthorityError as exc:
            raise refuse_equity_read_query(_INVALID, "the issuance authority is invalid") from exc
        except EquityRecommendationIssuanceUnavailableError as exc:
            raise refuse_equity_read_query(
                _R.DEPENDENCY_UNAVAILABLE, "the issuance authority is unavailable"
            ) from exc
        if recorded is not None:
            issued = getattr(recorded, "recommendation", None)
            if not isinstance(recorded, EquityRecordedIssuance) or (
                issued.strategy,  # type: ignore[union-attr]
                issued.listing,  # type: ignore[union-attr]
                issued.trading_date,  # type: ignore[union-attr]
            ) != (strategy, listing, trading_date):
                raise refuse_equity_read_query(
                    _INVALID, "the issuance authority answered another strategy, listing or date"
                )
        return recorded

    def _disposition(self, issuance: EquityRecordedIssuance) -> Any:
        try:
            with refusing_equity_read_storage_failure():  # EI7-SI1
                return self._generations.load_disposition(issuance.recommendation.generation_key)
        except InvalidEquityRecommendationRecordError as exc:
            raise refuse_equity_read_query(
                _INVALID, "the issuance's generation record is invalid"
            ) from exc
        except EquityRecommendationGenerationStoreUnavailableError as exc:
            raise refuse_equity_read_query(
                _R.DEPENDENCY_UNAVAILABLE, "the generation records are unavailable"
            ) from exc


def _require_scope(strategy: object, listing: object, trading_date: object) -> None:
    if not isinstance(strategy, StrategyIdentity):
        raise refuse_equity_read_query(_R.UNSUPPORTED_SCOPE, "the strategy is not supported")
    if not isinstance(listing, ListingReference) or (
        listing.exchange_code not in EQUITY_READ_SUPPORTED_EXCHANGES
    ):
        raise refuse_equity_read_query(_R.UNSUPPORTED_SCOPE, "the listing is not supported")
    if isinstance(trading_date, datetime) or not isinstance(trading_date, date):
        raise refuse_equity_read_query(_R.UNSUPPORTED_SCOPE, "the trading date is not a date")
