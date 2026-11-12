"""Synthetic equity recommendation lifecycle evaluation (EI-5C, ADR-024).

Connects a verified stored EI-4D proposal, and immutable EI-2B evidence
snapshots, to the EI-5B Core lifecycle
(``northstar_core.recommendations.equity.lifecycle``). Core decides every
transition; this module owns only what Core deliberately does not: which
proposal, which snapshots, and whether they belong to the proposal's listing,
session, research window and chronology.

Synthetic, never issuance
-------------------------
No issuance authority exists (EI-5D). A lifecycle needs a ``RECORDED`` start,
so evaluation runs only under an explicit ``synthetic-`` label, and its result
says plainly that it establishes no issuance:

- an EI-4D generation record is a proposal, never an issued recommendation;
- no ``eiris1`` issuance identity is computed and nothing is recorded in an
  issued history;
- nothing is persisted: lifecycle events live in the returned value only, and
  a caller continues a lifecycle by passing them back;
- the result is never actionable and is not a published recommendation.

Verification
------------
The disposition is reloaded and verified by the existing
``VerifyStoredEquityRecommendationGenerationUseCase`` (EI-4D-A, levels A to C).
Its typed failures propagate unchanged. Only a ``PROPOSED`` disposition is
evaluated. Verification does not establish issuance eligibility under the
pending versioned EI-4C contract (C-2).

Snapshots
---------
Each snapshot is read by identity from the existing EI-2B snapshot store,
which verifies it on load. Before Core sees it, it must:

- be the snapshot asked for, of SYNTHETIC evidence class;
- belong to the proposal's listing, trading date and venue reference;
- use the proposal's research window, version, five-minute timeframe and currency;
- be taken at or after the modeled issuance;
- follow the previous evaluation: a later ``as_of``, and a watermark that
  never regresses. Repeating the last evaluated snapshot is idempotent; a
  different snapshot at the same instant is refused (the EI3-D9 pattern).

Its intervals from the lifecycle's cursor through ``modeled_exit_at`` become
Core intervals one for one, in the EI-2A vocabulary. Not-ended intervals stay
``NOT_ENDED``, and no price is carried for any outcome but ``ACCEPTABLE``.
Nothing is filled, repaired or relinked.

Revisions of cited evidence (EI5-D8)
------------------------------------
After each snapshot, every interval an earlier event cited as acceptable is
compared with the same interval in the new snapshot. A different set of
acceptable candles, or ``CONFLICTING_VALUES`` or ``UNRESOLVED_REVISIONS``, is
recorded once per new candle set as a non-state ``SOURCE_REVISION_RECORDED``
event, materiality ``UNASSESSED``. A cited interval that is no longer ended or
received contradicts the append-only journal, and the snapshot is refused.
Late evidence after terminal ``INCOMPLETE_EVIDENCE`` is not recorded (O-3).

Nothing reads a clock, writes to any store or imports Infrastructure.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from northstar_core.foundation.value_objects import PointInTime, Timeframe
from northstar_core.recommendations.equity.lifecycle import (
    EquityLifecycleAdvanceStatus,
    EquityLifecycleEvent,
    EquityLifecycleEventKind,
    EquityLifecycleEvidence,
    EquityLifecycleInterval,
    EquityLifecycleIntervalOutcome,
    EquityLifecycleObservation,
    EquityLifecycleState,
    EquityLifecycleWindow,
    EquityRecommendationLifecycle,
    InvalidEquityRecommendationLifecycleError,
)

from northstar_application.application_services.equity_candle_evidence_snapshot import (
    EquityCandleEvidenceClass,
    EquityCandleEvidenceSnapshot,
    EquityCandleSnapshotInterval,
)
from northstar_application.application_services.equity_intraday_candle_validation import (
    EquityCandleIntervalOutcome,
)
from northstar_application.application_services.equity_recommendation_generation_evidence import (
    EquityRecommendationCandidateProvenance,
)
from northstar_application.application_services.equity_recommendation_generation_records import (
    EquityRecommendationDispositionKind,
)
from northstar_application.application_services.equity_trade_recommendation_eligibility import (
    EquityTradeRecommendationProposal,
)
from northstar_application.application_services.verify_equity_recommendation_generation import (
    VerifyStoredEquityRecommendationGenerationUseCase,
)
from northstar_application.ports.equity_candle_evidence_journal import (
    EquityCandleEvidenceSnapshotStore,
)

_SYNTHETIC_PREFIX = "synthetic-"
_FIVE_MINUTES = Timeframe("5m")
_REVISED = frozenset(
    {
        EquityCandleIntervalOutcome.CONFLICTING_VALUES,
        EquityCandleIntervalOutcome.UNRESOLVED_REVISIONS,
    }
)


class EquitySyntheticLifecycleRefusal(StrEnum):
    """Why a synthetic lifecycle evaluation was refused. Nothing is evaluated after it."""

    GENERATION_NOT_FOUND = "GENERATION_NOT_FOUND"
    DISPOSITION_NOT_PROPOSED = "DISPOSITION_NOT_PROPOSED"
    INVALID_LIFECYCLE_HISTORY = "INVALID_LIFECYCLE_HISTORY"
    PRIOR_EVALUATION_INCONSISTENT = "PRIOR_EVALUATION_INCONSISTENT"
    SNAPSHOT_NOT_FOUND = "SNAPSHOT_NOT_FOUND"
    SNAPSHOT_IDENTITY_MISMATCH = "SNAPSHOT_IDENTITY_MISMATCH"
    SNAPSHOT_NOT_SYNTHETIC = "SNAPSHOT_NOT_SYNTHETIC"
    SNAPSHOT_LISTING_MISMATCH = "SNAPSHOT_LISTING_MISMATCH"
    SNAPSHOT_SESSION_MISMATCH = "SNAPSHOT_SESSION_MISMATCH"
    SNAPSHOT_WINDOW_MISMATCH = "SNAPSHOT_WINDOW_MISMATCH"
    SNAPSHOT_BEFORE_ISSUANCE = "SNAPSHOT_BEFORE_ISSUANCE"
    SNAPSHOT_OUT_OF_ORDER = "SNAPSHOT_OUT_OF_ORDER"
    SNAPSHOT_WATERMARK_REGRESSION = "SNAPSHOT_WATERMARK_REGRESSION"
    SNAPSHOT_EVIDENCE_INCONSISTENT = "SNAPSHOT_EVIDENCE_INCONSISTENT"
    SNAPSHOT_CONTRADICTS_CITED_EVIDENCE = "SNAPSHOT_CONTRADICTS_CITED_EVIDENCE"


class EquitySyntheticLifecycleRefusedError(ValueError):
    """A synthetic lifecycle evaluation was refused; ``reason`` says why."""

    def __init__(self, reason: EquitySyntheticLifecycleRefusal, message: str) -> None:
        super().__init__(f"{reason}: {message}")
        self.reason = reason


class InvalidEquitySyntheticLifecycleRequestError(ValueError):
    """Raised when a synthetic lifecycle request is malformed."""


def _refuse(
    reason: EquitySyntheticLifecycleRefusal, message: str
) -> EquitySyntheticLifecycleRefusedError:
    return EquitySyntheticLifecycleRefusedError(reason, message)


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise InvalidEquitySyntheticLifecycleRequestError(
            f"{name} must be non-empty text without surrounding whitespace."
        )
    return value


@dataclass(frozen=True, slots=True)
class EquitySyntheticLifecycleSnapshotPosition:
    """The last evaluated snapshot: its identity, ``as_of`` and journal watermark."""

    snapshot_id: str
    as_of: PointInTime
    watermark: int

    def __post_init__(self) -> None:
        _text(self.snapshot_id, "Snapshot position snapshot_id")
        if not isinstance(self.as_of, PointInTime):
            raise InvalidEquitySyntheticLifecycleRequestError(
                "Snapshot position as_of must be a PointInTime."
            )
        watermark = self.watermark
        if isinstance(watermark, bool) or not isinstance(watermark, int) or watermark < 0:
            raise InvalidEquitySyntheticLifecycleRequestError(
                "Snapshot position watermark must be a non-negative integer."
            )

    @classmethod
    def of(cls, snapshot: EquityCandleEvidenceSnapshot) -> EquitySyntheticLifecycleSnapshotPosition:
        """Return the position of ``snapshot``."""
        return cls(snapshot.snapshot_id, snapshot.as_of, snapshot.watermark)


@dataclass(frozen=True, slots=True)
class EquitySyntheticLifecycleRequest:
    """One synthetic evaluation: a proposal, the snapshots to evaluate, and any prior state.

    ``synthetic_label`` must begin ``synthetic-``: it names a synthetic,
    non-authoritative evaluation, never an issuance. ``snapshot_ids`` are
    evaluated in the given order. ``prior_events`` and ``last_evaluated``
    continue an earlier evaluation of the same proposal (see ``following``);
    both are in-memory values, never stored records.
    """

    synthetic_label: str
    generation_key: str
    snapshot_ids: tuple[str, ...]
    prior_events: tuple[EquityLifecycleEvent, ...] = ()
    last_evaluated: EquitySyntheticLifecycleSnapshotPosition | None = None

    def __post_init__(self) -> None:
        label = _text(self.synthetic_label, "synthetic_label")
        if not label.startswith(_SYNTHETIC_PREFIX) or label == _SYNTHETIC_PREFIX:
            raise InvalidEquitySyntheticLifecycleRequestError(
                "A lifecycle evaluation is synthetic only: its label must begin 'synthetic-'."
            )
        _text(self.generation_key, "generation_key")
        if not isinstance(self.snapshot_ids, tuple) or not self.snapshot_ids:
            raise InvalidEquitySyntheticLifecycleRequestError(
                "snapshot_ids must be a non-empty tuple."
            )
        for snapshot_id in self.snapshot_ids:
            _text(snapshot_id, "snapshot_ids entry")
        if not isinstance(self.prior_events, tuple) or not all(
            isinstance(event, EquityLifecycleEvent) for event in self.prior_events
        ):
            raise InvalidEquitySyntheticLifecycleRequestError(
                "prior_events must be a tuple of EquityLifecycleEvent."
            )
        if self.last_evaluated is not None and not isinstance(
            self.last_evaluated, EquitySyntheticLifecycleSnapshotPosition
        ):
            raise InvalidEquitySyntheticLifecycleRequestError(
                "last_evaluated must be an EquitySyntheticLifecycleSnapshotPosition."
            )

    @classmethod
    def following(
        cls, evaluation: EquitySyntheticLifecycleEvaluation, snapshot_ids: tuple[str, ...]
    ) -> EquitySyntheticLifecycleRequest:
        """Return the request that continues ``evaluation`` with more snapshots."""
        return cls(
            synthetic_label=evaluation.synthetic_label,
            generation_key=evaluation.generation_key,
            snapshot_ids=snapshot_ids,
            prior_events=evaluation.lifecycle.events,
            last_evaluated=evaluation.last_evaluated,
        )


@dataclass(frozen=True, slots=True)
class EquitySyntheticLifecycleStep:
    """What one snapshot did.

    ``status`` is Core's advance status, or None when the snapshot repeated the
    last evaluated one and nothing was evaluated. ``events`` are the events it
    added, in order: Core's lifecycle events, then any revision events.
    ``halted_at`` is the interval Core stopped at, when it stopped short.
    """

    snapshot_id: str
    as_of: PointInTime
    watermark: int
    repeated: bool
    status: EquityLifecycleAdvanceStatus | None
    events: tuple[EquityLifecycleEvent, ...]
    halted_at: EquityLifecycleInterval | None


@dataclass(frozen=True, slots=True)
class EquitySyntheticLifecycleEvaluation:
    """A synthetic, non-authoritative lifecycle evaluation of one verified proposal.

    It is not an issued or published recommendation: ``issuance_established``
    and ``is_authoritative`` are always False. ``lifecycle`` is the Core
    lifecycle, in memory only; its events are the whole history so far.
    """

    synthetic_label: str
    generation_key: str
    proposal_fingerprint: str
    window: EquityLifecycleWindow
    lifecycle: EquityRecommendationLifecycle
    steps: tuple[EquitySyntheticLifecycleStep, ...]
    last_evaluated: EquitySyntheticLifecycleSnapshotPosition | None

    @property
    def state(self) -> EquityLifecycleState:
        """Return the Core lifecycle state."""
        return self.lifecycle.state

    @property
    def new_events(self) -> tuple[EquityLifecycleEvent, ...]:
        """Return the events this evaluation added, in order."""
        return tuple(event for step in self.steps for event in step.events)

    @property
    def issuance_established(self) -> bool:
        """Return False: a synthetic evaluation never establishes an issuance."""
        return False

    @property
    def is_authoritative(self) -> bool:
        """Return False: nothing here is an authoritative recommendation record."""
        return False


def _lifecycle_window(
    proposal: EquityTradeRecommendationProposal,
    candidate: EquityRecommendationCandidateProvenance,
) -> EquityLifecycleWindow:
    """Return the Core window from the verified proposal and its candidate's grid."""
    entry = proposal.entry_window
    openings = candidate.entry_bar_openings()
    latest = entry.latest_entry_bar_opens_at
    if latest not in openings or entry.modeled_exit_at not in candidate.bar_ends():
        raise _refuse(
            EquitySyntheticLifecycleRefusal.SNAPSHOT_EVIDENCE_INCONSISTENT,
            "The proposal's entry window does not lie on its research-window grid.",
        )
    index = openings.index(latest)
    ends = openings[index + 1] if index + 1 < len(openings) else candidate.research_window_closes_at
    return EquityLifecycleWindow(
        levels=proposal.definition.levels,
        eligible_from=entry.timing.eligible_from,
        latest_entry_bar_opens_at=latest,
        entry_window_ends_at=ends,
        modeled_exit_at=entry.modeled_exit_at,
    )


def _candle_references(
    snapshot: EquityCandleEvidenceSnapshot, sequences: tuple[int, ...]
) -> tuple[str, ...]:
    by_sequence = {item.sequence: item for item in snapshot.observations}
    references = []
    for sequence in sequences:
        item = by_sequence.get(sequence)
        if item is None:
            raise _refuse(
                EquitySyntheticLifecycleRefusal.SNAPSHOT_EVIDENCE_INCONSISTENT,
                f"Interval names observation {sequence}, which the snapshot does not hold.",
            )
        observation = item.record.observation
        references.append(
            f"{sequence}:{observation.source_id}:{observation.source_event_id}:"
            f"{observation.source_revision}"
        )
    return tuple(references)


def _identities(references: tuple[str, ...]) -> frozenset[str]:
    """Return the candle identities (source, event, revision) the references name."""
    return frozenset(reference.split(":", 1)[1] for reference in references)


def _core_interval(
    snapshot: EquityCandleEvidenceSnapshot, interval: EquityCandleSnapshotInterval
) -> EquityLifecycleInterval:
    outcome = EquityLifecycleIntervalOutcome(interval.outcome.value)
    if outcome is not EquityLifecycleIntervalOutcome.ACCEPTABLE:
        return EquityLifecycleInterval(interval.opens_at, interval.closes_at, outcome)
    by_sequence = {item.sequence: item for item in snapshot.observations}
    ranges = {
        (by_sequence[s].record.observation.bar.low, by_sequence[s].record.observation.bar.high)
        for s in interval.acceptable_sequences
        if s in by_sequence
    }
    if len(ranges) != 1:
        raise _refuse(
            EquitySyntheticLifecycleRefusal.SNAPSHOT_EVIDENCE_INCONSISTENT,
            f"The acceptable candles of the interval opening {interval.opens_at} do not "
            "agree on one range, or are not in the snapshot.",
        )
    ((low, high),) = ranges
    return EquityLifecycleInterval(
        interval.opens_at,
        interval.closes_at,
        outcome,
        low,
        high,
        _candle_references(snapshot, interval.acceptable_sequences),
    )


def _observation(
    snapshot: EquityCandleEvidenceSnapshot,
    window: EquityLifecycleWindow,
    start: PointInTime,
) -> EquityLifecycleObservation:
    """Return the snapshot's intervals from ``start`` through ``modeled_exit_at``, for Core."""
    selected = [
        interval
        for interval in snapshot.intervals
        if interval.opens_at.compare(start) >= 0
        and interval.closes_at.compare(window.modeled_exit_at) <= 0
    ]
    contiguous = all(
        later.opens_at == earlier.closes_at
        for earlier, later in zip(selected, selected[1:], strict=False)
    )
    if (
        not selected
        or selected[0].opens_at != start
        or selected[-1].closes_at != window.modeled_exit_at
        or not contiguous
    ):
        raise _refuse(
            EquitySyntheticLifecycleRefusal.SNAPSHOT_EVIDENCE_INCONSISTENT,
            f"The snapshot does not hold every interval from {start} to "
            f"{window.modeled_exit_at} contiguously.",
        )
    return EquityLifecycleObservation(
        snapshot.as_of,
        snapshot.snapshot_id,
        tuple(_core_interval(snapshot, interval) for interval in selected),
    )


class EvaluateSyntheticEquityRecommendationLifecycleUseCase:
    """Evaluates a verified stored proposal's synthetic lifecycle over EI-2B snapshots.

    Read only: it reloads and verifies the disposition, reads snapshots by
    identity, and returns an in-memory result. It never issues, records,
    persists or reads a clock. See the module docstring.
    """

    def __init__(
        self,
        generations: VerifyStoredEquityRecommendationGenerationUseCase,
        snapshots: EquityCandleEvidenceSnapshotStore,
    ) -> None:
        if not isinstance(generations, VerifyStoredEquityRecommendationGenerationUseCase):
            raise TypeError(
                "generations must be a VerifyStoredEquityRecommendationGenerationUseCase."
            )
        if not isinstance(snapshots, EquityCandleEvidenceSnapshotStore):
            raise TypeError("snapshots must be an EquityCandleEvidenceSnapshotStore.")
        self._generations = generations
        self._snapshots = snapshots

    def execute(
        self, request: EquitySyntheticLifecycleRequest
    ) -> EquitySyntheticLifecycleEvaluation:
        """Return the synthetic evaluation of ``request``, or raise a typed refusal."""
        if not isinstance(request, EquitySyntheticLifecycleRequest):
            raise TypeError("request must be an EquitySyntheticLifecycleRequest.")
        verified = self._generations.execute(request.generation_key)
        if verified is None:
            raise _refuse(
                EquitySyntheticLifecycleRefusal.GENERATION_NOT_FOUND,
                f"No stored disposition exists for {request.generation_key}.",
            )
        disposition = verified.disposition
        if disposition.kind is not EquityRecommendationDispositionKind.PROPOSED:
            raise _refuse(
                EquitySyntheticLifecycleRefusal.DISPOSITION_NOT_PROPOSED,
                f"The disposition is {disposition.kind}: only a PROPOSED disposition has a "
                "proposal to evaluate.",
            )
        proposal = disposition.proposal
        assert proposal is not None
        candidate = disposition.candidate
        window = _lifecycle_window(proposal, candidate)
        try:
            lifecycle = EquityRecommendationLifecycle(window, request.prior_events)
        except InvalidEquityRecommendationLifecycleError as exc:
            raise _refuse(
                EquitySyntheticLifecycleRefusal.INVALID_LIFECYCLE_HISTORY, str(exc)
            ) from exc
        position = request.last_evaluated
        self._require_prior(lifecycle, position)
        steps: list[EquitySyntheticLifecycleStep] = []
        for snapshot_id in request.snapshot_ids:
            snapshot = self._load(snapshot_id)
            self._require_session(snapshot, proposal, candidate)
            if position is not None and snapshot_id == position.snapshot_id:
                steps.append(
                    EquitySyntheticLifecycleStep(
                        snapshot_id, snapshot.as_of, snapshot.watermark, True, None, (), None
                    )
                )
                continue
            self._require_order(snapshot, position, lifecycle)
            start = lifecycle.cursor or window.eligible_from
            advance = lifecycle.advance(_observation(snapshot, window, start))
            lifecycle, revisions = _record_revisions(advance.lifecycle, snapshot)
            steps.append(
                EquitySyntheticLifecycleStep(
                    snapshot_id,
                    snapshot.as_of,
                    snapshot.watermark,
                    False,
                    advance.status,
                    advance.events + revisions,
                    advance.halted_at,
                )
            )
            position = EquitySyntheticLifecycleSnapshotPosition.of(snapshot)
        return EquitySyntheticLifecycleEvaluation(
            synthetic_label=request.synthetic_label,
            generation_key=disposition.generation_key,
            proposal_fingerprint=proposal.fingerprint,
            window=window,
            lifecycle=lifecycle,
            steps=tuple(steps),
            last_evaluated=position,
        )

    @staticmethod
    def _require_prior(
        lifecycle: EquityRecommendationLifecycle,
        position: EquitySyntheticLifecycleSnapshotPosition | None,
    ) -> None:
        last = lifecycle.last_evaluated_at
        if last is None:
            return
        if position is None or position.as_of.compare(last) < 0:
            raise _refuse(
                EquitySyntheticLifecycleRefusal.PRIOR_EVALUATION_INCONSISTENT,
                "Prior events need the last evaluated snapshot, at or after their last event.",
            )

    def _load(self, snapshot_id: str) -> EquityCandleEvidenceSnapshot:
        snapshot = self._snapshots.get_snapshot(snapshot_id)
        if snapshot is None:
            raise _refuse(
                EquitySyntheticLifecycleRefusal.SNAPSHOT_NOT_FOUND,
                f"No EI-2B snapshot is stored under {snapshot_id}.",
            )
        if snapshot.snapshot_id != snapshot_id:
            raise _refuse(
                EquitySyntheticLifecycleRefusal.SNAPSHOT_IDENTITY_MISMATCH,
                f"The store answered {snapshot.snapshot_id} for {snapshot_id}.",
            )
        return snapshot

    @staticmethod
    def _require_session(
        snapshot: EquityCandleEvidenceSnapshot,
        proposal: EquityTradeRecommendationProposal,
        candidate: EquityRecommendationCandidateProvenance,
    ) -> None:
        reason = EquitySyntheticLifecycleRefusal
        if snapshot.evidence_class is not EquityCandleEvidenceClass.SYNTHETIC:
            raise _refuse(reason.SNAPSHOT_NOT_SYNTHETIC, "Only synthetic evidence is evaluated.")
        if snapshot.listing != candidate.source.listing:
            raise _refuse(
                reason.SNAPSHOT_LISTING_MISMATCH,
                f"The snapshot is for {snapshot.listing}, not {candidate.source.listing}.",
            )
        if (
            snapshot.trading_date != candidate.source.trading_date
            or snapshot.venue_reference != candidate.venue_reference
        ):
            raise _refuse(
                reason.SNAPSHOT_SESSION_MISMATCH,
                "The snapshot is for another session or venue-day reference: no next-session "
                "or differently resolved session evidence is evaluated.",
            )
        if (
            snapshot.window_policy_version != candidate.research_window_policy_version
            or snapshot.window_opens_at != candidate.research_window_opens_at
            or snapshot.window_closes_at != candidate.research_window_closes_at
            or snapshot.timeframe != _FIVE_MINUTES
            or snapshot.currency != proposal.definition.levels.currency
        ):
            raise _refuse(
                reason.SNAPSHOT_WINDOW_MISMATCH,
                "The snapshot's research window, timeframe or currency is not the proposal's.",
            )
        if snapshot.as_of.compare(proposal.entry_window.timing.issued_at) < 0:
            raise _refuse(
                reason.SNAPSHOT_BEFORE_ISSUANCE,
                f"The snapshot at {snapshot.as_of} precedes the modeled issuance.",
            )

    @staticmethod
    def _require_order(
        snapshot: EquityCandleEvidenceSnapshot,
        position: EquitySyntheticLifecycleSnapshotPosition | None,
        lifecycle: EquityRecommendationLifecycle,
    ) -> None:
        reason = EquitySyntheticLifecycleRefusal
        last = lifecycle.last_evaluated_at
        if last is not None and snapshot.as_of.compare(last) < 0:
            raise _refuse(reason.SNAPSHOT_OUT_OF_ORDER, "The snapshot precedes the last event.")
        if position is None:
            return
        if snapshot.as_of.compare(position.as_of) <= 0:
            raise _refuse(
                reason.SNAPSHOT_OUT_OF_ORDER,
                f"The snapshot at {snapshot.as_of} does not follow the last evaluated one at "
                f"{position.as_of}; a different snapshot at the same instant is refused.",
            )
        if snapshot.watermark < position.watermark:
            raise _refuse(
                reason.SNAPSHOT_WATERMARK_REGRESSION,
                f"Watermark {snapshot.watermark} regresses from {position.watermark}.",
            )


def _record_revisions(
    lifecycle: EquityRecommendationLifecycle, snapshot: EquityCandleEvidenceSnapshot
) -> tuple[EquityRecommendationLifecycle, tuple[EquityLifecycleEvent, ...]]:
    """Record corrections, in ``snapshot``, to evidence earlier events cited (EI5-D8)."""
    intervals = {(i.opens_at, i.closes_at): i for i in snapshot.intervals}
    before = len(lifecycle.events)
    for cited in tuple(lifecycle.events):
        if not cited.kind.is_bar_driven or cited.evidence.reference == snapshot.snapshot_id:
            continue
        interval = intervals.get((cited.evidence.opens_at, cited.evidence.closes_at))
        if interval is None:
            raise _refuse(
                EquitySyntheticLifecycleRefusal.SNAPSHOT_EVIDENCE_INCONSISTENT,
                f"The snapshot lacks the cited interval opening {cited.evidence.opens_at}.",
            )
        if interval.outcome is EquityCandleIntervalOutcome.ACCEPTABLE:
            current = _candle_references(snapshot, interval.acceptable_sequences)
            # An exact repeat of the cited candle is the same evidence, not a correction.
            if _identities(current) == _identities(cited.evidence.observations):
                continue
        elif interval.outcome in _REVISED:
            current = _candle_references(snapshot, interval.observation_sequences)
        else:
            raise _refuse(
                EquitySyntheticLifecycleRefusal.SNAPSHOT_CONTRADICTS_CITED_EVIDENCE,
                f"The cited interval opening {cited.evidence.opens_at} is {interval.outcome} "
                "in a later snapshot, which an append-only journal cannot produce.",
            )
        if any(
            event.kind is EquityLifecycleEventKind.SOURCE_REVISION_RECORDED
            and event.cited_sequence == cited.sequence
            and event.evidence.observations == current
            for event in lifecycle.events
        ):
            continue
        evidence = EquityLifecycleEvidence(
            snapshot.snapshot_id, cited.evidence.opens_at, cited.evidence.closes_at, current
        )
        lifecycle = lifecycle.record_source_revision(cited.sequence, evidence, snapshot.as_of)
    return lifecycle, lifecycle.events[before:]
