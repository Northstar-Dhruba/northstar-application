"""Authoritative synthetic equity intraday research timeline advancement (EI-3C-A).

Advances the durable research timeline of one strategy configuration, listing
and trading date over a finite batch of issued EI-2B snapshots, under the
owner decisions recorded on 2026-11-07 in the EI-3 Decision Register
(northstar-docs, product/Equity-Intraday-V1-EI3-Decision-Register.md,
section 11): EI3-D11, EI3-D12 and EI3-D13. The caller supplies no history:
the EquityIntradayResearchTimelineStore is the source of truth. Research and
candidate selection are the unchanged EI-3A evaluator and EI-3B scheduler.
It is synthetic research only; nothing here is a recommendation, order or fill.

Establishment (EI3-D11)
-----------------------
A timeline, with its full configuration, becomes authoritative when its first
entry commits. An advance that appends nothing writes nothing, so no empty
timeline is ever stored, and an in-memory timeline is never authoritative.
A stored timeline under another canonical configuration document refuses
every input with CONFIGURATION_MISMATCH and writes nothing.

Rehydration (EI3-D13)
---------------------
Stored history is accepted only whole. For each stored entry, in sequence
order: its decision document must be canonical JSON whose SHA-256 is its
``eir1`` fingerprint; its EI-2B snapshot must load through the snapshot store,
which verifies it, and match the entry's snapshot ID, manifest fingerprint,
decision instant, watermark, window close, listing and date; the current EI-3A
evaluator must reproduce the decision document exactly from that snapshot; and
the rebuilt EI-3B entry must re-derive its stored ``eirt1`` fingerprint. The
EI-3B timeline then re-verifies sequence, chain, instants, watermarks and the
single CANDIDATE. There is no historical evaluator registry: the only
compatible behaviour is the current evaluator for the stored decision format.
Another format is EquityIntradayResearchHistoricalEvaluationUnavailableError;
a different reproduction is EquityIntradayResearchDecisionNotReproducedError.
Nothing is skipped, repaired or migrated, and no market data is read.

Advancement and commit (EI3-D12)
--------------------------------
Each attempt loads and rehydrates the stored history, runs EI-3B over the
whole submitted batch, and -- when EI-3B appended anything -- calls the store's
compare-and-append with the loaded head. A stale head discards the attempt:
the next attempt reloads and re-runs the batch, so roles always come from
committed history. At most EQUITY_INTRADAY_RESEARCH_TIMELINE_COMMIT_ATTEMPTS
attempts are made; then the result is STALE_AFTER_RETRIES and claims nothing.

An unconfirmed commit is reconciled by one authoritative reload. The attempt
committed exactly when the reloaded history holds every attempted entry, by
fingerprint, at its sequence -- whatever another writer appended afterwards.
It did not commit when the history still extends what the attempt loaded but
lacks those entries: the attempt is then retried like a stale one. When the
reload fails, or the history no longer extends what the attempt loaded,
EquityIntradayResearchTimelineCommitUnresolvedError is raised: no success and
no rollback is claimed. Nothing reads a clock, sleeps or polls.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum
from typing import Any, NoReturn

from northstar_core.domain.value_objects import ListingReference
from northstar_core.foundation.value_objects import PointInTime
from northstar_core.strategy import StrategyIdentity

from northstar_application.application_services.equity_candle_evidence_journal import (
    canonical_equity_candle_json,
)
from northstar_application.application_services.evaluate_equity_intraday_research import (
    EQUITY_INTRADAY_RESEARCH_DECISION_PREFIX,
    EquityIntradayResearchOutcome,
    EquityOpeningRangeBreakoutConfiguration,
    EquityOpeningRangeBreakoutEvaluator,
)
from northstar_application.application_services.schedule_equity_intraday_research import (
    EquityIntradayResearchScheduleItem,
    EquityIntradayResearchScheduleOutcome,
    EquityIntradayResearchScheduleRefusal,
    EquityIntradayResearchScheduleResult,
    EquityIntradayResearchTimeline,
    EquityIntradayResearchTimelineEntry,
    EquityIntradayResearchTimelineIdentity,
    EquityIntradayResearchTimelineRole,
    InvalidEquityIntradayResearchTimelineError,
    ScheduleEquityIntradayResearchUseCase,
)
from northstar_application.ports.equity_candle_evidence_journal import (
    EquityCandleEvidenceSnapshotStore,
    EquityCandleEvidenceUnavailableError,
)
from northstar_application.ports.equity_intraday_research_timeline_store import (
    EquityIntradayResearchDecisionNotReproducedError,
    EquityIntradayResearchHistoricalEvaluationUnavailableError,
    EquityIntradayResearchTimelineCommitUnconfirmedError,
    EquityIntradayResearchTimelineCommitUnresolvedError,
    EquityIntradayResearchTimelineConfigurationConflictError,
    EquityIntradayResearchTimelineStaleError,
    EquityIntradayResearchTimelineStore,
    EquityIntradayResearchTimelineUnavailableError,
    InvalidStoredEquityIntradayResearchTimelineError,
)

EQUITY_INTRADAY_RESEARCH_TIMELINE_COMMIT_ATTEMPTS = 3


def _bad(message: str) -> NoReturn:
    raise InvalidStoredEquityIntradayResearchTimelineError(message)


def _text(value: object) -> bool:
    return isinstance(value, str) and bool(value)


def _count(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


# ---------------------------------------------------------------------------
# Stored records
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StoredEquityIntradayResearchTimelineEntry:
    """One committed timeline entry exactly as stored.

    ``decision_document`` is the EI-3A decision's canonical JSON text; its
    SHA-256 is ``decision_fingerprint``. Explanation prose is not stored: it
    is not part of any fingerprint, and rehydration re-derives it.
    """

    sequence: int
    role: EquityIntradayResearchTimelineRole
    decision_outcome: EquityIntradayResearchOutcome
    snapshot_id: str
    manifest_fingerprint: str
    decision_instant: PointInTime
    watermark: int
    research_window_closes_at: PointInTime
    decision_document: str
    decision_fingerprint: str
    previous_fingerprint: str | None
    entry_fingerprint: str

    def __post_init__(self) -> None:
        if not _count(self.sequence) or self.sequence < 1:
            _bad("Stored entry sequence must be an integer of at least 1.")
        if not _count(self.watermark) or self.watermark < 0:
            _bad("Stored entry watermark must be a non-negative integer.")
        if not isinstance(self.role, EquityIntradayResearchTimelineRole):
            _bad("Stored entry role must be an EquityIntradayResearchTimelineRole.")
        if not isinstance(self.decision_outcome, EquityIntradayResearchOutcome):
            _bad("Stored entry decision_outcome must be an EquityIntradayResearchOutcome.")
        for name in (
            "snapshot_id",
            "manifest_fingerprint",
            "decision_document",
            "decision_fingerprint",
            "entry_fingerprint",
        ):
            if not _text(getattr(self, name)):
                _bad(f"Stored entry {name} must be non-empty text.")
        for name in ("decision_instant", "research_window_closes_at"):
            if not isinstance(getattr(self, name), PointInTime):
                _bad(f"Stored entry {name} must be a PointInTime.")
        if self.previous_fingerprint is not None and not _text(self.previous_fingerprint):
            _bad("Stored entry previous_fingerprint must be None or non-empty text.")


@dataclass(frozen=True, slots=True)
class StoredEquityIntradayResearchTimeline:
    """An authoritative timeline exactly as stored: its key, configuration and entries.

    ``configuration_document`` is the configuration's canonical JSON text.
    There is always at least one entry: a timeline is established by its
    first committed entry (EI3-D11).
    """

    strategy: StrategyIdentity
    listing: ListingReference
    trading_date: date
    configuration_document: str
    entries: tuple[StoredEquityIntradayResearchTimelineEntry, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.strategy, StrategyIdentity):
            _bad("Stored timeline strategy must be a StrategyIdentity.")
        if not isinstance(self.listing, ListingReference):
            _bad("Stored timeline listing must be a ListingReference.")
        if type(self.trading_date) is not date:
            _bad("Stored timeline trading_date must be a plain date.")
        if not _text(self.configuration_document):
            _bad("Stored timeline configuration_document must be non-empty text.")
        if not isinstance(self.entries, tuple) or not all(
            isinstance(e, StoredEquityIntradayResearchTimelineEntry) for e in self.entries
        ):
            _bad("Stored timeline entries must be a tuple of stored entries.")
        if not self.entries:
            _bad("A stored timeline has at least one committed entry (EI3-D11).")

    @property
    def head_fingerprint(self) -> str:
        """Return the last committed entry's fingerprint."""
        return self.entries[-1].entry_fingerprint


def _configuration_document(configuration: EquityOpeningRangeBreakoutConfiguration) -> str:
    return canonical_equity_candle_json(configuration.document())


def _stored_entry(
    entry: EquityIntradayResearchTimelineEntry,
) -> StoredEquityIntradayResearchTimelineEntry:
    return StoredEquityIntradayResearchTimelineEntry(
        sequence=entry.sequence,
        role=entry.role,
        decision_outcome=entry.decision.outcome,
        snapshot_id=entry.snapshot_id,
        manifest_fingerprint=entry.manifest_fingerprint,
        decision_instant=entry.decision_instant,
        watermark=entry.watermark,
        research_window_closes_at=entry.research_window_closes_at,
        decision_document=canonical_equity_candle_json(entry.decision.document()),
        decision_fingerprint=entry.decision.fingerprint,
        previous_fingerprint=entry.previous_fingerprint,
        entry_fingerprint=entry.fingerprint,
    )


def stored_equity_intraday_research_timeline(
    timeline: EquityIntradayResearchTimeline,
) -> StoredEquityIntradayResearchTimeline:
    """Return the stored record of a verified, non-empty ``timeline``."""
    if not isinstance(timeline, EquityIntradayResearchTimeline):
        raise TypeError("timeline must be an EquityIntradayResearchTimeline.")
    timeline.verify()
    if not timeline.entries:
        raise ValueError("An empty timeline is never stored (EI3-D11).")
    identity = timeline.identity
    return StoredEquityIntradayResearchTimeline(
        strategy=identity.strategy,
        listing=identity.listing,
        trading_date=identity.trading_date,
        configuration_document=_configuration_document(identity.configuration),
        entries=tuple(_stored_entry(entry) for entry in timeline.entries),
    )


# ---------------------------------------------------------------------------
# Rehydration (EI3-D13)
# ---------------------------------------------------------------------------


def _refuse_float(value: str) -> NoReturn:
    raise ValueError(f"Stored decision documents hold no floating-point numbers ({value}).")


def _canonical_document(text: str, where: str) -> dict[str, Any]:
    try:
        document = json.loads(text, parse_float=_refuse_float, parse_constant=_refuse_float)
        canonical = isinstance(document, dict) and (canonical_equity_candle_json(document) == text)
    except (ValueError, TypeError) as exc:
        _bad(f"{where}'s decision document is not canonical JSON: {exc}")
    if not canonical:
        _bad(f"{where}'s decision document is not canonical JSON.")
    return document  # type: ignore[no-any-return]


def _rehydrate_entry(
    identity: EquityIntradayResearchTimelineIdentity,
    evaluator: EquityOpeningRangeBreakoutEvaluator,
    snapshots: EquityCandleEvidenceSnapshotStore,
    index: int,
    record: StoredEquityIntradayResearchTimelineEntry,
) -> EquityIntradayResearchTimelineEntry:
    where = f"Stored entry {index}"
    if record.sequence != index:
        _bad(f"{where} has sequence {record.sequence}.")
    stored = _canonical_document(record.decision_document, where)
    digest = hashlib.sha256(record.decision_document.encode("utf-8")).hexdigest()
    if EQUITY_INTRADAY_RESEARCH_DECISION_PREFIX + digest != record.decision_fingerprint:
        _bad(f"{where}'s decision document does not match its decision fingerprint.")
    snapshot = snapshots.get_snapshot(record.snapshot_id)
    if snapshot is None:
        _bad(f"{where}'s source snapshot {record.snapshot_id} is unavailable.")
    if (
        snapshot.snapshot_id,
        snapshot.manifest_fingerprint,
        snapshot.as_of,
        snapshot.watermark,
        snapshot.window_closes_at,
        snapshot.listing,
        snapshot.trading_date,
    ) != (
        record.snapshot_id,
        record.manifest_fingerprint,
        record.decision_instant,
        record.watermark,
        record.research_window_closes_at,
        identity.listing,
        identity.trading_date,
    ):
        _bad(f"{where} does not match its source snapshot {record.snapshot_id}.")
    decision = evaluator.evaluate(
        identity.listing, identity.trading_date, snapshot, record.snapshot_id
    )
    rebuilt = decision.document()
    if stored.get("format") != rebuilt["format"]:
        raise EquityIntradayResearchHistoricalEvaluationUnavailableError(
            f"{where} is a {stored.get('format')!r} decision; only {rebuilt['format']!r} "
            "can be evaluated. Nothing is migrated."
        )
    if (
        canonical_equity_candle_json(rebuilt) != record.decision_document
        or decision.fingerprint != record.decision_fingerprint
        or decision.outcome is not record.decision_outcome
    ):
        raise EquityIntradayResearchDecisionNotReproducedError(
            f"{where}'s decision does not re-derive from snapshot {record.snapshot_id}."
        )
    try:
        entry = EquityIntradayResearchTimelineEntry(
            identity,
            record.sequence,
            record.role,
            decision,
            record.research_window_closes_at,
            record.previous_fingerprint,
        )
    except InvalidEquityIntradayResearchTimelineError as exc:
        _bad(f"{where} is not a valid timeline entry: {exc}")
    if entry.fingerprint != record.entry_fingerprint:
        _bad(f"{where} does not match its entry fingerprint.")
    return entry


def rehydrate_equity_intraday_research_timeline(
    identity: EquityIntradayResearchTimelineIdentity,
    stored: StoredEquityIntradayResearchTimeline,
    snapshots: EquityCandleEvidenceSnapshotStore,
) -> EquityIntradayResearchTimeline:
    """Rebuild and verify the authoritative timeline from ``stored``, whole or not at all.

    Raises EquityIntradayResearchTimelineConfigurationConflictError when the
    stored configuration document is not ``identity``'s, and
    InvalidStoredEquityIntradayResearchTimelineError (or one of its EI3-D13
    subclasses) when anything fails verification. A snapshot store failure
    propagates as EquityCandleEvidenceUnavailableError.
    """
    if not isinstance(identity, EquityIntradayResearchTimelineIdentity):
        raise TypeError("identity must be an EquityIntradayResearchTimelineIdentity.")
    if not isinstance(stored, StoredEquityIntradayResearchTimeline):
        raise TypeError("stored must be a StoredEquityIntradayResearchTimeline.")
    if not isinstance(snapshots, EquityCandleEvidenceSnapshotStore):
        raise TypeError("snapshots must be an EquityCandleEvidenceSnapshotStore.")
    if (stored.strategy, stored.listing, stored.trading_date) != (
        identity.strategy,
        identity.listing,
        identity.trading_date,
    ):
        _bad("The stored timeline is for another strategy, listing or date.")
    if stored.configuration_document != _configuration_document(identity.configuration):
        raise EquityIntradayResearchTimelineConfigurationConflictError(
            "The stored timeline holds another configuration; it is never changed."
        )
    evaluator = EquityOpeningRangeBreakoutEvaluator(identity.configuration)
    entries = tuple(
        _rehydrate_entry(identity, evaluator, snapshots, index, record)
        for index, record in enumerate(stored.entries, start=1)
    )
    try:
        return EquityIntradayResearchTimeline(identity, entries)
    except InvalidEquityIntradayResearchTimelineError as exc:
        _bad(f"The stored timeline is invalid: {exc}")


# ---------------------------------------------------------------------------
# Advancement (EI3-D11, EI3-D12)
# ---------------------------------------------------------------------------


class EquityIntradayResearchTimelineAdvanceStatus(StrEnum):
    """What one authoritative advancement established."""

    COMMITTED = "COMMITTED"
    NOTHING_TO_COMMIT = "NOTHING_TO_COMMIT"
    STALE_AFTER_RETRIES = "STALE_AFTER_RETRIES"
    CONFIGURATION_CONFLICT = "CONFIGURATION_CONFLICT"


@dataclass(frozen=True, slots=True)
class EquityIntradayResearchTimelineAdvance:
    """The result of one authoritative advancement.

    ``timeline`` is verified authoritative history, or None when none exists
    or none could be established. After an unconfirmed commit is reconciled
    (``reconciled``) it is the reloaded history, which may also hold entries
    another writer appended later. ``schedule`` is EI-3B's result for the
    attempt reported, and ``items`` its per-input outcomes; under
    CONFIGURATION_CONFLICT every input is REFUSED with CONFIGURATION_MISMATCH.
    STALE_AFTER_RETRIES reports no schedule and no items: no attempt
    committed.
    """

    status: EquityIntradayResearchTimelineAdvanceStatus
    timeline: EquityIntradayResearchTimeline | None
    schedule: EquityIntradayResearchScheduleResult | None
    items: tuple[EquityIntradayResearchScheduleItem, ...]
    attempts: int
    commit_confirmed: bool
    reconciled: bool
    detail: str

    def __post_init__(self) -> None:
        status = self.status
        if not isinstance(status, EquityIntradayResearchTimelineAdvanceStatus):
            raise ValueError("status must be an EquityIntradayResearchTimelineAdvanceStatus.")
        if not _count(self.attempts) or not (
            1 <= self.attempts <= EQUITY_INTRADAY_RESEARCH_TIMELINE_COMMIT_ATTEMPTS
        ):
            raise ValueError("attempts must be within the commit attempt limit.")
        committed = status is EquityIntradayResearchTimelineAdvanceStatus.COMMITTED
        if committed != self.commit_confirmed:
            raise ValueError("Exactly a COMMITTED advance has a confirmed commit.")
        if committed and (self.timeline is None or self.schedule is None):
            raise ValueError("A committed advance reports its timeline and schedule.")
        if self.reconciled and not committed:
            raise ValueError("Only a committed advance can be reconciled.")
        if status is EquityIntradayResearchTimelineAdvanceStatus.STALE_AFTER_RETRIES and (
            self.timeline is not None or self.schedule is not None or self.items
        ):
            raise ValueError("STALE_AFTER_RETRIES reports nothing as committed.")

    @property
    def appended(self) -> tuple[EquityIntradayResearchTimelineEntry, ...]:
        """Return the entries this advance committed."""
        if self.schedule is None or not self.commit_confirmed:
            return ()
        return self.schedule.appended


class _Reconciled(StrEnum):
    ABSENT = "ABSENT"
    CONFLICT = "CONFLICT"


def _snapshot_ids(snapshot_ids: Sequence[str]) -> tuple[str, ...]:
    if isinstance(snapshot_ids, str) or not isinstance(snapshot_ids, Sequence):
        raise TypeError("snapshot_ids must be a sequence of snapshot identities.")
    ids = tuple(snapshot_ids)
    if not all(_text(i) for i in ids):
        raise TypeError("Every snapshot identity must be non-empty text.")
    return ids


def _prints(entries: tuple[EquityIntradayResearchTimelineEntry, ...]) -> list[str]:
    return [entry.fingerprint for entry in entries]


class AdvanceEquityIntradayResearchTimelineUseCase:
    """Advances the authoritative research timeline of one configuration."""

    def __init__(
        self,
        timelines: EquityIntradayResearchTimelineStore,
        snapshots: EquityCandleEvidenceSnapshotStore,
        configuration: EquityOpeningRangeBreakoutConfiguration,
    ) -> None:
        if not isinstance(timelines, EquityIntradayResearchTimelineStore):
            raise TypeError("timelines must be an EquityIntradayResearchTimelineStore.")
        self._scheduler = ScheduleEquityIntradayResearchUseCase(snapshots, configuration)
        self._timelines = timelines
        self._snapshots = snapshots
        self._configuration = configuration
        self._configuration_document = _configuration_document(configuration)

    def execute(
        self, listing: ListingReference, trading_date: date, snapshot_ids: Sequence[str]
    ) -> EquityIntradayResearchTimelineAdvance:
        """Evaluate ``snapshot_ids`` into the stored timeline and commit what EI-3B appends.

        Storage, snapshot and verification failures propagate; nothing is
        then claimed as committed.
        """
        if not isinstance(listing, ListingReference):
            raise TypeError("listing must be a ListingReference.")
        if not isinstance(trading_date, date) or isinstance(trading_date, datetime):
            raise TypeError("trading_date must be a plain date.")
        ids = _snapshot_ids(snapshot_ids)
        identity = EquityIntradayResearchTimelineIdentity(
            listing, trading_date, self._configuration
        )
        detail = ""
        limit = EQUITY_INTRADAY_RESEARCH_TIMELINE_COMMIT_ATTEMPTS
        for attempt in range(1, limit + 1):
            stored = self._timelines.load(identity.strategy, listing, trading_date)
            if stored is not None and (
                stored.configuration_document != self._configuration_document
            ):
                return self._conflict(
                    ids, attempt, "The stored timeline holds another configuration."
                )
            current = (
                EquityIntradayResearchTimeline(identity)
                if stored is None
                else rehydrate_equity_intraday_research_timeline(identity, stored, self._snapshots)
            )
            schedule = self._scheduler.advance(current, ids)
            if not schedule.appended:
                return EquityIntradayResearchTimelineAdvance(
                    EquityIntradayResearchTimelineAdvanceStatus.NOTHING_TO_COMMIT,
                    None if stored is None else current,
                    schedule,
                    schedule.items,
                    attempt,
                    False,
                    False,
                    "No entry was appended; nothing was written.",
                )
            expected = current.head_fingerprint
            try:
                self._timelines.append(
                    stored_equity_intraday_research_timeline(schedule.timeline), expected
                )
            except EquityIntradayResearchTimelineStaleError as exc:
                detail = f"Attempt {attempt} was stale: {exc}"
                continue
            except EquityIntradayResearchTimelineConfigurationConflictError as exc:
                return self._conflict(ids, attempt, str(exc))
            except EquityIntradayResearchTimelineCommitUnconfirmedError as exc:
                found = self._reconcile(identity, current, schedule, exc)
                if found is _Reconciled.CONFLICT:
                    return self._conflict(
                        ids, attempt, "Another configuration established the timeline."
                    )
                if found is _Reconciled.ABSENT:
                    detail = f"Attempt {attempt}'s unconfirmed commit is absent; re-evaluated."
                    continue
                return EquityIntradayResearchTimelineAdvance(
                    EquityIntradayResearchTimelineAdvanceStatus.COMMITTED,
                    found,
                    schedule,
                    schedule.items,
                    attempt,
                    True,
                    True,
                    "The unconfirmed commit was found in the reloaded authoritative history.",
                )
            return EquityIntradayResearchTimelineAdvance(
                EquityIntradayResearchTimelineAdvanceStatus.COMMITTED,
                schedule.timeline,
                schedule,
                schedule.items,
                attempt,
                True,
                False,
                f"Committed {len(schedule.appended)} entries.",
            )
        return EquityIntradayResearchTimelineAdvance(
            EquityIntradayResearchTimelineAdvanceStatus.STALE_AFTER_RETRIES,
            None,
            None,
            (),
            limit,
            False,
            False,
            f"No attempt of {limit} committed. {detail}",
        )

    def _reconcile(
        self,
        identity: EquityIntradayResearchTimelineIdentity,
        loaded: EquityIntradayResearchTimeline,
        schedule: EquityIntradayResearchScheduleResult,
        error: EquityIntradayResearchTimelineCommitUnconfirmedError,
    ) -> EquityIntradayResearchTimeline | _Reconciled:
        attempted = schedule.timeline

        def unresolved(reason: str) -> EquityIntradayResearchTimelineCommitUnresolvedError:
            return EquityIntradayResearchTimelineCommitUnresolvedError(
                f"The commit outcome is unknown: {reason}",
                loaded.head_fingerprint,
                attempted.head_fingerprint,
            )

        try:
            stored = self._timelines.load(
                identity.strategy, identity.listing, identity.trading_date
            )
            if stored is None:
                if loaded.entries:
                    raise unresolved("committed history has disappeared.") from error
                return _Reconciled.ABSENT
            if stored.configuration_document != self._configuration_document:
                if loaded.entries:
                    raise unresolved("the stored configuration has changed.") from error
                return _Reconciled.CONFLICT
            history = rehydrate_equity_intraday_research_timeline(identity, stored, self._snapshots)
        except (
            EquityIntradayResearchTimelineUnavailableError,
            EquityCandleEvidenceUnavailableError,
            InvalidStoredEquityIntradayResearchTimelineError,
        ) as exc:
            raise unresolved(f"the authoritative reload failed ({exc}).") from exc
        held = _prints(history.entries)
        if held[: len(loaded.entries)] != _prints(loaded.entries):
            raise unresolved("stored history no longer extends what the attempt loaded.") from error
        if held[: len(attempted.entries)] == _prints(attempted.entries):
            return history
        return _Reconciled.ABSENT

    @staticmethod
    def _conflict(
        ids: tuple[str, ...], attempt: int, detail: str
    ) -> EquityIntradayResearchTimelineAdvance:
        return EquityIntradayResearchTimelineAdvance(
            EquityIntradayResearchTimelineAdvanceStatus.CONFIGURATION_CONFLICT,
            None,
            None,
            tuple(
                EquityIntradayResearchScheduleItem(
                    position,
                    snapshot_id,
                    EquityIntradayResearchScheduleOutcome.REFUSED,
                    EquityIntradayResearchScheduleRefusal.CONFIGURATION_MISMATCH,
                    None,
                    detail,
                )
                for position, snapshot_id in enumerate(ids)
            ),
            attempt,
            False,
            False,
            detail,
        )
