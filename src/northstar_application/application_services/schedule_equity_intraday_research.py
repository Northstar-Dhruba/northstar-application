"""Deterministic synthetic equity intraday research timeline and scheduling (EI-3B).

Advances one immutable research timeline -- one strategy configuration, one
listing, one trading date -- over a finite batch of issued EI-2B evidence
snapshots, evaluating each accepted snapshot with the unchanged EI-3A
evaluator. It applies the owner decisions recorded on 2026-11-06 in the EI-3
Decision Register (northstar-docs, product/Equity-Intraday-V1-EI3-Decision-
Register.md, section 10): EI3-D6, EI3-D7 and EI3-D9. It is synthetic research
only: no entry is a recommendation, order, fill, position or P&L.

Scheduling (EI3-D6)
-------------------
Research uses the approved 09:15 to 15:00 IST window (CE-D17) through EI-3A,
so no signal candle ends after 15:00 IST. The decision instant is the
snapshot's ``as_of`` (EI3-D1) and may fall after 15:00 IST once earlier
candles become final; such an entry is historical research, flagged by
``decided_after_research_window``, and never an actionable intraday signal.
No 14:25 signal cutoff, entry cutoff or exit cutoff is applied (EI0-D7 stays
proposed), and no maximum finality latency is assumed. EI-3A's strict
completeness (EI3-D2) is unchanged: whether a trailing unfinalized candle may
be ignored is EI3-D10, which is pending. Under it, sustained finality latency
of five minutes or more can prevent evaluations during the active session;
post-window evaluations may still become possible once every required candle
is final.

Candidate frequency (EI3-D7)
----------------------------
In decision-instant order, the first EI-3A LONG_CANDIDATE of a timeline is
CANDIDATE; every later one is REPEAT_QUALIFYING and keeps its full decision.
NO_SIGNAL and REJECTED keep their outcomes. The full configuration is fixed
for the timeline: a scheduler with another configuration is refused, never a
second candidate. No later correction withdraws an earlier entry (EI0-D21 is
pending).

Conflicts and ordering (EI3-D9)
-------------------------------
Each submitted snapshot is loaded and evaluated by EI-3A. A snapshot that is
missing, untrustworthy, not synthetic, for another listing or date, or for
another timeframe, window policy or grid is REFUSED: it is not part of this
history, and is never recorded as a research rejection. A trustworthy, in-scope
snapshot whose market evidence EI-3A rejects becomes a REJECTED entry.

Accepted snapshots are then classified against the timeline as supplied:

- a snapshot at an instant the timeline already holds is DUPLICATE when its
  manifest fingerprint is the entry's, and otherwise REFUSED with
  CONFLICTING_EVIDENCE_FOR_INSTANT -- the entry is never replaced;
- a snapshot before the timeline head is REFUSED with BEFORE_TIMELINE_HEAD, so
  late input never creates a retroactive candidate;
- the rest are grouped by decision instant and taken in ascending instant
  order. A group with more than one manifest is refused entirely, with
  CONFLICTING_EVIDENCE_FOR_INSTANT: no winner is chosen. A group with one
  manifest is equivalent evidence -- the manifest fingerprint covers the as-of
  instant, watermark and every record, so only the issuance instant, and
  hence the snapshot ID, can differ. Its representative is the snapshot with
  the lexicographically smallest ID (then the earliest position); the others
  are DUPLICATE of its entry. This chooses among identical evidence only. The
  group is REFUSED with EVIDENCE_REGRESSION when its watermark is below the
  head's, and is appended otherwise.

The order is therefore as_of, then manifest (which fixes the watermark), then
snapshot ID. Equivalence uses manifest fingerprints, never EI-3A decision
fingerprints, which include the snapshot ID. Manifests that differ only in
watermark are distinct manifests and conflict.

Determinism guarantee
---------------------
Within one ``advance`` call the result does not depend on the order of the
batch: the same timeline, entries and fingerprints. Across calls, how
snapshots are split between calls never changes the research: which manifest
decides each instant, each entry's role and outcome, its research values, or
the candidate count. It can change which of several equivalent snapshots
represents an instant. The first one established in an earlier call stays the
entry -- history is never rewritten -- and an equivalent snapshot with a
smaller ID arriving later is a DUPLICATE of it. Because the EI-3A decision
fingerprint includes the snapshot ID, that entry's decision fingerprint,
entry fingerprint and every later entry and head fingerprint then differ from
those of a single-batch run. Timeline fingerprints are therefore
reproducible for the same sequence of calls, not across batch partitions.

Integrity
---------
Each entry is fingerprinted (``eirt1-sha256:``) over the timeline identity,
its sequence, role, previous entry fingerprint, the EI-3A decision
fingerprint and its evidence references -- never explanation prose. A timeline
re-verifies every entry before it is advanced; an invalid one is refused
whole. The chain shows that entries were not altered after they were chained.
It does not prove that a supplied timeline is complete or authenticate whoever
supplied it: authoritative history is EI-3C.

Nothing here reads a clock, generates random identifiers, persists anything,
or calls a provider, broker or network.
"""

from __future__ import annotations

import dataclasses
import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from enum import StrEnum
from types import MappingProxyType
from typing import Any, NoReturn

from northstar_core.domain.value_objects import ListingReference
from northstar_core.foundation.value_objects import PointInTime
from northstar_core.strategy import StrategyIdentity

from northstar_application.application_services.equity_candle_evidence_journal import (
    canonical_equity_candle_json,
)
from northstar_application.application_services.equity_candle_evidence_snapshot import (
    EquityCandleEvidenceSnapshot,
)
from northstar_application.application_services.evaluate_equity_intraday_research import (
    EquityIntradayResearchDecision,
    EquityIntradayResearchOutcome,
    EquityIntradayResearchRejection,
    EquityOpeningRangeBreakoutConfiguration,
    EquityOpeningRangeBreakoutEvaluator,
)
from northstar_application.ports.equity_candle_evidence_journal import (
    EquityCandleEvidenceSnapshotStore,
)

EQUITY_INTRADAY_RESEARCH_TIMELINE_ENTRY_PREFIX = "eirt1-sha256:"
_IDENTITY_FORMAT = "northstar.equity-intraday-research-timeline/1"
_ENTRY_FORMAT = "northstar.equity-intraday-research-timeline-entry/1"


class InvalidEquityIntradayResearchTimelineError(ValueError):
    """Raised when a research timeline or one of its entries breaks an invariant."""


def _invalid(message: str) -> NoReturn:
    raise InvalidEquityIntradayResearchTimelineError(message)


def _instant(value: PointInTime) -> datetime:
    return datetime.fromisoformat(value.value.replace("Z", "+00:00")).astimezone(UTC)


class EquityIntradayResearchTimelineRole(StrEnum):
    """An entry's role in its timeline (EI3-D7). None of them is an order or a recommendation."""

    CANDIDATE = "CANDIDATE"
    REPEAT_QUALIFYING = "REPEAT_QUALIFYING"
    NO_SIGNAL = "NO_SIGNAL"
    REJECTED = "REJECTED"


class EquityIntradayResearchScheduleOutcome(StrEnum):
    """What happened to one submitted snapshot."""

    APPENDED = "APPENDED"
    DUPLICATE = "DUPLICATE"
    REFUSED = "REFUSED"


class EquityIntradayResearchScheduleRefusal(StrEnum):
    """Why a submitted snapshot cannot enter the timeline.

    A refusal is not a research rejection: a REFUSED snapshot is not part of
    this history, whereas a REJECTED entry records EI-3A rejecting evidence
    that does belong to it.
    """

    INVALID_TIMELINE = "INVALID_TIMELINE"
    CONFIGURATION_MISMATCH = "CONFIGURATION_MISMATCH"
    SNAPSHOT_UNAVAILABLE = "SNAPSHOT_UNAVAILABLE"
    UNTRUSTED_SNAPSHOT = "UNTRUSTED_SNAPSHOT"
    EVIDENCE_NOT_SYNTHETIC = "EVIDENCE_NOT_SYNTHETIC"
    OTHER_LISTING_OR_DATE = "OTHER_LISTING_OR_DATE"
    RESEARCH_CONTEXT_MISMATCH = "RESEARCH_CONTEXT_MISMATCH"
    BEFORE_TIMELINE_HEAD = "BEFORE_TIMELINE_HEAD"
    CONFLICTING_EVIDENCE_FOR_INSTANT = "CONFLICTING_EVIDENCE_FOR_INSTANT"
    EVIDENCE_REGRESSION = "EVIDENCE_REGRESSION"


_RESEARCH = EquityIntradayResearchRejection
_REFUSE = EquityIntradayResearchScheduleRefusal
# EI-3A rejections that say a snapshot is outside this timeline's scope.
_SCOPE_REFUSALS = MappingProxyType(
    {
        _RESEARCH.EVIDENCE_NOT_SYNTHETIC: _REFUSE.EVIDENCE_NOT_SYNTHETIC,
        _RESEARCH.LISTING_MISMATCH: _REFUSE.OTHER_LISTING_OR_DATE,
        _RESEARCH.TRADING_DATE_MISMATCH: _REFUSE.OTHER_LISTING_OR_DATE,
        _RESEARCH.UNSUPPORTED_TIMEFRAME: _REFUSE.RESEARCH_CONTEXT_MISMATCH,
        _RESEARCH.RESEARCH_WINDOW_POLICY_MISMATCH: _REFUSE.RESEARCH_CONTEXT_MISMATCH,
        _RESEARCH.INVALID_INTERVAL_GRID: _REFUSE.RESEARCH_CONTEXT_MISMATCH,
    }
)
_ROLE_OUTCOMES = MappingProxyType(
    {
        EquityIntradayResearchTimelineRole.CANDIDATE: EquityIntradayResearchOutcome.LONG_CANDIDATE,
        EquityIntradayResearchTimelineRole.REPEAT_QUALIFYING: (
            EquityIntradayResearchOutcome.LONG_CANDIDATE
        ),
        EquityIntradayResearchTimelineRole.NO_SIGNAL: EquityIntradayResearchOutcome.NO_SIGNAL,
        EquityIntradayResearchTimelineRole.REJECTED: EquityIntradayResearchOutcome.REJECTED,
    }
)


def _scope_rejection(
    decision: EquityIntradayResearchDecision,
) -> EquityIntradayResearchScheduleRefusal | None:
    if decision.outcome is not EquityIntradayResearchOutcome.REJECTED:
        return None
    first = decision.reasons[0]
    if not isinstance(first, EquityIntradayResearchRejection):
        return None
    return _SCOPE_REFUSALS.get(first)


# ---------------------------------------------------------------------------
# Timeline contract
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EquityIntradayResearchTimelineIdentity:
    """What one timeline is about: a full synthetic configuration, a listing and a date.

    Identities are compared by ``document()``, so configurations that differ
    in any recorded value -- even ``1.5`` against ``1.50`` -- differ.
    """

    listing: ListingReference
    trading_date: date
    configuration: EquityOpeningRangeBreakoutConfiguration

    def __post_init__(self) -> None:
        if not isinstance(self.listing, ListingReference):
            _invalid("Timeline listing must be a ListingReference.")
        if type(self.trading_date) is not date:
            _invalid("Timeline trading_date must be a plain date.")
        if not isinstance(self.configuration, EquityOpeningRangeBreakoutConfiguration):
            _invalid("Timeline configuration must be an EquityOpeningRangeBreakoutConfiguration.")

    @property
    def strategy(self) -> StrategyIdentity:
        """Return the configuration's strategy identity."""
        return self.configuration.strategy

    def document(self) -> dict[str, Any]:
        """Return the identity's canonical content."""
        return {
            "format": _IDENTITY_FORMAT,
            "strategy": self.configuration.strategy.identity,
            "configuration": self.configuration.document(),
            "listing": {
                "symbol": self.listing.symbol.value,
                "exchange_code": self.listing.exchange_code.value,
            },
            "trading_date": self.trading_date.isoformat(),
        }


@dataclass(frozen=True, slots=True)
class EquityIntradayResearchTimelineEntry:
    """One immutable, chained research decision in a timeline.

    ``decision`` is the unchanged EI-3A decision for the entry's snapshot;
    ``research_window_closes_at`` is that snapshot's window close.
    ``fingerprint`` is derived and cannot be supplied.
    """

    identity: EquityIntradayResearchTimelineIdentity
    sequence: int
    role: EquityIntradayResearchTimelineRole
    decision: EquityIntradayResearchDecision
    research_window_closes_at: PointInTime
    previous_fingerprint: str | None
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        self._check()
        object.__setattr__(self, "fingerprint", self._digest())

    def _check(self) -> None:
        if not isinstance(self.identity, EquityIntradayResearchTimelineIdentity):
            _invalid("Entry identity must be an EquityIntradayResearchTimelineIdentity.")
        if not isinstance(self.sequence, int) or isinstance(self.sequence, bool):
            _invalid("Entry sequence must be an integer.")
        if self.sequence < 1:
            _invalid("Entry sequence starts at 1.")
        if not isinstance(self.role, EquityIntradayResearchTimelineRole):
            _invalid("Entry role must be an EquityIntradayResearchTimelineRole.")
        if not isinstance(self.research_window_closes_at, PointInTime):
            _invalid("Entry research_window_closes_at must be a PointInTime.")
        previous = self.previous_fingerprint
        if self.sequence == 1:
            if previous is not None:
                _invalid("The first entry has no previous entry.")
        elif not isinstance(previous, str) or not previous.startswith(
            EQUITY_INTRADAY_RESEARCH_TIMELINE_ENTRY_PREFIX
        ):
            _invalid(f"Entry {self.sequence} must name its previous entry's fingerprint.")
        decision = self.decision
        if not isinstance(decision, EquityIntradayResearchDecision):
            _invalid("Entry decision must be an EquityIntradayResearchDecision.")
        evidence = decision.evidence
        if evidence is None:
            _invalid("A decision on an untrusted snapshot can never be an entry.")
        if decision.outcome is not _ROLE_OUTCOMES[self.role]:
            _invalid(f"A {decision.outcome.value} decision cannot have role {self.role.value}.")
        if _scope_rejection(decision) is not None:
            _invalid(f"A snapshot rejected for {decision.reasons[0].value} is out of scope.")
        if (decision.listing, decision.trading_date) != (
            self.identity.listing,
            self.identity.trading_date,
        ) or decision.configuration.document() != self.identity.configuration.document():
            _invalid("Entry decision is for another listing, date or configuration.")
        if decision.requested_snapshot_id != evidence.snapshot_id:
            _invalid("Entry decision must be for the snapshot it names.")
        ends = evidence.required_interval_ends
        if ends and ends[-1].compare(self.research_window_closes_at) > 0:
            _invalid("A required interval ends after the research window.")

    def document(self) -> dict[str, Any]:
        """Return the entry's canonical structural content -- the fingerprint's input."""
        evidence = self.decision.evidence
        assert evidence is not None
        return {
            "format": _ENTRY_FORMAT,
            "timeline": self.identity.document(),
            "sequence": self.sequence,
            "role": self.role.value,
            "previous_fingerprint": self.previous_fingerprint,
            "decision_fingerprint": self.decision.fingerprint,
            "decision_outcome": self.decision.outcome.value,
            "decision_instant": evidence.as_of.value,
            "watermark": evidence.watermark,
            "snapshot_id": evidence.snapshot_id,
            "manifest_fingerprint": evidence.manifest_fingerprint,
            "research_window_closes_at": self.research_window_closes_at.value,
        }

    def _digest(self) -> str:
        encoded = canonical_equity_candle_json(self.document()).encode("utf-8")
        return EQUITY_INTRADAY_RESEARCH_TIMELINE_ENTRY_PREFIX + hashlib.sha256(encoded).hexdigest()

    def verify(self) -> None:
        """Re-derive the decision and this entry; raise if either was altered."""
        self._check()
        try:
            rebuilt = dataclasses.replace(self.decision)
        except (ValueError, TypeError) as exc:
            _invalid(f"Entry {self.sequence}'s decision no longer validates: {exc}")
        if rebuilt != self.decision:
            _invalid(f"Entry {self.sequence}'s decision does not match its fingerprint.")
        if self._digest() != self.fingerprint:
            _invalid(f"Entry {self.sequence} does not match its fingerprint.")

    @property
    def decision_instant(self) -> PointInTime:
        """Return the decision instant: the snapshot's ``as_of`` (EI3-D1)."""
        assert self.decision.evidence is not None
        return self.decision.evidence.as_of

    @property
    def watermark(self) -> int:
        """Return the snapshot's journal watermark."""
        assert self.decision.evidence is not None
        return self.decision.evidence.watermark

    @property
    def snapshot_id(self) -> str:
        """Return the identity of the snapshot the decision rests on."""
        assert self.decision.evidence is not None
        return self.decision.evidence.snapshot_id

    @property
    def manifest_fingerprint(self) -> str:
        """Return the snapshot's manifest fingerprint."""
        assert self.decision.evidence is not None
        return self.decision.evidence.manifest_fingerprint

    @property
    def decided_after_research_window(self) -> bool:
        """Return whether the decision was made after the window closed: historical research."""
        return self.decision_instant.compare(self.research_window_closes_at) > 0


@dataclass(frozen=True, slots=True)
class EquityIntradayResearchTimeline:
    """One immutable research timeline for one configuration, listing and trading date.

    Construction verifies every invariant. A timeline is what its caller
    supplies: it cannot show that no other history exists (EI-3C).
    """

    identity: EquityIntradayResearchTimelineIdentity
    entries: tuple[EquityIntradayResearchTimelineEntry, ...] = ()

    def __post_init__(self) -> None:
        self.verify()

    @classmethod
    def start(
        cls,
        listing: ListingReference,
        trading_date: date,
        configuration: EquityOpeningRangeBreakoutConfiguration,
    ) -> EquityIntradayResearchTimeline:
        """Return a new, empty timeline. Never use it to continue existing history."""
        return cls(EquityIntradayResearchTimelineIdentity(listing, trading_date, configuration))

    def verify(self) -> None:
        """Re-check every invariant and re-derive every fingerprint; raise if any fails."""
        if not isinstance(self.identity, EquityIntradayResearchTimelineIdentity):
            _invalid("Timeline identity must be an EquityIntradayResearchTimelineIdentity.")
        if not isinstance(self.entries, tuple) or not all(
            isinstance(e, EquityIntradayResearchTimelineEntry) for e in self.entries
        ):
            _invalid("Timeline entries must be a tuple of EquityIntradayResearchTimelineEntry.")
        identity = self.identity.document()
        previous: EquityIntradayResearchTimelineEntry | None = None
        qualified = False
        for index, entry in enumerate(self.entries, start=1):
            entry.verify()
            if entry.identity.document() != identity:
                _invalid(f"Entry {index} belongs to another timeline.")
            if entry.sequence != index:
                _invalid(f"Entry {index} has sequence {entry.sequence}.")
            if entry.previous_fingerprint != (None if previous is None else previous.fingerprint):
                _invalid(f"Entry {index} does not chain to its previous entry.")
            if previous is not None:
                if entry.decision_instant.compare(previous.decision_instant) <= 0:
                    _invalid(f"Entry {index}'s decision instant does not follow the previous one.")
                if entry.watermark < previous.watermark:
                    _invalid(f"Entry {index}'s journal watermark regresses.")
            if entry.decision.outcome is EquityIntradayResearchOutcome.LONG_CANDIDATE:
                expected = (
                    EquityIntradayResearchTimelineRole.REPEAT_QUALIFYING
                    if qualified
                    else EquityIntradayResearchTimelineRole.CANDIDATE
                )
                if entry.role is not expected:
                    _invalid(f"Entry {index} must be {expected.value}.")
                qualified = True
            previous = entry

    @property
    def listing(self) -> ListingReference:
        """Return the timeline's listing."""
        return self.identity.listing

    @property
    def trading_date(self) -> date:
        """Return the timeline's trading date."""
        return self.identity.trading_date

    @property
    def configuration(self) -> EquityOpeningRangeBreakoutConfiguration:
        """Return the timeline's fixed configuration."""
        return self.identity.configuration

    @property
    def head(self) -> EquityIntradayResearchTimelineEntry | None:
        """Return the latest entry, or None for an empty timeline."""
        return self.entries[-1] if self.entries else None

    @property
    def head_fingerprint(self) -> str | None:
        """Return the latest entry's fingerprint, or None for an empty timeline."""
        return self.entries[-1].fingerprint if self.entries else None

    @property
    def candidate(self) -> EquityIntradayResearchTimelineEntry | None:
        """Return the timeline's one CANDIDATE entry, if it has one."""
        return next(
            (e for e in self.entries if e.role is EquityIntradayResearchTimelineRole.CANDIDATE),
            None,
        )

    def document(self) -> dict[str, Any]:
        """Return the timeline's canonical content, decisions included."""
        return {
            "timeline": self.identity.document(),
            "head_fingerprint": self.head_fingerprint,
            "entries": [
                {
                    **{k: v for k, v in e.document().items() if k != "timeline"},
                    "fingerprint": e.fingerprint,
                    "decision": e.decision.document(),
                }
                for e in self.entries
            ],
        }


# ---------------------------------------------------------------------------
# Scheduling
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EquityIntradayResearchScheduleItem:
    """The outcome for one submitted snapshot, at its position in the batch.

    ``entry`` is the appended entry, or for a DUPLICATE the established entry
    it repeats; a REFUSED snapshot has a refusal and no entry.
    """

    position: int
    snapshot_id: str
    outcome: EquityIntradayResearchScheduleOutcome
    refusal: EquityIntradayResearchScheduleRefusal | None
    entry: EquityIntradayResearchTimelineEntry | None
    detail: str

    def __post_init__(self) -> None:
        if not isinstance(self.outcome, EquityIntradayResearchScheduleOutcome):
            raise ValueError("outcome must be an EquityIntradayResearchScheduleOutcome.")
        refused = self.outcome is EquityIntradayResearchScheduleOutcome.REFUSED
        if refused != isinstance(self.refusal, EquityIntradayResearchScheduleRefusal):
            raise ValueError("Exactly a REFUSED item has a refusal.")
        if refused != (self.entry is None):
            raise ValueError("Exactly an APPENDED or DUPLICATE item names an entry.")


@dataclass(frozen=True, slots=True)
class EquityIntradayResearchScheduleResult:
    """The advanced timeline and one item per submitted snapshot, in submission order."""

    timeline: EquityIntradayResearchTimeline
    items: tuple[EquityIntradayResearchScheduleItem, ...]

    @property
    def appended(self) -> tuple[EquityIntradayResearchTimelineEntry, ...]:
        """Return the entries this batch appended, in timeline order."""
        appended = [
            i.entry
            for i in self.items
            if i.outcome is EquityIntradayResearchScheduleOutcome.APPENDED and i.entry is not None
        ]
        return tuple(sorted(appended, key=lambda e: e.sequence))


@dataclass(frozen=True, slots=True)
class _Accepted:
    position: int
    snapshot_id: str
    decision: EquityIntradayResearchDecision
    window_closes_at: PointInTime

    @property
    def instant(self) -> PointInTime:
        assert self.decision.evidence is not None
        return self.decision.evidence.as_of

    @property
    def manifest(self) -> str:
        assert self.decision.evidence is not None
        return self.decision.evidence.manifest_fingerprint

    @property
    def watermark(self) -> int:
        assert self.decision.evidence is not None
        return self.decision.evidence.watermark


def _role(
    outcome: EquityIntradayResearchOutcome, qualified: bool
) -> EquityIntradayResearchTimelineRole:
    if outcome is EquityIntradayResearchOutcome.LONG_CANDIDATE:
        return (
            EquityIntradayResearchTimelineRole.REPEAT_QUALIFYING
            if qualified
            else EquityIntradayResearchTimelineRole.CANDIDATE
        )
    if outcome is EquityIntradayResearchOutcome.NO_SIGNAL:
        return EquityIntradayResearchTimelineRole.NO_SIGNAL
    return EquityIntradayResearchTimelineRole.REJECTED


def _snapshot_ids(snapshot_ids: Sequence[str]) -> tuple[str, ...]:
    if isinstance(snapshot_ids, str) or not isinstance(snapshot_ids, Sequence):
        raise TypeError("snapshot_ids must be a sequence of snapshot identities.")
    ids = tuple(snapshot_ids)
    if not all(isinstance(i, str) and i for i in ids):
        raise TypeError("Every snapshot identity must be non-empty text.")
    return ids


class ScheduleEquityIntradayResearchUseCase:
    """Advances an explicit research timeline over issued snapshots; nothing is persisted."""

    def __init__(
        self,
        snapshots: EquityCandleEvidenceSnapshotStore,
        configuration: EquityOpeningRangeBreakoutConfiguration,
    ) -> None:
        if not isinstance(snapshots, EquityCandleEvidenceSnapshotStore):
            raise TypeError("snapshots must be an EquityCandleEvidenceSnapshotStore.")
        self._snapshots = snapshots
        self._evaluator = EquityOpeningRangeBreakoutEvaluator(configuration)
        self._configuration = configuration

    def start(
        self, listing: ListingReference, trading_date: date
    ) -> EquityIntradayResearchTimeline:
        """Return a new, empty timeline under this scheduler's configuration."""
        return EquityIntradayResearchTimeline.start(listing, trading_date, self._configuration)

    def advance(
        self, timeline: EquityIntradayResearchTimeline, snapshot_ids: Sequence[str]
    ) -> EquityIntradayResearchScheduleResult:
        """Evaluate ``snapshot_ids`` into ``timeline`` and return the advanced timeline.

        ``timeline`` must be the history to continue; it is never replaced by a
        fresh one. A storage failure propagates; nothing is returned, and the
        caller's timeline is unchanged.
        """
        if not isinstance(timeline, EquityIntradayResearchTimeline):
            raise TypeError("timeline must be an EquityIntradayResearchTimeline.")
        ids = _snapshot_ids(snapshot_ids)
        try:
            timeline.verify()
        except InvalidEquityIntradayResearchTimelineError as exc:
            return self._refuse_all(timeline, ids, _REFUSE.INVALID_TIMELINE, str(exc))
        if timeline.configuration.document() != self._configuration.document():
            return self._refuse_all(
                timeline,
                ids,
                _REFUSE.CONFIGURATION_MISMATCH,
                f"The timeline's configuration is {timeline.configuration.version}; this "
                f"scheduler's is {self._configuration.version}. A timeline's configuration "
                "never changes.",
            )

        items: dict[int, EquityIntradayResearchScheduleItem] = {}

        def record(
            position: int,
            snapshot_id: str,
            outcome: EquityIntradayResearchScheduleOutcome,
            detail: str,
            *,
            refusal: EquityIntradayResearchScheduleRefusal | None = None,
            entry: EquityIntradayResearchTimelineEntry | None = None,
        ) -> None:
            items[position] = EquityIntradayResearchScheduleItem(
                position, snapshot_id, outcome, refusal, entry, detail
            )

        def refuse(
            item: _Accepted, refusal: EquityIntradayResearchScheduleRefusal, detail: str
        ) -> None:
            record(
                item.position,
                item.snapshot_id,
                EquityIntradayResearchScheduleOutcome.REFUSED,
                detail,
                refusal=refusal,
            )

        def duplicate(item: _Accepted, entry: EquityIntradayResearchTimelineEntry) -> None:
            record(
                item.position,
                item.snapshot_id,
                EquityIntradayResearchScheduleOutcome.DUPLICATE,
                f"Manifest {item.manifest} is already entry {entry.sequence} "
                f"(snapshot {entry.snapshot_id}); nothing was appended.",
                entry=entry,
            )

        accepted: list[_Accepted] = []
        loaded: dict[str, EquityCandleEvidenceSnapshot | None] = {}
        for position, snapshot_id in enumerate(ids):
            if snapshot_id not in loaded:
                loaded[snapshot_id] = self._snapshots.get_snapshot(snapshot_id)
            snapshot = loaded[snapshot_id]
            decision = self._evaluator.evaluate(
                timeline.listing, timeline.trading_date, snapshot, snapshot_id
            )
            refusal = self._scope_refusal(decision)
            if refusal is not None:
                record(
                    position,
                    snapshot_id,
                    EquityIntradayResearchScheduleOutcome.REFUSED,
                    f"Not part of this timeline: EI-3A found {decision.reasons[0].value}.",
                    refusal=refusal,
                )
                continue
            assert snapshot is not None
            accepted.append(_Accepted(position, snapshot_id, decision, snapshot.window_closes_at))

        established = {_instant(e.decision_instant): e for e in timeline.entries}
        head = timeline.head
        fresh: dict[datetime, list[_Accepted]] = {}
        for item in accepted:
            entry = established.get(_instant(item.instant))
            if entry is not None:
                if entry.manifest_fingerprint == item.manifest:
                    duplicate(item, entry)
                else:
                    refuse(
                        item,
                        _REFUSE.CONFLICTING_EVIDENCE_FOR_INSTANT,
                        f"Entry {entry.sequence} already decided {item.instant} on manifest "
                        f"{entry.manifest_fingerprint}; {item.manifest} differs. The entry is "
                        "kept and no winner is chosen.",
                    )
            elif head is not None and item.instant.compare(head.decision_instant) < 0:
                refuse(
                    item,
                    _REFUSE.BEFORE_TIMELINE_HEAD,
                    f"{item.instant} precedes the timeline head at {head.decision_instant}; "
                    "history is never rewritten.",
                )
            else:
                fresh.setdefault(_instant(item.instant), []).append(item)

        entries = list(timeline.entries)
        qualified = any(
            e.decision.outcome is EquityIntradayResearchOutcome.LONG_CANDIDATE for e in entries
        )
        for instant in sorted(fresh):
            group = sorted(fresh[instant], key=lambda i: (i.snapshot_id, i.position))
            manifests = sorted({i.manifest for i in group})
            if len(manifests) > 1:
                for item in group:
                    refuse(
                        item,
                        _REFUSE.CONFLICTING_EVIDENCE_FOR_INSTANT,
                        f"{len(manifests)} different manifests claim {item.instant}: "
                        f"{', '.join(manifests)}. None is chosen.",
                    )
                continue
            representative = group[0]
            if entries and representative.watermark < entries[-1].watermark:
                for item in group:
                    refuse(
                        item,
                        _REFUSE.EVIDENCE_REGRESSION,
                        f"Watermark {item.watermark} at {item.instant} is below the head's "
                        f"{entries[-1].watermark}.",
                    )
                continue
            role = _role(representative.decision.outcome, qualified)
            entry = EquityIntradayResearchTimelineEntry(
                timeline.identity,
                len(entries) + 1,
                role,
                representative.decision,
                representative.window_closes_at,
                entries[-1].fingerprint if entries else None,
            )
            entries.append(entry)
            qualified = qualified or role is EquityIntradayResearchTimelineRole.CANDIDATE
            record(
                representative.position,
                representative.snapshot_id,
                EquityIntradayResearchScheduleOutcome.APPENDED,
                f"Entry {entry.sequence}: {role.value} at {representative.instant}.",
                entry=entry,
            )
            for item in group[1:]:
                duplicate(item, entry)

        return EquityIntradayResearchScheduleResult(
            EquityIntradayResearchTimeline(timeline.identity, tuple(entries)),
            tuple(items[position] for position in range(len(ids))),
        )

    @staticmethod
    def _scope_refusal(
        decision: EquityIntradayResearchDecision,
    ) -> EquityIntradayResearchScheduleRefusal | None:
        if decision.evidence is None:
            if decision.reasons[0] is _RESEARCH.SNAPSHOT_UNAVAILABLE:
                return _REFUSE.SNAPSHOT_UNAVAILABLE
            return _REFUSE.UNTRUSTED_SNAPSHOT
        return _scope_rejection(decision)

    @staticmethod
    def _refuse_all(
        timeline: EquityIntradayResearchTimeline,
        ids: tuple[str, ...],
        refusal: EquityIntradayResearchScheduleRefusal,
        detail: str,
    ) -> EquityIntradayResearchScheduleResult:
        return EquityIntradayResearchScheduleResult(
            timeline,
            tuple(
                EquityIntradayResearchScheduleItem(
                    position,
                    snapshot_id,
                    EquityIntradayResearchScheduleOutcome.REFUSED,
                    refusal,
                    None,
                    detail,
                )
                for position, snapshot_id in enumerate(ids)
            ),
        )
