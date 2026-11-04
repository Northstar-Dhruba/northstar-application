"""Append-only journal records for 5-minute equity candle evidence (EI-2B).

EI-2A validates candle observations held in memory. EI-2B makes the evidence
durable and auditable: every candle observation and every finality statement
Northstar receives becomes one immutable journal entry, and nothing is ever
updated, replaced or deleted. This module defines what an entry holds, how it
is canonically encoded and fingerprinted, the append rules every journal
applies, and how a window's evidence is composed for EI-2A. The port is
``northstar_application.ports.equity_candle_evidence_journal``.

Four different times
--------------------
- ``source_published_at``: when the source says it published the candle, if
  it says so. It may not be after the receipt.
- ``received_at``: when Northstar received the record.
- the journal ``sequence`` and ``journaled_at``: when the journal appended it.
  The sequence is assigned by the journal, strictly increasing, and is the
  only authority on insertion order; ``journaled_at`` is informative.
- a snapshot's ``issued_at`` (equity_candle_evidence_snapshot).

No source or receipt time orders the journal. A record appended late carries a
late sequence whatever it claims, so it cannot enter a snapshot taken at an
earlier watermark.

Finality is a record of its own
-------------------------------
An observation record carries no finality: its EI-2A ``finality`` must be
None. A finality statement is journaled as its own record, with its own
receipt time and sequence, so it may arrive before or after its candle. It is
composed onto its candle only when it is within the reader's watermark and was
received by the observation instant; a source confirmation time that is
earlier cannot make it visible earlier. Journaling a statement approves
nothing: approval is the EI-2A finality policy's, at evaluation.

Identity, retries and conflicts (fail closed)
---------------------------------------------
- ``ingestion_id`` is Northstar's stable identity for one receipt. The same
  record appended again under it is a retry: nothing is written and the
  original entry is returned. Any difference under it -- payload or receipt
  time -- raises EquityCandleIngestionConflictError; nothing is written.
- A source identity (source, source event, revision) received again under a
  new ingestion id is a distinct receipt and is appended. Its first receipt
  binds the identity to its listing, timeframe, interval end, correction
  reference, publication time and synthetic classification; a receipt that
  disagrees raises EquityCandleSourceIdentityConflictError and nothing is
  written, because it would place one source record in two intervals or give
  it two provenances, and no EI-2A outcome can represent that.
- Different OHLCV values under one source identity are kept: the receipt is
  appended, the append result names the earlier entries it contradicts, and
  EI-2A reports the interval CONFLICTING_VALUES. Nothing is overwritten,
  merged or preferred.
- A finality statement must repeat its identity's earlier statements exactly,
  and agree with its candle's synthetic classification; otherwise it raises
  EquityCandleSourceIdentityConflictError.
- A correction (``corrects_revision``) is a new record holding a reference.
  It is never applied, trusted or used to order revisions; EI-2A reports such
  an interval UNRESOLVED_REVISIONS.

Encoding
--------
Canonical JSON: sorted keys, no whitespace, decimals as their exact Core
fixed-point text (never float), instants as canonical UTC PointInTime text.
``ecp1-sha256:`` fingerprints a record's payload -- what the source asserted.
``ecr1-sha256:`` fingerprints the whole record -- payload, ingestion id and
receipt time. Decoding requires the text to re-encode exactly.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from decimal import Decimal
from enum import StrEnum
from typing import Any

from northstar_core.domain.value_objects import ListingReference
from northstar_core.foundation.value_objects import (
    Currency,
    ExchangeCode,
    PointInTime,
    Price,
    Quantity,
    Symbol,
    Timeframe,
)
from northstar_core.market_data import HistoricalOHLCVBar

from northstar_application.application_services.equity_intraday_candle_validation import (
    EquityCandleFinalityEvidence,
    EquityCandleObservation,
)
from northstar_application.ports.equity_candle_evidence_journal import (
    EquityCandleIngestionConflictError,
    EquityCandleSourceIdentityConflictError,
)

EQUITY_CANDLE_PAYLOAD_FINGERPRINT_PREFIX = "ecp1-sha256:"
EQUITY_CANDLE_RECORD_FINGERPRINT_PREFIX = "ecr1-sha256:"
_FORMAT = "northstar.equity-candle-evidence/1"


class InvalidEquityCandleJournalRecordError(ValueError):
    """Raised when a journal record, entry or window is malformed or inconsistent."""


def _require_text(value: object, subject: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise InvalidEquityCandleJournalRecordError(f"{subject} must be a non-empty string.")


def _require_instant(value: object, subject: str) -> None:
    if not isinstance(value, PointInTime):
        raise InvalidEquityCandleJournalRecordError(f"{subject} must be a PointInTime.")


def _require_sequence(value: object, subject: str, minimum: int = 1) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise InvalidEquityCandleJournalRecordError(f"{subject} must be an integer >= {minimum}.")


class EquityCandleJournalEntryKind(StrEnum):
    """What one journal entry records."""

    OBSERVATION = "OBSERVATION"
    FINALITY = "FINALITY"


@dataclass(frozen=True, slots=True)
class EquityCandleObservationRecord:
    """One received candle observation, as journaled.

    ``observation`` is the EI-2A observation without finality; ``is_synthetic``
    marks invented test evidence and is never changed.
    """

    ingestion_id: str
    observation: EquityCandleObservation
    is_synthetic: bool
    source_published_at: PointInTime | None = None

    def __post_init__(self) -> None:
        _require_text(self.ingestion_id, "Observation record ingestion_id")
        if not isinstance(self.observation, EquityCandleObservation):
            raise InvalidEquityCandleJournalRecordError(
                "Observation record observation must be an EquityCandleObservation."
            )
        if self.observation.finality is not None:
            raise InvalidEquityCandleJournalRecordError(
                "A journaled observation carries no finality; finality is journaled as its own "
                "EquityCandleFinalityRecord so its receipt and journal position are kept."
            )
        if not isinstance(self.is_synthetic, bool):
            raise InvalidEquityCandleJournalRecordError(
                "Observation record is_synthetic must be a bool."
            )
        if self.source_published_at is not None:
            _require_instant(self.source_published_at, "Observation record source_published_at")
            if self.source_published_at.compare(self.observation.received_at) > 0:
                raise InvalidEquityCandleJournalRecordError(
                    "A candle cannot be published by its source after Northstar received it."
                )

    @property
    def kind(self) -> EquityCandleJournalEntryKind:
        """Return OBSERVATION."""
        return EquityCandleJournalEntryKind.OBSERVATION

    @property
    def source_identity(self) -> tuple[str, str, str]:
        """Return the source, source event and revision this record is."""
        o = self.observation
        return (o.source_id, o.source_event_id, o.source_revision)

    @property
    def received_at(self) -> PointInTime:
        """Return when Northstar received the record."""
        return self.observation.received_at


@dataclass(frozen=True, slots=True)
class EquityCandleFinalityRecord:
    """One received source-finality statement, as journaled."""

    ingestion_id: str
    evidence: EquityCandleFinalityEvidence
    received_at: PointInTime

    def __post_init__(self) -> None:
        _require_text(self.ingestion_id, "Finality record ingestion_id")
        if not isinstance(self.evidence, EquityCandleFinalityEvidence):
            raise InvalidEquityCandleJournalRecordError(
                "Finality record evidence must be EquityCandleFinalityEvidence."
            )
        _require_instant(self.received_at, "Finality record received_at")
        if self.evidence.confirmed_at.compare(self.received_at) > 0:
            raise InvalidEquityCandleJournalRecordError(
                "A finality statement cannot be confirmed after Northstar received it."
            )

    @property
    def kind(self) -> EquityCandleJournalEntryKind:
        """Return FINALITY."""
        return EquityCandleJournalEntryKind.FINALITY

    @property
    def source_identity(self) -> tuple[str, str, str]:
        """Return the source, source event and revision this statement confirms."""
        e = self.evidence
        return (e.source_id, e.source_event_id, e.source_revision)

    @property
    def is_synthetic(self) -> bool:
        """Return whether the statement is synthetic test evidence."""
        return self.evidence.is_synthetic


EquityCandleJournalRecord = EquityCandleObservationRecord | EquityCandleFinalityRecord


@dataclass(frozen=True, slots=True)
class EquityCandleJournalEntry:
    """One immutable journal entry: a record and the sequence it was appended at."""

    sequence: int
    journaled_at: PointInTime
    record: EquityCandleJournalRecord

    def __post_init__(self) -> None:
        _require_sequence(self.sequence, "Journal sequence")
        _require_instant(self.journaled_at, "Journal journaled_at")
        if not isinstance(self.record, EquityCandleObservationRecord | EquityCandleFinalityRecord):
            raise InvalidEquityCandleJournalRecordError("Journal record has an unknown type.")


class EquityCandleAppendOutcome(StrEnum):
    """What an append did."""

    APPENDED = "APPENDED"
    ALREADY_APPENDED = "ALREADY_APPENDED"


@dataclass(frozen=True, slots=True)
class EquityCandleAppendResult:
    """The entry an append wrote, or the original entry a retry repeats.

    ``conflicting_values`` lists earlier observation entries of the same
    source identity whose OHLCV values differ. Both are kept; none is chosen.
    """

    outcome: EquityCandleAppendOutcome
    entry: EquityCandleJournalEntry
    conflicting_values: tuple[int, ...] = ()


def equity_candle_in_window(
    record: EquityCandleJournalRecord,
    listing: ListingReference,
    timeframe: Timeframe,
    opens_after: PointInTime,
    closes_at: PointInTime,
) -> bool:
    """Return whether ``record`` is an observation of the listing within the window."""
    if not isinstance(record, EquityCandleObservationRecord):
        return False
    bar = record.observation.bar
    return (
        bar.symbol == listing.symbol
        and bar.exchange_code == listing.exchange_code
        and bar.timeframe == timeframe
        and opens_after.compare(bar.point_in_time) < 0
        and bar.point_in_time.compare(closes_at) <= 0
    )


@dataclass(frozen=True, slots=True)
class EquityCandleJournalWindow:
    """A research window's journal evidence at one watermark, in sequence order."""

    listing: ListingReference
    timeframe: Timeframe
    opens_after: PointInTime
    closes_at: PointInTime
    watermark: int
    observations: tuple[EquityCandleJournalEntry, ...]
    finalities: tuple[EquityCandleJournalEntry, ...]

    def __post_init__(self) -> None:
        _require_sequence(self.watermark, "Journal window watermark", minimum=0)
        for entries, kind in (
            (self.observations, EquityCandleJournalEntryKind.OBSERVATION),
            (self.finalities, EquityCandleJournalEntryKind.FINALITY),
        ):
            if not isinstance(entries, tuple) or not all(
                isinstance(entry, EquityCandleJournalEntry) and entry.record.kind is kind
                for entry in entries
            ):
                raise InvalidEquityCandleJournalRecordError(
                    f"Journal window {kind.value.lower()}s must be a tuple of {kind.value} entries."
                )
            sequences = [entry.sequence for entry in entries]
            if sequences != sorted(set(sequences)) or any(s > self.watermark for s in sequences):
                raise InvalidEquityCandleJournalRecordError(
                    "Journal window entries must be in ascending sequence order, without "
                    "repeats, at or below the watermark."
                )
        for entry in self.observations:
            if not equity_candle_in_window(
                entry.record, self.listing, self.timeframe, self.opens_after, self.closes_at
            ):
                raise InvalidEquityCandleJournalRecordError(
                    f"Journal entry {entry.sequence} is not an observation in this window."
                )
        identities = {entry.record.source_identity for entry in self.observations}
        for entry in self.finalities:
            if entry.record.source_identity not in identities:
                raise InvalidEquityCandleJournalRecordError(
                    f"Finality entry {entry.sequence} confirms no observation in this window."
                )


@dataclass(frozen=True, slots=True)
class EquityCandleComposedObservation:
    """A journaled observation with the finality visible for it at one instant.

    ``finality_sequences`` names every finality entry composed onto it; they
    all state the same evidence.
    """

    entry: EquityCandleJournalEntry
    observation: EquityCandleObservation
    finality_sequences: tuple[int, ...]


def compose_equity_candle_observations(
    window: EquityCandleJournalWindow, as_of: PointInTime
) -> tuple[EquityCandleComposedObservation, ...]:
    """Attach to each observation the finality statements visible at ``as_of``.

    A statement is visible when it is in ``window`` (so within its watermark)
    and was received at or before ``as_of``. Every observation in the window is
    returned, in sequence order; EI-2A then judges its receipt time. Visible
    statements that disagree are an integrity failure and raise.
    """
    if not isinstance(window, EquityCandleJournalWindow):
        raise TypeError("window must be an EquityCandleJournalWindow.")
    if not isinstance(as_of, PointInTime):
        raise TypeError("as_of must be a PointInTime.")
    composed: list[EquityCandleComposedObservation] = []
    for entry in window.observations:
        record = entry.record
        if not isinstance(record, EquityCandleObservationRecord):
            raise InvalidEquityCandleJournalRecordError("Window observations must be observations.")
        visible = [
            finality
            for finality in window.finalities
            if isinstance(finality.record, EquityCandleFinalityRecord)
            and finality.record.source_identity == record.source_identity
            and finality.record.received_at.compare(as_of) <= 0
        ]
        statements = {
            finality.record.evidence
            for finality in visible
            if isinstance(finality.record, EquityCandleFinalityRecord)
        }
        if len(statements) > 1:
            raise EquityCandleSourceIdentityConflictError(
                f"Journal entry {entry.sequence} has disagreeing visible finality statements."
            )
        composed.append(
            EquityCandleComposedObservation(
                entry,
                replace(record.observation, finality=statements.pop() if statements else None),
                tuple(finality.sequence for finality in visible),
            )
        )
    return tuple(composed)


# ---------------------------------------------------------------------------
# Append rules
# ---------------------------------------------------------------------------


def equity_candle_retry(
    record: EquityCandleJournalRecord, same_ingestion: EquityCandleJournalEntry | None
) -> EquityCandleJournalEntry | None:
    """Return the entry ``record`` retries, None when its ingestion id is new.

    Raises EquityCandleIngestionConflictError when the ingestion id is already
    journaled with any other content.
    """
    if same_ingestion is None:
        return None
    if equity_candle_record_fingerprint(same_ingestion.record) != (
        equity_candle_record_fingerprint(record)
    ):
        raise EquityCandleIngestionConflictError(
            f"Ingestion {record.ingestion_id!r} is already journaled as entry "
            f"{same_ingestion.sequence} with different content; nothing was written."
        )
    return same_ingestion


def _binding(record: EquityCandleObservationRecord) -> tuple[object, ...]:
    bar, observation = record.observation.bar, record.observation
    return (
        bar.symbol,
        bar.exchange_code,
        bar.timeframe,
        bar.point_in_time,
        observation.corrects_revision,
        record.source_published_at,
    )


def equity_candle_identity_conflicts(
    record: EquityCandleJournalRecord, same_identity: tuple[EquityCandleJournalEntry, ...]
) -> tuple[int, ...]:
    """Check ``record`` against its source identity's earlier entries.

    Returns the sequences of earlier observations whose OHLCV values differ.
    Raises EquityCandleSourceIdentityConflictError when an earlier entry binds
    the identity to other facts.
    """
    conflicting: list[int] = []
    for entry in same_identity:
        earlier = entry.record
        if earlier.source_identity != record.source_identity:
            raise ValueError(f"Journal entry {entry.sequence} is another source identity.")

        def refuse(what: str, sequence: int = entry.sequence) -> None:
            raise EquityCandleSourceIdentityConflictError(
                f"Source identity {'/'.join(record.source_identity)} is bound by entry "
                f"{sequence} to other {what}; nothing was written."
            )

        if earlier.is_synthetic != record.is_synthetic:
            refuse("synthetic classification")
        if isinstance(record, EquityCandleFinalityRecord):
            if isinstance(earlier, EquityCandleFinalityRecord) and earlier.evidence != (
                record.evidence
            ):
                refuse("finality evidence")
            continue
        if not isinstance(earlier, EquityCandleObservationRecord):
            continue
        if _binding(earlier) != _binding(record):
            refuse("listing, timeframe, interval, correction or publication facts")
        if earlier.observation.values != record.observation.values:
            conflicting.append(entry.sequence)
    return tuple(sorted(conflicting))


# ---------------------------------------------------------------------------
# Canonical encoding
# ---------------------------------------------------------------------------


def canonical_equity_candle_json(document: dict[str, Any]) -> str:
    """Return ``document`` as canonical JSON: sorted keys, no whitespace."""
    return json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(prefix: str, document: dict[str, Any]) -> str:
    encoded = canonical_equity_candle_json(document).encode("utf-8")
    return prefix + hashlib.sha256(encoded).hexdigest()


def _decimal(value: Decimal) -> str:
    return format(value, "f")


def _optional_instant(value: PointInTime | None) -> str | None:
    return None if value is None else value.value


def _payload(record: EquityCandleJournalRecord) -> dict[str, Any]:
    if isinstance(record, EquityCandleFinalityRecord):
        evidence = record.evidence
        return {
            "kind": EquityCandleJournalEntryKind.FINALITY.value,
            "source_id": evidence.source_id,
            "source_event_id": evidence.source_event_id,
            "source_revision": evidence.source_revision,
            "confirmed_at": evidence.confirmed_at.value,
            "evidence_reference": evidence.evidence_reference,
            "is_synthetic": evidence.is_synthetic,
        }
    observation = record.observation
    bar = observation.bar
    return {
        "kind": EquityCandleJournalEntryKind.OBSERVATION.value,
        "symbol": bar.symbol.value,
        "exchange_code": bar.exchange_code.value,
        "timeframe": bar.timeframe.value,
        "interval_end": bar.point_in_time.value,
        "currency": bar.open.currency.value,
        "open": _decimal(bar.open.amount),
        "high": _decimal(bar.high.amount),
        "low": _decimal(bar.low.amount),
        "close": _decimal(bar.close.amount),
        "volume": _decimal(bar.volume.value),
        "adjusted_close": (
            None if bar.adjusted_close is None else _decimal(bar.adjusted_close.amount)
        ),
        "source_id": observation.source_id,
        "source_event_id": observation.source_event_id,
        "source_revision": observation.source_revision,
        "corrects_revision": observation.corrects_revision,
        "source_published_at": _optional_instant(record.source_published_at),
        "is_synthetic": record.is_synthetic,
    }


def equity_candle_record_document(record: EquityCandleJournalRecord) -> dict[str, Any]:
    """Return the canonical document of ``record``."""
    return {
        "format": _FORMAT,
        "ingestion_id": record.ingestion_id,
        "received_at": record.received_at.value,
        "payload": _payload(record),
    }


def equity_candle_payload_fingerprint(record: EquityCandleJournalRecord) -> str:
    """Return ``ecp1-sha256:`` over what the source asserted."""
    return _digest(
        EQUITY_CANDLE_PAYLOAD_FINGERPRINT_PREFIX, {"format": _FORMAT, "payload": _payload(record)}
    )


def equity_candle_record_fingerprint(record: EquityCandleJournalRecord) -> str:
    """Return ``ecr1-sha256:`` over the whole record, receipt included."""
    return _digest(EQUITY_CANDLE_RECORD_FINGERPRINT_PREFIX, equity_candle_record_document(record))


def _text(value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError("Encoded equity candle values must be text.")
    return value


def _optional_text(value: Any) -> str | None:
    return None if value is None else _text(value)


def _flag(value: Any) -> bool:
    if not isinstance(value, bool):
        raise TypeError("Encoded equity candle flags must be booleans.")
    return value


def equity_candle_record_from_document(document: Any) -> EquityCandleJournalRecord:
    """Rebuild a record from its canonical document; the document must re-encode exactly.

    Raises ValueError, TypeError or KeyError for anything else.
    """
    if not isinstance(document, dict) or document.get("format") != _FORMAT:
        raise ValueError("Not an equity candle evidence document.")
    payload = document["payload"]
    if not isinstance(payload, dict):
        raise TypeError("Equity candle evidence payload must be an object.")
    ingestion_id = _text(document["ingestion_id"])
    received_at = PointInTime(_text(document["received_at"]))
    record: EquityCandleJournalRecord
    kind = payload["kind"]
    if kind == EquityCandleJournalEntryKind.FINALITY.value:
        record = EquityCandleFinalityRecord(
            ingestion_id,
            EquityCandleFinalityEvidence(
                source_id=_text(payload["source_id"]),
                source_event_id=_text(payload["source_event_id"]),
                source_revision=_text(payload["source_revision"]),
                confirmed_at=PointInTime(_text(payload["confirmed_at"])),
                evidence_reference=_text(payload["evidence_reference"]),
                is_synthetic=_flag(payload["is_synthetic"]),
            ),
            received_at,
        )
    elif kind == EquityCandleJournalEntryKind.OBSERVATION.value:
        currency = Currency(_text(payload["currency"]))

        def price(name: str) -> Price:
            return Price(Decimal(_text(payload[name])), currency)

        adjusted = payload["adjusted_close"]
        bar = HistoricalOHLCVBar(
            symbol=Symbol(_text(payload["symbol"])),
            exchange_code=ExchangeCode(_text(payload["exchange_code"])),
            point_in_time=PointInTime(_text(payload["interval_end"])),
            timeframe=Timeframe(_text(payload["timeframe"])),
            open=price("open"),
            high=price("high"),
            low=price("low"),
            close=price("close"),
            volume=Quantity(Decimal(_text(payload["volume"]))),
            adjusted_close=None if adjusted is None else price("adjusted_close"),
        )
        published = _optional_text(payload["source_published_at"])
        record = EquityCandleObservationRecord(
            ingestion_id,
            EquityCandleObservation(
                bar=bar,
                source_id=_text(payload["source_id"]),
                source_event_id=_text(payload["source_event_id"]),
                source_revision=_text(payload["source_revision"]),
                received_at=received_at,
                finality=None,
                corrects_revision=_optional_text(payload["corrects_revision"]),
            ),
            _flag(payload["is_synthetic"]),
            None if published is None else PointInTime(published),
        )
    else:
        raise ValueError(f"Unknown equity candle evidence kind {kind!r}.")
    if equity_candle_record_document(record) != document:
        raise ValueError("Equity candle evidence document is not in canonical form.")
    return record
