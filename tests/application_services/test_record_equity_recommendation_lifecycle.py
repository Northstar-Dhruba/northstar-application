"""EI-6B Application: durable lifecycle contracts, canonical events and verified recording.

The Application-owned parts of the EI-6B Durable Recommendation Lifecycle
Persistence Design v1.0 acceptance matrix (section 12), at layers C
(contract, with a labelled in-memory store), E (synthetic end to end) and A
(source scans). Test names carry the scenario identifiers. Real-SQLite parts
(S) belong to EI-6B Infrastructure PR-2 and are not claimed here.

Everything is synthetic and in memory:

- the world, snapshots, verified EI-4D records and the EI-5D issuance
  authority are the EI-5C and EI-5E test fixtures, reused unchanged
  (``lifecycle``, ``e2e``, ``ei5d``);
- ``_Lifecycles`` below is a LABELLED SYNTHETIC IN-MEMORY TEST STORE: not
  durable and not an adapter. It holds canonical documents and uses the
  production rules (``equity_recorded_lifecycle``,
  ``equity_lifecycle_append_decision``) and the production validator;
- every authorization comes from the real recording use case; the tests never
  construct one through the private issuer.

Nothing reads a clock, opens a connection, places an order or records a fill.
"""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import inspect
import json
import pathlib
import subprocess
import sys
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest
import test_equity_recommendation_issuance_end_to_end as e2e
import test_evaluate_synthetic_equity_recommendation_lifecycle as lifecycle
import test_issue_equity_trade_recommendation as ei5d
from northstar_core.foundation.value_objects import PointInTime
from northstar_core.recommendations.equity.lifecycle import (
    EquityEvidenceDeficiency,
    EquityEvidenceDeficiencyKind,
    EquityIntrabarOrdering,
    EquityLifecycleCancellationReason,
    EquityLifecycleEvent,
    EquityLifecycleEventKind,
    EquityLifecycleEvidence,
    EquityLifecycleLevel,
    EquityLifecycleState,
    EquityRecommendationLifecycle,
    EquityRevisionMateriality,
    InvalidEquityRecommendationLifecycleError,
)

from northstar_application import application_services
from northstar_application import ports as ports_package
from northstar_application.application_services import (
    EQUITY_RECOMMENDATION_LIFECYCLE_EVENT_FORMAT,
    EquityIssuedRecommendation,
    EquityLifecycleRecordingOutcome,
    EquityLifecycleRecordingRefusal,
    EquityLifecycleRecordingRefusedError,
    EquityLifecycleRecordingResult,
    EquityLifecycleRecordingUnresolvedError,
    IncompatibleEquityLifecycleEventDocumentError,
    InvalidEquityLifecycleEventDocumentError,
    InvalidEquityLifecycleWindowProvenanceError,
    RecordEquityRecommendationLifecycleUseCase,
    VerifiedEquityRecommendationLifecycleAppend,
    decode_equity_lifecycle_event_document,
    equity_issued_lifecycle_window,
    equity_lifecycle_append_decision,
    equity_lifecycle_event_document,
    equity_lifecycle_window,
    equity_recommendation_disposition,
    equity_recorded_lifecycle,
    require_verified_equity_recommendation_lifecycle_append,
)
from northstar_application.ports import (
    EquityRecommendationIssuanceReader,
    EquityRecommendationLifecycleCommitUnconfirmedError,
    EquityRecommendationLifecycleStore,
    EquityRecommendationLifecycleStoreUnavailableError,
    EquityRecommendationLifecycleWrite,
    EquityRecommendationLifecycleWriteOutcome,
    InvalidStoredEquityRecommendationIssuanceAuthorityError,
    InvalidStoredEquityRecommendationLifecycleError,
    UnverifiedEquityRecommendationLifecycleAppendError,
)

K = EquityLifecycleEventKind
S = EquityLifecycleState
L = EquityLifecycleLevel
_O = EquityLifecycleRecordingOutcome
_R = EquityLifecycleRecordingRefusal
_W = EquityRecommendationLifecycleWriteOutcome
_Integrity = InvalidStoredEquityRecommendationLifecycleError
_FORMAT = "northstar.equity-recommendation-lifecycle-event/1"


# ----------------------------------------------------------------------
# The labelled synthetic in-memory test store
# ----------------------------------------------------------------------


class _Lifecycles(EquityRecommendationLifecycleStore):
    """LABELLED SYNTHETIC IN-MEMORY TEST STORE -- not durable, not an adapter.

    It holds canonical documents per issuance and applies the production
    rehydration, validator and append rule. ``script`` injects one behaviour
    per append, consumed in order: ``unavailable`` (nothing written),
    ``lost_before`` (a lost COMMIT with no effect), ``lost_after`` (committed,
    then the acknowledgment lost) and ``lost_after_corrupt`` (committed, then
    the stored history corrupted, then the acknowledgment lost).
    ``unreadable`` makes every read fail. ``rows`` may be planted TEST-ONLY.
    ``before_append`` runs once, inside the first append after validation, to
    stand for a competing writer that commits first.
    """

    def __init__(self, issuances: EquityRecommendationIssuanceReader, generations: Any) -> None:
        self.issuances = issuances
        self.generations = generations
        self.rows: dict[str, list[str]] = {}
        self.script: list[str] = []
        self.unreadable = False
        self.appends = 0
        self.authorizations: list[Any] = []
        self.before_append: Any = None

    def load_lifecycle(self, issuance_id: str):  # type: ignore[no-untyped-def]
        if self.unreadable:
            raise EquityRecommendationLifecycleStoreUnavailableError("test: store unreadable")
        recorded = self.issuances.load_issuance(issuance_id)
        if recorded is None:
            return None
        disposition = self.generations.load_disposition(recorded.recommendation.generation_key)
        try:
            window = equity_issued_lifecycle_window(recorded.recommendation, disposition)
        except InvalidEquityLifecycleWindowProvenanceError as exc:
            raise _Integrity(_Integrity.WINDOW, f"test: {exc}") from exc
        return equity_recorded_lifecycle(recorded, window, self.rows.get(issuance_id, []))

    def append(self, authorization):  # type: ignore[no-untyped-def]
        verified = require_verified_equity_recommendation_lifecycle_append(authorization)
        self.appends += 1
        self.authorizations.append(verified)
        if self.before_append is not None:
            competing, self.before_append = self.before_append, None
            competing()
        mode = self.script.pop(0) if self.script else "ok"
        if mode == "unavailable":
            raise EquityRecommendationLifecycleStoreUnavailableError("test: store unavailable")
        if mode == "lost_before":
            raise EquityRecommendationLifecycleCommitUnconfirmedError("test: COMMIT lost")
        stored = self.load_lifecycle(verified.issuance_id)
        assert stored is not None
        outcome = equity_lifecycle_append_decision(stored, verified)
        if outcome is _W.APPENDED:
            self.rows.setdefault(verified.issuance_id, []).extend(verified.event_documents)
            stored = self.load_lifecycle(verified.issuance_id)
        if mode == "lost_after_corrupt":
            self.rows[verified.issuance_id][-1] = self.rows[verified.issuance_id][-1] + " "
        if mode in ("lost_after", "lost_after_corrupt"):
            raise EquityRecommendationLifecycleCommitUnconfirmedError("test: ack lost")
        assert stored is not None
        return EquityRecommendationLifecycleWrite(outcome, stored)


@dataclass
class _Recording:
    """The EI-5E chain with the issuance ISSUED, and an empty lifecycle store."""

    world: ei5d._World
    scenario: lifecycle._Scenario
    issued: EquityIssuedRecommendation
    store: _Lifecycles
    log: list[EquityLifecycleRecordingResult] = field(default_factory=list)

    @property
    def issuance_id(self) -> str:
        return self.issued.issuance_id

    def recorder(self, store: Any = None, max_writes: int = 3, issuances: Any = None) -> Any:
        return RecordEquityRecommendationLifecycleUseCase(
            issuances or self.world.authority,
            self.world.verifier(),
            self.scenario.world.snapshots,
            store or self.store,
            max_writes,
        )

    def record(self, *snapshot_ids: str, **options: Any) -> EquityLifecycleRecordingResult:
        result = self.recorder(**options).execute(self.issuance_id, tuple(snapshot_ids))
        self.log.append(result)
        return result

    def refusal(self, *snapshot_ids: str, **options: Any) -> EquityLifecycleRecordingRefusedError:
        with pytest.raises(EquityLifecycleRecordingRefusedError) as raised:
            self.record(*snapshot_ids, **options)
        return raised.value

    def snap(self, clock: str) -> str:
        return self.scenario.snap(clock)

    def documents(self) -> list[str]:
        return list(self.store.rows.get(self.issuance_id, []))


def _recording() -> _Recording:
    world, scenario = e2e._chain()
    issued = world.issue(scenario.key)
    assert issued.outcome is ei5d._O.ISSUED
    return _Recording(world, scenario, issued.issuance, _Lifecycles(world.authority, world.store))


def _kinds(result: EquityLifecycleRecordingResult) -> list[EquityLifecycleEventKind]:
    return [event.kind for event in result.new_events]


def _active(recording: _Recording) -> EquityLifecycleRecordingResult:
    """Activate on the bar 09:45-09:50, recorded durably; the cursor is then 09:50."""
    recording.scenario.fill("09:40", "09:50", {"09:45": lifecycle._ENTRY})
    result = recording.record(recording.snap("09:51"))
    assert (result.outcome, result.state) == (_O.APPENDED, S.ACTIVE)
    return result


def _captured(recording: _Recording, *snapshot_ids: str) -> Any:
    """A genuine authorization from the real use case, captured by an unavailable store."""

    class _Capture(_Lifecycles):
        def append(self, authorization):  # type: ignore[no-untyped-def]
            self.authorizations.append(authorization)
            raise EquityRecommendationLifecycleStoreUnavailableError("test: captured")

    capture = _Capture(recording.world.authority, recording.world.store)
    capture.rows = recording.store.rows  # the same stored history, read only
    refused = recording.refusal(*snapshot_ids, store=capture)
    assert (refused.reason, refused.retryable) == (_R.STORE_UNAVAILABLE, True)
    (authorization,) = capture.authorizations
    return authorization


def _tamper(obj: object, name: str, value: object) -> None:
    """TEST-ONLY privileged tampering with a frozen object (outside the trust boundary)."""
    object.__setattr__(obj, name, value)


# ----------------------------------------------------------------------
# LS-09: canonical lifecycle-event documents (format /1)
# ----------------------------------------------------------------------

_P = PointInTime


def _ev(reference: str, opens: str, closes: str, *observations: str) -> EquityLifecycleEvidence:
    return EquityLifecycleEvidence(reference, _P(opens), _P(closes), observations)


_BAR = ("2026-08-12T04:15:00Z", "2026-08-12T04:20:00Z")
_AT = _P("2026-08-12T04:21:00Z")
_SNAP = "ecs1-sha256:" + "a" * 64
_LATER = "ecs1-sha256:" + "b" * 64
_CANDLE = "1:synthetic-source:evt-13:r1"


def _event(kind: EquityLifecycleEventKind, **fields: Any) -> EquityLifecycleEvent:
    values: dict[str, Any] = {
        "sequence": 2,
        "kind": kind,
        "evaluated_at": _AT,
        "evidence": _ev(_SNAP, *_BAR, _CANDLE),
    }
    values.update(fields)
    return EquityLifecycleEvent(**values)


# One structurally valid Core event per kind (SYNTHETIC TEST DATA).
_EVENTS: dict[EquityLifecycleEventKind, EquityLifecycleEvent] = {
    K.ENTRY_WINDOW_OPENED: _event(
        K.ENTRY_WINDOW_OPENED,
        sequence=1,
        from_state=S.RECORDED,
        to_state=S.WAITING_FOR_ENTRY,
        evidence=_ev(_SNAP, *_BAR),
    ),
    K.ENTRY_ACTIVATED: _event(
        K.ENTRY_ACTIVATED, from_state=S.WAITING_FOR_ENTRY, to_state=S.ACTIVE, touched=(L.ENTRY,)
    ),
    K.ENTRY_INDETERMINATE: _event(
        K.ENTRY_INDETERMINATE,
        from_state=S.WAITING_FOR_ENTRY,
        to_state=S.INDETERMINATE_ENTRY,
        touched=(L.ENTRY, L.STOP, L.TARGET),
        ordering=EquityIntrabarOrdering.UNKNOWN,
    ),
    K.ENTRY_CANCELLED: _event(
        K.ENTRY_CANCELLED,
        from_state=S.WAITING_FOR_ENTRY,
        to_state=S.CANCELLED_BEFORE_ENTRY,
        touched=(L.STOP,),
        reason=EquityLifecycleCancellationReason.STOP_TOUCHED_BEFORE_ENTRY,
    ),
    K.ENTRY_WINDOW_EXPIRED: _event(
        K.ENTRY_WINDOW_EXPIRED,
        from_state=S.WAITING_FOR_ENTRY,
        to_state=S.EXPIRED_BEFORE_ENTRY,
        touched=(L.TARGET,),
    ),
    K.TARGET_REACHED: _event(
        K.TARGET_REACHED, from_state=S.ACTIVE, to_state=S.CLOSED_TARGET, touched=(L.TARGET,)
    ),
    K.STOP_REACHED: _event(
        K.STOP_REACHED, from_state=S.ACTIVE, to_state=S.CLOSED_STOP, touched=(L.STOP,)
    ),
    K.EXIT_INDETERMINATE: _event(
        K.EXIT_INDETERMINATE,
        from_state=S.ACTIVE,
        to_state=S.INDETERMINATE_EXIT,
        touched=(L.STOP, L.TARGET),
        ordering=EquityIntrabarOrdering.UNKNOWN,
    ),
    K.SESSION_CUTOFF_REACHED: _event(
        K.SESSION_CUTOFF_REACHED, from_state=S.ACTIVE, to_state=S.SESSION_CUTOFF_REACHED
    ),
    K.EVIDENCE_INCOMPLETE: _event(
        K.EVIDENCE_INCOMPLETE,
        from_state=S.ACTIVE,
        to_state=S.INCOMPLETE_EVIDENCE,
        evidence=_ev(_SNAP, *_BAR),
        deficiencies=(
            EquityEvidenceDeficiency(
                _P(_BAR[0]), _P(_BAR[1]), EquityEvidenceDeficiencyKind.MISSING
            ),
            EquityEvidenceDeficiency(
                _P("2026-08-12T04:20:00Z"),
                _P("2026-08-12T04:20:00.5Z"),
                EquityEvidenceDeficiencyKind.UNFINALIZED,
            ),
        ),
    ),
    K.SOURCE_REVISION_RECORDED: _event(
        K.SOURCE_REVISION_RECORDED,
        sequence=3,
        from_state=S.ACTIVE,
        to_state=S.ACTIVE,
        evidence=_ev(_LATER, *_BAR, _CANDLE, "2:synthetic-source:evt-13:r2"),
        cited_sequence=2,
        superseded_evidence=_ev(_SNAP, *_BAR, _CANDLE),
        materiality=EquityRevisionMateriality.UNASSESSED,
    ),
}

# Golden vectors: SHA-256 of each canonical document, frozen (a test digest, not a format
# fingerprint: the format has none).
_GOLDEN: dict[EquityLifecycleEventKind, str] = {
    K.ENTRY_WINDOW_OPENED: "8b7d335765c3e05fa621b9ecdd0d94fd011ab8ab49f5d80ea855494ec0300876",
    K.ENTRY_ACTIVATED: "557eda5ffebd4a1e3c4b25f2474f7486e2415a3f9af3a42c7ec2eff66e546faa",
    K.ENTRY_INDETERMINATE: "726fef2d4f9381982c02ab03ef14dbea56d0fe9f6fbbae66739eb42247c1fb92",
    K.ENTRY_CANCELLED: "82c9b19925302964461956294a2c2d6f38d74bdb2292d7aa1a5dcade5efc3ae9",
    K.ENTRY_WINDOW_EXPIRED: "78d8a8b1ce7fb457530de45abc626099388461cf04dd49f6d7ced921b52941f3",
    K.TARGET_REACHED: "1636f3bce29495506d28bc49caadd8192eda36c13bf497f4ad9740a0cf2e8975",
    K.STOP_REACHED: "91647f5119d4cbf2cc12737e201320ba436275d5167060012972597f12abaaa4",
    K.EXIT_INDETERMINATE: "6f67eb679d31e8880b235d1e28e8bf9b4444f75bb242ae03f00b11e67798baee",
    K.SESSION_CUTOFF_REACHED: "2e2f2e4fb94b4d14b0028f4722648d5d174facd3e90bf0f66760a8835853b69a",
    K.EVIDENCE_INCOMPLETE: "c6b5e050efd2fb5dece51cb196d2b12b40a4da99a9d23f4bb06639e74561d4d5",
    K.SOURCE_REVISION_RECORDED: "123efdc3f750d865619f5d285fc849b7ef71cf0b96c4a628643336d85305a194",
}

# One full golden document, frozen byte for byte.
_GOLDEN_CANCELLED = (
    '{"cited_sequence":null,"deficiencies":[],"evaluated_at":"2026-08-12T04:21:00Z",'
    '"evidence":{"closes_at":"2026-08-12T04:20:00Z","observations":'
    '["1:synthetic-source:evt-13:r1"],"opens_at":"2026-08-12T04:15:00Z","reference":'
    '"ecs1-sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"},'
    '"format":"northstar.equity-recommendation-lifecycle-event/1",'
    '"from_state":"WAITING_FOR_ENTRY","kind":"ENTRY_CANCELLED","materiality":null,'
    '"ordering":"NOT_REQUIRED","reason":"STOP_TOUCHED_BEFORE_ENTRY","sequence":2,'
    '"superseded_evidence":null,"to_state":"CANCELLED_BEFORE_ENTRY","touched":["STOP"]}'
)


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_ls09_every_core_event_kind_has_a_frozen_canonical_document() -> None:
    assert set(_EVENTS) == set(EquityLifecycleEventKind)  # all 11 kinds
    assert EQUITY_RECOMMENDATION_LIFECYCLE_EVENT_FORMAT == _FORMAT
    for kind, event in _EVENTS.items():
        document = equity_lifecycle_event_document(event)
        assert _digest(document) == _GOLDEN[kind], kind
        assert equity_lifecycle_event_document(event) == document  # deterministic
        assert decode_equity_lifecycle_event_document(document) == event  # strict round trip
        parsed = json.loads(document)
        assert parsed["format"] == _FORMAT and len(parsed) == 14
        assert set(parsed) - {"format"} == {f.name for f in dataclasses.fields(event)}
        assert "issuance_id" not in parsed and "fingerprint" not in document
    assert equity_lifecycle_event_document(_EVENTS[K.ENTRY_CANCELLED]) == _GOLDEN_CANCELLED


def test_ls09_the_golden_documents_reproduce_in_a_fresh_interpreter() -> None:
    script = (
        "import json, sys\n"
        "from northstar_application.application_services import "
        "decode_equity_lifecycle_event_document, equity_lifecycle_event_document\n"
        "text = sys.argv[1]\n"
        "event = decode_equity_lifecycle_event_document(text)\n"
        "print(equity_lifecycle_event_document(event) == text)\n"
    )
    document = equity_lifecycle_event_document(_EVENTS[K.EVIDENCE_INCOMPLETE])
    completed = subprocess.run(
        [sys.executable, "-c", script, document],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        env={**__import__("os").environ, "PYTHONHASHSEED": "1234"},
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "True"


def _mutated(document: str, change: Any) -> str:
    """Return ``document`` with ``change`` applied to its parsed object, canonically re-encoded."""
    parsed = json.loads(document)
    change(parsed)
    return json.dumps(parsed, sort_keys=True, separators=(",", ":"))


_REVISION = equity_lifecycle_event_document(_EVENTS[K.SOURCE_REVISION_RECORDED])
_INCOMPLETE = equity_lifecycle_event_document(_EVENTS[K.EVIDENCE_INCOMPLETE])


@pytest.mark.parametrize(
    ("document", "error"),
    [
        ("not json", InvalidEquityLifecycleEventDocumentError),
        ("[]", InvalidEquityLifecycleEventDocumentError),
        (
            _mutated(_REVISION, lambda d: d.update(format=_FORMAT[:-1] + "2")),
            IncompatibleEquityLifecycleEventDocumentError,
        ),
        (
            _mutated(_REVISION, lambda d: d.pop("format")),
            IncompatibleEquityLifecycleEventDocumentError,
        ),
        (
            _mutated(_REVISION, lambda d: d.pop("materiality")),
            InvalidEquityLifecycleEventDocumentError,
        ),
        (
            _mutated(_REVISION, lambda d: d.update(extra=1)),
            InvalidEquityLifecycleEventDocumentError,
        ),
        (
            _mutated(_REVISION, lambda d: d["evidence"].pop("observations")),
            InvalidEquityLifecycleEventDocumentError,
        ),
        (
            _mutated(_REVISION, lambda d: d["superseded_evidence"].update(note="x")),
            InvalidEquityLifecycleEventDocumentError,
        ),
        (
            _mutated(_INCOMPLETE, lambda d: d["deficiencies"][0].pop("kind")),
            InvalidEquityLifecycleEventDocumentError,
        ),
        (
            _mutated(_REVISION, lambda d: d.update(sequence="3")),
            InvalidEquityLifecycleEventDocumentError,
        ),
        (
            _mutated(_REVISION, lambda d: d.update(sequence=True)),
            InvalidEquityLifecycleEventDocumentError,
        ),
        (
            _mutated(_REVISION, lambda d: d.update(touched="")),
            InvalidEquityLifecycleEventDocumentError,
        ),
        (
            _mutated(_REVISION, lambda d: d.update(kind="REOPENED")),
            InvalidEquityLifecycleEventDocumentError,
        ),
        (
            _mutated(_INCOMPLETE, lambda d: d["deficiencies"][0].update(kind="LATE")),
            InvalidEquityLifecycleEventDocumentError,
        ),
        (
            _mutated(_REVISION, lambda d: d.update(evaluated_at="2026-08-12T09:51:00+05:30")),
            InvalidEquityLifecycleEventDocumentError,
        ),
        (
            _mutated(
                _INCOMPLETE,
                lambda d: d["deficiencies"][1].update(closes_at="2026-08-12T04:20:00.50Z"),
            ),
            InvalidEquityLifecycleEventDocumentError,
        ),
        (
            _mutated(_REVISION, lambda d: d.update(materiality=None)),
            InvalidEquityLifecycleEventDocumentError,
        ),
        (
            _mutated(_REVISION, lambda d: d.update(touched=["STOP"])),
            InvalidEquityLifecycleEventDocumentError,
        ),
        (_REVISION.replace(",", ", ", 1), InvalidEquityLifecycleEventDocumentError),
        (
            _REVISION.replace('{"cited_sequence":2,', '{"cited_sequence":2,"cited_sequence":2,', 1),
            InvalidEquityLifecycleEventDocumentError,
        ),
        (
            _REVISION.replace('"cited_sequence":2', '"cited_sequence":2.0', 1),
            InvalidEquityLifecycleEventDocumentError,
        ),
    ],
    ids=[
        "malformed JSON",
        "not an object",
        "unsupported version",
        "missing format",
        "missing field",
        "extra field",
        "nested evidence missing key",
        "nested evidence extra key",
        "deficiency missing key",
        "integer as text",
        "boolean as integer",
        "array as text",
        "unknown kind",
        "unknown deficiency kind",
        "offset instant",
        "trailing fractional zero",
        "kind-specific field nulled",
        "shape Core refuses",
        "whitespace",
        "duplicate key",
        "float",
    ],
)
def test_ls09_strict_decoding_refuses_every_non_canonical_or_invalid_document(
    document: str, error: type[Exception]
) -> None:
    with pytest.raises(error):
        decode_equity_lifecycle_event_document(document)


# ----------------------------------------------------------------------
# LS-17: the authoritative lifecycle window
# ----------------------------------------------------------------------


def test_ls17_a_b_the_window_derives_from_the_issued_record_and_equals_ei5c() -> None:
    recording = _recording()
    disposition = recording.world.store.dispositions[recording.scenario.key]
    window = equity_issued_lifecycle_window(recording.issued, disposition)
    assert (window.levels, window.eligible_from) == (
        recording.issued.levels,
        recording.issued.eligible_from,
    )
    assert (window.latest_entry_bar_opens_at, window.modeled_exit_at) == (
        recording.issued.latest_entry_bar_opens_at,
        recording.issued.modeled_exit_at,
    )
    assert window.entry_window_ends_at == lifecycle._pit("10:50")  # the next grid opening
    synthetic = recording.scenario.run(recording.snap("09:41"))  # EI-5C, unchanged
    assert synthetic.window == window
    assert recording.store.load_lifecycle(recording.issuance_id).lifecycle.window == window


def test_ls17_c_inconsistent_or_unissuable_provenance_fails_closed() -> None:
    recording = _recording()
    disposition = recording.world.store.dispositions[recording.scenario.key]
    other_audit = recording.world.generate(policy_set_version="set/2")
    other = equity_recommendation_disposition(other_audit)
    assert other is not None and other.generation_key != disposition.generation_key
    moved = dataclasses.replace(recording.issued, modeled_exit_at=lifecycle._pit("13:00"))
    assert moved.issuance_id == recording.issued.issuance_id  # identity parts unchanged
    narrowed = dataclasses.replace(recording.issued, not_assessed=recording.issued.not_assessed[:1])
    assert narrowed.issuance_id == recording.issued.issuance_id
    version1 = lifecycle._scenario().disposition  # the same candidate under evaluator /1
    assert version1.evaluator_version.endswith("/1")
    for issuance, source in (
        (recording.issued, other),  # another disposition
        (moved, disposition),  # an issued record that does not re-derive
        (narrowed, disposition),  # differs only outside the window: re-derivation catches it
        (recording.issued, version1),  # not evaluator /2
        (recording.issued, object()),  # not a disposition
    ):
        with pytest.raises(InvalidEquityLifecycleWindowProvenanceError):
            equity_issued_lifecycle_window(issuance, source)  # type: ignore[arg-type]
    off_grid = SimpleNamespace(
        entry_window=SimpleNamespace(
            latest_entry_bar_opens_at=lifecycle._pit("10:47"),
            modeled_exit_at=disposition.proposal.entry_window.modeled_exit_at,
            timing=disposition.proposal.entry_window.timing,
        ),
        definition=disposition.proposal.definition,
    )
    with pytest.raises(InvalidEquityLifecycleWindowProvenanceError, match="research-window grid"):
        equity_lifecycle_window(off_grid, disposition.candidate)  # type: ignore[arg-type]


def test_ls17_c_d_the_recorder_refuses_unprovable_windows_and_takes_no_window_input() -> None:
    recording = _recording()
    moved = dataclasses.replace(recording.issued, modeled_exit_at=lifecycle._pit("13:00"))
    planted = ei5d._Authority()
    planted.plant(moved)  # TEST-ONLY: an issued record that does not re-derive from its source
    refused = recording.refusal(recording.snap("09:41"), issuances=planted)
    assert (refused.reason, refused.retryable) == (_R.WINDOW_PROVENANCE_INVALID, False)
    parameters = inspect.signature(RecordEquityRecommendationLifecycleUseCase.execute).parameters
    assert list(parameters) == ["self", "issuance_id", "snapshot_ids"]
    window_parameters = inspect.signature(equity_issued_lifecycle_window).parameters
    assert list(window_parameters) == ["issuance", "disposition"]

    # A store whose own window derivation fails reports WINDOW, never a raw history.
    class _OtherSource:
        def load_disposition(self, key: str) -> Any:
            return equity_recommendation_disposition(
                recording.world.generate(policy_set_version="set/2")
            )

    broken = _Lifecycles(recording.world.authority, _OtherSource())
    with pytest.raises(_Integrity) as raised:
        broken.load_lifecycle(recording.issuance_id)
    assert raised.value.category == "WINDOW" and raised.value.retryable is False


# ----------------------------------------------------------------------
# LS-01 to LS-03: recording, loading, empty and unknown lifecycles
# ----------------------------------------------------------------------


def test_ls03_a_known_issuance_without_events_is_an_empty_recorded_lifecycle() -> None:
    recording = _recording()
    loaded = recording.store.load_lifecycle(recording.issuance_id)
    assert loaded is not None and loaded.lifecycle.events == ()
    assert loaded.lifecycle.state is S.RECORDED and loaded.position == 0
    assert loaded.issuance.recommendation == recording.issued


def test_ls03_b_an_unknown_issuance_is_none_and_the_recorder_refuses() -> None:
    recording = _recording()
    unknown = "eiris1-sha256:" + "0" * 64
    assert recording.store.load_lifecycle(unknown) is None
    with pytest.raises(EquityLifecycleRecordingRefusedError) as raised:
        recording.recorder().execute(unknown, (recording.snap("09:41"),))
    assert (raised.value.reason, raised.value.retryable) == (_R.ISSUANCE_NOT_FOUND, False)


def test_ls01_the_first_append_records_core_events_and_reloads_replayed() -> None:
    recording = _recording()
    result = _active(recording)
    assert _kinds(result) == [K.ENTRY_WINDOW_OPENED, K.ENTRY_ACTIVATED]
    assert (result.writes, result.reconciled) == (1, False)
    assert [e.sequence for e in result.recorded.lifecycle.events] == [1, 2]
    loaded = recording.store.load_lifecycle(recording.issuance_id)
    assert loaded == result.recorded and loaded.lifecycle.state is S.ACTIVE
    assert recording.documents() == [equity_lifecycle_event_document(e) for e in result.new_events]


def test_ls02_a_an_append_after_existing_history_continues_the_sequence() -> None:
    recording = _recording()
    _active(recording)
    recording.scenario.fill("09:50", "10:05", {"10:00": ("110", "111")})
    closed = recording.record(recording.snap("10:06"))
    assert (closed.outcome, closed.state, _kinds(closed)) == (
        _O.APPENDED,
        S.CLOSED_TARGET,
        [K.TARGET_REACHED],
    )
    assert [e.sequence for e in closed.recorded.lifecycle.events] == [1, 2, 3]


def test_ls02_b_repeated_and_quiet_evaluations_write_and_fabricate_nothing() -> None:
    recording = _recording()
    first = recording.snap("09:41")
    opened = recording.record(first)
    assert _kinds(opened) == [K.ENTRY_WINDOW_OPENED]
    before = recording.documents()
    again = recording.record(first)  # the same snapshot again, in a later call
    assert (again.outcome, again.new_events, again.writes) == (_O.NO_NEW_EVENTS, (), 0)
    recording.scenario.fill("09:40", "10:15", {"10:00": None})  # a gap: a temporary block
    blocked = recording.snap("10:16")
    for _ in range(2):  # deterministic, and safe to repeat
        quiet = recording.record(blocked)
        assert (quiet.outcome, quiet.state, quiet.writes) == (
            _O.NO_NEW_EVENTS,
            S.WAITING_FOR_ENTRY,
            0,
        )
    assert recording.documents() == before and recording.store.appends == 1
    within = recording.record(blocked, blocked)  # repeated within one call: idempotent
    assert within.outcome is _O.NO_NEW_EVENTS


# ----------------------------------------------------------------------
# LS-04, LS-05: parent provenance and the authorization boundary
# ----------------------------------------------------------------------


def test_ls04_b_c_corrupt_or_mismatched_parent_authority_fails_closed_and_distinct() -> None:
    recording = _recording()

    class _CorruptParent(ei5d._Authority):
        def load_issuance(self, issuance_id):  # type: ignore[no-untyped-def]
            raise InvalidStoredEquityRecommendationIssuanceAuthorityError(
                "ISSUANCE", "test: corrupt", issuance_id=issuance_id
            )

    corrupt = _Lifecycles(_CorruptParent(), recording.world.store)
    with pytest.raises(InvalidStoredEquityRecommendationIssuanceAuthorityError):
        corrupt.load_lifecycle(recording.issuance_id)  # the D6-5 error, unchanged
    refused = recording.refusal(recording.snap("09:41"), store=corrupt)
    assert (refused.reason, refused.retryable) == (_R.STORED_AUTHORITY_INVALID, False)
    # (c) A matching identity alone is not authority: a store whose parent differs is refused.
    planted = ei5d._Authority()
    planted.plant(recording.issued)  # the same identity, recorded at another sequence (1, not 2)
    mismatched = _Lifecycles(planted, recording.world.store)
    assert mismatched.load_lifecycle(recording.issuance_id).issuance != (
        recording.world.authority.load_issuance(recording.issuance_id)
    )
    refused = recording.refusal(recording.snap("09:41"), store=mismatched)
    assert (refused.reason, refused.retryable) == (_R.STORED_AUTHORITY_INVALID, False)


def test_ls05_unauthorized_or_changed_appends_are_refused_before_any_write() -> None:
    recording = _recording()
    snapshot = recording.snap("09:41")
    genuine = _captured(recording, snapshot)
    assert require_verified_equity_recommendation_lifecycle_append(genuine) is genuine
    copies = {
        "forged": lambda: VerifiedEquityRecommendationLifecycleAppend(
            genuine.issuance, genuine.window, genuine.prior, genuine.events
        ),
        "replaced": lambda: dataclasses.replace(genuine),
    }
    for name, make in copies.items():
        with pytest.raises(UnverifiedEquityRecommendationLifecycleAppendError):
            recording.store.append(make())
        assert recording.store.appends == 0, name
    for value in (None, object(), "authorization"):
        with pytest.raises(UnverifiedEquityRecommendationLifecycleAppendError):
            recording.store.append(value)
    later_event = _EVENTS[K.ENTRY_ACTIVATED]
    changes = {
        "events": (genuine.events[0], later_event),
        "prior": (genuine.events[0],),
        "covered": genuine.covered + " ",
        "event_documents": (),
    }
    for name, value in changes.items():
        changed = _captured(recording, snapshot)
        _tamper(changed, name, value)
        with pytest.raises(UnverifiedEquityRecommendationLifecycleAppendError):
            recording.store.append(changed)
        assert recording.documents() == [], name
    nested = _captured(recording, snapshot)
    _tamper(nested.events[0].evidence, "reference", "ecs1-sha256:" + "f" * 64)
    with pytest.raises(UnverifiedEquityRecommendationLifecycleAppendError):
        recording.store.append(nested)
    window_changed = _captured(recording, snapshot)
    _tamper(window_changed.window, "entry_window_ends_at", lifecycle._pit("10:55"))
    with pytest.raises(UnverifiedEquityRecommendationLifecycleAppendError):
        recording.store.append(window_changed)
    issuance_changed = _captured(recording, snapshot)
    other = dataclasses.replace(
        issuance_changed.issuance,
        recording_sequence=issuance_changed.issuance.recording_sequence + 1,
    )
    _tamper(issuance_changed, "issuance", other)
    with pytest.raises(UnverifiedEquityRecommendationLifecycleAppendError):
        recording.store.append(issuance_changed)
    assert recording.documents() == [] and recording.store.appends == 0
    assert recording.store.append(genuine).outcome is _W.APPENDED  # the genuine one still writes


def test_ls05_an_authorization_refuses_a_gap_an_invalid_transition_and_an_empty_batch() -> None:
    recording = _recording()
    genuine = _captured(recording, recording.snap("09:41"))
    opened = genuine.events[0]
    gap = dataclasses.replace(opened, sequence=2)
    skipping = dataclasses.replace(_EVENTS[K.ENTRY_ACTIVATED], sequence=1)  # RECORDED -> ACTIVE
    for prior, events in (((), (gap,)), ((), ()), ((opened,), (opened,)), ((), (skipping,))):
        with pytest.raises(UnverifiedEquityRecommendationLifecycleAppendError):
            VerifiedEquityRecommendationLifecycleAppend(
                genuine.issuance, genuine.window, prior, events
            )
    with pytest.raises(ValueError):  # a shape Core refuses cannot even be built
        dataclasses.replace(opened, from_state=S.WAITING_FOR_ENTRY)


# ----------------------------------------------------------------------
# LS-10: corrupt stored history (planted, TEST-ONLY)
# ----------------------------------------------------------------------


def test_ls10_rr05_a_corrupt_or_unreplayable_stored_history_is_refused_and_preserved() -> None:
    recording = _recording()
    _active(recording)
    good = recording.documents()
    cases = {
        "EVENT": [good[0], good[1].replace('"ACTIVE"', '"ACTIVE" ', 1)],
        "EVENT unsupported": [good[0], good[1].replace("lifecycle-event/1", "lifecycle-event/9")],
        "HISTORY order": [good[1], good[0]],
        "HISTORY gap": [good[1]],
    }
    for name, rows in cases.items():
        recording.store.rows[recording.issuance_id] = list(rows)
        with pytest.raises(_Integrity) as raised:
            recording.store.load_lifecycle(recording.issuance_id)
        assert raised.value.category == name.split()[0] and raised.value.retryable is False, name
        refused = recording.refusal(recording.snap("10:00"))
        assert refused.reason is _R.STORED_LIFECYCLE_INVALID and not refused.retryable
        assert recording.store.rows[recording.issuance_id] == rows  # preserved, never repaired


# ----------------------------------------------------------------------
# LS-12: commit uncertainty and bounded reconciliation
# ----------------------------------------------------------------------


def test_ls12_an_acknowledgment_lost_after_commit_reconciles_as_equivalent() -> None:
    recording = _recording()
    recording.store.script = ["lost_after"]
    result = recording.record(recording.snap("09:41"))
    assert (result.outcome, result.writes, result.reconciled) == (_O.EQUIVALENT, 1, True)
    assert len(recording.documents()) == 1  # never duplicated


def test_ls12_a_commit_lost_before_effect_is_retried_within_the_bound() -> None:
    recording = _recording()
    recording.store.script = ["lost_before"]
    result = recording.record(recording.snap("09:41"))
    assert (result.outcome, result.writes, result.reconciled) == (_O.APPENDED, 2, False)
    recording = _recording()
    recording.store.script = ["lost_before"] * 3
    with pytest.raises(EquityLifecycleRecordingUnresolvedError):
        recording.record(recording.snap("09:41"))
    assert recording.documents() == []


def test_ls12_an_unreadable_reload_or_corruption_on_reload_claims_nothing() -> None:
    recording = _recording()

    class _LostThenUnreadable(_Lifecycles):
        def append(self, authorization):  # type: ignore[no-untyped-def]
            require_verified_equity_recommendation_lifecycle_append(authorization)
            self.unreadable = True
            raise EquityRecommendationLifecycleCommitUnconfirmedError("test: COMMIT lost")

    lost = _LostThenUnreadable(recording.world.authority, recording.world.store)
    with pytest.raises(EquityLifecycleRecordingUnresolvedError):
        recording.record(recording.snap("09:41"), store=lost)
    recording.store.script = ["lost_after_corrupt"]
    with pytest.raises(_Integrity) as raised:  # permanent, raised unchanged
        recording.record(recording.snap("09:41"))
    assert raised.value.category == "EVENT"


def test_ls12_a_reload_showing_our_events_then_later_ones_is_confirmed_equivalent() -> None:
    recording = _recording()
    snapshot = recording.snap("09:41")

    class _LostThenOvertaken(_Lifecycles):
        def append(self, authorization):  # type: ignore[no-untyped-def]
            written = super().append(authorization)
            assert written.outcome is _W.APPENDED
            other = _Lifecycles(self.issuances, self.generations)
            other.rows = self.rows  # another writer on the same stored history
            recording.scenario.fill("09:40", "09:50", {"09:45": lifecycle._ENTRY})
            follow = recording.record(recording.snap("09:51"), store=other)
            assert follow.outcome is _O.APPENDED
            raise EquityRecommendationLifecycleCommitUnconfirmedError("test: ack lost")

    overtaken = _LostThenOvertaken(recording.world.authority, recording.world.store)
    recording.store = overtaken
    result = recording.record(snapshot)
    assert (result.outcome, result.reconciled) == (_O.EQUIVALENT, True)
    assert [e.kind for e in result.recorded.lifecycle.events] == [
        K.ENTRY_WINDOW_OPENED,
        K.ENTRY_ACTIVATED,
    ]


def test_ls12_store_outages_are_retryable_refusals_with_nothing_written() -> None:
    recording = _recording()
    recording.store.script = ["unavailable"]
    refused = recording.refusal(recording.snap("09:41"))
    assert (refused.reason, refused.retryable) == (_R.STORE_UNAVAILABLE, True)
    recording.store.unreadable = True
    refused = recording.refusal(recording.snap("09:41"))
    assert (refused.reason, refused.retryable) == (_R.STORE_UNAVAILABLE, True)
    recording.store.unreadable = False
    recording.world.authority.unreadable = True
    refused = recording.refusal(recording.snap("09:41"))
    assert (refused.reason, refused.retryable) == (_R.AUTHORITY_UNAVAILABLE, True)
    assert recording.documents() == []


# ----------------------------------------------------------------------
# LS-13, LS-14, RR-06: identical and conflicting appends
# ----------------------------------------------------------------------


def test_ls13_an_identical_retry_and_an_identical_concurrent_request_are_equivalent() -> None:
    recording = _recording()
    snapshot = recording.snap("09:41")
    genuine = _captured(recording, snapshot)
    assert recording.store.append(genuine).outcome is _W.APPENDED
    again = recording.store.append(genuine)  # the same authorization retried
    assert again.outcome is _W.EQUIVALENT and len(recording.documents()) == 1
    # Concurrent: the competitor commits the identical batch first, inside our append.
    racing = _recording()
    first = racing.snap("09:41")
    racing.store.before_append = lambda: _commit_competitor(racing, first)
    result = racing.record(first)
    assert (result.outcome, result.writes) == (_O.EQUIVALENT, 1)
    assert len(racing.documents()) == 1  # no duplicates
    assert racing.log[0].outcome is _O.APPENDED  # the competitor's, committed first


def _commit_competitor(recording: _Recording, *snapshot_ids: str) -> None:
    """Another writer records ``snapshot_ids`` into the same store before we commit."""
    recording.record(*snapshot_ids)


def test_ls14_rr06_a_conflicting_concurrent_or_stale_append_is_refused_and_not_adopted() -> None:
    recording = _recording()
    ours = recording.snap("09:41")
    recording.scenario.fill("09:40", "09:50", {"09:45": lifecycle._ENTRY})
    theirs = recording.snap("09:51")
    recording.store.before_append = lambda: _commit_competitor(recording, theirs)
    result = recording.record(ours)
    assert result.outcome is _O.CONFLICT and result.writes == 1
    competitor = recording.log[0]
    assert competitor.outcome is _O.APPENDED
    assert result.recorded.lifecycle.events == competitor.recorded.lifecycle.events
    assert recording.documents() == [
        equity_lifecycle_event_document(e) for e in competitor.recorded.lifecycle.events
    ]
    assert all(e.evidence.reference == theirs for e in result.recorded.lifecycle.events)
    # A stale authorization (history advanced since it was issued) is a CONFLICT too.
    stale = _recording()
    old = _captured(stale, stale.snap("09:41"))
    stale.scenario.fill("09:40", "09:50", {"09:45": lifecycle._ENTRY})
    stale.record(stale.snap("09:51"))
    written = stale.store.append(old)
    assert written.outcome is _W.CONFLICT and len(stale.documents()) == 2


def test_ls14_a_prior_history_of_the_same_length_but_other_content_is_a_conflict() -> None:
    recording = _recording()
    assert recording.record(recording.snap("09:41")).outcome is _O.APPENDED  # stored: [A]
    elsewhere = _recording()  # the same issuance identity and window, another history: [B]
    assert elsewhere.issuance_id == recording.issuance_id
    assert elsewhere.record(elsewhere.snap("09:43")).outcome is _O.APPENDED
    elsewhere.scenario.fill("09:40", "09:50", {"09:45": lifecycle._ENTRY})
    other_prior = _captured(elsewhere, elsewhere.snap("09:51"))  # prior [B], events [C]
    assert len(other_prior.prior) == len(recording.documents()) == 1
    assert other_prior.prior_documents != tuple(recording.documents())
    stored = recording.store.load_lifecycle(recording.issuance_id)
    assert equity_lifecycle_append_decision(stored, other_prior) is _W.CONFLICT
    before = recording.documents()
    assert recording.store.append(other_prior).outcome is _W.CONFLICT
    assert recording.documents() == before  # nothing written, nothing adopted


def test_the_append_rule_refuses_another_parent_or_window_as_an_invalid_write() -> None:
    recording = _recording()
    genuine = _captured(recording, recording.snap("09:41"))
    stored = recording.store.load_lifecycle(recording.issuance_id)
    assert equity_lifecycle_append_decision(stored, genuine) is _W.APPENDED
    other_parent = dataclasses.replace(
        stored, issuance=dataclasses.replace(stored.issuance, recording_sequence=9)
    )
    with pytest.raises(ValueError, match="another parent"):
        equity_lifecycle_append_decision(other_parent, genuine)


# ----------------------------------------------------------------------
# LS-15, RV-04: terminal authority and O-3 option (b)
# ----------------------------------------------------------------------


def test_ls15_a_a_terminal_lifecycle_records_no_state_change() -> None:
    recording = _recording()
    recording.scenario.fill("09:40", "09:55", {"09:50": ("97", "98")})
    cancelled = recording.record(recording.snap("09:56"))
    assert cancelled.state is S.CANCELLED_BEFORE_ENTRY
    recording.scenario.fill("09:55", "10:30", {"10:00": lifecycle._ENTRY})
    later = recording.record(recording.snap("10:31"))
    assert (later.outcome, later.state, later.writes) == (
        _O.NO_NEW_EVENTS,
        S.CANCELLED_BEFORE_ENTRY,
        0,
    )
    events = cancelled.recorded.lifecycle.events
    reopening = dataclasses.replace(
        _EVENTS[K.ENTRY_ACTIVATED], sequence=len(events) + 1, evaluated_at=events[-1].evaluated_at
    )
    with pytest.raises(UnverifiedEquityRecommendationLifecycleAppendError):
        VerifiedEquityRecommendationLifecycleAppend(
            cancelled.recorded.issuance, cancelled.recorded.lifecycle.window, events, (reopening,)
        )


def test_ls15_b_a_core_permitted_revision_after_terminality_is_recorded() -> None:
    recording = _recording()
    _active(recording)
    recording.scenario.fill("09:50", "10:05", {"10:00": ("110", "111")})
    closed = recording.record(recording.snap("10:06"))
    assert closed.state is S.CLOSED_TARGET
    recording.scenario.put("10:00", ("105", "108"), revision="r2")
    revised = recording.record(recording.snap("16:00"))
    assert (revised.outcome, _kinds(revised), revised.state) == (
        _O.APPENDED,
        [K.SOURCE_REVISION_RECORDED],
        S.CLOSED_TARGET,
    )
    assert revised.new_events[0].materiality is EquityRevisionMateriality.UNASSESSED


def test_ls15_c_rv04_late_gap_filling_evidence_after_incomplete_evidence_records_nothing() -> None:
    recording = _recording()
    _active(recording)
    recording.scenario.fill("09:50", "14:00", {"12:00": None})
    incomplete = recording.record(recording.snap("14:01"))
    assert (incomplete.state, _kinds(incomplete)) == (
        S.INCOMPLETE_EVIDENCE,
        [K.EVIDENCE_INCOMPLETE],
    )
    before = recording.documents()
    recording.scenario.put("12:00", lifecycle._QUIET)  # the late bar that fills the gap
    late = recording.record(recording.snap("16:00"))
    assert (late.outcome, late.state, late.new_events, late.writes) == (
        _O.NO_NEW_EVENTS,
        S.INCOMPLETE_EVIDENCE,
        (),
        0,
    )
    assert recording.documents() == before  # terminal state and history unchanged (O-3 (b))
    terminal = incomplete.recorded.lifecycle
    cited = terminal.events[-1]
    assert cited.kind is K.EVIDENCE_INCOMPLETE
    with pytest.raises(InvalidEquityRecommendationLifecycleError):  # no revision cites it
        terminal.record_source_revision(
            cited.sequence,
            EquityLifecycleEvidence(_LATER, cited.evidence.opens_at, cited.evidence.closes_at),
            lifecycle._pit("16:00"),
        )


# ----------------------------------------------------------------------
# Refusals of the recording use case
# ----------------------------------------------------------------------


def test_snapshot_rules_are_ei5cs_and_only_synthetic_evidence_is_recorded() -> None:
    recording = _recording()
    real = recording.scenario.world.issue(lifecycle._ist("09:51"))
    emptied = tuple(
        lifecycle._INTERVAL(
            i.opens_at,
            i.closes_at,
            i.outcome if i.outcome is lifecycle._EI2A.NOT_ENDED else lifecycle._EI2A.MISSING,
            (),
            (),
        )
        for i in real.intervals
    )
    accepted = lifecycle.snapshot_module.EquityReferenceProvenanceKind.ACCEPTED_PRODUCTION
    production = lifecycle._rebuilt(
        real,
        emptied,
        observations=(),
        finalities=(),
        unmatched_sequences=(),
        venue_provenance=accepted,
        applicability_provenance=accepted,
        finality_policy=lifecycle.NO_APPROVED_EQUITY_CANDLE_FINALITY,
    )
    refused = recording.refusal(lifecycle._plant(recording.scenario, production))
    assert (refused.reason, refused.retryable) == (_R.SNAPSHOT_NOT_SYNTHETIC, False)
    assert recording.refusal("ecs1-sha256:" + "0" * 64).reason is _R.SNAPSHOT_NOT_FOUND
    _active(recording)
    older = recording.snap("09:41")
    assert recording.refusal(older).reason is _R.SNAPSHOT_OUT_OF_ORDER
    assert recording.store.appends == 1


def test_a_missing_disposition_and_typed_inputs_are_refused() -> None:
    recording = _recording()
    snapshot = recording.snap("09:41")
    del recording.world.store.dispositions[recording.scenario.key]
    refused = recording.refusal(snapshot)
    assert (refused.reason, refused.retryable) == (_R.GENERATION_NOT_FOUND, False)
    with pytest.raises(TypeError):
        recording.recorder().execute(recording.issuance_id, [snapshot])  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        recording.recorder().execute(recording.issuance_id, ())
    with pytest.raises(ValueError):
        recording.recorder(max_writes=0)


# ----------------------------------------------------------------------
# E: a synthetic end-to-end recorded lifecycle
# ----------------------------------------------------------------------


def test_e2e_an_issued_recommendation_lifecycle_is_recorded_retried_replayed_and_closed() -> None:
    recording = _recording()
    issuances_before = list(recording.world.authority.issuances)
    dispositions_before = dict(recording.world.store.dispositions)
    # An authoritative MODELED issuance starts with an empty RECORDED lifecycle.
    assert recording.store.load_lifecycle(recording.issuance_id).lifecycle.state is S.RECORDED
    opened = recording.record(recording.snap("09:41"))
    assert (opened.outcome, opened.state) == (_O.APPENDED, S.WAITING_FOR_ENTRY)
    # An identical concurrent request: EQUIVALENT, no duplicate.
    recording.scenario.fill("09:40", "09:50", {"09:45": lifecycle._ENTRY})
    entry = recording.snap("09:51")
    recording.store.before_append = lambda: _commit_competitor(recording, entry)
    activated = recording.record(entry)
    assert (activated.outcome, activated.state) == (_O.EQUIVALENT, S.ACTIVE)
    # Rehydrated replay equals the recorded history, through Core.
    loaded = recording.store.load_lifecycle(recording.issuance_id)
    replayed = EquityRecommendationLifecycle(loaded.lifecycle.window, loaded.lifecycle.events)
    assert replayed == loaded.lifecycle and replayed.state is S.ACTIVE
    # A conflicting stale append changes nothing: it closes on another snapshot than the one
    # committed first.
    recording.scenario.fill("09:50", "10:05", {"10:00": ("110", "111")})
    stale = _captured(recording, recording.snap("10:06"))
    closed = recording.record(recording.snap("10:07"))
    assert (closed.outcome, closed.state) == (_O.APPENDED, S.CLOSED_TARGET)
    history = recording.documents()
    assert recording.store.append(stale).outcome is _W.CONFLICT
    assert recording.documents() == history
    # Terminal: later evidence records nothing.
    assert recording.record(recording.snap("11:00")).outcome is _O.NO_NEW_EVENTS
    # No order, fill, position, issuance or generation record was created anywhere.
    assert recording.world.authority.issuances == issuances_before
    assert recording.world.store.dispositions == dispositions_before
    forbidden = {"price", "quantity", "order", "fill", "position", "pnl", "exit_price"}
    for event in recording.store.load_lifecycle(recording.issuance_id).lifecycle.events:
        assert not forbidden & {f.name for f in dataclasses.fields(event)}


# ----------------------------------------------------------------------
# A: ownership and the write boundary
# ----------------------------------------------------------------------


def _root() -> pathlib.Path:
    source = inspect.getsourcefile(application_services)
    assert source is not None
    return pathlib.Path(source).parents[1]


_LIFECYCLE_MODULES = (
    "ports/equity_recommendation_lifecycle_store.py",
    "application_services/equity_recommendation_lifecycle_events.py",
    "application_services/equity_recommendation_lifecycle_window.py",
    "application_services/equity_recommendation_lifecycle_authorization.py",
    "application_services/record_equity_recommendation_lifecycle.py",
)


def _trees() -> dict[str, ast.Module]:
    root = _root()
    return {
        path.relative_to(root).as_posix(): ast.parse(path.read_text(encoding="utf-8"))
        for path in sorted(root.rglob("*.py"))
    }


def _names(tree: ast.AST) -> set[str]:
    found = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    found |= {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names}
    return found


def test_ar_only_the_recording_use_case_depends_on_the_store_or_issues_authorizations() -> None:
    allowed_store = {
        "ports/equity_recommendation_lifecycle_store.py",
        "ports/__init__.py",
        "application_services/record_equity_recommendation_lifecycle.py",
    }
    allowed_issuer = {
        "application_services/equity_recommendation_lifecycle_authorization.py",
        "application_services/record_equity_recommendation_lifecycle.py",
    }
    violations = []
    for path, tree in _trees().items():
        names = _names(tree)
        if "EquityRecommendationLifecycleStore" in names and path not in allowed_store:
            violations.append(f"{path}: depends on the lifecycle store")
        if "_issue_lifecycle_append" in names and path not in allowed_issuer:
            violations.append(f"{path}: uses the private lifecycle-append issuer")
    assert violations == []
    assert "_issue_lifecycle_append" not in application_services.__all__
    assert EquityRecommendationLifecycleStore.__abstractmethods__ == frozenset(
        {"load_lifecycle", "append"}
    )
    for name in (
        "EquityRecommendationLifecycleStore",
        "InvalidStoredEquityRecommendationLifecycleError",
    ):
        assert name in ports_package.__all__


def test_ar_lifecycle_modules_import_no_infrastructure_execution_or_clock() -> None:
    trees = _trees()
    forbidden = (
        "infrastructure",
        "sqlite",
        "sqlalchemy",
        "broker",
        "execution",
        "paper",
        "futures",
        "order",
        "provider",
        "monitor",
    )
    violations = []
    for path in _LIFECYCLE_MODULES:
        for node in ast.walk(trees[path]):
            if isinstance(node, ast.ImportFrom | ast.Import):
                modules = (
                    [node.module or ""]
                    if isinstance(node, ast.ImportFrom)
                    else [alias.name for alias in node.names]
                )
                violations += [f"{path}: {m}" for m in modules if any(w in m for w in forbidden)]
            if isinstance(node, ast.Attribute) and node.attr in {"now", "utcnow", "today"}:
                violations.append(f"{path}:{node.lineno}: reads a clock")
            if isinstance(node, ast.Name) and node.id in {"random", "uuid4"}:
                violations.append(f"{path}:{node.lineno}: uses randomness")
    assert violations == []
