"""EI-7C ports: the three read-only reader contracts and the query refusal type.

Port and contract tests for the EI-7 Decision Register, sections 13.3, 13.8
and 13.12 (EI7-P4, EI7-FC4, EI7-FC6). They prove the exact reader-only
surfaces, the immutable recorded-session result, the discovery-limit error and
the stable refusal vocabulary. No adapter, database or clock is involved.
"""

from __future__ import annotations

import dataclasses
import inspect
import typing
from datetime import date

import pytest
from northstar_core.domain.value_objects import ListingReference
from northstar_core.foundation.value_objects import ExchangeCode, PointInTime, Symbol

from northstar_application import ports
from northstar_application.application_services import (
    EQUITY_RECOMMENDATION_ISSUANCE_AUTHORITY_REFERENCE,
    EquityIssuanceCompletenessDeclaration,
    EquityReadQueryRefusal,
    EquityReadQueryRefusedError,
    EquityRecordedCompletenessDeclaration,
)
from northstar_application.ports import (
    EQUITY_RECORDED_SESSION_DISCOVERY_LIMIT,
    EQUITY_RECORDING_POSITION_MAXIMUM,
    EquityCandleEvidenceSnapshotReader,
    EquityCandleEvidenceSnapshotStore,
    EquityCandleEvidenceUnavailableError,
    EquityIntradayResearchTimelineReader,
    EquityIntradayResearchTimelineStore,
    EquityRecordedSessionDiscoveryLimitExceededError,
    EquityRecordedSessionIssuanceReader,
    EquityRecordedSessionIssuances,
    InvalidEquityRecordedSessionIssuancesError,
    InvalidStoredEquityCandleEvidenceSnapshotError,
)

_NSE = ExchangeCode("NSE")
_DAY = date(2026, 8, 12)
_REF = EQUITY_RECOMMENDATION_ISSUANCE_AUTHORITY_REFERENCE


class _SnapshotsOnly:
    def get_snapshot(self, snapshot_id: str) -> None:
        return None


class _TimelinesOnly:
    def load(self, strategy, listing, trading_date) -> None:  # type: ignore[no-untyped-def]
        return None


class _WritingSnapshots(EquityCandleEvidenceSnapshotStore):
    def issue(self, snapshot):  # type: ignore[no-untyped-def]
        return 1

    def get_snapshot(self, snapshot_id):  # type: ignore[no-untyped-def]
        return None


class _WritingTimelines(EquityIntradayResearchTimelineStore):
    def load(self, strategy, listing, trading_date):  # type: ignore[no-untyped-def]
        return None

    def append(self, timeline, expected_head_fingerprint):  # type: ignore[no-untyped-def]
        return None


def _declaration(sequence: int, through: str = "2026-08-12T04:06:00Z", day: date = _DAY):
    return EquityRecordedCompletenessDeclaration(
        EquityIssuanceCompletenessDeclaration(_NSE, day, PointInTime(through)), sequence
    )


# ----------------------------------------------------------------------
# Reader-only surfaces (EI7-P4, EI7-FC6)
# ----------------------------------------------------------------------


def test_p01_the_reader_protocols_name_exactly_one_read_operation_each() -> None:
    assert typing.get_protocol_members(EquityCandleEvidenceSnapshotReader) == frozenset(
        {"get_snapshot"}
    )
    assert typing.get_protocol_members(EquityIntradayResearchTimelineReader) == frozenset({"load"})
    for protocol in (EquityCandleEvidenceSnapshotReader, EquityIntradayResearchTimelineReader):
        assert not {"issue", "append", "publish", "write"} & set(dir(protocol))
    assert EquityRecordedSessionIssuanceReader.__abstractmethods__ == frozenset(
        {"recorded_session_issuances"}
    )


def test_p02_existing_writer_capable_stores_and_read_only_fakes_both_satisfy_the_readers() -> None:
    assert isinstance(_WritingSnapshots(), EquityCandleEvidenceSnapshotReader)
    assert isinstance(_SnapshotsOnly(), EquityCandleEvidenceSnapshotReader)
    assert isinstance(_WritingTimelines(), EquityIntradayResearchTimelineReader)
    assert isinstance(_TimelinesOnly(), EquityIntradayResearchTimelineReader)
    # A read-only fake is not a store: it cannot be handed to a write path.
    assert not isinstance(_SnapshotsOnly(), EquityCandleEvidenceSnapshotStore)
    assert not isinstance(_TimelinesOnly(), EquityIntradayResearchTimelineStore)


def test_p03_incompatible_objects_are_rejected() -> None:
    for candidate in (object(), _TimelinesOnly(), "get_snapshot"):
        assert not isinstance(candidate, EquityCandleEvidenceSnapshotReader)
    for candidate in (object(), _SnapshotsOnly()):
        assert not isinstance(candidate, EquityIntradayResearchTimelineReader)

    class _Partial(EquityRecordedSessionIssuanceReader):
        pass

    with pytest.raises(TypeError):
        _Partial()  # type: ignore[abstract]


def test_p04_a_corrupt_snapshot_error_stays_in_the_ei2b_family_and_is_permanent() -> None:
    error = InvalidStoredEquityCandleEvidenceSnapshotError("test: does not verify")
    assert isinstance(error, EquityCandleEvidenceUnavailableError)  # existing handlers unchanged
    assert error.retryable is False


def test_p05_the_recorded_session_port_signature_is_bounded_and_keyword_only() -> None:
    signature = inspect.signature(EquityRecordedSessionIssuanceReader.recorded_session_issuances)
    parameters = signature.parameters
    assert list(parameters) == ["self", "exchange_code", "trading_date", "limit", "watermark"]
    assert parameters["limit"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["watermark"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["watermark"].default is None
    assert EQUITY_RECORDED_SESSION_DISCOVERY_LIMIT == 50
    assert EQUITY_RECORDING_POSITION_MAXIMUM == 2**63 - 1


# ----------------------------------------------------------------------
# The immutable recorded-session result
# ----------------------------------------------------------------------


def test_p06_an_undeclared_empty_session_is_a_result_never_none() -> None:
    result = EquityRecordedSessionIssuances(_NSE, _DAY, 7, 3, (), (), _REF)
    assert (result.recording_position, result.recording_watermark) == (7, 3)
    assert (result.complete_through, result.is_declared, result.issuances) == (None, False, ())
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.recording_watermark = 7  # type: ignore[misc]


def test_p07_declarations_through_w_state_the_greatest_complete_through() -> None:
    first, second = _declaration(1, "2026-08-12T04:00:00Z"), _declaration(3, "2026-08-12T04:30:00Z")
    result = EquityRecordedSessionIssuances(_NSE, _DAY, 3, 3, (first, second), (), _REF)
    assert result.is_declared and result.complete_through == PointInTime("2026-08-12T04:30:00Z")


@pytest.mark.parametrize(
    "changes",
    [
        {"recording_watermark": 8},  # W beyond N
        {"recording_position": -1},
        {"recording_watermark": True},
        {"recording_position": EQUITY_RECORDING_POSITION_MAXIMUM + 1},
        {"reference": "another-authority/1"},
        {"trading_date": "2026-08-12"},
        {"exchange_code": "NSE"},
        {"declarations": [_declaration(1)]},
        {"declarations": (_declaration(1, day=date(2026, 8, 13)),)},
        {"declarations": (_declaration(2), _declaration(1, "2026-08-12T03:00:00Z"))},
        {"declarations": (_declaration(9),)},  # recorded after W
    ],
)
def test_p08_a_result_that_violates_its_contract_cannot_be_constructed(changes: dict) -> None:
    fields = {
        "exchange_code": _NSE,
        "trading_date": _DAY,
        "recording_position": 7,
        "recording_watermark": 7,
        "declarations": (),
        "issuances": (),
        "reference": _REF,
        **changes,
    }
    with pytest.raises(InvalidEquityRecordedSessionIssuancesError):
        EquityRecordedSessionIssuances(**fields)


def test_p09_the_discovery_limit_error_is_permanent_and_carries_no_total() -> None:
    error = EquityRecordedSessionDiscoveryLimitExceededError(50)
    assert (error.limit, error.retryable) == (50, False)
    assert not hasattr(error, "count")
    for bad in (0, -1, True, "50"):
        with pytest.raises(ValueError):
            EquityRecordedSessionDiscoveryLimitExceededError(bad)  # type: ignore[arg-type]


# ----------------------------------------------------------------------
# The refusal vocabulary (EI7-CF2, EI7-FC4)
# ----------------------------------------------------------------------


def test_p10_the_refusal_reasons_are_exactly_the_approved_codes() -> None:
    assert {r.value for r in EquityReadQueryRefusal} == {
        "ISSUANCE_NOT_FOUND",
        "WATERMARK_AHEAD",
        "DISCOVERY_LIMIT_EXCEEDED",
        "EVIDENCE_CLASS_UNSUPPORTED",
        "UNSUPPORTED_SCOPE",
        "PERSISTED_AUTHORITY_INVALID",
        "LIFECYCLE_HISTORY_INVALID",
        "DEPENDENCY_UNAVAILABLE",
    }
    # EQUITY_NOT_CONFIGURED is runtime configuration (EI7-CF5), never a query refusal.
    assert "EQUITY_NOT_CONFIGURED" not in {r.value for r in EquityReadQueryRefusal}


def test_p11_only_a_dependency_outage_is_retryable_and_the_reason_is_typed() -> None:
    for reason in EquityReadQueryRefusal:
        refusal = EquityReadQueryRefusedError(reason, "test")
        assert refusal.retryable is (reason is EquityReadQueryRefusal.DEPENDENCY_UNAVAILABLE)
        assert str(refusal).startswith(reason.value)
    with pytest.raises(TypeError):
        EquityReadQueryRefusedError("PERSISTED_AUTHORITY_INVALID", "test")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        EquityReadQueryRefusedError(EquityReadQueryRefusal.UNSUPPORTED_SCOPE, "")


def test_p12_the_new_ports_are_exported() -> None:
    for name in (
        "EquityCandleEvidenceSnapshotReader",
        "EquityIntradayResearchTimelineReader",
        "EquityRecordedSessionIssuanceReader",
        "EquityRecordedSessionIssuances",
        "EquityRecordedSessionDiscoveryLimitExceededError",
        "InvalidStoredEquityCandleEvidenceSnapshotError",
        "EQUITY_RECORDED_SESSION_DISCOVERY_LIMIT",
        "EQUITY_RECORDING_POSITION_MAXIMUM",
    ):
        assert name in ports.__all__ and hasattr(ports, name)
    assert ListingReference(Symbol("M&M"), _NSE).exchange_code == _NSE
