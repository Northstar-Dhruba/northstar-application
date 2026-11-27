"""EI-7E P1: the Application-owned read-only dependency-probe contract (EI7E-D02).

Port and contract tests for the EI-7 Decision Register, section 15.3, and the
design, section 25.5. They prove the one narrow probe operation, the
immutable READABLE result, the safe retryable unavailability error, the
unchanged propagation of the SI-A permanent storage-integrity contract, and
that the three outcomes stay distinct. Probe implementations here are
TEST-ONLY fakes: which storage condition is which is decided on real SQLite
by EI-7E P2, and the HTTP mapping belongs to EI-7E E1. No database, adapter,
clock or web framework is involved.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import typing
from abc import ABC
from pathlib import Path

import pytest

from northstar_application import application_services, ports
from northstar_application.application_services import (
    EquityReadQueryRefusal,
    EquityReadQueryRefusedError,
)
from northstar_application.ports import (
    EquityCandleEvidenceSnapshotReader,
    EquityCandleEvidenceUnavailableError,
    EquityIntradayResearchTimelineReader,
    EquityIntradayResearchTimelineUnavailableError,
    EquityReadDependencyProbe,
    EquityReadDependencyProbeOutcome,
    EquityReadDependencyProbeResult,
    EquityReadDependencyUnavailableError,
    EquityReadStorageIntegrityCondition,
    EquityReadStorageIntegrityError,
    EquityRecommendationGenerationRecordReader,
    EquityRecommendationGenerationStoreUnavailableError,
    EquityRecommendationIssuanceReader,
    EquityRecommendationIssuanceUnavailableError,
    EquityRecommendationLifecycleReader,
    EquityRecommendationLifecycleStoreUnavailableError,
    EquityRecordedSessionIssuanceReader,
)
from northstar_application.ports import equity_read_dependency_probe as probe_module

_READABLE = EquityReadDependencyProbeOutcome.READABLE
_PORTS_DIR = Path(ports.__file__).resolve().parent
_SERVICES_DIR = Path(application_services.__file__).resolve().parent
_MODULE = Path(inspect.getfile(probe_module))
_SENSITIVE = (
    "C:/secret/northstar-equity.sqlite3",
    "SELECT name FROM sqlite_master WHERE type = 'table'",
    "connection-id 7f3a-91",
    "unable to open database file",
)
_FAMILY_UNAVAILABLE = (
    EquityRecommendationIssuanceUnavailableError,
    EquityRecommendationLifecycleStoreUnavailableError,
    EquityRecommendationGenerationStoreUnavailableError,
    EquityIntradayResearchTimelineUnavailableError,
    EquityCandleEvidenceUnavailableError,
)


def _readable() -> EquityReadDependencyProbeResult:
    return EquityReadDependencyProbeResult(_READABLE)


def _chained(error_type: type[Exception], *args: object) -> Exception:
    """The error as an implementation raises it: chained to a sensitive engine-level cause."""
    try:
        raise RuntimeError(" | ".join(_SENSITIVE))
    except RuntimeError as engine:
        try:
            raise error_type(*args) from engine
        except error_type as error:
            return error


class _FakeProbe(EquityReadDependencyProbe):
    """TEST-ONLY: answers READABLE, or raises exactly the error it was given."""

    def __init__(self, failure: Exception | None = None) -> None:
        self._failure = failure
        self.calls = 0

    def probe(self) -> EquityReadDependencyProbeResult:
        self.calls += 1
        if self._failure is not None:
            raise self._failure
        return _readable()


def _public(error: BaseException) -> tuple[str, ...]:
    return (str(error), repr(error), *(str(a) for a in error.args))


# ======================================================================
# 12.1 The port
# ======================================================================


def test_p01_the_port_is_an_abc_with_exactly_one_abstract_read_operation() -> None:
    assert inspect.isabstract(EquityReadDependencyProbe)
    assert EquityReadDependencyProbe.__mro__[1:] == (ABC, object)
    assert EquityReadDependencyProbe.__abstractmethods__ == frozenset({"probe"})
    public = [name for name in dir(EquityReadDependencyProbe) if not name.startswith("_")]
    assert public == ["probe"]


def test_p02_probe_takes_no_identity_session_path_connection_or_clock() -> None:
    signature = inspect.signature(EquityReadDependencyProbe.probe)
    assert list(signature.parameters) == ["self"]
    hints = typing.get_type_hints(EquityReadDependencyProbe.probe)
    assert hints == {"return": EquityReadDependencyProbeResult}


def test_p03_the_port_cannot_be_used_without_an_implementation() -> None:
    with pytest.raises(TypeError):
        EquityReadDependencyProbe()  # type: ignore[abstract]

    class _NoProbe(EquityReadDependencyProbe):
        pass

    with pytest.raises(TypeError):
        _NoProbe()  # type: ignore[abstract]


def test_p04_an_implementation_returns_the_typed_result() -> None:
    fake = _FakeProbe()
    assert isinstance(fake, EquityReadDependencyProbe)
    result = fake.probe()
    assert result == _readable() and result.status is _READABLE
    assert fake.calls == 1


@pytest.mark.parametrize(
    "name",
    [
        "load",
        "save",
        "append",
        "record",
        "issue",
        "repair",
        "recover",
        "migrate",
        "integrity_check",
        "monitor",
        "initialize_database",
        "write",
        "execute",
        "connect",
    ],
)
def test_p05_the_port_exposes_no_write_recovery_or_query_operation(name: str) -> None:
    assert not hasattr(EquityReadDependencyProbe, name)


@pytest.mark.parametrize(
    "reader",
    [
        EquityRecordedSessionIssuanceReader,
        EquityRecommendationIssuanceReader,
        EquityRecommendationGenerationRecordReader,
        EquityRecommendationLifecycleReader,
        EquityIntradayResearchTimelineReader,
        EquityCandleEvidenceSnapshotReader,
    ],
)
def test_p06_the_port_is_separate_from_every_existing_reader(reader: type) -> None:
    assert not issubclass(EquityReadDependencyProbe, reader)
    assert not isinstance(_FakeProbe(), reader)
    assert "probe" not in vars(reader)


def test_p07_the_contract_is_exported_from_the_application_ports_namespace() -> None:
    names = (
        "EquityReadDependencyProbe",
        "EquityReadDependencyProbeResult",
        "EquityReadDependencyProbeOutcome",
        "EquityReadDependencyUnavailableError",
    )
    for name in names:
        assert name in ports.__all__, name
        exported = getattr(ports, name)
        assert exported is getattr(probe_module, name), name
        assert exported.__module__ == probe_module.__name__, name
    assert not any(name in application_services.__all__ for name in names)


# ======================================================================
# 12.2 The READABLE result
# ======================================================================


def test_r01_readable_is_the_one_approved_status() -> None:
    assert [s.value for s in EquityReadDependencyProbeOutcome] == ["READABLE"]
    result = _readable()
    assert result.status is _READABLE and result.status.value == "READABLE"


def test_r02_the_result_is_immutable_and_slotted() -> None:
    result = _readable()
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.status = _READABLE  # type: ignore[misc]
    assert not hasattr(result, "__dict__")
    with pytest.raises((AttributeError, TypeError)):
        result.checked_at = "2026-11-28T00:00:00Z"  # type: ignore[attr-defined]


def test_r03_the_result_holds_only_its_status() -> None:
    assert [f.name for f in dataclasses.fields(EquityReadDependencyProbeResult)] == ["status"]
    for fact in ("path", "database", "checked_at", "timestamp", "version", "tables", "count"):
        assert not hasattr(_readable(), fact), fact


def test_r04_equality_hash_and_representation_are_deterministic() -> None:
    first, second = _readable(), _readable()
    assert first == second and hash(first) == hash(second)
    outcome = "<EquityReadDependencyProbeOutcome.READABLE: 'READABLE'>"
    assert repr(first) == repr(second) == f"EquityReadDependencyProbeResult(status={outcome})"


@pytest.mark.parametrize("status", ["READABLE", "readable", "UNKNOWN", "OK", None, 1, True])
def test_r05_any_other_status_is_refused(status: object) -> None:
    with pytest.raises(ValueError):
        EquityReadDependencyProbeResult(status)  # type: ignore[arg-type]


def test_r06_the_status_is_never_defaulted() -> None:
    with pytest.raises(TypeError):
        EquityReadDependencyProbeResult()  # type: ignore[call-arg]


# ======================================================================
# 12.3 Dependency unavailability
# ======================================================================


def test_u01_the_error_takes_no_free_text_and_has_one_fixed_message() -> None:
    assert list(inspect.signature(EquityReadDependencyUnavailableError.__init__).parameters) == [
        "self"
    ]
    with pytest.raises(TypeError):
        EquityReadDependencyUnavailableError("unable to open database file")  # type: ignore[call-arg]
    error = EquityReadDependencyUnavailableError()
    assert str(error) == "The Equity read dependency is unavailable; nothing was read."
    assert str(error) == str(EquityReadDependencyUnavailableError())  # stable


def test_u02_it_is_retryable_and_a_runtime_error_like_the_reader_outages() -> None:
    error = EquityReadDependencyUnavailableError()
    assert error.retryable is True
    assert EquityReadDependencyUnavailableError.__mro__[1:] == RuntimeError.__mro__


@pytest.mark.parametrize(
    "other",
    [*_FAMILY_UNAVAILABLE, EquityReadStorageIntegrityError, EquityReadQueryRefusedError],
)
def test_u03_it_is_distinct_from_reader_family_storage_integrity_and_refusal_errors(
    other: type,
) -> None:
    assert not issubclass(EquityReadDependencyUnavailableError, other)
    assert not issubclass(other, EquityReadDependencyUnavailableError)


def test_u04_a_probe_raises_it_distinct_from_readable_and_chains_the_cause_privately() -> None:
    error = _chained(EquityReadDependencyUnavailableError)
    fake = _FakeProbe(error)
    with pytest.raises(EquityReadDependencyUnavailableError) as raised:
        fake.probe()
    assert raised.value is error
    assert not isinstance(raised.value, EquityReadStorageIntegrityError)
    assert isinstance(raised.value.__cause__, RuntimeError)  # kept for diagnosis only
    for fragment in _SENSITIVE:
        assert not any(fragment in text for text in _public(raised.value)), fragment


def test_u05_it_asserts_nothing_about_stored_data() -> None:
    error = EquityReadDependencyUnavailableError()
    assert vars(error) == {}
    for record_fact in ("condition", "category", "issuance_id", "snapshot_id", "record", "path"):
        assert not hasattr(error, record_fact), record_fact
    for word in ("sqlite", "SELECT", "/", "\\", "issuance", "snapshot", "empty", "missing"):
        assert word not in str(error), word


# ======================================================================
# 12.4 SI-A permanent storage integrity, unchanged
# ======================================================================


@pytest.mark.parametrize("condition", list(EquityReadStorageIntegrityCondition))
def test_i01_the_si_a_contract_propagates_through_a_probe_unchanged(
    condition: EquityReadStorageIntegrityCondition,
) -> None:
    error = _chained(EquityReadStorageIntegrityError, condition)
    with pytest.raises(EquityReadStorageIntegrityError) as raised:
        _FakeProbe(error).probe()
    assert raised.value is error and type(raised.value) is EquityReadStorageIntegrityError
    assert raised.value.condition is condition and raised.value.retryable is False
    assert not isinstance(raised.value, EquityReadDependencyUnavailableError)
    for fragment in _SENSITIVE:
        assert not any(fragment in text for text in _public(raised.value)), fragment


def test_i02_the_three_outcomes_never_stand_in_for_one_another() -> None:
    outcomes = {
        "readable": _readable(),
        "unavailable": EquityReadDependencyUnavailableError(),
        "integrity": EquityReadStorageIntegrityError(EquityReadStorageIntegrityCondition.CORRUPT),
    }
    assert not isinstance(outcomes["readable"], BaseException)
    assert outcomes["unavailable"].retryable is True
    assert outcomes["integrity"].retryable is False
    assert type(outcomes["unavailable"]) is not type(outcomes["integrity"])


def test_i03_no_second_storage_integrity_contract_exists() -> None:
    defining = []
    for path in sorted(_PORTS_DIR.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and "StorageIntegrity" in node.name:
                defining.append((path.name, node.name))
    assert defining == [
        ("equity_read_storage_integrity.py", "EquityReadStorageIntegrityCondition"),
        ("equity_read_storage_integrity.py", "EquityReadStorageIntegrityError"),
    ]


def test_i04_the_probe_module_defines_exactly_one_error() -> None:
    errors = [
        obj
        for obj in vars(probe_module).values()
        if isinstance(obj, type)
        and issubclass(obj, BaseException)
        and obj.__module__ == probe_module.__name__
    ]
    assert errors == [EquityReadDependencyUnavailableError]


# ======================================================================
# 13 Architecture
# ======================================================================


def _tree() -> ast.Module:
    return ast.parse(_MODULE.read_text(encoding="utf-8"))


def test_a01_the_module_imports_only_the_standard_abstractions_it_needs() -> None:
    tree = _tree()
    modules = {n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    modules |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert modules == {"__future__", "abc", "dataclasses", "enum"}


def test_a02_the_module_reads_no_environment_clock_or_database() -> None:
    tree = _tree()
    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    used |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    for name in (
        "os",
        "environ",
        "getenv",
        "time",
        "datetime",
        "now",
        "sqlite3",
        "connect",
        "execute",
        "Path",
        "open",
    ):
        assert name not in used, name


def test_a03_no_write_recovery_or_monitoring_name_enters_the_module() -> None:
    words = (
        "Writer",
        "Recorder",
        "Store",
        "issue",
        "append",
        "write",
        "record",
        "monitor",
        "initialize",
        "migrate",
        "repair",
        "recover",
        "integrity_check",
        "schedule",
    )
    tree = _tree()
    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    used |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    used |= {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef | ast.ClassDef)}
    assert not [u for u in used if any(word in u for word in words)]


def test_a04_no_public_refusal_reason_is_added() -> None:
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


def test_a05_the_existing_reader_ports_are_unchanged() -> None:
    expected = {
        EquityRecordedSessionIssuanceReader: {"recorded_session_issuances"},
        EquityRecommendationIssuanceReader: {
            "load_issuance",
            "load_session_issuance",
            "session_view",
            "session_view_at",
        },
        EquityRecommendationGenerationRecordReader: {"load_attempt", "load_disposition"},
        EquityRecommendationLifecycleReader: {"load_lifecycle"},
        EquityIntradayResearchTimelineReader: {"load"},
        EquityCandleEvidenceSnapshotReader: {"get_snapshot"},
    }
    for reader, operations in expected.items():
        declared = {n for n, v in vars(reader).items() if callable(v) and not n.startswith("_")}
        assert declared == operations, reader


def test_a06_p1_adds_no_application_service_around_the_probe() -> None:
    for path in sorted(_SERVICES_DIR.glob("*.py")):
        assert "EquityReadDependency" not in path.read_text(encoding="utf-8"), path.name
