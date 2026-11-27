"""SI-A: the permanent read-storage integrity contract and its EI-7C query propagation.

Acceptance of the EI-7 Decision Register, section 14.4 (EI7-SI1, approved
2026-11-28) at the Application layer: ``EquityReadStorageIntegrityError`` is
an additive, Application-owned contract that a reader raises for a permanent
failure of the storage read boundary; every EI-7C query maps it to the
existing refusal PERSISTED_AUTHORITY_INVALID -- whole response, not
retryable, one fixed message -- and never to DEPENDENCY_UNAVAILABLE, UNKNOWN,
NOT_QUERYABLE, an evidence panel or a lifecycle summary status. It says
nothing about any one record.

Readers here are the accepted EI-7C TEST-ONLY read wrappers over the accepted
in-memory worlds (test_read_equity_*), with the new error injected at one
reader boundary at a time. No SQLite: which engine condition raises the
contract is SI-B's, on real SQLite. The underlying exception injected below
is deliberately sensitive -- a fake path, SQL and driver text -- to prove
that nothing of it reaches a public string.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from typing import Any

import pytest
import test_read_equity_recommendation_detail as detail
import test_read_equity_research_timeline as research
import test_read_equity_session_issuances as sessions
import test_record_equity_recommendation_lifecycle as ei6b

from northstar_application import application_services, ports
from northstar_application.application_services import (
    EquityEvidencePanelReason,
    EquityEvidencePanelStatus,
    EquityLifecycleSummaryStatus,
    EquityReadQueryRefusal,
    EquityReadQueryRefusedError,
    GetEquityLifecycleSummaryUseCase,
    GetEquityRecommendationDetailUseCase,
    GetEquityResearchTimelineUseCase,
    ListEquityDeclaredSessionRecommendationsUseCase,
    ListEquityRecordedSessionIssuancesUseCase,
)
from northstar_application.application_services import (
    equity_read_query_errors as query_errors,
)
from northstar_application.ports import (
    EquityCandleEvidenceUnavailableError,
    EquityIntradayResearchTimelineUnavailableError,
    EquityReadStorageIntegrityCondition,
    EquityReadStorageIntegrityError,
    EquityRecommendationGenerationStoreUnavailableError,
    EquityRecommendationIssuanceUnavailableError,
    EquityRecommendationLifecycleStoreUnavailableError,
    EquityRecordedSessionDiscoveryLimitExceededError,
    InvalidEquityRecordedSessionIssuancesError,
    InvalidStoredEquityCandleEvidenceSnapshotError,
    InvalidStoredEquityIntradayResearchTimelineError,
    InvalidStoredEquityRecommendationIssuanceAuthorityError,
    InvalidStoredEquityRecommendationLifecycleError,
)
from northstar_application.ports import (
    equity_read_storage_integrity as contract_module,
)

_R = EquityReadQueryRefusal
_C = EquityReadStorageIntegrityCondition
_SRC = Path(application_services.__file__).resolve().parents[1]
_SENSITIVE = (
    "C:/secret/northstar-equity.sqlite3",
    "SELECT issuance_id FROM equity_recommendation_issuances WHERE 1",
    "database disk image is malformed",
    "internal-diagnostic-7f3a",
)
_FIXED = f"PERSISTED_AUTHORITY_INVALID: {query_errors.EQUITY_READ_STORAGE_FAILURE_MESSAGE}"


def _storage_failure(condition: _C = _C.CORRUPT) -> EquityReadStorageIntegrityError:
    """The contract as SI-B will raise it: chained to a sensitive engine-level exception."""
    try:
        raise RuntimeError(" | ".join(_SENSITIVE))
    except RuntimeError as engine:
        try:
            raise EquityReadStorageIntegrityError(condition) from engine
        except EquityReadStorageIntegrityError as error:
            return error


def _assert_storage_refusal(refused: EquityReadQueryRefusedError) -> None:
    """PERSISTED_AUTHORITY_INVALID, not retryable, fixed text, chained, nothing leaked."""
    assert refused.reason is _R.PERSISTED_AUTHORITY_INVALID
    assert refused.retryable is False
    assert str(refused) == _FIXED
    assert isinstance(refused.__cause__, EquityReadStorageIntegrityError)
    assert isinstance(refused.__cause__.__cause__, RuntimeError)  # kept for diagnosis only
    public = (str(refused), repr(refused), str(refused.__cause__), repr(refused.__cause__))
    for fragment in _SENSITIVE:
        assert not any(fragment in text for text in public), fragment


def _refused(call: Any) -> EquityReadQueryRefusedError:
    with pytest.raises(EquityReadQueryRefusedError) as raised:
        call()
    return raised.value


# ======================================================================
# 13.1 The contract
# ======================================================================


@pytest.mark.parametrize("condition", list(_C))
def test_c01_the_error_is_constructed_with_a_condition_and_a_fixed_safe_message(
    condition: _C,
) -> None:
    error = EquityReadStorageIntegrityError(condition)
    assert error.condition is condition and error.retryable is False
    assert str(error) == str(EquityReadStorageIntegrityError(condition))  # stable
    assert "nothing was read" in str(error)
    for word in ("sqlite", "SQLITE", "SELECT", "/", "\\", ".sqlite3", "issuance", "snapshot"):
        assert word not in str(error), word
    assert {c.value for c in _C} == {"CORRUPT", "UNUSABLE"}  # the two approved classifications


@pytest.mark.parametrize("condition", ["CORRUPT", "corrupt", None, 1, ""])
def test_c01_the_condition_must_be_the_enum_never_text(condition: object) -> None:
    with pytest.raises(TypeError):
        EquityReadStorageIntegrityError(condition)  # type: ignore[arg-type]


def test_c01_the_constructor_takes_no_free_text() -> None:
    parameters = inspect.signature(EquityReadStorageIntegrityError.__init__).parameters
    assert list(parameters) == ["self", "condition"]
    with pytest.raises(TypeError):
        EquityReadStorageIntegrityError(_C.CORRUPT, "database disk image is malformed")  # type: ignore[call-arg]


@pytest.mark.parametrize(
    "other",
    [
        EquityRecommendationIssuanceUnavailableError,
        EquityRecommendationLifecycleStoreUnavailableError,
        EquityRecommendationGenerationStoreUnavailableError,
        EquityIntradayResearchTimelineUnavailableError,
        EquityCandleEvidenceUnavailableError,
        EquityRecordedSessionDiscoveryLimitExceededError,
        InvalidStoredEquityRecommendationIssuanceAuthorityError,
        InvalidStoredEquityRecommendationLifecycleError,
        InvalidStoredEquityIntradayResearchTimelineError,
        InvalidStoredEquityCandleEvidenceSnapshotError,
        InvalidEquityRecordedSessionIssuancesError,
        EquityReadQueryRefusedError,
        RuntimeError,
        ValueError,
    ],
)
def test_c02_it_is_neither_an_unavailability_nor_a_record_integrity_error(other: type) -> None:
    assert not issubclass(EquityReadStorageIntegrityError, other)
    assert not issubclass(other, EquityReadStorageIntegrityError)


def test_c02_it_derives_from_exception_directly() -> None:
    assert EquityReadStorageIntegrityError.__mro__[1:] == Exception.__mro__


def test_c03_it_names_no_record_category_or_identity() -> None:
    error = EquityReadStorageIntegrityError(_C.UNUSABLE)
    assert set(vars(error)) == {"condition"}
    for record_fact in ("category", "issuance_id", "snapshot_id", "generation_key", "record"):
        assert not hasattr(error, record_fact), record_fact


def test_c04_it_is_exported_from_the_application_ports_namespace() -> None:
    assert ports.EquityReadStorageIntegrityError is EquityReadStorageIntegrityError
    assert ports.EquityReadStorageIntegrityCondition is EquityReadStorageIntegrityCondition
    assert "EquityReadStorageIntegrityError" in ports.__all__
    assert "EquityReadStorageIntegrityCondition" in ports.__all__
    assert EquityReadStorageIntegrityError.__module__ == contract_module.__name__


def test_c05_the_contract_module_needs_no_database_driver_or_web_framework() -> None:
    tree = ast.parse(Path(inspect.getfile(contract_module)).read_text(encoding="utf-8"))
    modules = {n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    modules |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert modules <= {"__future__", "enum"}, modules


def test_c06_the_engine_exception_is_chained_but_never_in_the_public_string() -> None:
    error = _storage_failure()
    assert isinstance(error.__cause__, RuntimeError)
    for fragment in _SENSITIVE:
        assert fragment not in str(error) and fragment not in repr(error)


def test_c07_no_refusal_reason_is_added() -> None:
    assert [r.value for r in EquityReadQueryRefusal] == [
        "ISSUANCE_NOT_FOUND",
        "WATERMARK_AHEAD",
        "DISCOVERY_LIMIT_EXCEEDED",
        "EVIDENCE_CLASS_UNSUPPORTED",
        "UNSUPPORTED_SCOPE",
        "PERSISTED_AUTHORITY_INVALID",
        "LIFECYCLE_HISTORY_INVALID",
        "DEPENDENCY_UNAVAILABLE",
    ]


def test_c08_the_shared_mapping_refuses_only_the_storage_failure() -> None:
    with pytest.raises(EquityReadQueryRefusedError) as raised:
        with query_errors.refusing_equity_read_storage_failure():
            raise _storage_failure(_C.UNUSABLE)
    _assert_storage_refusal(raised.value)
    for other in (
        EquityRecommendationIssuanceUnavailableError("test: down"),
        InvalidStoredEquityRecommendationLifecycleError("HISTORY", "test: gap"),
        KeyError("test"),
    ):
        with pytest.raises(type(other)):
            with query_errors.refusing_equity_read_storage_failure():
                raise other
    with query_errors.refusing_equity_read_storage_failure():
        pass  # no failure, nothing raised


# ======================================================================
# 13.2 Session discovery: recorded and declared, every watermark
# ======================================================================


class _FailingSessions(sessions._Sessions):
    """The accepted EI-7C in-memory session reader whose read meets a storage failure."""

    def __init__(self, declared: bool) -> None:
        super().__init__()
        self.issue(sessions._base())
        if declared:
            self.declare(sessions._covering())
        self.failure: EquityReadStorageIntegrityError | None = None

    def recorded_session_issuances(self, exchange_code, trading_date, *, limit, watermark=None):  # type: ignore[no-untyped-def]
        if self.failure is not None:
            self.calls.append((exchange_code, trading_date, limit, watermark))
            raise self.failure
        return super().recorded_session_issuances(
            exchange_code, trading_date, limit=limit, watermark=watermark
        )


@pytest.mark.parametrize(
    "use_case",
    [ListEquityRecordedSessionIssuancesUseCase, ListEquityDeclaredSessionRecommendationsUseCase],
)
@pytest.mark.parametrize("watermark", [None, 0, 1], ids=["latest", "historical-0", "historical-1"])
@pytest.mark.parametrize("declared", [True, False], ids=["declared", "undeclared"])
@pytest.mark.parametrize("condition", list(_C))
def test_s_a_storage_failure_refuses_both_session_views_at_any_watermark(
    use_case: type, watermark: int | None, declared: bool, condition: _C
) -> None:
    reader = _FailingSessions(declared)
    query = use_case(reader)
    assert query.execute(sessions._NSE, sessions._day(), watermark) is not None  # control
    reader.failure = _storage_failure(condition)
    refused = _refused(lambda: query.execute(sessions._NSE, sessions._day(), watermark))
    _assert_storage_refusal(refused)
    assert reader.calls[-1][3] == watermark  # the watermark was passed on, never fabricated


def test_s_an_ordinary_outage_is_still_dependency_unavailable_and_retryable() -> None:
    reader = _FailingSessions(declared=True)
    reader.unavailable = True
    refused = _refused(lambda: sessions._recorded(reader))
    assert (refused.reason, refused.retryable) == (_R.DEPENDENCY_UNAVAILABLE, True)
    reader.unavailable, reader.ledger_corrupt = False, True
    assert _refused(lambda: sessions._declared(reader)).reason is _R.PERSISTED_AUTHORITY_INVALID


# ======================================================================
# 13.3 Recommendation detail: every reader boundary fails the whole detail
# ======================================================================


def test_d_a_storage_failure_of_the_lifecycle_read_refuses_the_whole_detail() -> None:
    recording = ei6b._recording()
    lifecycles = detail._Lifecycles(recording.store)
    lifecycles.error = _storage_failure()
    _assert_storage_refusal(detail._refused(recording, lifecycles=lifecycles))
    assert lifecycles.calls == [recording.issuance_id]  # not ISSUANCE_NOT_FOUND


def test_d_a_storage_failure_of_the_generation_read_refuses_the_whole_detail() -> None:
    recording = ei6b._recording()
    generations = detail._Generations(recording.world.store)
    generations.error = _storage_failure(_C.UNUSABLE)
    _assert_storage_refusal(detail._refused(recording, generations=generations))
    assert generations.keys == [recording.issued.generation_key]  # never "no disposition"


@pytest.mark.parametrize("which", ["candidate", "lifecycle-cited"])
def test_d_a_storage_failure_of_a_panel_snapshot_is_never_a_panel_status(which: str) -> None:
    recording = ei6b._recording()
    ei6b._active(recording)
    target = (
        detail._candidate_id(recording)
        if which == "candidate"
        else detail._lifecycle_ids(detail._detail(recording))[0]
    )
    snapshots = detail._Snapshots(recording.scenario.world.snapshots, {target: _storage_failure()})
    _assert_storage_refusal(detail._refused(recording, snapshots=snapshots))


def test_d_the_accepted_panel_states_stay_distinct_from_a_storage_failure() -> None:
    recording = ei6b._recording()
    ei6b._active(recording)
    cited = detail._lifecycle_ids(detail._detail(recording))[0]
    store = recording.scenario.world.snapshots
    panel_answers = {
        None: (EquityEvidencePanelStatus.UNAVAILABLE, EquityEvidencePanelReason.EVIDENCE_NOT_FOUND),
        InvalidStoredEquityCandleEvidenceSnapshotError("test: corrupt"): (
            EquityEvidencePanelStatus.INVALID,
            EquityEvidencePanelReason.EVIDENCE_INVALID,
        ),
        EquityCandleEvidenceUnavailableError("test: down"): (
            EquityEvidencePanelStatus.UNAVAILABLE,
            EquityEvidencePanelReason.DEPENDENCY_UNAVAILABLE,
        ),
    }
    for answer, expected in panel_answers.items():
        view = detail._detail(recording, snapshots=detail._Snapshots(store, {cited: answer}))
        panel = next(p for p in view.evidence if p.snapshot_id == cited)
        assert (panel.status, panel.reason) == expected
    # The non-SYNTHETIC refusal (EVIDENCE_CLASS_UNSUPPORTED) is pinned, unchanged, by test_d13.


# ======================================================================
# 13.5 Lifecycle summary: never INVALID or UNAVAILABLE for a storage failure
# ======================================================================


def test_l_a_storage_failure_refuses_the_summary_never_a_summary_status() -> None:
    recording = ei6b._recording()
    lifecycles = detail._Lifecycles(recording.store)
    summary = GetEquityLifecycleSummaryUseCase(lifecycles)
    assert summary.execute(recording.issuance_id).status is EquityLifecycleSummaryStatus.AVAILABLE
    lifecycles.error = _storage_failure()
    _assert_storage_refusal(_refused(lambda: summary.execute(recording.issuance_id)))


@pytest.mark.parametrize(
    ("error", "status"),
    [
        (InvalidStoredEquityRecommendationLifecycleError("HISTORY", "test: gap"), "INVALID"),
        (EquityRecommendationLifecycleStoreUnavailableError("test: down"), "UNAVAILABLE"),
        (EquityRecommendationIssuanceUnavailableError("test: down"), "UNAVAILABLE"),
    ],
)
def test_l_accepted_summary_statuses_are_unchanged(error: Exception, status: str) -> None:
    recording = ei6b._recording()
    lifecycles = detail._Lifecycles(recording.store)
    lifecycles.error = error
    summary = GetEquityLifecycleSummaryUseCase(lifecycles).execute(recording.issuance_id)
    assert summary.status.value == status and summary.state is None


# ======================================================================
# 13.4 Research: every reader boundary fails the whole response
# ======================================================================


def test_r_a_storage_failure_of_the_timeline_read_is_never_unknown() -> None:
    subject = research._unissued()
    timelines = research._Timelines(subject.world.timelines)
    timelines.error = _storage_failure()
    _assert_storage_refusal(subject.refused(timelines=timelines))


def test_r_a_storage_failure_of_a_research_snapshot_refuses_before_any_decision() -> None:
    subject = research._unissued()
    first = research._stored(subject).entries[0].snapshot_id
    snapshots = detail._Snapshots(subject.world.snapshots, {first: _storage_failure()})
    _assert_storage_refusal(subject.refused(snapshots=snapshots))
    assert snapshots.reads == [first]  # no further snapshot, no re-derived decision


def test_r_a_storage_failure_of_the_issuance_read_is_never_not_queryable() -> None:
    subject = research._unissued()
    assert subject.view().generation_status.value == "NOT_QUERYABLE"  # control
    subject.issuances.error = _storage_failure(_C.UNUSABLE)
    _assert_storage_refusal(subject.refused())


def test_r_a_storage_failure_of_the_generation_read_refuses_an_issued_response() -> None:
    subject, _recording = research._issued()
    assert subject.view().generation_status.value == "PROPOSED"  # control
    generations = detail._Generations(subject.generations.inner)
    generations.error = _storage_failure()
    _assert_storage_refusal(subject.refused(generations=generations))


def test_r_ordinary_research_reader_failures_keep_their_classification() -> None:
    subject = research._unissued()
    timelines = research._Timelines(subject.world.timelines)
    timelines.error = EquityIntradayResearchTimelineUnavailableError("test: down")
    refused = subject.refused(timelines=timelines)
    assert (refused.reason, refused.retryable) == (_R.DEPENDENCY_UNAVAILABLE, True)
    timelines.error = None
    first = research._stored(subject).entries[0].snapshot_id
    for answer, reason in (
        (EquityCandleEvidenceUnavailableError("test: down"), _R.DEPENDENCY_UNAVAILABLE),
        (
            InvalidStoredEquityCandleEvidenceSnapshotError("test: bad"),
            _R.PERSISTED_AUTHORITY_INVALID,
        ),
    ):
        snapshots = detail._Snapshots(subject.world.snapshots, {first: answer})
        assert subject.refused(snapshots=snapshots).reason is reason


# ======================================================================
# 13.7 Architecture
# ======================================================================

_QUERY_MODULES = (
    "application_services/read_equity_session_issuances.py",
    "application_services/read_equity_recommendation_detail.py",
    "application_services/read_equity_research_timeline.py",
)
_READER_CALLS = {
    "recorded_session_issuances",
    "load_lifecycle",
    "load_disposition",
    "load_attempt",
    "get_snapshot",
    "load",
    "load_issuance",
    "load_session_issuance",
}


def _tree(relative: str) -> ast.Module:
    return ast.parse((_SRC / relative).read_text(encoding="utf-8"))


def _mapped(node: ast.With) -> bool:
    return any(
        isinstance(item.context_expr, ast.Call)
        and getattr(item.context_expr.func, "id", "") == "refusing_equity_read_storage_failure"
        for item in node.items
    )


@pytest.mark.parametrize("module", _QUERY_MODULES)
def test_a01_every_reader_call_in_a_query_module_runs_inside_the_storage_mapping(
    module: str,
) -> None:
    tree = _tree(module)
    covered: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.With) and _mapped(node):
            covered |= {id(n) for n in ast.walk(node)}
    reader_calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr in _READER_CALLS
        and not (isinstance(n.func.value, ast.Attribute) and n.func.value.attr == "_snapshots")
        and not (isinstance(n.func.value, ast.Name) and n.func.value.id == "json")
    ]
    assert reader_calls, module
    for call in reader_calls:
        assert id(call) in covered, f"{module}:{call.lineno} {ast.unparse(call)}"


def test_a02_the_changed_read_modules_import_no_infrastructure_driver_or_web_framework() -> None:
    forbidden = ("infrastructure", "sqlite", "fastapi", "starlette", "uvicorn", "httpx")
    for module in (*_QUERY_MODULES, "application_services/equity_read_query_errors.py",
                   "ports/equity_read_storage_integrity.py"):  # fmt: skip
        tree = _tree(module)
        modules = {n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
        modules |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        assert not [m for m in modules if any(word in m for word in forbidden)], module


def test_a03_the_query_use_cases_keep_their_signatures_and_take_readers_only() -> None:
    session = (["self", "reader"], ["self", "exchange_code", "trading_date", "watermark"])
    expected = {
        ListEquityRecordedSessionIssuancesUseCase: session,
        ListEquityDeclaredSessionRecommendationsUseCase: session,
        GetEquityRecommendationDetailUseCase: (
            ["self", "lifecycles", "generations", "snapshots"],
            ["self", "issuance_id"],
        ),
        GetEquityLifecycleSummaryUseCase: (["self", "lifecycles"], ["self", "issuance_id"]),
        GetEquityResearchTimelineUseCase: (
            ["self", "timelines", "snapshots", "issuances", "generations"],
            ["self", "strategy", "listing", "trading_date"],
        ),
    }
    for use_case, (init, execute) in expected.items():
        assert list(inspect.signature(use_case.__init__).parameters) == init, use_case
        assert list(inspect.signature(use_case.execute).parameters) == execute, use_case


def test_a04_no_write_or_monitoring_name_enters_the_changed_modules() -> None:
    names = {"Writer", "Recorder", "Store", "issue", "append", "write_", "record_", "monitor"}
    for module in ("application_services/equity_read_query_errors.py",
                   "ports/equity_read_storage_integrity.py"):  # fmt: skip
        tree = _tree(module)
        used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        used |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        used |= {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names}
        assert not [u for u in used if any(word in u for word in names)], module


def test_a05_the_mapping_is_not_exported_as_a_public_route_or_wire_field() -> None:
    assert "EquityReadStorageIntegrityError" not in application_services.__all__
    assert not hasattr(application_services, "refusing_equity_read_storage_failure")
