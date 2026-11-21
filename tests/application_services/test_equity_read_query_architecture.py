"""EI-7C architecture: the read-query layer holds readers only and imports nothing forbidden.

Source scans for the EI-7 Decision Register, sections 13.3 and 13.8, and EI-7
R-2: no Infrastructure, SQLite or web framework; no writer, recorder, issuer,
monitor, publisher, store write method, scheduler, broker or clock in any
EI-7C read module; every use-case dependency is a reader port; and the EI-6C
monitor is never given the recorded-session discovery port (EI6C-P7).
"""

from __future__ import annotations

import ast
import inspect
import pathlib

from northstar_application import application_services, ports
from northstar_application.application_services import (
    GetEquityLifecycleSummaryUseCase,
    GetEquityRecommendationDetailUseCase,
    GetEquityResearchTimelineUseCase,
    ListEquityDeclaredSessionRecommendationsUseCase,
    ListEquityRecordedSessionIssuancesUseCase,
)

_READ_MODULES = (
    "ports/equity_candle_evidence_snapshot_reader.py",
    "ports/equity_intraday_research_timeline_reader.py",
    "ports/equity_recorded_session_issuance_reader.py",
    "application_services/equity_read_query_errors.py",
    "application_services/equity_recommendation_read_lineage.py",
    "application_services/read_equity_session_issuances.py",
    "application_services/read_equity_recommendation_detail.py",
    "application_services/read_equity_research_timeline.py",
)

_FORBIDDEN_IMPORTS = (
    "infrastructure",
    "sqlite",
    "sqlalchemy",
    "fastapi",
    "starlette",
    "httpx",
    "uvicorn",
    "requests",
    "socket",
    "broker",
    "execution",
    "paper",
    "futures",
    "provider",
    "scheduler",
    "monitor",
    "random",
    "uuid",
)

_FORBIDDEN_NAMES = {
    "EquityRecommendationIssuanceWriter",
    "EquityRecommendationGenerationRecordWriter",
    "EquityRecommendationGenerationStore",
    "EquityRecommendationLifecycleStore",
    "EquityIntradayResearchTimelineStore",
    "EquityCandleEvidenceSnapshotStore",
    "EquityCandleEvidenceJournal",
    "EquitySyntheticPolicyCatalogStore",
    "EquityCandleEvidenceSnapshotCatalog",
    "IssueEquityTradeRecommendationUseCase",
    "DeclareEquitySessionIssuanceCompletenessUseCase",
    "RecordEquityRecommendationGenerationUseCase",
    "RecordEquityRecommendationLifecycleUseCase",
    "MonitorEquityRecommendationLifecyclesUseCase",
    "AdvanceEquityIntradayResearchTimelineUseCase",
    "GenerateEquityTradeRecommendationUseCase",
    "IssueEquityCandleEvidenceSnapshotUseCase",
    "PublishEquitySyntheticPolicyBundleUseCase",
}

_FORBIDDEN_CALLS = {
    "append",
    "issue",
    "publish",
    "write_issuance",
    "write_declaration",
    "write_generation",
    "write_attempt",
    "now",
    "utcnow",
    "today",
    "time",
    "sleep",
}


def _root() -> pathlib.Path:
    source = inspect.getsourcefile(application_services)
    assert source is not None
    return pathlib.Path(source).parent.parent


def _tree(relative: str) -> ast.Module:
    return ast.parse((_root() / relative).read_text(encoding="utf-8"))


def test_ar01_read_modules_import_no_infrastructure_web_broker_scheduler_or_randomness() -> None:
    violations = []
    for path in _READ_MODULES:
        for node in ast.walk(_tree(path)):
            if isinstance(node, ast.ImportFrom | ast.Import):
                modules = (
                    [node.module or ""]
                    if isinstance(node, ast.ImportFrom)
                    else [alias.name for alias in node.names]
                )
                violations += [
                    f"{path}: {m}" for m in modules if any(w in m for w in _FORBIDDEN_IMPORTS)
                ]
    assert violations == []


def test_ar02_read_modules_name_no_writer_store_recorder_issuer_monitor_or_publisher() -> None:
    violations = []
    for path in _READ_MODULES:
        tree = _tree(path)
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        names |= {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names}
        violations += [f"{path}: {name}" for name in sorted(names & _FORBIDDEN_NAMES)]
    assert violations == []


def test_ar03_read_modules_call_no_write_operation_and_read_no_clock() -> None:
    violations = []
    for path in _READ_MODULES:
        for node in ast.walk(_tree(path)):
            if isinstance(node, ast.Call):
                func = node.func
                name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
                if name in _FORBIDDEN_CALLS:
                    violations.append(f"{path}:{node.lineno}: calls {name}")
    assert violations == []


def test_ar04_every_use_case_dependency_is_a_reader_port() -> None:
    for use_case in (
        ListEquityRecordedSessionIssuancesUseCase,
        ListEquityDeclaredSessionRecommendationsUseCase,
        GetEquityRecommendationDetailUseCase,
        GetEquityLifecycleSummaryUseCase,
        GetEquityResearchTimelineUseCase,
    ):
        constructor = use_case.__init__
        if use_case in (
            ListEquityRecordedSessionIssuancesUseCase,
            ListEquityDeclaredSessionRecommendationsUseCase,
        ):
            constructor = use_case.__mro__[1].__init__
        annotations = [
            p.annotation
            for name, p in inspect.signature(constructor).parameters.items()
            if name != "self"
        ]
        assert annotations, use_case
        assert all(str(a).endswith("Reader") for a in annotations), (use_case, annotations)


def test_ar05_the_ei6c_monitor_is_never_given_the_discovery_port() -> None:
    tree = _tree("application_services/monitor_equity_recommendation_lifecycles.py")
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    names |= {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names}
    modules = {n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert "EquityRecordedSessionIssuanceReader" not in names
    assert not {m for m in modules if "recorded_session" in m or ".read_equity" in m}


def test_ar06_the_only_change_to_existing_ei3c_code_is_the_rehydration_reader_check() -> None:
    tree = _tree("application_services/advance_equity_intraday_research_timeline.py")
    rehydrate = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef)
        and n.name == "rehydrate_equity_intraday_research_timeline"
    )
    checked = {
        n.args[1].id
        for n in ast.walk(rehydrate)
        if isinstance(n, ast.Call)
        and getattr(n.func, "id", "") == "isinstance"
        and isinstance(n.args[1], ast.Name)
    }
    assert "EquityCandleEvidenceSnapshotReader" in checked
    assert "EquityCandleEvidenceSnapshotStore" not in checked
    advance = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.ClassDef) and n.name == "AdvanceEquityIntradayResearchTimelineUseCase"
    )
    source = ast.unparse(advance)
    assert (
        "snapshots: EquityCandleEvidenceSnapshotStore" in source
    )  # the write side keeps its store


def test_ar07_the_read_layer_is_exported_and_no_read_name_leaks_a_writer() -> None:
    for name in (
        "GetEquityRecommendationDetailUseCase",
        "GetEquityResearchTimelineUseCase",
        "GetEquityLifecycleSummaryUseCase",
        "ListEquityRecordedSessionIssuancesUseCase",
        "ListEquityDeclaredSessionRecommendationsUseCase",
        "EquityReadQueryRefusedError",
        "EquityReadQueryRefusal",
    ):
        assert name in application_services.__all__
    for name in ("EquityRecordedSessionIssuanceReader", "EquityCandleEvidenceSnapshotReader"):
        assert name in ports.__all__
