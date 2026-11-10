"""EI4-D11-W3: the generation write boundary as an architecture rule (ADR-023).

After W3 the Application persistence port carries no raw write: the combined
``EquityRecommendationGenerationStore`` is exactly the read port and the
authorized-write port, every operation abstract, with no compatibility
default. These tests pin that contract, show that the verified recorder is the
only Application code that writes -- and only what it issued -- and scan the
production source so a raw-write call, a raw port method, a write default or
a writer dependency cannot be reintroduced silently.

The scan reads production source only (``src/northstar_application``); test
code, including labelled historical-corruption setup, is outside it by
construction. It works on the syntax tree, so comments, docstrings and
strings never count. It is focused on the EI4-D11 boundary:
``record_generation`` and ``record_attempt`` are reserved names there, and
similarly named operations of other domains (``record_audit``,
``record_progress``, ``record_count``) are not flagged. The one exemption is
structural, not a path list: a class calling its OWN method of a reserved
name on ``self`` -- the recorder's public ``record_attempt`` use-case entry,
which itself writes through ``write_attempt`` -- unless that class implements
a generation port.

Everything is synthetic and reuses the EI-4C research world (``ei4c``) and
the EI-4D-A store double (``ei4d._Store``). The boundary is not tamper-proof
against privileged code or direct database access, and nothing here claims so.
"""

from __future__ import annotations

import ast
import inspect
import pathlib
from typing import Any

import pytest
import test_generate_equity_trade_recommendation as ei4c
import test_record_equity_recommendation_generation as ei4d

from northstar_application import application_services, ports
from northstar_application.application_services import (
    AuditEquityRecommendationPolicyBindingsUseCase,
    EquityRecommendationRecordingOutcome,
    VerifiedEquityRecommendationAttemptWrite,
    VerifiedEquityRecommendationGenerationWrite,
    VerifyStoredEquityRecommendationGenerationUseCase,
    equity_recommendation_disposition,
    require_verified_equity_recommendation_attempt_write,
    require_verified_equity_recommendation_generation_write,
)
from northstar_application.application_services import (
    record_equity_recommendation_generation as record_module,
)
from northstar_application.ports import (
    EquityRecommendationGenerationRecordReader,
    EquityRecommendationGenerationRecordWriter,
    EquityRecommendationGenerationStore,
)
from northstar_application.ports import (
    equity_recommendation_generation_store as store_port_module,
)

_O = EquityRecommendationRecordingOutcome
_Reader = EquityRecommendationGenerationRecordReader
_Writer = EquityRecommendationGenerationRecordWriter
_Store = EquityRecommendationGenerationStore

_RAW_WRITES = frozenset({"record_generation", "record_attempt"})
_AUTHORIZED_WRITES = frozenset({"write_generation", "write_attempt"})
_REQUIRED = frozenset({"load_disposition", "load_attempt"}) | _AUTHORIZED_WRITES
_GENERATION_PORTS = frozenset(
    {
        "EquityRecommendationGenerationRecordReader",
        "EquityRecommendationGenerationRecordWriter",
        "EquityRecommendationGenerationStore",
    }
)
_WRITE_CAPABLE_PORTS = _GENERATION_PORTS - {"EquityRecommendationGenerationRecordReader"}
_PORT_MODULE = "ports/equity_recommendation_generation_store.py"
_PORTS_PACKAGE = "ports/__init__.py"
_RECORDER_MODULE = "application_services/record_equity_recommendation_generation.py"
_ISSUERS = {
    "write_generation": "_issue_generation_write",
    "write_attempt": "_issue_attempt_write",
}


# ----------------------------------------------------------------------
# The source scan
# ----------------------------------------------------------------------


def _package_root() -> pathlib.Path:
    source = inspect.getsourcefile(application_services)
    assert source is not None
    return pathlib.Path(source).parents[1]


def _base_names(node: ast.ClassDef) -> set[str]:
    names = set()
    for base in node.bases:
        if isinstance(base, ast.Name):
            names.add(base.id)
        elif isinstance(base, ast.Attribute):
            names.add(base.attr)
    return names


def _is_abstract_stub(function: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    decorated = any(
        (isinstance(d, ast.Name) and d.id == "abstractmethod")
        or (isinstance(d, ast.Attribute) and d.attr == "abstractmethod")
        for d in function.decorator_list
    )
    docstring_only = len(function.body) == 1 and (
        isinstance(function.body[0], ast.Expr)
        and isinstance(function.body[0].value, ast.Constant)
        and isinstance(function.body[0].value.value, str)
    )
    return decorated and docstring_only


class _BoundaryScan(ast.NodeVisitor):
    """Collects EI4-D11 write-boundary violations in one production module."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.violations: list[str] = []
        self._classes: list[ast.ClassDef] = []

    def _flag(self, node: ast.AST, message: str) -> None:
        self.violations.append(f"{self.path}:{getattr(node, 'lineno', '?')}: {message}")

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        bases = _base_names(node) & _GENERATION_PORTS
        methods = [n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        if node.name in _GENERATION_PORTS:
            if self.path != _PORT_MODULE:
                self._flag(node, f"redefines the generation port {node.name}")
            for method in methods:
                if not _is_abstract_stub(method):
                    self._flag(
                        method,
                        f"{node.name}.{method.name} is a concrete port default; every "
                        "generation port operation is abstract (EI4-D11-W3)",
                    )
        elif bases:
            self._flag(
                node,
                f"{node.name} implements {sorted(bases)}; Application ships no generation "
                "store implementation (adapters belong in Infrastructure)",
            )
        if node.name in _GENERATION_PORTS or bases:
            for method in methods:
                if method.name in _RAW_WRITES:
                    self._flag(method, f"{node.name} declares raw generation write {method.name}")
        self._classes.append(node)
        self.generic_visit(node)
        self._classes.pop()

    def _own_method(self, name: str) -> bool:
        """True when the enclosing class defines ``name`` and implements no generation port."""
        if not self._classes:
            return False
        owner = self._classes[-1]
        if owner.name in _GENERATION_PORTS or _base_names(owner) & _GENERATION_PORTS:
            return False
        return any(
            isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name
            for n in owner.body
        )

    def _write_call(self, node: ast.Call, name: str, *, on_self: bool) -> None:
        if name in _RAW_WRITES and not (on_self and self._own_method(name)):
            self._flag(
                node,
                f"calls raw generation write {name}(); generation records are written only "
                "by the verified recorder through write_generation / write_attempt",
            )
        if name in _AUTHORIZED_WRITES and self.path != _RECORDER_MODULE:
            self._flag(node, f"calls {name}() outside the verified recorder")

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Attribute):
            on_self = isinstance(func.value, ast.Name) and func.value.id == "self"
            self._write_call(node, func.attr, on_self=on_self)
        elif isinstance(func, ast.Name) and func.id in _RAW_WRITES | _AUTHORIZED_WRITES:
            self._write_call(node, func.id, on_self=False)
        elif (
            isinstance(func, ast.Name)
            and func.id == "getattr"
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and node.args[1].value in _RAW_WRITES | _AUTHORIZED_WRITES
        ):
            self._write_call(node, str(node.args[1].value), on_self=False)
        self.generic_visit(node)

    def _dependency(self, node: ast.AST, name: str) -> None:
        if name in _WRITE_CAPABLE_PORTS and self.path not in {
            _PORT_MODULE,
            _PORTS_PACKAGE,
            _RECORDER_MODULE,
        }:
            self._flag(node, f"depends on the write-capable port {name}; read with the reader")

    def visit_Name(self, node: ast.Name) -> None:
        self._dependency(node, node.id)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            self._dependency(node, alias.name)


def boundary_violations(source: str, path: str) -> list[str]:
    """The EI4-D11 write-boundary violations in ``source``, a module at ``path``.

    ``path`` is relative to the ``northstar_application`` package, with ``/``.
    """
    scan = _BoundaryScan(path)
    scan.visit(ast.parse(source))
    return scan.violations


def _production_modules() -> list[tuple[str, str]]:
    root = _package_root()
    return [
        (path.relative_to(root).as_posix(), path.read_text(encoding="utf-8"))
        for path in sorted(root.rglob("*.py"))
    ]


# ----------------------------------------------------------------------
# A. The port contract
# ----------------------------------------------------------------------


def test_a_the_combined_port_requires_exactly_reads_and_authorized_writes() -> None:
    assert _Store.__abstractmethods__ == _REQUIRED
    assert _Store.__mro__[1:3] == (_Reader, _Writer)
    assert {n for n in vars(_Store) if not n.startswith("_")} == set()  # it adds nothing
    for name in _AUTHORIZED_WRITES:
        assert getattr(_Store, name) is getattr(_Writer, name), name  # inherited, not redefined
        assert getattr(_Store, name).__isabstractmethod__ is True, name
    for port in (_Reader, _Writer, _Store):
        for name in _RAW_WRITES:
            assert not hasattr(port, name), (port.__name__, name)
    assert _Reader.__abstractmethods__ == {"load_disposition", "load_attempt"}
    assert {n for n in vars(_Reader) if not n.startswith("_")} == {
        "load_disposition",
        "load_attempt",
    }


def _never(*_: Any) -> Any:
    raise AssertionError("test: never called")


class _ReadOnly(_Store):
    load_disposition = load_attempt = _never


class _RawOnly(_Store):
    """An implementation of the pre-W3 contract: reads and raw writes, no authorized write."""

    load_disposition = load_attempt = record_generation = record_attempt = _never


class _GenerationOnly(_Store):
    load_disposition = load_attempt = write_generation = _never


class _RawWriter(_Writer):
    record_generation = record_attempt = _never


@pytest.mark.parametrize(
    ("implementation", "missing"),
    [
        (_ReadOnly, {"write_generation", "write_attempt"}),
        (_RawOnly, {"write_generation", "write_attempt"}),
        (_GenerationOnly, {"write_attempt"}),
        (_RawWriter, {"write_generation", "write_attempt"}),
    ],
    ids=["read-only", "raw-only", "generation-only", "raw-writer"],
)
def test_a_an_implementation_without_both_authorized_writes_stays_abstract(
    implementation: type, missing: set[str]
) -> None:
    assert implementation.__abstractmethods__ == missing
    with pytest.raises(TypeError, match="abstract"):
        implementation()


def test_a_an_implementation_of_reads_and_authorized_writes_satisfies_the_port() -> None:
    store = ei4d._Store()
    assert isinstance(store, _Store) and isinstance(store, _Reader) and isinstance(store, _Writer)
    for name in _REQUIRED:
        assert name in vars(ei4d._Store), name  # its own, never an inherited default
    assert not any(hasattr(store, name) for name in _RAW_WRITES)


# ----------------------------------------------------------------------
# B. No compatibility fallback
# ----------------------------------------------------------------------


def test_b_the_port_module_has_no_write_default_and_calls_nothing() -> None:
    source = inspect.getsource(store_port_module)
    assert boundary_violations(source, _PORT_MODULE) == []
    tree = ast.parse(source)
    ports_ = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in _GENERATION_PORTS]
    assert {n.name for n in ports_} == _GENERATION_PORTS
    for port in ports_:
        assert not [n for n in ast.walk(port) if isinstance(n, ast.Call)], port.name
        assert not [n for n in ast.walk(port) if isinstance(n, (ast.Import, ast.ImportFrom))]


def test_b_no_public_application_api_exposes_a_raw_generation_write() -> None:
    for module in (application_services, ports):
        assert not set(module.__all__) & (_RAW_WRITES | _AUTHORIZED_WRITES)
        for name in module.__all__:
            value = getattr(module, name)
            if isinstance(value, type) and issubclass(value, (_Reader, _Writer)):
                assert not any(hasattr(value, raw) for raw in _RAW_WRITES), name


# ----------------------------------------------------------------------
# C. The recorder writes only through authorizations it issued
# ----------------------------------------------------------------------


class _StillOffersRawWrites(ei4d._Store):
    """A complete store that ALSO still offers raw writes -- as the W2-I adapter does
    until W4-I. Every call is logged; a raw call is a test failure."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[str, Any]] = []

    def write_generation(self, authorization):  # type: ignore[no-untyped-def]
        self.calls.append(("write_generation", authorization))
        return super().write_generation(authorization)

    def write_attempt(self, authorization):  # type: ignore[no-untyped-def]
        self.calls.append(("write_attempt", authorization))
        return super().write_attempt(authorization)

    def record_generation(self, *records: Any) -> Any:
        self.calls.append(("record_generation", records))
        raise AssertionError("test: the recorder called a raw generation write")

    def record_attempt(self, *records: Any) -> Any:
        self.calls.append(("record_attempt", records))
        raise AssertionError("test: the recorder called a raw attempt write")


_SCRIPTS = {
    "committed": ([], 1),
    "ack-lost-after-commit": (["lost_after"], 1),
    "commit-lost-then-retried": (["lost_before"], 2),
    "retry-limit": (["lost_before"] * 3, 3),
    "rolled-back": (["rollback"], 1),
    "unavailable": (["unavailable"], 1),
    "stored-row-invalid": (["undecodable"], 1),
}


@pytest.mark.parametrize("path", ["generation", "attempt-only"])
@pytest.mark.parametrize("script", list(_SCRIPTS))
def test_c_every_recorder_branch_writes_only_through_its_one_issued_authorization(
    path: str, script: str
) -> None:
    if path == "generation":
        ports_, audit = ei4c._proposed()
        expected, kind, validate = (
            "write_generation",
            VerifiedEquityRecommendationGenerationWrite,
            require_verified_equity_recommendation_generation_write,
        )
    else:
        ports_ = ei4c._ports()
        audit = ports_.run(listing_eligibility_version=ei4c._v("eligibility/unknown"))
        assert equity_recommendation_disposition(audit) is None
        expected, kind, validate = (
            "write_attempt",
            VerifiedEquityRecommendationAttemptWrite,
            require_verified_equity_recommendation_attempt_write,
        )
    store = _StillOffersRawWrites()
    store.script, writes = list(_SCRIPTS[script][0]), _SCRIPTS[script][1]
    result = ei4d._recorder(ports_, store).execute(audit)
    assert {name for name, _ in store.calls} == {expected}
    assert result.writes == writes == len(store.calls) == store.writes
    (authorization,) = {id(a): a for _, a in store.calls}.values()  # one, reused
    assert type(authorization) is kind
    assert validate(authorization) is authorization  # genuinely issued, content unchanged


def test_c_conflict_and_equivalence_also_write_only_through_authorizations() -> None:
    ports_, audit = ei4c._proposed()
    store = _StillOffersRawWrites()
    assert ei4d._recorder(ports_, store).execute(ei4d._unresolved(ports_)).is_stored
    assert ei4d._recorder(ports_, store).execute(audit).outcome is _O.CONTENT_CONFLICT
    assert ei4d._recorder(ports_, store).execute(ei4d._unresolved(ports_)).outcome is (
        _O.EQUIVALENT_COMMITTED
    )
    assert [name for name, _ in store.calls] == ["write_generation"] * 3


def _bound_names(node: ast.AST) -> set[str]:
    """The names ``node`` binds as targets -- never the names its value reads."""
    if isinstance(node, ast.Assign):
        targets: list[ast.expr] = list(node.targets)
    elif isinstance(node, (ast.AnnAssign, ast.AugAssign, ast.NamedExpr, ast.For)):
        targets = [node.target]
    elif isinstance(node, ast.withitem) and node.optional_vars is not None:
        targets = [node.optional_vars]
    else:
        return set()
    return {n.id for target in targets for n in ast.walk(target) if isinstance(n, ast.Name)}


def test_c_the_recorder_writes_only_what_it_issued_in_the_same_operation() -> None:
    """Structural: each authorized write passes a local bound once from the matching issuer."""
    tree = ast.parse(inspect.getsource(record_module))
    seen = set()
    for function in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
        for call in [n for n in ast.walk(function) if isinstance(n, ast.Call)]:
            name = getattr(call.func, "attr", None)
            if name not in _AUTHORIZED_WRITES:
                continue
            seen.add(name)
            assert isinstance(call.func, ast.Attribute)
            assert ast.unparse(call.func.value) == "self._store", function.name
            (argument,) = call.args
            assert isinstance(argument, ast.Name) and not call.keywords, function.name
            bindings = [n for n in ast.walk(function) if argument.id in _bound_names(n)]
            assert len(bindings) == 1, function.name  # bound once, never rebound in a retry
            (binding,) = bindings
            assert isinstance(binding, ast.Assign) and isinstance(binding.value, ast.Call)
            assert ast.unparse(binding.value.func) == _ISSUERS[name], function.name
            assert binding.lineno < call.lineno  # issued before the bounded write loop
    assert seen == _AUTHORIZED_WRITES


# ----------------------------------------------------------------------
# D. Read-only consumers depend on the reader alone
# ----------------------------------------------------------------------


_READ_ONLY_CONSUMERS = [
    (VerifyStoredEquityRecommendationGenerationUseCase, "store"),
    (AuditEquityRecommendationPolicyBindingsUseCase, "records"),
]


@pytest.mark.parametrize(
    ("use_case", "parameter"), _READ_ONLY_CONSUMERS, ids=lambda v: getattr(v, "__name__", v)
)
def test_d_read_only_consumers_take_the_reader_and_never_write(
    use_case: type, parameter: str
) -> None:
    annotation = inspect.signature(use_case).parameters[parameter].annotation
    assert annotation == "EquityRecommendationGenerationRecordReader"
    module = inspect.getmodule(use_case)
    assert module is not None
    source = inspect.getsource(module)
    path = pathlib.Path(inspect.getfile(module)).relative_to(_package_root()).as_posix()
    assert boundary_violations(source, path) == []
    names = {n.id for n in ast.walk(ast.parse(source)) if isinstance(n, ast.Name)}
    assert not names & ({"_issue_generation_write", "_issue_attempt_write"} | _WRITE_CAPABLE_PORTS)

    class _WriterOnly(_Writer):
        write_generation = write_attempt = _never

    with pytest.raises(TypeError, match="RecordReader"):
        use_case(_WriterOnly(), None)  # a writer is no reader; refused before anything else


# ----------------------------------------------------------------------
# E. The production source scan
# ----------------------------------------------------------------------


def test_e_application_production_source_respects_the_write_boundary() -> None:
    modules = _production_modules()
    assert any(path == _RECORDER_MODULE for path, _ in modules)
    violations = [v for path, source in modules for v in boundary_violations(source, path)]
    assert violations == [], "\n".join(violations)


_SERVICE = "application_services/a_new_service.py"

_FORBIDDEN = {
    "raw-generation-call": (
        _SERVICE,
        "class Service:\n"
        "    def run(self, d, a):\n"
        "        return self._store.record_generation(d, a)\n",
        "calls raw generation write record_generation()",
    ),
    "raw-attempt-call-on-a-local": (
        _SERVICE,
        "def run(store, a):\n    return store.record_attempt(a)\n",
        "calls raw generation write record_attempt()",
    ),
    "raw-call-through-getattr": (
        _SERVICE,
        "def run(store, a):\n    return getattr(store, 'record_attempt')(a)\n",
        "calls raw generation write record_attempt()",
    ),
    "authorized-write-outside-the-recorder": (
        _SERVICE,
        "class Service:\n"
        "    def run(self, token):\n"
        "        return self._store.write_generation(token)\n",
        "calls write_generation() outside the verified recorder",
    ),
    "raw-method-on-the-port": (
        _PORT_MODULE,
        "class EquityRecommendationGenerationStore(Reader, Writer):\n"
        "    @abstractmethod\n"
        "    def record_generation(self, d, a):\n"
        '        """Raw."""\n',
        "declares raw generation write record_generation",
    ),
    "write-default-delegating-to-raw": (
        _PORT_MODULE,
        "class EquityRecommendationGenerationStore(Reader, Writer):\n"
        "    def write_generation(self, authorization):\n"
        "        return self.record_generation(authorization.disposition, authorization.attempt)\n",
        "concrete port default",
    ),
    "application-side-fallback-store": (
        _SERVICE,
        "class Fallback(EquityRecommendationGenerationStore):\n"
        "    def record_generation(self, d, a):\n"
        "        return self.record_generation(d, a)\n",
        "Application ships no generation store implementation",
    ),
    "consumer-depending-on-the-writer": (
        "application_services/a_new_reader.py",
        "from northstar_application.ports import EquityRecommendationGenerationStore\n",
        "depends on the write-capable port EquityRecommendationGenerationStore",
    ),
}


@pytest.mark.parametrize("case", list(_FORBIDDEN))
def test_e_the_scan_detects_a_deliberately_introduced_violation(case: str) -> None:
    path, source, expected = _FORBIDDEN[case]
    violations = boundary_violations(source, path)
    assert any(expected in v for v in violations), violations
    assert all(v.startswith(f"{path}:") for v in violations)  # it names the path and line


_HARMLESS = {
    "comment-docstring-and-string": (
        '"""Raw writes such as store.record_generation(d, a) were removed (W3)."""\n'
        "# self._store.record_attempt(attempt) -- historical, see ADR-023\n"
        "MESSAGE = 'record_generation is not an authorized write path'\n"
    ),
    "similarly-named-operations-of-other-domains": (
        "class FuturesService:\n"
        "    def run(self, audit, progress, report):\n"
        "        self._audits.record_audit(audit)\n"
        "        self._lifecycle.record_progress(progress)\n"
        "        return report.record_count()\n"
    ),
    "a-use-case-calling-its-own-attempt-entry": (
        "class Recorder:\n"
        "    def execute(self, attempt):\n"
        "        return self.record_attempt(attempt)\n"
        "    def record_attempt(self, attempt):\n"
        "        return attempt\n"
    ),
    "reading-through-the-reader": (
        "from northstar_application.ports import EquityRecommendationGenerationRecordReader\n"
        "class Audit:\n"
        "    def __init__(self, records: EquityRecommendationGenerationRecordReader):\n"
        "        self._records = records\n"
        "    def run(self, key):\n"
        "        return self._records.load_disposition(key)\n"
    ),
}


@pytest.mark.parametrize("case", list(_HARMLESS))
def test_e_the_scan_ignores_text_and_unrelated_operations(case: str) -> None:
    assert boundary_violations(_HARMLESS[case], _SERVICE) == []


def test_e_the_own_method_exemption_never_covers_a_port_implementation() -> None:
    source = (
        "class Adapter(EquityRecommendationGenerationStore):\n"
        "    def write_attempt(self, authorization):\n"
        "        return self.record_attempt(authorization.attempt)\n"
        "    def record_attempt(self, attempt):\n"
        "        return attempt\n"
    )
    violations = boundary_violations(source, _SERVICE)
    assert any("calls raw generation write record_attempt()" in v for v in violations)
    assert any("declares raw generation write record_attempt" in v for v in violations)
