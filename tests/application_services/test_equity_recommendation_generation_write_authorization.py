"""EI4-D11-W1: verified-write authorizations and the migrated recorder (ADR-023).

Contract tests for the Application layer of the EI4-D11 boundary: the read
and authorized-write ports, the recorder-issued authorizations, their exact
content binding, and the recorder's writes through ``write_generation`` and
``write_attempt``.

Everything is synthetic and reuses the existing helpers: the EI-4C research
world (``ei4c``) with its explicitly published policy catalog, and the EI-4D-A
generation-store double (``ei4d._Store``). Authorizations are only ever
obtained from the real recorder, by capturing what it passes to the store;
no test constructs an issued authorization itself. Where a test alters an
already-issued authorization, it does so with ``object.__setattr__`` -- a
labelled simulation of privileged tampering, outside the approved trust
boundary -- to show the validator still refuses what it can detect.

Since EI4-D11-W3 the store port has no raw write and no compatibility
default; the port contract and the architecture scans are in
test_equity_recommendation_generation_write_architecture.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import json
import pathlib
import types
from typing import Any

import pytest
import test_generate_equity_trade_recommendation as ei4c
import test_record_equity_recommendation_generation as ei4d

from northstar_application import application_services, ports
from northstar_application.application_services import (
    AuditEquityRecommendationPolicyBindingsUseCase,
    EquityRecommendationGenerationAttempt,
    EquityRecommendationGenerationDisposition,
    EquityRecommendationRecordingOutcome,
    EquityVerifiedRecommendationGeneration,
    VerifiedEquityRecommendationAttemptWrite,
    VerifiedEquityRecommendationGenerationWrite,
    VerifyStoredEquityRecommendationGenerationUseCase,
    equity_recommendation_attempt,
    equity_recommendation_disposition,
    require_verified_equity_recommendation_attempt_write,
    require_verified_equity_recommendation_generation_write,
)
from northstar_application.application_services import (
    equity_recommendation_generation_write_authorization as authorization_module,
)
from northstar_application.ports import (
    EquityRecommendationGenerationRecordReader,
    EquityRecommendationGenerationRecordWriter,
    EquityRecommendationGenerationStore,
    UnverifiedEquityRecommendationGenerationWriteError,
)

_O = EquityRecommendationRecordingOutcome
_Unverified = UnverifiedEquityRecommendationGenerationWriteError
Disposition = EquityRecommendationGenerationDisposition
Attempt = EquityRecommendationGenerationAttempt


class _Capturing(ei4d._Store):
    """The EI-4D-A store double, recording every authorization it is offered.

    The capture runs before the double's own ``write_*``, which validate the
    authorization first; ``writes`` counts only writes that passed validation.
    """

    def __init__(self) -> None:
        super().__init__()
        self.authorizations: list[Any] = []

    def write_generation(self, authorization):  # type: ignore[no-untyped-def]
        self.authorizations.append(authorization)
        return super().write_generation(authorization)

    def write_attempt(self, authorization):  # type: ignore[no-untyped-def]
        self.authorizations.append(authorization)
        return super().write_attempt(authorization)


def _issued() -> tuple[ei4c._Ports, _Capturing, VerifiedEquityRecommendationGenerationWrite]:
    """A PROPOSED generation recorded through the real recorder; its issued authorization."""
    ports_, audit = ei4c._proposed()
    store = _Capturing()
    assert ei4d._recorder(ports_, store).execute(audit).outcome is _O.COMMITTED
    (authorization,) = store.authorizations
    assert isinstance(authorization, VerifiedEquityRecommendationGenerationWrite)
    return ports_, store, authorization


def _issued_attempt() -> tuple[ei4c._Ports, VerifiedEquityRecommendationAttemptWrite]:
    ports_ = ei4c._ports()
    audit = ports_.run(listing_eligibility_version=ei4c._v("eligibility/unknown"))
    assert equity_recommendation_disposition(audit) is None  # attempt-only
    store = _Capturing()
    assert ei4d._recorder(ports_, store).execute(audit).outcome is _O.COMMITTED
    (authorization,) = store.authorizations
    assert isinstance(authorization, VerifiedEquityRecommendationAttemptWrite)
    return ports_, authorization


def _tamper(obj: object, name: str, value: object) -> None:
    """TEST-ONLY privileged tampering with a frozen record (outside the trust boundary)."""
    object.__setattr__(obj, name, value)


def _relabelled(disposition: Disposition, **changes: Any) -> Disposition:
    """A copy of ``disposition`` whose persisted fields are then tampered with in place."""
    copy = ei4d._rebuilt(disposition)
    for name, value in changes.items():
        _tamper(copy, name, value)
    return copy


# ----------------------------------------------------------------------
# A. The ports
# ----------------------------------------------------------------------


def test_a_the_combined_port_is_exactly_the_reader_and_the_authorized_writer() -> None:
    store = EquityRecommendationGenerationStore
    assert store.__abstractmethods__ == frozenset(
        {"load_disposition", "load_attempt", "write_generation", "write_attempt"}
    )
    for name, parameters in (
        ("load_disposition", ["self", "generation_key"]),
        ("load_attempt", ["self", "attempt_identity"]),
        ("write_generation", ["self", "authorization"]),
        ("write_attempt", ["self", "authorization"]),
    ):
        assert list(inspect.signature(getattr(store, name)).parameters) == parameters, name
    assert issubclass(store, EquityRecommendationGenerationRecordReader)
    assert issubclass(store, EquityRecommendationGenerationRecordWriter)
    assert EquityRecommendationGenerationRecordReader.__abstractmethods__ == frozenset(
        {"load_disposition", "load_attempt"}
    )
    assert EquityRecommendationGenerationRecordWriter.__abstractmethods__ == frozenset(
        {"write_generation", "write_attempt"}
    )
    reader_api = {n for n in vars(EquityRecommendationGenerationRecordReader) if n[0] != "_"}
    assert reader_api == {"load_disposition", "load_attempt"}
    ei4d._Store()  # an existing implementation stays constructible, unchanged


# ----------------------------------------------------------------------
# B. Issuance after verification; the recorder writes only through it
# ----------------------------------------------------------------------


def test_b_a_verified_disposition_is_written_through_its_issued_authorization() -> None:
    ports_, store, authorization = _issued()
    disposition = authorization.disposition
    assert isinstance(authorization.verified, EquityVerifiedRecommendationGeneration)
    assert authorization.verified.replayed.assessment_fingerprint == (
        disposition.assessment_fingerprint
    )
    assert authorization.policy.is_verified
    assert authorization.policy.checks[-1].offered_fingerprint == disposition.policy_set_fingerprint
    assert authorization.attempt.generation_key == disposition.generation_key
    assert require_verified_equity_recommendation_generation_write(authorization) is authorization
    assert store.writes == 1  # one validated write, through the issued authorization
    assert store.dispositions[disposition.generation_key] == disposition


def _attempt_of(disposition: Disposition, template: Attempt) -> Attempt:
    """``template``'s logical attempt, naming ``disposition``'s assessment fingerprint."""
    document = json.loads(template.attempt_document)
    document["assessment_fingerprint"] = disposition.assessment_fingerprint
    return Attempt(
        application_services.equity_recommendation_fingerprint(document),
        application_services.canonical_equity_recommendation_json(document),
    )


@pytest.mark.parametrize("failure", ["source evidence", "policy catalog"])
def test_b_no_authorization_is_issued_when_a_mandatory_check_fails(failure: str) -> None:
    ports_, audit = ei4c._proposed()
    disposition = equity_recommendation_disposition(audit)
    assert disposition is not None
    attempt = equity_recommendation_attempt(audit)
    store = _Capturing()
    if failure == "source evidence":  # a forgery whose candidate evidence does not re-establish
        forged = ei4d._manifest_relinked(disposition)
        result = ei4d._recorder(ports_, store).record(forged, _attempt_of(forged, attempt))
        assert result.outcome is _O.SOURCE_EVIDENCE_INVALID
    else:  # the catalog does not establish the frozen policy set
        ports_.catalog = ei4c._Catalog()
        result = ei4d._recorder(ports_, store).record(disposition, attempt)
        assert result.outcome is _O.POLICY_REFERENCE_NOT_ESTABLISHED
    assert store.authorizations == [] and store.writes == 0


# ----------------------------------------------------------------------
# C. Exact content binding
# ----------------------------------------------------------------------


def test_c_an_authorization_accepts_exactly_its_own_content() -> None:
    _, _, authorization = _issued()
    covered = authorization.covered
    disposition, attempt = authorization.disposition, authorization.attempt
    assert covered == (
        disposition.generation_key,
        disposition.kind.value,
        disposition.evaluator_version,
        disposition.assessment_fingerprint,
        disposition.assessment_document,
        disposition.proposal_fingerprint,
        disposition.proposal_document,
        attempt.attempt_identity,
        attempt.attempt_document,
    )
    assert require_verified_equity_recommendation_generation_write(authorization) is authorization
    store = ei4d._Store()
    assert store.write_generation(authorization).outcome.value == "COMMITTED"


@pytest.mark.parametrize(
    "field_name",
    [
        "assessment_document",
        "assessment_fingerprint",
        "proposal_document",
        "proposal_fingerprint",
        "kind",
        "evaluator_version",
    ],
)
def test_c_a_disposition_changed_after_issuance_is_refused(field_name: str) -> None:
    ports_, _, authorization = _issued()
    other = ei4d._unresolved(ports_)  # the same generation key, other content and kind
    other_disposition = equity_recommendation_disposition(other)
    assert other_disposition is not None
    assert other_disposition.generation_key == authorization.disposition.generation_key
    value = {
        "evaluator_version": "northstar.equity-recommendation-evaluator/2",
    }.get(field_name, getattr(other_disposition, field_name))
    _tamper(authorization.disposition, field_name, value)
    store = ei4d._Store()
    with pytest.raises(_Unverified):
        store.write_generation(authorization)
    assert store.writes == 0 and store.dispositions == {}


def test_c_changed_nested_canonical_content_is_refused_without_privilege() -> None:
    _, _, authorization = _issued()
    # Ordinary code can mutate the decoded document a store reads columns from.
    authorization.disposition.assessment.document["reasons"] = ["CORE_FORMATION_REJECTED"]
    store = ei4d._Store()
    with pytest.raises(_Unverified, match="canonical document"):
        store.write_generation(authorization)
    assert store.writes == 0


def test_c_another_attempt_or_policy_check_is_refused() -> None:
    ports_, _, authorization = _issued()
    other = equity_recommendation_attempt(ei4d._unresolved(ports_))
    _tamper(authorization, "attempt", other)
    with pytest.raises(_Unverified):
        require_verified_equity_recommendation_generation_write(authorization)
    _, _, fresh = _issued()
    unpublished = application_services.verify_equity_synthetic_policy_set(
        ei4c._Catalog(), fresh.disposition.inputs.policy_set
    )
    _tamper(fresh, "policy", unpublished)
    with pytest.raises(_Unverified, match="policy-binding"):
        require_verified_equity_recommendation_generation_write(fresh)


def test_c_one_operation_never_authorizes_another_under_the_same_key() -> None:
    ports_, _, proposed = _issued()
    unresolved_store = _Capturing()
    assert ei4d._recorder(ports_, unresolved_store).execute(ei4d._unresolved(ports_)).is_stored
    (unresolved,) = unresolved_store.authorizations
    assert unresolved.disposition.generation_key == proposed.disposition.generation_key
    # Not by swapping parts in place ...
    _tamper(proposed, "verified", unresolved.verified)
    _tamper(proposed, "attempt", unresolved.attempt)
    with pytest.raises(_Unverified, match="differs"):
        require_verified_equity_recommendation_generation_write(proposed)
    # ... and not by building a consistent copy: it was never issued.
    copy = dataclasses.replace(unresolved)
    with pytest.raises(_Unverified, match="not issued"):
        require_verified_equity_recommendation_generation_write(copy)


def test_c_matching_identifiers_without_issuance_are_insufficient() -> None:
    _, _, authorization = _issued()
    rebuilt = VerifiedEquityRecommendationGenerationWrite(
        authorization.verified, authorization.policy, authorization.attempt
    )
    assert rebuilt.covered == authorization.covered  # identical content and identifiers
    with pytest.raises(_Unverified, match="not issued"):
        require_verified_equity_recommendation_generation_write(rebuilt)


# ----------------------------------------------------------------------
# D. Issuer validation, before any delegation
# ----------------------------------------------------------------------


def test_d_missing_foreign_and_unissued_authorizations_are_refused_before_any_write() -> None:
    _, _, authorization = _issued()
    _, attempt_authorization = _issued_attempt()
    lookalike = types.SimpleNamespace(
        verified=authorization.verified,
        policy=authorization.policy,
        attempt=authorization.attempt,
        covered=authorization.covered,
        disposition=authorization.disposition,
    )
    store = _Capturing()
    for candidate in (
        None,
        authorization.disposition,
        attempt_authorization,
        lookalike,
        dataclasses.replace(authorization),
    ):
        with pytest.raises(_Unverified):
            store.write_generation(candidate)
    for candidate in (
        None,
        authorization,
        authorization.attempt,
        dataclasses.replace(attempt_authorization),
    ):
        with pytest.raises(_Unverified):
            store.write_attempt(candidate)
    assert store.writes == 0 and store.attempts == {}


def test_d_an_inconsistent_authorization_cannot_even_be_constructed() -> None:
    ports_, _, authorization = _issued()
    other = equity_recommendation_attempt(ei4d._unresolved(ports_))
    with pytest.raises(_Unverified, match="attempt of this disposition"):
        VerifiedEquityRecommendationGenerationWrite(
            authorization.verified, authorization.policy, other
        )
    with pytest.raises(_Unverified):
        VerifiedEquityRecommendationGenerationWrite(
            authorization.verified,
            "verified",
            authorization.attempt,  # type: ignore[arg-type]
        )
    with pytest.raises(_Unverified, match="written with its disposition"):
        VerifiedEquityRecommendationAttemptWrite(authorization.attempt)


# ----------------------------------------------------------------------
# E. Attempt-only authorization
# ----------------------------------------------------------------------


def test_e_a_genuine_attempt_only_result_is_authorized_without_policy_evidence() -> None:
    ports_ = ei4c._ports()
    ports_.publish = False  # no policy is published: an attempt-only result needs none
    audit = ports_.run()
    assert equity_recommendation_disposition(audit) is None
    assert audit.reasons and audit.status.value == "DEPENDENCY_UNAVAILABLE"
    store = _Capturing()
    loads = ports_.catalog.loads
    result = ei4d._recorder(ports_, store).execute(audit)
    assert (result.outcome, result.attempt_recorded) == (_O.COMMITTED, True)
    (authorization,) = store.authorizations
    assert isinstance(authorization, VerifiedEquityRecommendationAttemptWrite)
    assert require_verified_equity_recommendation_attempt_write(authorization) is authorization
    attempt = equity_recommendation_attempt(audit)
    assert authorization.attempt == attempt
    assert authorization.covered == (attempt.attempt_identity, attempt.attempt_document)
    assert store.writes == 1 and store.dispositions == {}
    assert ports_.catalog.loads == loads  # recording consulted no policy evidence


def test_e_attempt_and_generation_authorizations_are_not_interchangeable() -> None:
    _, _, generation = _issued()
    _, attempt = _issued_attempt()
    store = _Capturing()
    with pytest.raises(_Unverified):
        store.write_generation(attempt)
    with pytest.raises(_Unverified):
        store.write_attempt(generation)
    assert store.writes == 0


def test_e_a_fabricated_or_changed_attempt_write_is_refused() -> None:
    ports_, authorization = _issued_attempt()
    fabricated = VerifiedEquityRecommendationAttemptWrite(authorization.attempt)  # never issued
    store = _Capturing()
    with pytest.raises(_Unverified, match="not issued"):
        store.write_attempt(fabricated)
    other = equity_recommendation_attempt(
        ports_.run(listing_eligibility_version=ei4c._v("eligibility/other-unknown"))
    )
    assert other.attempt_identity != authorization.attempt.attempt_identity
    _tamper(authorization, "attempt", other)
    with pytest.raises(_Unverified, match="differs"):
        store.write_attempt(authorization)
    assert store.writes == 0 and store.attempts == {}


# ----------------------------------------------------------------------
# G. Bounded reconciliation reuses the one authorization
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("script", "outcome", "writes"),
    [
        ([], _O.COMMITTED, 1),
        (["lost_after"], _O.EQUIVALENT_COMMITTED, 1),
        (["lost_before"], _O.COMMITTED, 2),
        (["lost_before"] * 3, _O.RETRY_LIMIT_REACHED, 3),
    ],
)
def test_g_retries_reuse_the_same_authorization_within_the_bound(
    script: list[str], outcome: EquityRecommendationRecordingOutcome, writes: int
) -> None:
    ports_, audit = ei4c._proposed()
    store = _Capturing()
    store.script = list(script)
    issued_before = len(authorization_module._ISSUED)
    result = ei4d._recorder(ports_, store).execute(audit)
    assert (result.outcome, result.writes) == (outcome, writes)
    assert len(store.authorizations) == writes
    assert len({id(a) for a in store.authorizations}) == 1  # one authorization, reused
    assert len(authorization_module._ISSUED) <= issued_before + 1


def test_g_a_conflicting_stored_record_is_unchanged_by_the_boundary() -> None:
    ports_, audit = ei4c._proposed()
    store = _Capturing()
    assert ei4d._recorder(ports_, store).execute(ei4d._unresolved(ports_)).is_stored
    result = ei4d._recorder(ports_, store).execute(audit)
    assert result.outcome is _O.CONTENT_CONFLICT and not result.is_stored


def test_g_a_retry_whose_content_changed_is_refused() -> None:
    ports_, audit = ei4c._proposed()

    class _MutatingBetweenWrites(_Capturing):
        def write_generation(self, authorization):  # type: ignore[no-untyped-def]
            if self.authorizations:  # the retry: its decoded content was changed meanwhile
                authorization.disposition.assessment.document["reasons"] = [
                    "CORE_FORMATION_REJECTED"
                ]
            return super().write_generation(authorization)

    store = _MutatingBetweenWrites()
    store.script = ["lost_before"]
    with pytest.raises(_Unverified):
        ei4d._recorder(ports_, store).execute(audit)
    assert store.writes == 1  # only the first, valid write passed validation
    assert store.dispositions == {}


# ----------------------------------------------------------------------
# H. Read-only consumers need only the reader
# ----------------------------------------------------------------------


class _Reader(EquityRecommendationGenerationRecordReader):
    """A read-only port double: it has no write method at all."""

    def __init__(self, source: ei4d._Store) -> None:
        self._source = source

    def load_disposition(self, generation_key: str):  # type: ignore[no-untyped-def]
        return self._source.dispositions.get(generation_key)

    def load_attempt(self, attempt_identity: str):  # type: ignore[no-untyped-def]
        return self._source.attempts.get(attempt_identity)


def test_h_historical_verification_and_audit_take_a_reader_and_issue_nothing() -> None:
    ports_, store, authorization = _issued()
    reader = _Reader(store)
    assert not hasattr(reader, "write_generation") and not hasattr(reader, "record_generation")
    issued = len(authorization_module._ISSUED)
    key = authorization.disposition.generation_key
    verified = VerifyStoredEquityRecommendationGenerationUseCase(
        reader, ei4d._verifier(ports_)
    ).execute(key)
    assert verified is not None and verified.disposition == authorization.disposition
    ports_.catalog.load_failure = "unavailable"  # replay never needs the catalog
    assert (
        VerifyStoredEquityRecommendationGenerationUseCase(reader, ei4d._verifier(ports_)).execute(
            key
        )
        is not None
    )
    ports_.catalog.load_failure = None
    audit = AuditEquityRecommendationPolicyBindingsUseCase(reader, ports_.catalog).execute([key])
    assert len(audit.records) == 1
    assert len(authorization_module._ISSUED) == issued  # reads authorize nothing
    with pytest.raises(TypeError, match="RecordReader"):
        VerifyStoredEquityRecommendationGenerationUseCase(object(), ei4d._verifier(ports_))  # type: ignore[arg-type]


# ----------------------------------------------------------------------
# Boundaries and public API
# ----------------------------------------------------------------------


def test_only_the_recorder_issues_and_nothing_imports_infrastructure() -> None:
    package = inspect.getsourcefile(application_services)
    assert package is not None
    root = pathlib.Path(package).parents[1]
    issuers = {"_issue_generation_write", "_issue_attempt_write", "_ISSUED"}
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {
            alias.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for alias in n.names
        }
        modules = {n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
        assert not any("infrastructure" in m or "sqlite" in m for m in modules), path
        if path.name not in (
            "equity_recommendation_generation_write_authorization.py",
            "record_equity_recommendation_generation.py",
        ):
            assert not names & issuers, path


def test_the_w1_contracts_are_exported_and_the_issuer_is_not() -> None:
    for name in (
        "VerifiedEquityRecommendationGenerationWrite",
        "VerifiedEquityRecommendationAttemptWrite",
        "require_verified_equity_recommendation_generation_write",
        "require_verified_equity_recommendation_attempt_write",
    ):
        assert name in application_services.__all__, name
    for name in (
        "EquityRecommendationGenerationRecordReader",
        "EquityRecommendationGenerationRecordWriter",
        "UnverifiedEquityRecommendationGenerationWriteError",
    ):
        assert name in ports.__all__, name
    exported = set(application_services.__all__) | set(ports.__all__)
    assert not {n for n in exported if "issue" in n.lower() and n.startswith("_")}
    assert "_issue_generation_write" not in exported and "_ISSUED" not in exported
