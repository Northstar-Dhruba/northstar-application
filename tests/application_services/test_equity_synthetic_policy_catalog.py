"""EI4-D10-S1: the Application policy-catalog contracts (ADR-022).

Contract tests for the immutable synthetic policy bindings, the catalog port,
explicit publication with bounded reconciliation, the catalog-backed reference
resolver and the read-only historical binding audit.

``_Catalog`` is a deterministic in-memory test double of the catalog port. It
stores canonical text and rebuilds every binding with ``from_stored`` on load,
as an adapter must. Its scripted failures exercise the Application's
semantics; nothing here establishes SQLite durability, atomicity or
cross-process correctness, which are S2's. Synthetic policies and stored
dispositions come from the existing EI-4C test helpers. Nothing is published
by a read, and nothing touches a network, clock or broker.
"""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import inspect
import json
from datetime import date, timedelta
from decimal import Decimal
from typing import Any

import pytest
import test_generate_equity_trade_recommendation as ei4c
from northstar_core.recommendations.equity import (
    EquityPriceIncrement,
    EquitySyntheticVersion,
    InvalidEquitySyntheticVersionError,
)

from northstar_application import application_services, ports
from northstar_application.application_services import (
    EQUITY_RECOMMENDATION_DOCUMENT_PREFIXES,
    EQUITY_SYNTHETIC_POLICY_COMPONENT_TYPES,
    EQUITY_SYNTHETIC_POLICY_DEFINITION_FORMAT,
    EQUITY_SYNTHETIC_POLICY_DEFINITION_PREFIX,
    EQUITY_SYNTHETIC_POLICY_NAMESPACE,
    EQUITY_SYNTHETIC_POLICY_PUBLICATION_WRITES,
    AuditEquityRecommendationPolicyBindingsUseCase,
    CatalogEquitySyntheticRecommendationReferences,
    EquityRecommendationGenerationDisposition,
    EquityRecommendationGenerationVerifier,
    EquityRecommendationPolicyBindingEvidence,
    EquityRecommendationPolicyBindingStatus,
    EquityResolvedSyntheticPolicySet,
    EquitySyntheticPolicyBindingIdentity,
    EquitySyntheticPolicyBundle,
    EquitySyntheticPolicyDefinitionBinding,
    EquitySyntheticPolicyPublicationOutcome,
    EquitySyntheticPolicyPublicationResult,
    EquitySyntheticPolicySetBinding,
    EquitySyntheticPolicyType,
    GenerateEquityTradeRecommendationUseCase,
    IncompatibleEquityRecommendationRecordError,
    IncompatibleEquitySyntheticPolicyBindingError,
    InvalidEquityRecommendationRecordError,
    InvalidEquitySyntheticPolicyBindingError,
    PublishEquitySyntheticPoliciesUseCase,
    RecordEquityRecommendationGenerationUseCase,
    canonical_equity_recommendation_json,
    equity_recommendation_disposition,
)
from northstar_application.application_services import (
    audit_equity_recommendation_policy_bindings as audit_module,
)
from northstar_application.application_services import (
    catalog_equity_synthetic_recommendation_references as resolver_module,
)
from northstar_application.application_services import (
    equity_synthetic_policy_bindings as bindings_module,
)
from northstar_application.application_services import (
    publish_equity_synthetic_policies as publish_module,
)
from northstar_application.ports import (
    EquityRecommendationGenerationStore,
    EquityRecommendationGenerationStoreUnavailableError,
    EquitySyntheticPolicyCatalogOutcome,
    EquitySyntheticPolicyCatalogUnavailableError,
    EquitySyntheticPolicyCatalogWrite,
    EquitySyntheticPolicyPublicationHistory,
    EquitySyntheticPolicyReferenceConflictError,
    EquitySyntheticRecommendationReferenceSource,
    EquitySyntheticReferenceUnavailableError,
    InvalidEquitySyntheticPolicyCatalogWriteError,
)
from northstar_application.ports import (
    equity_synthetic_policy_catalog_store as catalog_port_module,
)

_T = EquitySyntheticPolicyType
_P = EquitySyntheticPolicyPublicationOutcome
_A = EquityRecommendationPolicyBindingStatus
_W = EquitySyntheticPolicyCatalogOutcome
_NS = EQUITY_SYNTHETIC_POLICY_NAMESPACE
_v = ei4c._v
_Binding = EquitySyntheticPolicyDefinitionBinding | EquitySyntheticPolicySetBinding


def _resolved(
    set_version: str = "set/1",
    *,
    generation: Any = None,
    increment: Any = None,
    timing: Any = None,
    **labels: Any,
) -> EquityResolvedSyntheticPolicySet:
    """A resolved synthetic policy set from the EI-4C helpers (ei4c labels)."""
    return EquityResolvedSyntheticPolicySet(
        reference=ei4c._policy_set(set_version, **labels),
        generation_policy=generation or ei4c._generation(),
        risk_policy=ei4c._risk(),
        cost_schedule=ei4c._costs(),
        price_increment=increment or ei4c._increment(),
        timing_policy=timing or ei4c._timing_policy(),
    )


def _bundle(*args: Any, **kwargs: Any) -> EquitySyntheticPolicyBundle:
    return EquitySyntheticPolicyBundle.of(_resolved(*args, **kwargs))


def _relabelled_timing_bundle() -> EquitySyntheticPolicyBundle:
    """The same set and timing labels, with other timing content (the EI-4E AA-1 case)."""
    return _bundle(timing=ei4c._timing_policy(latency=timedelta(minutes=4)))


# The catalog test double lives with the EI-4C helpers, which the generator,
# the recorder and these tests share (EI4-D10-S3 wires it into both).
_Catalog = ei4c._Catalog


def _publisher(catalog: _Catalog, max_writes: int = 3) -> PublishEquitySyntheticPoliciesUseCase:
    return PublishEquitySyntheticPoliciesUseCase(catalog, max_writes)


# ----------------------------------------------------------------------
# S1-A: identity and canonical eipd1 bindings
# ----------------------------------------------------------------------

# The eipd1 digests of the EI-4C helper policies, pinned (prefix "eipd1-sha256:").
_GOLDEN_EIPD1 = {
    _T.GENERATION_POLICY: "41ce7ac08b4801dc18c2cf222475de1bb621c56c3911eb0988adbc61bba2aa52",
    _T.RISK_POLICY: "ea0f0b2cb49664bbc4777c2a88bf42979b890dfc60e05f5e83bb87c1d75fb492",
    _T.COST_SCHEDULE: "ebf2494cd9c339048051947b4cf113a83e6db40b5ce0aa73940207798cad898b",
    _T.PRICE_INCREMENT: "3d313451de5c2780838eac9a39f50c17d14cd8751c55cb93b311587234c6e9b0",
    _T.TIMING_POLICY: "56df4e188a7bad26f379c8c040f23881f855870f75ec96fe3a0bd479b1b964ca",
}
_GOLDEN_EIRS1 = "eirs1-sha256:7602c7204df699f0c18eed863b11e835d0eb1ecb720ab6ee979b9b950e959986"
_GOLDEN_EIRK1 = "eirk1-sha256:4161d4d3fa27fcf1ceffb413a2428e04cfa9bb8154aad02935778cc217a1ece6"


def test_eipd1_bindings_are_deterministic_canonical_and_pinned() -> None:
    first, again = _bundle(), _bundle()  # fresh objects throughout
    assert first == again
    assert EQUITY_SYNTHETIC_POLICY_DEFINITION_FORMAT == (
        "northstar.equity-synthetic-policy-definition/1"
    )
    assert EQUITY_SYNTHETIC_POLICY_DEFINITION_PREFIX == "eipd1-sha256:"
    assert EQUITY_SYNTHETIC_POLICY_COMPONENT_TYPES == tuple(_T)[:5]
    for binding in first.components:
        assert binding.fingerprint == "eipd1-sha256:" + _GOLDEN_EIPD1[binding.policy_type]
        document = binding.document()
        assert set(document) == {"format", "namespace", "policy_type", "version", "definition"}
        assert document["definition"] == binding.definition.document()  # existing canonical doc
        assert binding.canonical_json() == canonical_equity_recommendation_json(document)
        assert json.loads(binding.canonical_json()) == document
    increment = first.component(_T.PRICE_INCREMENT)
    assert increment.canonical_json() == (
        '{"definition":{"amount":"0.05","version":"synthetic-ei4c-increment/1"},'
        '"format":"northstar.equity-synthetic-policy-definition/1",'
        '"namespace":"northstar.equity-intraday-v1.synthetic",'
        '"policy_type":"PRICE_INCREMENT","version":"synthetic-ei4c-increment/1"}'
    )


def test_existing_eirs1_and_eirk1_are_unchanged() -> None:
    _, audit = ei4c._proposed()
    assert audit.inputs is not None
    assert audit.inputs.policy_set.fingerprint == _GOLDEN_EIRS1
    assert audit.generation_key == _GOLDEN_EIRK1
    bundle = EquitySyntheticPolicyBundle.of(audit.inputs.policy_set)
    assert bundle.policy_set.policy_set_fingerprint == _GOLDEN_EIRS1  # reused, never recomputed
    assert bundle.resolved == audit.inputs.policy_set
    prefixes = dict(EQUITY_RECOMMENDATION_DOCUMENT_PREFIXES)
    assert prefixes.pop(EQUITY_SYNTHETIC_POLICY_DEFINITION_FORMAT) == "eipd1-sha256:"
    assert prefixes == {  # every existing format keeps its prefix
        "northstar.equity-recommendation-policy-set/1": "eirs1-sha256:",
        "northstar.equity-recommendation-generation-inputs/1": "eiri1-sha256:",
        "northstar.equity-recommendation-assessment/1": "eira1-sha256:",
        "northstar.equity-recommendation-proposal/1": "eirp1-sha256:",
        "northstar.equity-recommendation-generation-attempt/1": "eirat1-sha256:",
    }


def test_identity_is_namespace_type_and_label_never_global() -> None:
    generation = EquitySyntheticPolicyDefinitionBinding.of(ei4c._generation("shared/1"))
    risk = EquitySyntheticPolicyDefinitionBinding.of(ei4c._risk("shared/1"))
    assert generation.version == risk.version  # one label, two types
    assert generation.identity != risk.identity  # two independent identities
    assert generation.identity == EquitySyntheticPolicyBindingIdentity(
        _NS, _T.GENERATION_POLICY, _v("shared/1")
    )
    assert generation.identity.document() == {
        "namespace": _NS,
        "policy_type": "GENERATION_POLICY",
        "version": "synthetic-ei4c-shared/1",
    }


def test_same_identity_equal_content_is_equivalent_and_other_content_conflicts() -> None:
    a = EquitySyntheticPolicyDefinitionBinding.of(ei4c._generation("generation/1"))
    b = EquitySyntheticPolicyDefinitionBinding.of(ei4c._generation("generation/1"))
    c = EquitySyntheticPolicyDefinitionBinding.of(ei4c._generation("generation/1", minimum="2"))
    assert a == b and a.fingerprint == b.fingerprint
    assert a.identity == c.identity and a != c and a.fingerprint != c.fingerprint


def test_different_labels_may_hold_identical_content() -> None:
    one = EquitySyntheticPolicyDefinitionBinding.of(ei4c._generation("generation/1"))
    two = EquitySyntheticPolicyDefinitionBinding.of(ei4c._generation("generation/copy"))

    def numbers(binding: EquitySyntheticPolicyDefinitionBinding) -> dict[str, Any]:
        return {k: v for k, v in binding.definition.document().items() if k != "version"}

    assert numbers(one) == numbers(two)  # identical policy content under two labels
    assert one.identity != two.identity and one.fingerprint != two.fingerprint  # both valid


def test_canonical_decimals_give_one_binding() -> None:
    plain = EquityPriceIncrement(Decimal("0.05"), _v("increment/1"))
    padded = EquityPriceIncrement(Decimal("0.0500"), _v("increment/1"))
    assert EquitySyntheticPolicyDefinitionBinding.of(plain) == (
        EquitySyntheticPolicyDefinitionBinding.of(padded)
    )


def test_one_price_increment_label_binds_one_amount() -> None:
    five = EquitySyntheticPolicyDefinitionBinding.of(ei4c._increment())
    ten = EquitySyntheticPolicyDefinitionBinding.of(
        EquityPriceIncrement(Decimal("0.10"), _v("increment/1"))
    )
    assert five.identity == ten.identity and five != ten  # a second amount is a conflict


def test_the_record_type_states_its_namespace_but_publication_is_approved_only() -> None:
    other = EquitySyntheticPolicyBundle.of(_resolved(), namespace="northstar.test.other")
    approved = _bundle()
    for a, b in zip(other.components, approved.components, strict=True):
        assert a.identity != b.identity and a.fingerprint != b.fingerprint
    catalog = _Catalog()
    with pytest.raises(InvalidEquitySyntheticPolicyBindingError, match="approved namespace"):
        _publisher(catalog).execute(other)
    assert catalog.publishes == 0


@pytest.mark.parametrize(
    "label", ["synthetic-ei4c-x/1 ", " synthetic-ei4c-x/1", "ei4c-x/1", "synthetic-"]
)
def test_invalid_labels_are_refused_and_never_rewritten(label: str) -> None:
    with pytest.raises(InvalidEquitySyntheticVersionError):
        EquitySyntheticVersion(label)


def test_labels_are_case_sensitive() -> None:
    lower = EquitySyntheticPolicyDefinitionBinding.of(ei4c._generation("case/1"))
    upper = EquitySyntheticPolicyDefinitionBinding.of(ei4c._generation("CASE/1"))
    assert lower.identity != upper.identity
    assert upper.version.value == "synthetic-ei4c-CASE/1"


@pytest.mark.parametrize(
    "build",
    [
        lambda b: EquitySyntheticPolicyBindingIdentity("", _T.RISK_POLICY, b.version),
        lambda b: EquitySyntheticPolicyBindingIdentity(" ns", _T.RISK_POLICY, b.version),
        lambda b: EquitySyntheticPolicyBindingIdentity(_NS, "BOGUS", b.version),
        lambda b: EquitySyntheticPolicyBindingIdentity(_NS, _T.RISK_POLICY, "synthetic-x"),
        lambda b: EquitySyntheticPolicyDefinitionBinding(
            _NS, _T.RISK_POLICY, b.definition, b.fingerprint
        ),
        lambda b: EquitySyntheticPolicyDefinitionBinding(
            _NS, _T.POLICY_SET, b.definition, b.fingerprint
        ),
        lambda b: EquitySyntheticPolicyDefinitionBinding(
            _NS, _T.GENERATION_POLICY, b.definition, "eipd1-sha256:" + "0" * 64
        ),
        lambda b: EquitySyntheticPolicyDefinitionBinding.of(object()),  # type: ignore[arg-type]
    ],
    ids=[
        "empty namespace",
        "spaced namespace",
        "unknown type",
        "label not a version",
        "type against definition",
        "set as definition",
        "fingerprint mismatch",
        "not a component",
    ],
)
def test_invalid_identities_and_bindings_are_refused(build: Any) -> None:
    binding = EquitySyntheticPolicyDefinitionBinding.of(ei4c._generation())
    with pytest.raises(InvalidEquitySyntheticPolicyBindingError):
        build(binding)


def test_a_stored_binding_rebuilds_exactly() -> None:
    for binding in _bundle().components:
        assert (
            EquitySyntheticPolicyDefinitionBinding.from_stored(
                binding.canonical_json(), binding.fingerprint
            )
            == binding
        )


def _restamped(document: dict[str, Any]) -> tuple[str, str]:
    """A canonical text with a fingerprint that verifies, for format tests."""
    text = canonical_equity_recommendation_json(document)
    prefix = EQUITY_RECOMMENDATION_DOCUMENT_PREFIXES.get(document.get("format"))
    return text, (prefix or "eipd1-sha256:") + hashlib.sha256(text.encode()).hexdigest()


def _stored_variants() -> dict[str, tuple[tuple[str, str], type[Exception]]]:
    binding = EquitySyntheticPolicyDefinitionBinding.of(ei4c._increment())
    text, fp = binding.canonical_json(), binding.fingerprint
    doc = binding.document()
    v2 = doc | {"format": "northstar.equity-synthetic-policy-definition/2"}
    invalid, incompatible = (
        InvalidEquitySyntheticPolicyBindingError,
        IncompatibleEquitySyntheticPolicyBindingError,
    )
    return {
        "unsupported version": ((canonical_equity_recommendation_json(v2), fp), incompatible),
        "foreign format": (
            _restamped(doc | {"format": "northstar.equity-recommendation-policy-set/1"}),
            invalid,
        ),
        "non-canonical text": ((json.dumps(doc, indent=1, sort_keys=True), fp), invalid),
        "fingerprint mismatch": ((text, "eipd1-sha256:" + "0" * 64), invalid),
        "extra field": (_restamped(doc | {"note": "x"}), invalid),
        "non-canonical decimal": (
            _restamped(doc | {"definition": {**doc["definition"], "amount": "0.050"}}),
            invalid,
        ),
        "label against definition": (
            _restamped(doc | {"version": "synthetic-ei4c-increment/2"}),
            invalid,
        ),
        "not json": (("{", fp), invalid),
        "float": (('{"amount":0.05}', fp), invalid),
    }


@pytest.mark.parametrize("case", list(_stored_variants()))
def test_a_stored_binding_that_does_not_verify_fails_closed(case: str) -> None:
    (text, fingerprint), error = _stored_variants()[case]
    with pytest.raises(error) as raised:
        EquitySyntheticPolicyDefinitionBinding.from_stored(text, fingerprint)
    if error is InvalidEquitySyntheticPolicyBindingError:
        assert not isinstance(raised.value, IncompatibleEquitySyntheticPolicyBindingError)


def test_a_policy_set_binding_reuses_the_existing_reference_and_eirs1() -> None:
    bundle = _bundle()
    stored = bundle.policy_set
    assert stored.identity == EquitySyntheticPolicyBindingIdentity(_NS, _T.POLICY_SET, _v("set/1"))
    assert stored.reference == ei4c._policy_set()
    assert stored.policy_set_fingerprint == bundle.resolved.fingerprint
    assert (
        EquitySyntheticPolicySetBinding.from_stored(
            _NS, stored.reference_json(), stored.policy_set_fingerprint
        )
        == stored
    )
    assert stored.component_identity(_T.TIMING_POLICY).version == _v("timing/1")
    for text, fp in (
        (stored.reference_json(), "eirk1-sha256:" + "0" * 64),
        (json.dumps(stored.reference.document(), indent=1), stored.policy_set_fingerprint),
        ("{}", stored.policy_set_fingerprint),
    ):
        with pytest.raises(InvalidEquitySyntheticPolicyBindingError):
            EquitySyntheticPolicySetBinding.from_stored(_NS, text, fp)


def test_a_bundle_holds_five_consistent_components_and_their_set() -> None:
    bundle = _bundle()
    components, policy_set = bundle.components, bundle.policy_set
    other = _bundle(timing=ei4c._timing_policy(latency=timedelta(minutes=4)))
    for parts in (
        (components[::-1], policy_set),  # wrong order
        (components[:4], policy_set),  # a component missing
        ((*components[:4], other.components[4]), policy_set),  # set eirs1 disagrees
        (components, other.policy_set),  # set eirs1 disagrees
        (  # a label the set does not name
            (
                *components[:4],
                EquitySyntheticPolicyDefinitionBinding.of(ei4c._timing_policy("t/9")),
            ),
            policy_set,
        ),
        (  # another namespace than the set
            components,
            EquitySyntheticPolicySetBinding(
                "northstar.test.other", policy_set.reference, policy_set.policy_set_fingerprint
            ),
        ),
    ):
        with pytest.raises(InvalidEquitySyntheticPolicyBindingError):
            EquitySyntheticPolicyBundle(*parts)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        EquitySyntheticPolicyBundle.of(object())  # type: ignore[arg-type]


# ----------------------------------------------------------------------
# S1-B, S1-C: the port and explicit publication
# ----------------------------------------------------------------------


def test_the_port_write_states_conflicts_exactly_for_conflict() -> None:
    binding = _bundle().policy_set
    with pytest.raises(InvalidEquitySyntheticPolicyCatalogWriteError):
        EquitySyntheticPolicyCatalogWrite(_W.CONFLICT, ())
    with pytest.raises(InvalidEquitySyntheticPolicyCatalogWriteError):
        EquitySyntheticPolicyCatalogWrite(_W.COMMITTED, (binding,))
    with pytest.raises(InvalidEquitySyntheticPolicyCatalogWriteError):
        EquitySyntheticPolicyCatalogWrite("COMMITTED", ())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="POLICY_REFERENCE_CONFLICT"):
        EquitySyntheticPolicyPublicationResult(
            _P.POLICY_REFERENCE_CONFLICT, _bundle(), (), 1, False, ""
        )


def test_first_publication_then_an_identical_one_is_idempotent() -> None:
    catalog = _Catalog()
    first = _publisher(catalog).execute(_bundle())
    assert (first.outcome, first.writes, first.reconciled) == (_P.PUBLISHED, 1, False)
    assert first.is_published and not first.is_permanent_conflict
    assert (len(catalog.definitions), len(catalog.sets)) == (5, 1)
    rows = catalog.rows()
    again = _publisher(catalog).execute(_bundle())
    assert (again.outcome, again.writes) == (_P.EQUIVALENT_PUBLISHED, 1)
    assert catalog.rows() == rows


def test_a_component_conflict_is_permanent_never_retried_and_writes_nothing() -> None:
    catalog = _Catalog()
    original = _bundle()
    _publisher(catalog).execute(original)
    rows = catalog.rows()
    result = _publisher(catalog).execute(_relabelled_timing_bundle())
    assert result.outcome is _P.POLICY_REFERENCE_CONFLICT and result.is_permanent_conflict
    assert not result.is_published and result.writes == 1 and catalog.publishes == 2
    assert {b.identity for b in result.conflicts} == {
        original.component(_T.TIMING_POLICY).identity,
        original.policy_set.identity,  # its eirs1 differs too
    }
    assert original.component(_T.TIMING_POLICY) in result.conflicts  # the stored one stays
    assert catalog.rows() == rows  # nothing overwritten, nothing republished


def test_a_set_conflict_writes_none_of_the_bundle() -> None:
    catalog = _Catalog()
    _publisher(catalog).execute(_bundle())
    rows = catalog.rows()
    other = _bundle(
        timing=ei4c._timing_policy("timing/2"), timing_policy_version=_v("timing/2")
    )  # same set label, another (new) component label
    result = _publisher(catalog).execute(other)
    assert result.outcome is _P.POLICY_REFERENCE_CONFLICT
    assert [b.identity for b in result.conflicts] == [other.policy_set.identity]
    assert catalog.rows() == rows  # the new timing/2 was not published either: all or nothing


def test_a_new_set_may_reuse_published_components() -> None:
    catalog = _Catalog()
    _publisher(catalog).execute(_bundle())
    result = _publisher(catalog).execute(_bundle("set/2"))
    assert result.outcome is _P.PUBLISHED
    assert (len(catalog.definitions), len(catalog.sets)) == (5, 2)


@pytest.mark.parametrize(
    ("mode", "outcome"),
    [
        ("unavailable", _P.CATALOG_UNAVAILABLE),
        ("rollback", _P.NOT_PUBLISHED),
        ("invalid", _P.STORED_BINDING_INVALID),
        ("incompatible", _P.STORED_BINDING_INCOMPATIBLE),
        ("contradicting", _P.STORED_BINDING_INVALID),
    ],
)
def test_failures_are_typed_and_never_retried(mode: str, outcome: Any) -> None:
    catalog = _Catalog()
    catalog.script = [mode]
    result = _publisher(catalog).execute(_bundle())
    assert (result.outcome, result.writes, catalog.publishes) == (outcome, 1, 1)
    assert not result.is_published and catalog.rows() == ({}, {})


def test_invalid_publication_input_fails_before_any_write() -> None:
    catalog = _Catalog()
    with pytest.raises(TypeError):
        _publisher(catalog).execute(_resolved())  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        PublishEquitySyntheticPoliciesUseCase(object())  # type: ignore[arg-type]
    for bad in (0, 4, True, "3"):
        with pytest.raises(ValueError, match="max_writes"):
            PublishEquitySyntheticPoliciesUseCase(catalog, bad)  # type: ignore[arg-type]
    assert EQUITY_SYNTHETIC_POLICY_PUBLICATION_WRITES == 3
    assert catalog.publishes == 0


# ----------------------------------------------------------------------
# Commit uncertainty and bounded reconciliation
# ----------------------------------------------------------------------


def test_a_lost_acknowledgment_reconciles_as_equivalent_by_reload() -> None:
    catalog = _Catalog()
    catalog.script = ["lost_after"]
    result = _publisher(catalog).execute(_bundle())
    assert (result.outcome, result.reconciled, result.writes) == (_P.EQUIVALENT_PUBLISHED, True, 1)
    assert catalog.publishes == 1 and catalog.loads == 6


def test_a_verified_absent_set_is_written_again_within_the_bound() -> None:
    catalog = _Catalog()
    catalog.script = ["lost_before"]
    result = _publisher(catalog).execute(_bundle())
    assert (result.outcome, result.writes, catalog.publishes) == (_P.PUBLISHED, 2, 2)


def test_components_that_predate_an_unconfirmed_write_are_not_partial() -> None:
    catalog = _Catalog()
    _publisher(catalog).execute(_bundle())  # set/1 and its components
    catalog.script = ["lost_before"]
    result = _publisher(catalog).execute(_bundle("set/2"))  # set/2 absent after reload
    assert (result.outcome, result.writes) == (_P.PUBLISHED, 2)


def test_a_competitor_s_conflicting_commit_found_on_reload_is_permanent() -> None:
    catalog = _Catalog()
    catalog.script = ["lost_before"]
    catalog.before_load = lambda: catalog.seed(_relabelled_timing_bundle())
    result = _publisher(catalog).execute(_bundle())
    assert (result.outcome, result.reconciled) == (_P.POLICY_REFERENCE_CONFLICT, True)
    assert catalog.publishes == 1  # a permanent conflict is never retried


def test_a_competitor_s_equivalent_commit_found_on_reload_is_equivalent() -> None:
    catalog = _Catalog()
    catalog.script = ["lost_before"]
    catalog.before_load = lambda: catalog.seed(_bundle())
    result = _publisher(catalog).execute(_bundle())
    assert (result.outcome, result.reconciled, catalog.publishes) == (
        _P.EQUIVALENT_PUBLISHED,
        True,
        1,
    )


def test_a_set_without_its_components_on_reload_is_unresolved() -> None:
    catalog = _Catalog()
    catalog.script = ["lost_before"]
    bundle = _bundle()

    def inconsistent() -> None:
        catalog.seed(bundle)
        del catalog.definitions[bundle.component(_T.RISK_POLICY).identity]

    catalog.before_load = inconsistent
    result = _publisher(catalog).execute(bundle)
    assert result.outcome is _P.PUBLICATION_RECONCILIATION_UNRESOLVED
    assert result.outcome is not _P.NOT_PUBLISHED and catalog.publishes == 1


@pytest.mark.parametrize("failure", ["unavailable", "invalid"])
def test_a_failed_reload_leaves_the_commit_unresolved(failure: str) -> None:
    catalog = _Catalog()
    catalog.script = ["lost_after"]
    catalog.load_failure = failure
    result = _publisher(catalog).execute(_bundle())
    assert (result.outcome, result.reconciled) == (_P.PUBLICATION_RECONCILIATION_UNRESOLVED, True)
    assert result.outcome is not _P.NOT_PUBLISHED and catalog.publishes == 1


@pytest.mark.parametrize("max_writes", [1, 2, 3])
def test_publication_writes_never_exceed_the_bound(max_writes: int) -> None:
    catalog = _Catalog()
    catalog.script = ["lost_before"] * 5
    result = _publisher(catalog, max_writes).execute(_bundle())
    assert (result.outcome, result.writes, catalog.publishes) == (
        _P.RETRY_LIMIT_REACHED,
        max_writes,
        max_writes,
    )
    assert catalog.rows() == ({}, {})


# ----------------------------------------------------------------------
# S1-D: the catalog-backed resolver
# ----------------------------------------------------------------------


def _published() -> tuple[_Catalog, EquitySyntheticPolicyBundle]:
    catalog, bundle = _Catalog(), _bundle()
    assert _publisher(catalog).execute(bundle).is_published
    return catalog, bundle


def test_the_resolver_answers_only_published_definitions() -> None:
    catalog, bundle = _published()
    resolver = CatalogEquitySyntheticRecommendationReferences(catalog)
    assert isinstance(resolver, EquitySyntheticRecommendationReferenceSource)
    assert resolver.namespace == _NS
    assert resolver.generation_policy(_v("generation/1")) == ei4c._generation()
    assert resolver.risk_policy(_v("risk/1")) == ei4c._risk()
    assert resolver.cost_schedule(_v("costs/1")) == ei4c._costs()
    assert resolver.timing_policy(_v("timing/1")) == ei4c._timing_policy()
    for listing, day in ((ei4c._LISTING, ei4c._DAY), (ei4c._OTHER, date(2026, 9, 1))):
        assert resolver.price_increment(_v("increment/1"), listing, day) == ei4c._increment()
    assert resolver.generation_policy(_v("generation/unknown")) is None  # never a default
    assert resolver.risk_policy(_v("generation/1")) is None  # a label is per type
    assert catalog.publishes == 1  # reads never publish


def test_an_unavailable_catalog_is_the_existing_outage_error() -> None:
    catalog, _ = _published()
    catalog.load_failure = "unavailable"
    with pytest.raises(EquitySyntheticReferenceUnavailableError):
        CatalogEquitySyntheticRecommendationReferences(catalog).timing_policy(_v("timing/1"))


def _corrupt(catalog: _Catalog, policy_type: Any, text: str, fp: str) -> None:
    identity = EquitySyntheticPolicyBindingIdentity(_NS, policy_type, _v("increment/1"))
    catalog.definitions[identity] = (text, fp)


@pytest.mark.parametrize("case", list(_stored_variants()))
def test_a_stored_binding_that_does_not_verify_is_a_permanent_conflict(case: str) -> None:
    catalog, _ = _published()
    (text, fp), error = _stored_variants()[case]
    _corrupt(catalog, _T.PRICE_INCREMENT, text, fp)
    resolver = CatalogEquitySyntheticRecommendationReferences(catalog)
    with pytest.raises(EquitySyntheticPolicyReferenceConflictError) as raised:
        resolver.price_increment(_v("increment/1"), ei4c._LISTING, ei4c._DAY)
    assert raised.value.reason == "POLICY_REFERENCE_CONFLICT"
    assert raised.value.incompatible is (error is IncompatibleEquitySyntheticPolicyBindingError)
    assert not isinstance(raised.value, EquitySyntheticReferenceUnavailableError)


def test_a_binding_for_another_identity_is_a_permanent_conflict() -> None:
    catalog, bundle = _published()
    other = EquitySyntheticPolicyDefinitionBinding.of(ei4c._timing_policy("timing/other"))
    identity = bundle.component(_T.TIMING_POLICY).identity
    catalog.definitions[identity] = (other.canonical_json(), other.fingerprint)
    with pytest.raises(EquitySyntheticPolicyReferenceConflictError, match="another binding"):
        CatalogEquitySyntheticRecommendationReferences(catalog).timing_policy(_v("timing/1"))


def test_the_resolver_refuses_bad_arguments() -> None:
    resolver = CatalogEquitySyntheticRecommendationReferences(_Catalog())
    with pytest.raises(TypeError):
        resolver.generation_policy("synthetic-ei4c-generation/1")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        resolver.price_increment(_v("increment/1"), "M&M", ei4c._DAY)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        CatalogEquitySyntheticRecommendationReferences(object())  # type: ignore[arg-type]


def test_the_unchanged_generator_accepts_the_resolver_as_its_reference_source() -> None:
    ports_, original = ei4c._proposed()
    catalog = _Catalog()
    assert original.inputs is not None
    bundle = EquitySyntheticPolicyBundle.of(original.inputs.policy_set)
    assert _publisher(catalog).execute(bundle).is_published
    ports_.references = CatalogEquitySyntheticRecommendationReferences(catalog)  # type: ignore[assignment]
    audit = ports_.run()
    assert audit.status is ei4c._S.PROPOSED
    assert audit.generation_key == original.generation_key == _GOLDEN_EIRK1
    assert audit.assessment_fingerprint == original.assessment_fingerprint
    assert catalog.publishes == 1  # generation never publishes


def test_an_unpublished_catalog_gives_the_existing_unavailable_reasons() -> None:
    ports_, _ = ei4c._proposed()
    catalog = _Catalog()
    ports_.references = CatalogEquitySyntheticRecommendationReferences(catalog)  # type: ignore[assignment]
    audit = ports_.run()
    assert audit.status is ei4c._S.DEPENDENCY_UNAVAILABLE and audit.generation_key is None
    assert ei4c._R.TIMING_POLICY_UNAVAILABLE in audit.reasons
    assert catalog.publishes == 0 and catalog.rows() == ({}, {})


def test_a_resolver_conflict_is_the_generation_reason_policy_reference_conflict() -> None:
    # Until S3 this conflict propagated out of generation as an exception; the
    # S3 generator reports it as its typed DQ6 reason instead.
    ports_, original = ei4c._proposed()
    catalog = _Catalog()
    assert original.inputs is not None
    _publisher(catalog).execute(EquitySyntheticPolicyBundle.of(original.inputs.policy_set))
    (text, fp), _ = _stored_variants()["fingerprint mismatch"]
    _corrupt(catalog, _T.PRICE_INCREMENT, text, fp)
    ports_.references = CatalogEquitySyntheticRecommendationReferences(catalog)  # type: ignore[assignment]
    ports_.catalog = catalog
    audit = ports_.run()
    ei4c._refused(audit, ei4c._S.DEPENDENCY_UNAVAILABLE, ei4c._R.POLICY_REFERENCE_CONFLICT)
    assert audit.generation_key is None and audit.inputs is None  # attempt-only
    assert catalog.publishes == 1


# ----------------------------------------------------------------------
# S1-E: the read-only historical binding audit
# ----------------------------------------------------------------------


class _Records(EquityRecommendationGenerationStore):
    """Read-only test double of the generation store; any write is a test failure."""

    def __init__(self, *dispositions: EquityRecommendationGenerationDisposition) -> None:
        self.held = {d.generation_key: d for d in dispositions}
        self.failures: dict[str, Exception] = {}

    def load_disposition(self, generation_key: str):  # type: ignore[no-untyped-def]
        if generation_key in self.failures:
            raise self.failures[generation_key]
        return self.held.get(generation_key)

    def load_attempt(self, attempt_identity: str):  # type: ignore[no-untyped-def]
        raise AssertionError("the audit reads dispositions only")

    def record_generation(self, disposition, attempt):  # type: ignore[no-untyped-def]
        raise AssertionError("the audit never writes")

    def record_attempt(self, attempt):  # type: ignore[no-untyped-def]
        raise AssertionError("the audit never writes")


class _History(EquitySyntheticPolicyPublicationHistory):
    def __init__(self, answer: Any) -> None:
        self.answer = answer
        self.asked: list[tuple[Any, str, str]] = []

    def published_before(self, identity, fingerprint, generation_key):  # type: ignore[no-untyped-def]
        self.asked.append((identity, fingerprint, generation_key))
        if self.answer == "unavailable":
            raise EquitySyntheticPolicyCatalogUnavailableError("test: history down")
        return self.answer


def _stored() -> tuple[ei4c._Ports, EquityRecommendationGenerationDisposition]:
    ports_, audit = ei4c._proposed()
    disposition = equity_recommendation_disposition(audit)
    assert disposition is not None
    return ports_, disposition


def _audit(records: _Records, catalog: _Catalog, history: Any = None, *keys: str) -> Any:
    use_case = AuditEquityRecommendationPolicyBindingsUseCase(records, catalog, history)
    return use_case.execute(keys or tuple(records.held))


def _statuses(record: Any) -> set[Any]:
    return {f.status for f in record.findings}


def test_an_unpublished_policy_is_binding_not_established() -> None:
    _, disposition = _stored()
    result = _audit(_Records(disposition), _Catalog())
    (record,) = result.records
    assert len(record.findings) == 6 and _statuses(record) == {_A.BINDING_NOT_ESTABLISHED}
    assert record.status is _A.BINDING_NOT_ESTABLISHED and result.missing == ()


def test_a_matching_binding_today_is_never_historical_proof() -> None:
    _, disposition = _stored()
    catalog, _ = _published()
    (record,) = _audit(_Records(disposition), catalog).records
    assert _statuses(record) == {_A.BINDING_NOT_ESTABLISHED}
    assert all(f.catalog_fingerprint == f.recorded_fingerprint for f in record.findings)
    assert all("no historical evidence" in f.detail for f in record.findings)


@pytest.mark.parametrize(
    ("answer", "status"),
    [
        (True, _A.HISTORICALLY_VERIFIED_BINDING),
        (False, _A.BINDING_NOT_ESTABLISHED),
        (None, _A.BINDING_NOT_ESTABLISHED),
        ("unavailable", _A.AUDIT_EVIDENCE_UNAVAILABLE),
    ],
)
def test_only_recorded_historical_evidence_verifies_a_binding(answer: Any, status: Any) -> None:
    _, disposition = _stored()
    catalog, _ = _published()
    history = _History(answer)
    (record,) = _audit(_Records(disposition), catalog, history).records
    assert _statuses(record) == {status} and record.status is status
    assert len(history.asked) == 6
    assert {key for _, _, key in history.asked} == {disposition.generation_key}
    if status is _A.HISTORICALLY_VERIFIED_BINDING:
        assert {f.evidence for f in record.findings} == {
            EquityRecommendationPolicyBindingEvidence.PUBLICATION_HISTORY
        }


def test_the_catalog_binding_other_content_is_a_historical_conflict() -> None:
    _, disposition = _stored()
    catalog = _Catalog()
    catalog.seed(_relabelled_timing_bundle())
    (record,) = _audit(_Records(disposition), catalog).records
    conflicts = {
        f.identity.policy_type
        for f in record.findings
        if f.status is (_A.HISTORICAL_POLICY_VERSION_CONFLICT)
    }
    assert conflicts == {_T.TIMING_POLICY, _T.POLICY_SET}
    assert record.status is _A.HISTORICAL_POLICY_VERSION_CONFLICT


def test_one_label_with_two_contents_across_records_is_a_historical_conflict() -> None:
    _, first = _stored()
    ports_, _ = ei4c._proposed()
    ports_.references.hold("timing", ei4c._timing_policy(latency=timedelta(minutes=4)))
    # A record made before enforcement: no single catalog bound the label then, so
    # this one is generated against its own catalog. Under one catalog, EI4-D10-S3
    # refuses the reuse at generation (POLICY_REFERENCE_CONFLICT).
    ports_.catalog = ei4c._Catalog()
    second = equity_recommendation_disposition(ports_.run())
    assert second is not None and second.generation_key != first.generation_key
    result = _audit(_Records(first, second), _Catalog())
    for record in result.records:
        reused = {
            f.identity.policy_type
            for f in record.findings
            if f.evidence is EquityRecommendationPolicyBindingEvidence.OTHER_STORED_RECORDS
        }
        # The timing label holds two contents, so the set label holds two eirs1 values.
        assert reused == {_T.TIMING_POLICY, _T.POLICY_SET}
        assert record.status is _A.HISTORICAL_POLICY_VERSION_CONFLICT


@pytest.mark.parametrize(
    ("error", "status"),
    [
        (InvalidEquityRecommendationRecordError("test"), _A.INVALID_CANONICAL_EVIDENCE),
        (IncompatibleEquityRecommendationRecordError("test"), _A.INCOMPATIBLE_HISTORICAL_FORMAT),
        (
            EquityRecommendationGenerationStoreUnavailableError("test"),
            _A.AUDIT_EVIDENCE_UNAVAILABLE,
        ),
    ],
)
def test_a_stored_record_that_cannot_be_read_is_classified_never_assumed(
    error: Exception, status: Any
) -> None:
    _, disposition = _stored()
    records = _Records(disposition)
    records.failures[disposition.generation_key] = error
    (record,) = _audit(records, _Catalog()).records
    assert record.findings == () and record.status is status


@pytest.mark.parametrize(
    ("case", "status"),
    [
        ("fingerprint mismatch", _A.INVALID_CANONICAL_EVIDENCE),
        ("unsupported version", _A.INCOMPATIBLE_HISTORICAL_FORMAT),
    ],
)
def test_a_catalog_binding_that_does_not_verify_is_classified(case: str, status: Any) -> None:
    _, disposition = _stored()
    catalog, _ = _published()
    (text, fp), _ = _stored_variants()[case]
    _corrupt(catalog, _T.PRICE_INCREMENT, text, fp)
    (record,) = _audit(_Records(disposition), catalog).records
    increment = next(f for f in record.findings if f.identity.policy_type is _T.PRICE_INCREMENT)
    assert increment.status is status and record.status is status
    assert increment.evidence is EquityRecommendationPolicyBindingEvidence.CATALOG


def test_an_unavailable_catalog_is_an_explicit_uncertainty() -> None:
    _, disposition = _stored()
    catalog = _Catalog()
    catalog.load_failure = "unavailable"
    (record,) = _audit(_Records(disposition), catalog).records
    assert _statuses(record) == {_A.AUDIT_EVIDENCE_UNAVAILABLE}


def test_missing_records_are_listed_and_the_audit_writes_nothing() -> None:
    _, disposition = _stored()
    catalog, _ = _published()
    rows = catalog.rows()
    records = _Records(disposition)
    result = _audit(records, catalog, None, disposition.generation_key, "eirk1-sha256:" + "f" * 64)
    assert result.missing == ("eirk1-sha256:" + "f" * 64,)
    assert len(result.records) == 1
    assert catalog.publishes == 1 and catalog.rows() == rows  # only the setup publication
    assert records.held == {disposition.generation_key: disposition}
    with pytest.raises(TypeError):
        AuditEquityRecommendationPolicyBindingsUseCase(records, catalog).execute([1])  # type: ignore[list-item]
    with pytest.raises(TypeError):
        AuditEquityRecommendationPolicyBindingsUseCase(records, catalog, object())  # type: ignore[arg-type]


def test_historical_replay_stays_independent_of_the_catalog() -> None:
    ports_, disposition = _stored()
    catalog = _Catalog()
    catalog.load_failure = "unavailable"  # the catalog is down
    verifier = EquityRecommendationGenerationVerifier(
        ports_.world.timelines, ports_.world.snapshots
    )
    verified = verifier.verify(disposition)
    assert verified.evaluation_reproduced and not verified.issuance_established
    assert catalog.loads == 0
    assert "catalog" not in inspect.signature(EquityRecommendationGenerationVerifier).parameters


# ----------------------------------------------------------------------
# Regression and boundaries
# ----------------------------------------------------------------------


def test_generation_and_recording_constructors_require_the_catalog_since_s3() -> None:
    # S1 left both constructors unchanged; EI4-D10-S3 adds the catalog, required.
    generator = inspect.signature(GenerateEquityTradeRecommendationUseCase).parameters
    assert list(generator) == [
        "timelines",
        "snapshots",
        "venue_days",
        "references",
        "listing_eligibility",
        "session_history",
        "policy_catalog",
    ]
    recorder = inspect.signature(RecordEquityRecommendationGenerationUseCase).parameters
    assert list(recorder) == ["store", "verifier", "policy_catalog", "max_writes"]
    for parameters in (generator, recorder):
        assert parameters["policy_catalog"].default is inspect.Parameter.empty
    # Historical replay still never reads the catalog (ADR-022 section 9).
    source = inspect.getsource(application_services.verify_equity_recommendation_generation)
    assert "PolicyCatalog" not in source and "policy_bindings" not in source


def test_the_new_contracts_are_exported() -> None:
    for name in (
        "AuditEquityRecommendationPolicyBindingsUseCase",
        "CatalogEquitySyntheticRecommendationReferences",
        "EQUITY_SYNTHETIC_POLICY_DEFINITION_FORMAT",
        "EQUITY_SYNTHETIC_POLICY_NAMESPACE",
        "EquitySyntheticPolicyBindingIdentity",
        "EquitySyntheticPolicyBundle",
        "EquitySyntheticPolicyDefinitionBinding",
        "EquitySyntheticPolicySetBinding",
        "EquitySyntheticPolicyType",
        "PublishEquitySyntheticPoliciesUseCase",
        "GenerateEquityTradeRecommendationUseCase",  # existing exports remain
        "RecordEquityRecommendationGenerationUseCase",
        "EquityResolvedSyntheticPolicySet",
    ):
        assert name in application_services.__all__, name
    for name in (
        "EquitySyntheticPolicyCatalogStore",
        "EquitySyntheticPolicyCatalogWrite",
        "EquitySyntheticPolicyPublicationHistory",
        "EquitySyntheticPolicyReferenceConflictError",
        "EquitySyntheticRecommendationReferenceSource",
    ):
        assert name in ports.__all__, name
    assert "decode_equity_synthetic_policy_component" not in application_services.__all__
    assert dataclasses.is_dataclass(EquitySyntheticPolicyDefinitionBinding)
    with pytest.raises(dataclasses.FrozenInstanceError):
        _bundle().policy_set.policy_set_fingerprint = "x"  # type: ignore[misc]


_MODULES = [bindings_module, publish_module, resolver_module, audit_module, catalog_port_module]


@pytest.mark.parametrize("module", _MODULES, ids=lambda m: m.__name__.rsplit(".", 1)[-1])
def test_no_sqlite_infrastructure_clock_retry_loop_or_issuance(module) -> None:  # type: ignore[no-untyped-def]
    source = inspect.getsource(module)
    tree = ast.parse(source)
    imported = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    imported |= {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    for name in imported:
        assert name is not None
        assert name.split(".")[0] not in {
            "sqlite3",
            "sqlalchemy",
            "time",
            "random",
            "uuid",
            "threading",
            "asyncio",
            "socket",
            "os",
        }, name
        for forbidden in (
            "northstar_infrastructure",
            "http",
            "urllib",
            "broker",
            "futures",
            "paper",
        ):
            assert forbidden not in name, name
    assert not any(isinstance(node, ast.While) for node in ast.walk(tree))  # no unbounded loop
    calls = {
        getattr(node.func, "attr", None) for node in ast.walk(tree) if isinstance(node, ast.Call)
    }
    assert not calls & {"sleep", "record_generation", "record_attempt", "issue"}
