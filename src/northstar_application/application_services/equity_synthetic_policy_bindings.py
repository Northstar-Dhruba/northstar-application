"""Immutable synthetic policy-version bindings (EI4-D10, ADR-022).

EI4-D10 (EI-4 Decision Register, section 10.2, approved 2026-11-08): within a
defined policy namespace and policy type, a published policy-version label
identifies exactly one immutable canonical policy definition. Conflicting
reuse fails closed, and a legitimate change needs a new label. ADR-022 and the
approved design decisions EI4-D10-DQ1 to EI4-D10-DQ9 (section 11) fix how.

Identity (DQ1, DQ2)
-------------------
A binding is identified by ``(namespace, policy_type, version)``:

- the approved namespace is the one closed constant
  ``EQUITY_SYNTHETIC_POLICY_NAMESPACE``. The record types accept any
  well-formed namespace text, so a binding states its own namespace; only the
  publication use case and the catalog-backed resolver restrict it to the
  approved one;
- ``EquitySyntheticPolicyType`` has exactly six members: the five components
  of a composite policy set and ``POLICY_SET`` itself;
- ``version`` is an ``EquitySyntheticVersion``, kept exactly: case-sensitive,
  never trimmed, ``synthetic-``-prefixed (Core validates it).

Uniqueness is per identity, never global: one label under two types is two
independent bindings, and two labels may hold identical content.

Canonical documents (DQ3, ADR-022 section 3)
--------------------------------------------
A component binding is the canonical document

    {"format": "northstar.equity-synthetic-policy-definition/1",
     "namespace": ..., "policy_type": ..., "version": ...,
     "definition": <the component's existing canonical document()>}

fingerprinted ``eipd1-sha256:`` with the existing canonical encoder and
``equity_recommendation_fingerprint``. Decimals are Core-canonical, so equal
values give one document. A PRICE_INCREMENT label binds exactly one amount.

A POLICY_SET binding reuses what exists: the policy-set reference document
(the set label and its five component labels) and the existing ``eirs1``
fingerprint of the resolved set. Nothing here computes a replacement ``eirs1``
or touches ``eirk1``; no binding document enters a generation record.

``EquitySyntheticPolicyBundle`` is one complete publication: the five
component bindings and the set binding, consistent with each other. A stored
binding that does not construct raises InvalidEquitySyntheticPolicyBindingError,
or IncompatibleEquitySyntheticPolicyBindingError for an unsupported format.
Nothing here persists, reads a clock or calls any external system.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, NoReturn

from northstar_core.recommendations.equity import (
    EquityPriceIncrement,
    EquitySyntheticCostSchedule,
    EquitySyntheticGenerationPolicy,
    EquitySyntheticRiskPolicy,
    EquitySyntheticVersion,
)

from northstar_application.application_services.equity_recommendation_generation_decoding import (
    InvalidEquityRecommendationRecordError,
    decode_equity_policy_set_reference,
    decode_equity_synthetic_policy_component,
)
from northstar_application.application_services.equity_recommendation_generation_evidence import (
    EQUITY_RECOMMENDATION_DOCUMENT_PREFIXES,
    EQUITY_SYNTHETIC_POLICY_DEFINITION_FORMAT,
    EquityResolvedSyntheticPolicySet,
    EquitySyntheticPolicySetReference,
    InvalidEquityRecommendationDocumentError,
    InvalidEquityRecommendationEvidenceError,
    canonical_equity_recommendation_json,
    equity_recommendation_fingerprint,
    verify_equity_recommendation_document,
)
from northstar_application.ports.equity_recommendation_references import (
    EquitySyntheticEntryTimingPolicy,
)

EQUITY_SYNTHETIC_POLICY_NAMESPACE = "northstar.equity-intraday-v1.synthetic"
EQUITY_SYNTHETIC_POLICY_DEFINITION_PREFIX = EQUITY_RECOMMENDATION_DOCUMENT_PREFIXES[
    EQUITY_SYNTHETIC_POLICY_DEFINITION_FORMAT
]
_POLICY_SET_PREFIX = "eirs1-sha256:"
_HEX = frozenset("0123456789abcdef")
_DEFINITION_FIELDS = ("format", "namespace", "policy_type", "version", "definition")


class InvalidEquitySyntheticPolicyBindingError(ValueError):
    """Raised when a policy binding is malformed, inconsistent or does not verify."""


class IncompatibleEquitySyntheticPolicyBindingError(InvalidEquitySyntheticPolicyBindingError):
    """Raised when a stored binding uses a format this code does not support. Never migrated."""


def _invalid(message: str) -> NoReturn:
    raise InvalidEquitySyntheticPolicyBindingError(message)


class EquitySyntheticPolicyType(StrEnum):
    """The six approved policy types (EI4-D10-DQ2). The list is closed."""

    GENERATION_POLICY = "GENERATION_POLICY"
    RISK_POLICY = "RISK_POLICY"
    COST_SCHEDULE = "COST_SCHEDULE"
    PRICE_INCREMENT = "PRICE_INCREMENT"
    TIMING_POLICY = "TIMING_POLICY"
    POLICY_SET = "POLICY_SET"


_T = EquitySyntheticPolicyType
# Component type -> (definition class, key in the policy-set document and in
# EquityResolvedSyntheticPolicySet, version field of the policy-set reference).
_COMPONENTS: dict[EquitySyntheticPolicyType, tuple[type, str, str]] = {
    _T.GENERATION_POLICY: (
        EquitySyntheticGenerationPolicy,
        "generation_policy",
        "generation_policy_version",
    ),
    _T.RISK_POLICY: (EquitySyntheticRiskPolicy, "risk_policy", "risk_policy_version"),
    _T.COST_SCHEDULE: (EquitySyntheticCostSchedule, "cost_schedule", "cost_schedule_version"),
    _T.PRICE_INCREMENT: (EquityPriceIncrement, "price_increment", "price_increment_version"),
    _T.TIMING_POLICY: (
        EquitySyntheticEntryTimingPolicy,
        "timing_policy",
        "timing_policy_version",
    ),
}
EQUITY_SYNTHETIC_POLICY_COMPONENT_TYPES = tuple(_COMPONENTS)

EquitySyntheticPolicyDefinition = (
    EquitySyntheticGenerationPolicy
    | EquitySyntheticRiskPolicy
    | EquitySyntheticCostSchedule
    | EquityPriceIncrement
    | EquitySyntheticEntryTimingPolicy
)


def _namespace(value: object) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        _invalid("A policy namespace is non-empty text without surrounding whitespace.")
    return value


def _policy_type(value: object) -> EquitySyntheticPolicyType:
    if isinstance(value, EquitySyntheticPolicyType):
        return value
    try:
        return EquitySyntheticPolicyType(value)
    except ValueError:
        _invalid(f"{value!r} is not an approved synthetic policy type.")


def _type_of(definition: object) -> EquitySyntheticPolicyType:
    for policy_type, (kind, _, _) in _COMPONENTS.items():
        if type(definition) is kind:
            return policy_type
    _invalid(f"A {type(definition).__name__} is not a synthetic policy component.")


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EquitySyntheticPolicyBindingIdentity:
    """``(namespace, policy_type, version)``: the unique identity of a binding."""

    namespace: str
    policy_type: EquitySyntheticPolicyType
    version: EquitySyntheticVersion

    def __post_init__(self) -> None:
        _namespace(self.namespace)
        object.__setattr__(self, "policy_type", _policy_type(self.policy_type))
        if not isinstance(self.version, EquitySyntheticVersion):
            _invalid("A binding identity's version must be an EquitySyntheticVersion.")

    def document(self) -> dict[str, str]:
        """Return the canonical identity."""
        return {
            "namespace": self.namespace,
            "policy_type": self.policy_type.value,
            "version": self.version.value,
        }


# ---------------------------------------------------------------------------
# Component bindings
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EquitySyntheticPolicyDefinitionBinding:
    """One published component definition under its identity, and its ``eipd1`` fingerprint.

    Construction verifies everything: the type matches the definition, and
    ``fingerprint`` re-derives from the canonical binding document. Use
    ``of`` to bind a definition, and ``from_stored`` to rebuild a stored one.
    """

    namespace: str
    policy_type: EquitySyntheticPolicyType
    definition: EquitySyntheticPolicyDefinition
    fingerprint: str

    def __post_init__(self) -> None:
        _namespace(self.namespace)
        policy_type = _policy_type(self.policy_type)
        object.__setattr__(self, "policy_type", policy_type)
        if policy_type is _T.POLICY_SET:
            _invalid("A POLICY_SET is bound by EquitySyntheticPolicySetBinding.")
        if _type_of(self.definition) is not policy_type:
            _invalid(f"A {type(self.definition).__name__} is not a {policy_type.value}.")
        if not isinstance(self.fingerprint, str) or self.fingerprint != (
            equity_recommendation_fingerprint(self.document())
        ):
            _invalid("The binding fingerprint does not re-derive from its canonical document.")

    @classmethod
    def of(
        cls,
        definition: EquitySyntheticPolicyDefinition,
        *,
        namespace: str = EQUITY_SYNTHETIC_POLICY_NAMESPACE,
    ) -> EquitySyntheticPolicyDefinitionBinding:
        """Bind ``definition`` under its own type and version label in ``namespace``."""
        policy_type = _type_of(definition)
        document = _definition_document(namespace, policy_type, definition)
        return cls(namespace, policy_type, definition, equity_recommendation_fingerprint(document))

    @classmethod
    def from_stored(cls, text: str, fingerprint: str) -> EquitySyntheticPolicyDefinitionBinding:
        """Rebuild a stored binding exactly, or fail closed.

        Another version of the ``northstar.equity-synthetic-policy-definition``
        family is incompatible, never migrated. A non-canonical encoding, a
        fingerprint that does not re-derive, a missing or extra field, or a
        definition that does not rebuild exactly is invalid.
        """
        if not isinstance(text, str) or not isinstance(fingerprint, str):
            _invalid("A stored binding and its fingerprint are text.")
        try:
            loose = json.loads(text)
        except json.JSONDecodeError as exc:
            raise InvalidEquitySyntheticPolicyBindingError(
                "The stored binding is not JSON."
            ) from exc
        if not isinstance(loose, dict):
            _invalid("The stored binding is not a JSON object.")
        stated = loose.get("format")
        if stated != EQUITY_SYNTHETIC_POLICY_DEFINITION_FORMAT:
            family = EQUITY_SYNTHETIC_POLICY_DEFINITION_FORMAT.rsplit("/", 1)[0] + "/"
            if isinstance(stated, str) and stated.startswith(family):
                raise IncompatibleEquitySyntheticPolicyBindingError(
                    f"The stored binding is {stated}; only "
                    f"{EQUITY_SYNTHETIC_POLICY_DEFINITION_FORMAT} is supported. "
                    "Nothing is migrated."
                )
            _invalid(f"The stored binding has format {stated!r}.")
        try:
            document = verify_equity_recommendation_document(text, fingerprint)
        except InvalidEquityRecommendationDocumentError as exc:
            raise InvalidEquitySyntheticPolicyBindingError(f"The stored binding: {exc}") from exc
        if set(document) != set(_DEFINITION_FIELDS):
            _invalid("The stored binding has missing or unexpected fields.")
        policy_type = _policy_type(document["policy_type"])
        if policy_type is _T.POLICY_SET:
            _invalid("A POLICY_SET is not stored as a definition binding.")
        try:
            definition = decode_equity_synthetic_policy_component(
                _COMPONENTS[policy_type][1], document["definition"]
            )
        except InvalidEquityRecommendationRecordError as exc:
            raise InvalidEquitySyntheticPolicyBindingError(
                f"The stored {policy_type.value} definition: {exc}"
            ) from exc
        if document["version"] != definition.version.value:
            _invalid("The stored binding names another version than its definition.")
        binding = cls(document["namespace"], policy_type, definition, fingerprint)
        if binding.document() != document:
            _invalid("The stored binding is not in canonical form.")
        return binding

    @property
    def version(self) -> EquitySyntheticVersion:
        """Return the published version label."""
        return self.definition.version

    @property
    def identity(self) -> EquitySyntheticPolicyBindingIdentity:
        """Return ``(namespace, policy_type, version)``."""
        return EquitySyntheticPolicyBindingIdentity(self.namespace, self.policy_type, self.version)

    def document(self) -> dict[str, Any]:
        """Return the canonical binding document."""
        return _definition_document(self.namespace, self.policy_type, self.definition)

    def canonical_json(self) -> str:
        """Return the canonical text that ``fingerprint`` names."""
        return canonical_equity_recommendation_json(self.document())


def _definition_document(
    namespace: str, policy_type: EquitySyntheticPolicyType, definition: Any
) -> dict[str, Any]:
    return {
        "format": EQUITY_SYNTHETIC_POLICY_DEFINITION_FORMAT,
        "namespace": namespace,
        "policy_type": policy_type.value,
        "version": definition.version.value,
        "definition": definition.document(),
    }


# ---------------------------------------------------------------------------
# Policy-set bindings
# ---------------------------------------------------------------------------


def _is_policy_set_fingerprint(value: object) -> bool:
    if not isinstance(value, str) or not value.startswith(_POLICY_SET_PREFIX):
        return False
    digest = value[len(_POLICY_SET_PREFIX) :]
    return len(digest) == 64 and set(digest) <= _HEX


@dataclass(frozen=True, slots=True)
class EquitySyntheticPolicySetBinding:
    """A published composite set label: its five component labels and its ``eirs1``.

    It reuses the existing policy-set reference and ``eirs1`` fingerprint; it
    computes no new set identity. Whether ``policy_set_fingerprint`` is the
    fingerprint of the bound components is verified with them, by
    EquitySyntheticPolicyBundle.
    """

    namespace: str
    reference: EquitySyntheticPolicySetReference
    policy_set_fingerprint: str

    def __post_init__(self) -> None:
        _namespace(self.namespace)
        if not isinstance(self.reference, EquitySyntheticPolicySetReference):
            _invalid("A policy-set binding needs an EquitySyntheticPolicySetReference.")
        if not _is_policy_set_fingerprint(self.policy_set_fingerprint):
            _invalid("A policy-set binding names an eirs1-sha256 fingerprint.")

    @classmethod
    def from_stored(
        cls, namespace: str, reference_text: str, policy_set_fingerprint: str
    ) -> EquitySyntheticPolicySetBinding:
        """Rebuild a stored set binding from its canonical reference text, or fail closed."""
        if not isinstance(reference_text, str):
            _invalid("A stored policy-set reference is text.")
        try:
            loose = json.loads(reference_text)
            reference = decode_equity_policy_set_reference(loose)
        except (json.JSONDecodeError, InvalidEquityRecommendationRecordError) as exc:
            raise InvalidEquitySyntheticPolicyBindingError(
                f"The stored policy-set reference does not decode: {exc}"
            ) from exc
        if canonical_equity_recommendation_json(reference.document()) != reference_text:
            _invalid("The stored policy-set reference is not canonically encoded.")
        return cls(namespace, reference, policy_set_fingerprint)

    @property
    def version(self) -> EquitySyntheticVersion:
        """Return the published set label."""
        return self.reference.version

    @property
    def identity(self) -> EquitySyntheticPolicyBindingIdentity:
        """Return ``(namespace, POLICY_SET, version)``."""
        return EquitySyntheticPolicyBindingIdentity(self.namespace, _T.POLICY_SET, self.version)

    def component_identity(
        self, policy_type: EquitySyntheticPolicyType
    ) -> EquitySyntheticPolicyBindingIdentity:
        """Return the identity of the component label the set names for ``policy_type``."""
        if policy_type not in _COMPONENTS:
            _invalid(f"{policy_type} is not a component type.")
        version = getattr(self.reference, _COMPONENTS[policy_type][2])
        return EquitySyntheticPolicyBindingIdentity(self.namespace, policy_type, version)

    def reference_json(self) -> str:
        """Return the canonical text of the stored reference."""
        return canonical_equity_recommendation_json(self.reference.document())


# ---------------------------------------------------------------------------
# One complete publication
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EquitySyntheticPolicyBundle:
    """Five component bindings and their set binding, published together or not at all.

    ``components`` are in EquitySyntheticPolicyType order, one per component
    type, each the label the set names. ``policy_set.policy_set_fingerprint``
    must be the ``eirs1`` that the existing EquityResolvedSyntheticPolicySet
    derives from exactly these components.
    """

    components: tuple[EquitySyntheticPolicyDefinitionBinding, ...]
    policy_set: EquitySyntheticPolicySetBinding
    resolved: EquityResolvedSyntheticPolicySet = field(init=False, compare=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.policy_set, EquitySyntheticPolicySetBinding):
            _invalid("A bundle needs an EquitySyntheticPolicySetBinding.")
        components = self.components
        if not isinstance(components, tuple) or not all(
            isinstance(c, EquitySyntheticPolicyDefinitionBinding) for c in components
        ):
            _invalid("A bundle's components are EquitySyntheticPolicyDefinitionBinding values.")
        if tuple(c.policy_type for c in components) != EQUITY_SYNTHETIC_POLICY_COMPONENT_TYPES:
            _invalid("A bundle holds exactly one binding per component type, in type order.")
        for component in components:
            if component.identity != self.policy_set.component_identity(component.policy_type):
                _invalid(
                    f"The {component.policy_type.value} binding {component.identity} is not "
                    "the label its policy set names, in its namespace."
                )
        try:
            resolved = EquityResolvedSyntheticPolicySet(
                reference=self.policy_set.reference,
                **{_COMPONENTS[c.policy_type][1]: c.definition for c in components},
            )
        except InvalidEquityRecommendationEvidenceError as exc:
            raise InvalidEquitySyntheticPolicyBindingError(
                f"The components do not form the policy set: {exc}"
            ) from exc
        if resolved.fingerprint != self.policy_set.policy_set_fingerprint:
            _invalid("The set binding's eirs1 is not the fingerprint of its components.")
        object.__setattr__(self, "resolved", resolved)

    @classmethod
    def of(
        cls,
        policy_set: EquityResolvedSyntheticPolicySet,
        *,
        namespace: str = EQUITY_SYNTHETIC_POLICY_NAMESPACE,
    ) -> EquitySyntheticPolicyBundle:
        """Bind every component of ``policy_set`` and the set itself in ``namespace``."""
        if not isinstance(policy_set, EquityResolvedSyntheticPolicySet):
            raise TypeError("policy_set must be an EquityResolvedSyntheticPolicySet.")
        return cls(
            tuple(
                EquitySyntheticPolicyDefinitionBinding.of(
                    getattr(policy_set, _COMPONENTS[t][1]), namespace=namespace
                )
                for t in EQUITY_SYNTHETIC_POLICY_COMPONENT_TYPES
            ),
            EquitySyntheticPolicySetBinding(
                namespace, policy_set.reference, policy_set.fingerprint
            ),
        )

    @property
    def namespace(self) -> str:
        """Return the namespace every binding of the bundle shares."""
        return self.policy_set.namespace

    def component(
        self, policy_type: EquitySyntheticPolicyType
    ) -> EquitySyntheticPolicyDefinitionBinding:
        """Return the component binding of ``policy_type``."""
        return next(c for c in self.components if c.policy_type is policy_type)
