"""Catalog verification of a resolved synthetic policy set (EI4-D10-S3, ADR-022 section 7).

``verify_equity_synthetic_policy_set`` checks one EquityResolvedSyntheticPolicySet
-- the policy content a generation applied or a disposition froze -- against
the immutable policy catalog (EquitySyntheticPolicyCatalogStore). It is the
one check behind two independent enforcement layers of ADR-022 section 7:

- D. generation validation: GenerateEquityTradeRecommendationUseCase runs it
  while gathering, before any assessment;
- E. the recording gate: RecordEquityRecommendationGenerationUseCase runs it
  before writing a disposition, whoever built that disposition.

What is verified
----------------
The policy set is bound exactly as publication binds it (S1's
EquitySyntheticPolicyBundle.of): five component bindings with their
``eipd1`` fingerprints and the POLICY_SET binding with the existing ``eirs1``.
Each of the six identities -- ``(EQUITY_SYNTHETIC_POLICY_NAMESPACE, type,
label)`` -- is loaded from the catalog exactly once and must be the published
binding of exactly that identity, equal to the offered one in every field:
namespace, type, label, canonical definition and fingerprint (for the set,
its reference document, five component labels and ``eirs1``). The catalog
rebuilds every stored binding with ``from_stored``, so its fingerprints are
re-derived, never trusted. When all six match, the published bindings must
also form one consistent bundle, which re-derives ``eirs1`` from the published
components: a syntactically valid ``eirs1`` alone is never accepted. A
label-only comparison is never made.

Verdicts, one per identity (EquitySyntheticPolicyBindingVerdict):

- VERIFIED: published with exactly the offered content;
- NOT_PUBLISHED: no binding is published under the identity. Absence is
  never publication evidence, and nothing is published here;
- CONFLICT: a valid binding is published under the identity with other
  content, or the catalog answered another identity -- a permanent integrity
  conflict (EI4-D10-DQ6), never retried;
- INVALID: the published binding does not verify, or the catalog answered
  something that is not a binding; it is not a proven alternative definition;
- INCOMPATIBLE: the published binding uses an unsupported format;
- UNAVAILABLE: the catalog could not answer.

The set's verdict is the most severe of its six: CONFLICT, INVALID,
INCOMPATIBLE, UNAVAILABLE, NOT_PUBLISHED, then VERIFIED. Callers map it to
their own vocabulary; nothing here is a generation status or recording
outcome.

The check only reads. It never publishes, repairs, retries, caches bindings
as authority, reads a clock or touches storage other than through the port.
Historical replay never calls it (ADR-022 section 9).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from northstar_application.application_services.equity_recommendation_generation_evidence import (
    EquityResolvedSyntheticPolicySet,
)
from northstar_application.application_services.equity_synthetic_policy_bindings import (
    EQUITY_SYNTHETIC_POLICY_COMPONENT_TYPES,
    EQUITY_SYNTHETIC_POLICY_NAMESPACE,
    EquitySyntheticPolicyBindingIdentity,
    EquitySyntheticPolicyBundle,
    EquitySyntheticPolicyDefinitionBinding,
    EquitySyntheticPolicySetBinding,
    EquitySyntheticPolicyType,
    IncompatibleEquitySyntheticPolicyBindingError,
    InvalidEquitySyntheticPolicyBindingError,
)
from northstar_application.ports.equity_synthetic_policy_catalog_store import (
    EquitySyntheticPolicyCatalogStore,
    EquitySyntheticPolicyCatalogUnavailableError,
)

_Binding = EquitySyntheticPolicyDefinitionBinding | EquitySyntheticPolicySetBinding


class EquitySyntheticPolicyBindingVerdict(StrEnum):
    """What the catalog establishes about one offered policy binding."""

    VERIFIED = "VERIFIED"
    NOT_PUBLISHED = "NOT_PUBLISHED"
    CONFLICT = "CONFLICT"
    INVALID = "INVALID"
    INCOMPATIBLE = "INCOMPATIBLE"
    UNAVAILABLE = "UNAVAILABLE"


_V = EquitySyntheticPolicyBindingVerdict
# Most severe first.
_SEVERITY = (
    _V.CONFLICT,
    _V.INVALID,
    _V.INCOMPATIBLE,
    _V.UNAVAILABLE,
    _V.NOT_PUBLISHED,
    _V.VERIFIED,
)


@dataclass(frozen=True, slots=True)
class EquitySyntheticPolicyBindingCheck:
    """The verdict on one of the six identities of a policy set.

    ``offered_fingerprint`` is the ``eipd1`` (component) or ``eirs1`` (set)
    fingerprint of the offered content, or None when the offered component
    cannot be bound at all; ``catalog_fingerprint`` is the published
    binding's, when a valid one was read. ``detail`` is diagnostic prose only:
    the verdict is the machine-readable result.
    """

    identity: EquitySyntheticPolicyBindingIdentity
    verdict: EquitySyntheticPolicyBindingVerdict
    offered_fingerprint: str | None
    catalog_fingerprint: str | None
    detail: str


@dataclass(frozen=True, slots=True)
class EquitySyntheticPolicySetVerification:
    """The six checks of one policy set: its five components in type order, then the set."""

    checks: tuple[EquitySyntheticPolicyBindingCheck, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.checks, tuple) or len(self.checks) != 6:
            raise ValueError("A policy-set verification holds exactly six checks.")
        if not all(isinstance(check, EquitySyntheticPolicyBindingCheck) for check in self.checks):
            raise ValueError("checks must be EquitySyntheticPolicyBindingCheck values.")
        expected = (*EQUITY_SYNTHETIC_POLICY_COMPONENT_TYPES, EquitySyntheticPolicyType.POLICY_SET)
        if tuple(check.identity.policy_type for check in self.checks) != expected:
            raise ValueError("checks are the five components in type order, then the set.")

    @property
    def verdict(self) -> EquitySyntheticPolicyBindingVerdict:
        """Return the most severe verdict of the six."""
        found = {check.verdict for check in self.checks}
        return next(verdict for verdict in _SEVERITY if verdict in found)

    @property
    def is_verified(self) -> bool:
        """Return whether every binding is published with exactly the offered content."""
        return self.verdict is _V.VERIFIED

    @property
    def failed(self) -> tuple[EquitySyntheticPolicyBindingCheck, ...]:
        """Return the checks that are not VERIFIED, in order."""
        return tuple(check for check in self.checks if check.verdict is not _V.VERIFIED)

    def describe(self) -> str:
        """Return the details of every failed check, for diagnostics."""
        return " ".join(check.detail for check in self.failed)


def _fingerprint(binding: _Binding) -> str:
    if isinstance(binding, EquitySyntheticPolicySetBinding):
        return binding.policy_set_fingerprint
    return binding.fingerprint


def _name(identity: EquitySyntheticPolicyBindingIdentity) -> str:
    return f"{identity.policy_type.value} {identity.version}"


def verify_equity_synthetic_policy_set(
    catalog: EquitySyntheticPolicyCatalogStore, policy_set: EquityResolvedSyntheticPolicySet
) -> EquitySyntheticPolicySetVerification:
    """Verify every binding of ``policy_set`` against ``catalog``; read only, never publish."""
    if not isinstance(catalog, EquitySyntheticPolicyCatalogStore):
        raise TypeError("catalog must be an EquitySyntheticPolicyCatalogStore.")
    if not isinstance(policy_set, EquityResolvedSyntheticPolicySet):
        raise TypeError("policy_set must be an EquityResolvedSyntheticPolicySet.")
    try:
        offered = EquitySyntheticPolicyBundle.of(
            policy_set, namespace=EQUITY_SYNTHETIC_POLICY_NAMESPACE
        )
    except InvalidEquitySyntheticPolicyBindingError as exc:
        return _unbindable(policy_set, exc)
    bindings: tuple[_Binding, ...] = (*offered.components, offered.policy_set)
    checks: list[EquitySyntheticPolicyBindingCheck] = []
    published: list[_Binding] = []
    for binding in bindings:
        check, found = _check(catalog, binding)
        checks.append(check)
        if found is not None:
            published.append(found)
    if all(check.verdict is _V.VERIFIED for check in checks):
        components = tuple(
            b for b in published if isinstance(b, EquitySyntheticPolicyDefinitionBinding)
        )
        (policy_set_binding,) = (
            b for b in published if isinstance(b, EquitySyntheticPolicySetBinding)
        )
        try:
            EquitySyntheticPolicyBundle(components, policy_set_binding)
        except InvalidEquitySyntheticPolicyBindingError as exc:
            set_check = checks[-1]
            checks[-1] = EquitySyntheticPolicyBindingCheck(
                set_check.identity,
                _V.INVALID,
                set_check.offered_fingerprint,
                set_check.catalog_fingerprint,
                f"The published bindings of {_name(set_check.identity)} do not form one "
                f"consistent policy set: {exc}",
            )
    return EquitySyntheticPolicySetVerification(tuple(checks))


def _unbindable(
    policy_set: EquityResolvedSyntheticPolicySet, error: InvalidEquitySyntheticPolicyBindingError
) -> EquitySyntheticPolicySetVerification:
    """The offered content cannot be bound at all: nothing can be verified, so all six fail."""
    named = EquitySyntheticPolicySetBinding(
        EQUITY_SYNTHETIC_POLICY_NAMESPACE, policy_set.reference, policy_set.fingerprint
    )
    identities = (
        *(named.component_identity(t) for t in EQUITY_SYNTHETIC_POLICY_COMPONENT_TYPES),
        named.identity,
    )
    return EquitySyntheticPolicySetVerification(
        tuple(
            EquitySyntheticPolicyBindingCheck(
                identity,
                _V.INVALID,
                policy_set.fingerprint if identity == named.identity else None,
                None,
                f"The offered policy set cannot be bound as published policies: {error}",
            )
            for identity in identities
        )
    )


def _check(
    catalog: EquitySyntheticPolicyCatalogStore, offered: _Binding
) -> tuple[EquitySyntheticPolicyBindingCheck, _Binding | None]:
    identity = offered.identity
    name = _name(identity)
    is_set = isinstance(offered, EquitySyntheticPolicySetBinding)
    kind = EquitySyntheticPolicySetBinding if is_set else EquitySyntheticPolicyDefinitionBinding

    def check(
        verdict: EquitySyntheticPolicyBindingVerdict, detail: str, catalog_fingerprint: str | None
    ) -> EquitySyntheticPolicyBindingCheck:
        return EquitySyntheticPolicyBindingCheck(
            identity, verdict, _fingerprint(offered), catalog_fingerprint, detail
        )

    try:
        found = catalog.load_policy_set(identity) if is_set else catalog.load_definition(identity)
    except EquitySyntheticPolicyCatalogUnavailableError as exc:
        return check(_V.UNAVAILABLE, f"The policy catalog cannot answer {name}: {exc}.", None), None
    except IncompatibleEquitySyntheticPolicyBindingError as exc:
        return (
            check(_V.INCOMPATIBLE, f"The published {name} uses an unsupported format: {exc}", None),
            None,
        )
    except InvalidEquitySyntheticPolicyBindingError as exc:
        return check(_V.INVALID, f"The published {name} does not verify: {exc}", None), None
    if found is None:
        return (
            check(
                _V.NOT_PUBLISHED,
                f"{name} is not published; it is never assumed or published here.",
                None,
            ),
            None,
        )
    if not isinstance(found, kind):
        return (
            check(_V.INVALID, f"The catalog answered {name} with something not a binding.", None),
            None,
        )
    if found.identity != identity:
        return (
            check(
                _V.CONFLICT,
                f"The catalog answered {name} with the binding of {found.identity.namespace} "
                f"{_name(found.identity)}: a permanent policy-reference conflict.",
                _fingerprint(found),
            ),
            None,
        )
    if found != offered:
        return (
            check(
                _V.CONFLICT,
                f"{name} is published as {_fingerprint(found)}, not as the offered "
                f"{_fingerprint(offered)}: a permanent policy-reference conflict (EI4-D10-DQ6).",
                _fingerprint(found),
            ),
            None,
        )
    return check(
        _V.VERIFIED, f"{name} is published with exactly this content.", _fingerprint(found)
    ), found
