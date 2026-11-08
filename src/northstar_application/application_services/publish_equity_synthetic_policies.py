"""Explicit publication of synthetic policy bindings (EI4-D10, ADR-022).

PublishEquitySyntheticPoliciesUseCase is the only way a policy binding is
created (ADR-022, sections 5 to 7). It is invoked explicitly by an operator or
by test setup, and never by a policy read, generation, recording, verification
or historical replay.

Flow
----
1. The caller supplies an EquitySyntheticPolicyBundle: the five component
   bindings and the set binding of one resolved policy set. Its construction
   validates every definition, derives every ``eipd1`` fingerprint and checks
   the set's existing ``eirs1`` against its components.
2. The bundle must be in the approved namespace (EI4-D10-DQ1). Anything else
   raises InvalidEquitySyntheticPolicyBindingError before any write.
3. ``store.publish(bundle)`` runs ONE atomic transaction.
4. Its outcome is interpreted:

   - COMMITTED: PUBLISHED;
   - EQUIVALENT: EQUIVALENT_PUBLISHED, idempotent;
   - CONFLICT: POLICY_REFERENCE_CONFLICT, a **permanent integrity conflict**
     (EI4-D10-DQ6). It is reported with the differing stored bindings and
     never retried; nothing is republished under the conflicting labels.

Failures
--------
- The catalog unavailable: CATALOG_UNAVAILABLE. No retry: no operational
  retry policy for ordinary downtime is approved.
- A definite rollback: NOT_PUBLISHED, without retry.
- A stored binding that does not construct: STORED_BINDING_INVALID or
  STORED_BINDING_INCOMPATIBLE. It is never repaired or migrated.
- An unconfirmed COMMIT is never treated as a rollback. The bundle's six
  identities are reloaded once:

  - any stored binding differs: POLICY_REFERENCE_CONFLICT (a competitor won);
  - the set binding is stored, equal, with all five components equal:
    EQUIVALENT_PUBLISHED, reconciled;
  - the set binding is stored but a component is not: the catalog is
    inconsistent, and the result is PUBLICATION_RECONCILIATION_UNRESOLVED;
  - the set binding is absent: the all-or-nothing transaction did not commit
    (components already stored equal may predate it), so the same bundle is
    written again, at most ``max_writes`` writes in all (never more than
    three), then RETRY_LIMIT_REACHED;
  - the reload fails or returns a binding that does not construct:
    PUBLICATION_RECONCILIATION_UNRESOLVED. Nothing is claimed.

Nothing here sleeps, reads a clock, retries an outage or a conflict, or
touches any generation record, ``eirs1`` or ``eirk1``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

from northstar_application.application_services.equity_synthetic_policy_bindings import (
    EQUITY_SYNTHETIC_POLICY_NAMESPACE,
    EquitySyntheticPolicyBindingIdentity,
    EquitySyntheticPolicyBundle,
    EquitySyntheticPolicyDefinitionBinding,
    EquitySyntheticPolicySetBinding,
    IncompatibleEquitySyntheticPolicyBindingError,
    InvalidEquitySyntheticPolicyBindingError,
)
from northstar_application.ports.equity_synthetic_policy_catalog_store import (
    EquitySyntheticPolicyCatalogCommitUnconfirmedError,
    EquitySyntheticPolicyCatalogNotCommittedError,
    EquitySyntheticPolicyCatalogOutcome,
    EquitySyntheticPolicyCatalogStore,
    EquitySyntheticPolicyCatalogUnavailableError,
    EquitySyntheticPolicyCatalogWrite,
)

EQUITY_SYNTHETIC_POLICY_PUBLICATION_WRITES = 3
_Binding = EquitySyntheticPolicyDefinitionBinding | EquitySyntheticPolicySetBinding


class EquitySyntheticPolicyPublicationOutcome(StrEnum):
    """How one publication ended. Only the first two mean the bundle is published."""

    PUBLISHED = "PUBLISHED"
    EQUIVALENT_PUBLISHED = "EQUIVALENT_PUBLISHED"
    POLICY_REFERENCE_CONFLICT = "POLICY_REFERENCE_CONFLICT"
    NOT_PUBLISHED = "NOT_PUBLISHED"
    CATALOG_UNAVAILABLE = "CATALOG_UNAVAILABLE"
    STORED_BINDING_INVALID = "STORED_BINDING_INVALID"
    STORED_BINDING_INCOMPATIBLE = "STORED_BINDING_INCOMPATIBLE"
    PUBLICATION_RECONCILIATION_UNRESOLVED = "PUBLICATION_RECONCILIATION_UNRESOLVED"
    RETRY_LIMIT_REACHED = "RETRY_LIMIT_REACHED"


_P = EquitySyntheticPolicyPublicationOutcome
_PUBLISHED = frozenset({_P.PUBLISHED, _P.EQUIVALENT_PUBLISHED})


@dataclass(frozen=True, slots=True)
class EquitySyntheticPolicyPublicationResult:
    """The outcome of publishing one bundle.

    ``conflicts`` holds the stored bindings that differ from the offered ones,
    for POLICY_REFERENCE_CONFLICT only. ``writes`` counts publish calls;
    ``reconciled`` says whether an unconfirmed commit was reconciled by reload.
    ``detail`` is diagnostic prose: callers decide on ``outcome``, never on it.
    """

    outcome: EquitySyntheticPolicyPublicationOutcome
    bundle: EquitySyntheticPolicyBundle
    conflicts: tuple[_Binding, ...]
    writes: int
    reconciled: bool
    detail: str

    def __post_init__(self) -> None:
        if not isinstance(self.outcome, EquitySyntheticPolicyPublicationOutcome):
            raise ValueError("outcome must be an EquitySyntheticPolicyPublicationOutcome.")
        if (self.outcome is _P.POLICY_REFERENCE_CONFLICT) != bool(self.conflicts):
            raise ValueError("Exactly a POLICY_REFERENCE_CONFLICT carries conflicting bindings.")

    @property
    def is_published(self) -> bool:
        """Return whether the whole bundle is durably published, by this write or an equal one."""
        return self.outcome in _PUBLISHED

    @property
    def is_permanent_conflict(self) -> bool:
        """Return whether this is a permanent integrity conflict: never retried."""
        return self.outcome is _P.POLICY_REFERENCE_CONFLICT


_Result = Callable[..., EquitySyntheticPolicyPublicationResult]


def _offered(
    bundle: EquitySyntheticPolicyBundle,
) -> dict[EquitySyntheticPolicyBindingIdentity, _Binding]:
    offered: dict[EquitySyntheticPolicyBindingIdentity, _Binding] = {
        c.identity: c for c in bundle.components
    }
    offered[bundle.policy_set.identity] = bundle.policy_set
    return offered


class PublishEquitySyntheticPoliciesUseCase:
    """Publishes one complete synthetic policy bundle, explicitly and atomically."""

    def __init__(
        self,
        store: EquitySyntheticPolicyCatalogStore,
        max_writes: int = EQUITY_SYNTHETIC_POLICY_PUBLICATION_WRITES,
    ) -> None:
        if not isinstance(store, EquitySyntheticPolicyCatalogStore):
            raise TypeError("store must be an EquitySyntheticPolicyCatalogStore.")
        if (
            isinstance(max_writes, bool)
            or not isinstance(max_writes, int)
            or not 1 <= max_writes <= EQUITY_SYNTHETIC_POLICY_PUBLICATION_WRITES
        ):
            raise ValueError(
                "max_writes must be an integer from 1 to "
                f"{EQUITY_SYNTHETIC_POLICY_PUBLICATION_WRITES}."
            )
        self._store = store
        self._max_writes = max_writes

    def execute(
        self, bundle: EquitySyntheticPolicyBundle
    ) -> EquitySyntheticPolicyPublicationResult:
        """Publish ``bundle`` -- all six bindings or none -- and report the typed outcome."""
        if not isinstance(bundle, EquitySyntheticPolicyBundle):
            raise TypeError("bundle must be an EquitySyntheticPolicyBundle.")
        if bundle.namespace != EQUITY_SYNTHETIC_POLICY_NAMESPACE:
            raise InvalidEquitySyntheticPolicyBindingError(
                f"Only the approved namespace {EQUITY_SYNTHETIC_POLICY_NAMESPACE} is published "
                f"(EI4-D10-DQ1), not {bundle.namespace!r}."
            )

        def result(
            outcome: EquitySyntheticPolicyPublicationOutcome,
            detail: str,
            *,
            writes: int,
            conflicts: tuple[_Binding, ...] = (),
            reconciled: bool = False,
        ) -> EquitySyntheticPolicyPublicationResult:
            return EquitySyntheticPolicyPublicationResult(
                outcome, bundle, conflicts, writes, reconciled, detail
            )

        for write in range(1, self._max_writes + 1):
            try:
                written = self._store.publish(bundle)
            except EquitySyntheticPolicyCatalogUnavailableError as exc:
                return result(_P.CATALOG_UNAVAILABLE, f"Catalog unavailable: {exc}", writes=write)
            except EquitySyntheticPolicyCatalogNotCommittedError as exc:
                return result(_P.NOT_PUBLISHED, f"Rolled back: {exc}", writes=write)
            except IncompatibleEquitySyntheticPolicyBindingError as exc:
                return result(_P.STORED_BINDING_INCOMPATIBLE, str(exc), writes=write)
            except InvalidEquitySyntheticPolicyBindingError as exc:
                return result(_P.STORED_BINDING_INVALID, str(exc), writes=write)
            except EquitySyntheticPolicyCatalogCommitUnconfirmedError as exc:
                settled = self._reconcile(bundle, str(exc), result, write)
                if settled is not None:
                    return settled
                continue  # the set is verifiably absent: write the same bundle again
            return self._interpret(bundle, written, result, write)
        return result(
            _P.RETRY_LIMIT_REACHED,
            f"No write of {self._max_writes} was confirmed; each reload found the set absent.",
            writes=self._max_writes,
            reconciled=True,
        )

    def _interpret(
        self,
        bundle: EquitySyntheticPolicyBundle,
        written: EquitySyntheticPolicyCatalogWrite,
        result: _Result,
        write: int,
    ) -> EquitySyntheticPolicyPublicationResult:
        if not isinstance(written, EquitySyntheticPolicyCatalogWrite):
            raise TypeError("The catalog must return an EquitySyntheticPolicyCatalogWrite.")
        if written.outcome is EquitySyntheticPolicyCatalogOutcome.COMMITTED:
            return result(_P.PUBLISHED, "Published the bundle.", writes=write)
        if written.outcome is EquitySyntheticPolicyCatalogOutcome.EQUIVALENT:
            return result(
                _P.EQUIVALENT_PUBLISHED, "The same bundle is already published.", writes=write
            )
        offered = _offered(bundle)
        for stored in written.conflicts:
            genuine = isinstance(
                stored, EquitySyntheticPolicyDefinitionBinding | EquitySyntheticPolicySetBinding
            ) and offered.get(stored.identity) not in (None, stored)
            if not genuine:
                return result(
                    _P.STORED_BINDING_INVALID,
                    "The catalog reported a conflict that its stored bindings contradict.",
                    writes=write,
                )
        return result(
            _P.POLICY_REFERENCE_CONFLICT,
            "Labels in the bundle are published with other content; the stored bindings stay "
            "authoritative. This is permanent: it is never retried or republished.",
            writes=write,
            conflicts=written.conflicts,
        )

    def _reconcile(
        self,
        bundle: EquitySyntheticPolicyBundle,
        unconfirmed: str,
        result: _Result,
        write: int,
    ) -> EquitySyntheticPolicyPublicationResult | None:
        """Reload once after an unconfirmed COMMIT; None means the set is verifiably absent."""
        offered = _offered(bundle)
        found: dict[EquitySyntheticPolicyBindingIdentity, _Binding | None] = {}
        try:
            for identity, binding in offered.items():
                load = (
                    self._store.load_policy_set
                    if isinstance(binding, EquitySyntheticPolicySetBinding)
                    else self._store.load_definition
                )
                found[identity] = load(identity)
        except (
            EquitySyntheticPolicyCatalogUnavailableError,
            InvalidEquitySyntheticPolicyBindingError,
        ) as exc:
            return result(
                _P.PUBLICATION_RECONCILIATION_UNRESOLVED,
                f"The commit is unconfirmed ({unconfirmed}) and the reload failed ({exc}); "
                "nothing is claimed.",
                writes=write,
                reconciled=True,
            )
        conflicts = tuple(
            stored
            for identity, stored in found.items()
            if stored is not None and stored != offered[identity]
        )
        if conflicts:
            return result(
                _P.POLICY_REFERENCE_CONFLICT,
                "After an unconfirmed commit, other content is published under labels of the "
                "bundle. This is permanent: it is never retried or republished.",
                writes=write,
                conflicts=conflicts,
                reconciled=True,
            )
        if found[bundle.policy_set.identity] is None:
            return None
        if any(found[c.identity] is None for c in bundle.components):
            return result(
                _P.PUBLICATION_RECONCILIATION_UNRESOLVED,
                "The set binding is stored without all of its components: the catalog is "
                "inconsistent, and nothing is claimed or repaired.",
                writes=write,
                reconciled=True,
            )
        return result(
            _P.EQUIVALENT_PUBLISHED,
            "The commit was unconfirmed; the reload finds the whole bundle published.",
            writes=write,
            reconciled=True,
        )
