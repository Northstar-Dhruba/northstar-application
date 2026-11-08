"""Application ports for the immutable synthetic policy catalog (EI4-D10, ADR-022).

``EquitySyntheticPolicyCatalogStore`` is the durable authority for published
synthetic policy bindings under EI4-D10 (EI-4 Decision Register, section 10.2)
and its approved design decisions EI4-D10-DQ1 to EI4-D10-DQ9 (section 11).
Infrastructure implements it with SQLite in the same database file as the
EI-2B evidence, the EI-3C timelines and the EI-4D generation records
(EI4-D10-DQ4). Application ships no implementation. It is a policy authority
only: never an issued-recommendation authority and never an issuance path.

Records
-------
Records are the Application's EquitySyntheticPolicyDefinitionBinding,
EquitySyntheticPolicySetBinding and EquitySyntheticPolicyBundle
(equity_synthetic_policy_bindings). An implementation returns a stored binding
only by constructing one, with ``from_stored``. A binding that does not
construct raises InvalidEquitySyntheticPolicyBindingError, or
IncompatibleEquitySyntheticPolicyBindingError for an unsupported format.
Nothing is repaired, migrated or skipped.

Operations
----------
``load_definition(identity)`` and ``load_policy_set(identity)`` return the
published binding with exactly that identity, or None when none is published.

``publish(bundle)`` is ONE atomic operation. In one transaction, holding the
write lock before reading (SQLite ``BEGIN IMMEDIATE``), the implementation:

1. reads, for each of the bundle's six identities (five components, then the
   set), the binding stored under it;
2. if any stored binding differs from the offered one -- any field -- writes
   NOTHING and reports CONFLICT with every differing stored binding;
3. otherwise inserts every offered binding not yet stored -- all or nothing --
   and reports COMMITTED when it inserted at least one, or EQUIVALENT when all
   six were already stored identically;
4. commits once.

Uniqueness of every identity, insert-only history and the requirement that a
set exist only with its five component bindings are enforced by the database,
never by an in-memory or process-local lock.

Outcomes and failures
---------------------
A returned EquitySyntheticPolicyCatalogWrite means the transaction committed
(or, for CONFLICT and EQUIVALENT, ended with nothing to write).

- EquitySyntheticPolicyCatalogUnavailableError: the catalog could not be
  opened or read; nothing was written.
- EquitySyntheticPolicyCatalogNotCommittedError: the transaction was rolled
  back before COMMIT; nothing was written. A definite rollback, not an outage.
- EquitySyntheticPolicyCatalogCommitUnconfirmedError: the failure came at or
  after COMMIT, or its acknowledgment was lost. The outcome is unknown and is
  never evidence of a rollback; the Application reconciles by reload.

The implementation never retries by itself, reads no clock and calls no
provider, broker or network.

Historical publication evidence
-------------------------------
``EquitySyntheticPolicyPublicationHistory`` is a narrowly scoped, optional,
read-only contract for the historical binding audit. It answers whether
recorded evidence shows that a binding was published before a generation
record was committed. No implementation exists, and no record format holds
such evidence yet, so until one is governed every audit answers without it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from northstar_application.application_services.equity_synthetic_policy_bindings import (
        EquitySyntheticPolicyBindingIdentity,
        EquitySyntheticPolicyBundle,
        EquitySyntheticPolicyDefinitionBinding,
        EquitySyntheticPolicySetBinding,
    )


class EquitySyntheticPolicyCatalogUnavailableError(RuntimeError):
    """Raised when the policy catalog cannot be read or opened; nothing was written."""


class EquitySyntheticPolicyCatalogNotCommittedError(RuntimeError):
    """Raised when a publication was definitely rolled back before COMMIT; nothing was written."""


class EquitySyntheticPolicyCatalogCommitUnconfirmedError(RuntimeError):
    """Raised when a publication's commit outcome is unknown. It is never a rollback."""


class InvalidEquitySyntheticPolicyCatalogWriteError(ValueError):
    """Raised when a catalog reports a write that its contract does not allow."""


class EquitySyntheticPolicyCatalogOutcome(StrEnum):
    """What one publication transaction established."""

    COMMITTED = "COMMITTED"
    EQUIVALENT = "EQUIVALENT"
    CONFLICT = "CONFLICT"


@dataclass(frozen=True, slots=True)
class EquitySyntheticPolicyCatalogWrite:
    """The outcome of one publication transaction.

    ``conflicts`` holds every stored binding that differs from the offered
    one, for CONFLICT only; it is empty for COMMITTED and EQUIVALENT.
    """

    outcome: EquitySyntheticPolicyCatalogOutcome
    conflicts: tuple[EquitySyntheticPolicyDefinitionBinding | EquitySyntheticPolicySetBinding, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.outcome, EquitySyntheticPolicyCatalogOutcome):
            raise InvalidEquitySyntheticPolicyCatalogWriteError(
                "outcome must be an EquitySyntheticPolicyCatalogOutcome."
            )
        if not isinstance(self.conflicts, tuple):
            raise InvalidEquitySyntheticPolicyCatalogWriteError("conflicts must be a tuple.")
        conflicted = self.outcome is EquitySyntheticPolicyCatalogOutcome.CONFLICT
        if conflicted != bool(self.conflicts):
            raise InvalidEquitySyntheticPolicyCatalogWriteError(
                "Exactly a CONFLICT reports the differing stored bindings."
            )


class EquitySyntheticPolicyCatalogStore(ABC):
    """Durable, insert-only catalog of published synthetic policy bindings."""

    @abstractmethod
    def load_definition(
        self, identity: EquitySyntheticPolicyBindingIdentity
    ) -> EquitySyntheticPolicyDefinitionBinding | None:
        """Return the component binding published under ``identity``, or None."""

    @abstractmethod
    def load_policy_set(
        self, identity: EquitySyntheticPolicyBindingIdentity
    ) -> EquitySyntheticPolicySetBinding | None:
        """Return the POLICY_SET binding published under ``identity``, or None."""

    @abstractmethod
    def publish(self, bundle: EquitySyntheticPolicyBundle) -> EquitySyntheticPolicyCatalogWrite:
        """Atomically publish the whole ``bundle``: all six bindings or none."""


class EquitySyntheticPolicyPublicationHistory(ABC):
    """Read-only historical evidence of when a binding was published (optional).

    ``published_before`` returns True only when recorded evidence shows that
    the binding with ``identity`` and ``fingerprint`` was published before the
    generation record ``generation_key`` was committed; False when recorded
    evidence shows it was not; and None when no such evidence exists. It
    never infers an answer from a binding existing today.
    """

    @abstractmethod
    def published_before(
        self,
        identity: EquitySyntheticPolicyBindingIdentity,
        fingerprint: str,
        generation_key: str,
    ) -> bool | None:
        """Return what recorded evidence establishes, or None when it establishes nothing."""
