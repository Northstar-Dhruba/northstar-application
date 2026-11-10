"""Application port for immutable recommendation-generation records (EI-4D-A).

``EquityRecommendationGenerationStore`` is the durable authority for EI-4D
generation records under the owner decisions EI4-D9-A to EI4-D9-J (EI-4
Decision Register, section 9, approved 2026-11-07) and ADR-021. It holds
synthetic generation dispositions, their proposals and logical generation
attempts. It never holds an issued recommendation, never writes the
issued-recommendation history and never a lifecycle state (EI4-D9-H,
EI4-D9-J). EI-4D-B implements it with SQLite in the same database authority as
the EI-2B evidence and EI-3C timelines; Application ships no implementation.

Records
-------
Records are the Application's EquityRecommendationGenerationDisposition and
EquityRecommendationGenerationAttempt (equity_recommendation_generation_
records). Their construction decodes and verifies every document, so an
implementation returns a stored row only by constructing one: a row that does
not construct raises InvalidEquityRecommendationRecordError, or its
IncompatibleEquityRecommendationRecordError subclass for an unsupported
format or evaluator version. Nothing is repaired, migrated or skipped.

Operations
----------
``load_disposition(key)`` and ``load_attempt(identity)`` return the committed
record, or None when none was committed.

Writes take a recorder-issued authorization (EI4-D11, ADR-023): a
VerifiedEquityRecommendationGenerationWrite or a
VerifiedEquityRecommendationAttemptWrite
(equity_recommendation_generation_write_authorization). The implementation
validates it with the matching public ``require_verified_*`` validator before
anything is opened or written, and persists exactly the content the validated
authorization covers. An authorization that is missing, of the wrong type,
not issued by the verified recorder, or whose content differs from what it
covers is refused with UnverifiedEquityRecommendationGenerationWriteError;
nothing is written.

``write_generation(authorization)`` is ONE atomic operation (EI4-D9-E) over
the disposition and attempt the authorization covers. In one transaction,
holding the write lock before reading, the implementation:

1. reads the disposition stored under ``disposition.generation_key``;
2. none stored: inserts the disposition, its proposal when it has one, and
   the attempt -- all or nothing -- and reports COMMITTED;
3. one stored that is the same record -- the same ``assessment_fingerprint``,
   ``proposal_fingerprint`` AND ``evaluator_version`` -- writes no
   disposition, inserts the attempt unless that identity is already stored,
   and reports EQUIVALENT with the stored disposition;
4. any other stored disposition -- other content, or the same content under
   another evaluator version -- is never replaced: the implementation inserts
   the conflicting attempt unless already stored and reports CONFLICT with the
   stored disposition (EI4-D9-D);
5. commits once.

Equivalence is equality of the whole record, never of the fingerprints alone:
the evaluator version is a stored field that no fingerprint covers, so a store
must compare it explicitly.

Uniqueness of the generation key and of the attempt identity is enforced by
the database, not by in-memory checks (EI4-D9-F).

``write_attempt(authorization)`` records the attempt-only result -- one with
no disposition -- that the authorization covers, inserting it unless the
identity is already stored: COMMITTED or EQUIVALENT. It never creates a
disposition.

Outcomes and failures
---------------------
A returned EquityRecommendationStoreWrite means the transaction committed.

- EquityRecommendationGenerationStoreUnavailableError: the store could not be
  read or opened; nothing was written.
- EquityRecommendationGenerationNotCommittedError: the transaction was rolled
  back before COMMIT -- a refused row, a lock not obtained; nothing was
  written. A definite rollback, not an outage.
- EquityRecommendationGenerationCommitUnconfirmedError: the failure came at or
  after COMMIT, or its acknowledgment was lost. The outcome is unknown and is
  never evidence of a rollback (EI4-D9-G); the caller reconciles by reload.

The implementation never retries by itself, reads no clock and calls no
provider, broker or network.

Read and authorized-write ports (EI4-D11, ADR-023)
--------------------------------------------------
``EquityRecommendationGenerationRecordReader`` is the read-only port:
``load_disposition`` and ``load_attempt``. Historical verification and the
policy-binding audit depend on it alone.

``EquityRecommendationGenerationRecordWriter`` is the authorized-write port:
``write_generation`` and ``write_attempt`` above, both abstract. The verified
recorder (RecordEquityRecommendationGenerationUseCase) is the only supported
caller and the only issuer of authorizations.

``EquityRecommendationGenerationStore`` is the combined port: exactly the
reader and the authorized writer, adding nothing. Its abstract operations are
``load_disposition``, ``load_attempt``, ``write_generation`` and
``write_attempt``; an implementation supplies all four explicitly. The port
has no raw write taking unverified records and no concrete write default
(EI4-D11-W3): an implementation that supplies only reads, or only raw
writes, cannot be instantiated. The boundary prevents unsupported ordinary
writes; it is not tamper-proof against privileged code or direct database
access (ADR-023).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from northstar_application.application_services.equity_recommendation_generation_records import (  # noqa: E501
        EquityRecommendationGenerationAttempt,
        EquityRecommendationGenerationDisposition,
    )
    from northstar_application.application_services.equity_recommendation_generation_write_authorization import (  # noqa: E501
        VerifiedEquityRecommendationAttemptWrite,
        VerifiedEquityRecommendationGenerationWrite,
    )


class EquityRecommendationGenerationStoreUnavailableError(RuntimeError):
    """Raised when generation storage cannot be read or opened; nothing was written."""


class EquityRecommendationGenerationNotCommittedError(RuntimeError):
    """Raised when a write was definitely rolled back before COMMIT; nothing was written."""


class EquityRecommendationGenerationCommitUnconfirmedError(RuntimeError):
    """Raised when a write's commit outcome is unknown. It is never a rollback."""


class EquityRecommendationStoreOutcome(StrEnum):
    """What one committed write established."""

    COMMITTED = "COMMITTED"
    EQUIVALENT = "EQUIVALENT"
    CONFLICT = "CONFLICT"


class InvalidEquityRecommendationStoreWriteError(ValueError):
    """Raised when a store reports a write that its contract does not allow."""


class UnverifiedEquityRecommendationGenerationWriteError(ValueError):
    """Raised when a generation write is not authorized by the verified recorder (EI4-D11).

    The authorization is missing, of the wrong type, was not issued by the
    verified recorder, or covers other content than offered. Nothing is
    written. No supported caller can produce it: it is a programming error,
    never a recording outcome.
    """


@dataclass(frozen=True, slots=True)
class EquityRecommendationStoreWrite:
    """A committed write: its outcome, the stored disposition when one already
    existed, and whether this write inserted the attempt.

    ``stored`` is None for COMMITTED (the offered disposition is now stored)
    and for every attempt-only write; it is the disposition already stored for
    EQUIVALENT and CONFLICT on ``write_generation``. ``attempt_inserted`` is
    False when an attempt with the same identity was already stored -- the
    attempt is durably recorded either way.
    """

    outcome: EquityRecommendationStoreOutcome
    stored: EquityRecommendationGenerationDisposition | None
    attempt_inserted: bool

    def __post_init__(self) -> None:
        if not isinstance(self.outcome, EquityRecommendationStoreOutcome):
            raise InvalidEquityRecommendationStoreWriteError(
                "outcome must be an EquityRecommendationStoreOutcome."
            )
        if not isinstance(self.attempt_inserted, bool):
            raise InvalidEquityRecommendationStoreWriteError("attempt_inserted must be a bool.")
        if self.outcome is EquityRecommendationStoreOutcome.COMMITTED and self.stored is not None:
            raise InvalidEquityRecommendationStoreWriteError(
                "A COMMITTED write reports no previously stored disposition."
            )


class EquityRecommendationGenerationRecordReader(ABC):
    """Read-only access to committed generation dispositions and logical attempts."""

    @abstractmethod
    def load_disposition(
        self, generation_key: str
    ) -> EquityRecommendationGenerationDisposition | None:
        """Return the committed disposition for ``generation_key``, or None."""

    @abstractmethod
    def load_attempt(self, attempt_identity: str) -> EquityRecommendationGenerationAttempt | None:
        """Return the committed attempt with ``attempt_identity``, or None."""


class EquityRecommendationGenerationRecordWriter(ABC):
    """Authorized writes of generation records: only recorder-issued authorizations."""

    @abstractmethod
    def write_generation(
        self, authorization: VerifiedEquityRecommendationGenerationWrite
    ) -> EquityRecommendationStoreWrite:
        """Atomically record exactly the disposition and attempt ``authorization`` covers."""

    @abstractmethod
    def write_attempt(
        self, authorization: VerifiedEquityRecommendationAttemptWrite
    ) -> EquityRecommendationStoreWrite:
        """Record exactly the attempt-only result ``authorization`` covers."""


class EquityRecommendationGenerationStore(
    EquityRecommendationGenerationRecordReader, EquityRecommendationGenerationRecordWriter
):
    """Durable, insert-only store of generation dispositions and logical attempts.

    Exactly the read port and the authorized-write port: every operation is
    abstract and inherited, so an implementation supplies all four itself.
    """
