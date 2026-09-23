"""Application port for freezing futures forward research decisions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from northstar_application.application_services.futures_forward_research_record import (
        FuturesForwardResearchRecord,
    )


class FuturesForwardResearchRecordConflictError(ValueError):
    """Raised when a frozen futures record would be replaced by a different record.

    It means an existing natural key holds a record that is not equal to the
    one being frozen. It is a research-persistence conflict, not a storage
    failure: the stored decision is evidence and must be kept as it is.
    """


class FuturesForwardResearchRecordStore(ABC):
    """Freezes futures forward research decisions into research storage.

    A frozen decision is immutable evidence of what a strategy advised about one
    concrete contract at one decision instant. Persistence is append-only with
    respect to its natural identity:

        (FuturesContract, Timeframe, decision instant, StrategyIdentity)

    Implementations must honour these freeze semantics for each natural key:

    - absent: the record is persisted.
    - present and equal to the record being frozen: an idempotent success, so a
      retried freeze is safe. Equality is full record value equality, never key
      equality alone.
    - present and different: raise FuturesForwardResearchRecordConflictError and
      leave the stored record exactly as it was. Later market data must never
      rewrite a frozen decision; this deliberately differs from market data
      stores, whose job is to hold observations rather than decisions.

    A batch containing two records sharing one natural key is a contract
    violation and must raise FuturesForwardResearchRecordConflictError, even
    when the two records are equal: a caller submitting one decision twice in a
    batch has a defect, and an idempotent retry is a separate call.

    Persisting an empty tuple must be a safe no-op returning zero.

    On success, store() must return the integer count of input records accepted
    as part of the logical batch (i.e. len(records)), counting an idempotent
    re-store as accepted. Partial batch success is not permitted; if any record
    cannot be persisted, the operation must fail by raising rather than
    returning a partial count.

    Measurements are not stored here. Outcomes are derived later from the
    frozen record and the persisted market history; a record never changes to
    reflect them. A session back-filled into history after a decision can shift
    which observation a horizon selects -- that is a known limitation of derived
    measurement, and it belongs to measurement semantics, not to this contract.
    """

    @abstractmethod
    def store(self, records: tuple[FuturesForwardResearchRecord, ...]) -> int:
        """Freeze a batch of records and return the number of records accepted."""
