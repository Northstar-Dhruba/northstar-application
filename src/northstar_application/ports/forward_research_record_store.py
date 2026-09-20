"""Application port for freezing forward research decisions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from northstar_application.application_services.record_forward_research_decision import (
        ForwardResearchRecord,
    )


class ForwardResearchRecordConflictError(ValueError):
    """Raised when a stored forward record would be replaced by a different record."""


class ForwardResearchRecordStore(ABC):
    """Freezes forward research decisions into research storage.

    A frozen decision is immutable evidence of what a strategy advised at one
    decision instant. Persistence is therefore append-only with respect to its
    natural identity:

        (Symbol, ExchangeCode, Timeframe, decision instant, StrategyIdentity)

    Implementations must honour these freeze semantics:

    - A record whose natural key is not yet stored is persisted.
    - Re-storing a record whose natural key exists and whose stored value is
      equal is an idempotent success, so a retried recording is safe.
    - Storing a record whose natural key exists but whose value differs must
      raise ForwardResearchRecordConflictError and must NOT overwrite the
      stored record. A frozen decision is never rewritten; this deliberately
      differs from HistoricalMarketDataStore, where later provider corrections
      legitimately replace earlier observations.

    A batch containing two records sharing one natural key is a contract
    violation and must raise rather than silently collapsing them.

    Persisting an empty tuple must be a safe no-op returning zero.

    On success, store() must return the integer count of input records accepted
    as part of the logical batch (i.e. len(records)), counting an idempotent
    re-store as accepted. Partial batch success is not permitted; if any record
    cannot be persisted, the operation must fail by raising an exception rather
    than returning a partial count.
    """

    @abstractmethod
    def store(self, records: tuple[ForwardResearchRecord, ...]) -> int:
        """Freeze a batch of records and return the number of records accepted."""
