"""Application port for persisting futures historical market data observations."""

from __future__ import annotations

from abc import ABC, abstractmethod

from northstar_core.futures import FuturesOHLCVBar


class FuturesHistoricalMarketDataConflictError(ValueError):
    """Raised when a stored futures bar would be replaced by a different bar."""


class FuturesHistoricalMarketDataStore(ABC):
    """Persists factual futures observations as immutable evidence.

    A bar's natural identity is:

        (FuturesContract, PointInTime, Timeframe)

    and a FuturesContract is itself ``(product reference, expiration date)``,
    so the key expands to product code, exchange, expiry, instant and
    timeframe. ES and MES at one instant occupy different keys, and so do
    March and June of one product.

    Implementations must honour these semantics:

    - A bar whose natural key is not yet stored is persisted.
    - Re-storing a bar whose natural key exists and whose stored value is equal
      is an idempotent success, so a retried ingestion is safe.
    - Storing a bar whose natural key exists but whose value differs must raise
      FuturesHistoricalMarketDataConflictError and must NOT overwrite the
      stored bar.

    Why this differs from HistoricalMarketDataStore
    -----------------------------------------------
    The equity store permits a later observation to replace an earlier one,
    because equity history is legitimately restated: splits and dividends cause
    providers to reissue corrected series, and the equity bar carries an
    adjusted_close field precisely to hold the restated value.

    None of that applies to a futures contract. It has no splits and no
    dividends, there is no adjusted close to restate, and once it expires its
    history is closed for good. A differing bar under an existing key therefore
    does not mean "a correction arrived"; it means two sources disagree about
    what a settled contract did, or that a back-adjusted continuous series has
    leaked in wearing a real contract's key. Both are conditions a research
    platform must be told about rather than have resolved silently in favour of
    whichever write happened to land last.

    This follows ForwardResearchRecordStore rather than
    HistoricalMarketDataStore. Copying the equity rule would make an ingestion
    bug indistinguishable from a legitimate correction, and would make a replay
    depend on write order.

    A batch containing two bars sharing one natural key is a contract violation
    and must raise rather than silently collapsing them, even when the two are
    equal: a caller that submits one bar twice in a batch has a defect the
    store should surface, and an idempotent re-store is a separate call rather
    than a duplicated element.

    Persisting an empty tuple must be a safe no-op returning zero.

    On success, store() must return the integer count of input bars accepted as
    part of the logical batch (i.e. len(bars)), counting an idempotent re-store
    as accepted. Partial batch success is not permitted: if any bar cannot be
    persisted, the operation must fail by raising rather than returning a
    partial count, leaving the stored history unchanged.

    The abstract contract documents these obligations but cannot enforce them
    at runtime.
    """

    @abstractmethod
    def store(self, bars: tuple[FuturesOHLCVBar, ...]) -> int:
        """Persist a batch of bars and return the number of bars accepted."""
