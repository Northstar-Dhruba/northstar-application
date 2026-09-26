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

    That particular mechanism does not apply to a futures contract: it has no
    splits and no dividends, and no adjusted close to restate.

    Futures observations can still be corrected. Exchanges revise settlement
    prices, trades are busted after the fact, session volume is finalised late,
    and providers reissue data they got wrong. A differing bar under an
    existing key may therefore be a genuine correction, or two sources
    disagreeing about what a settled contract did, or a back-adjusted
    continuous series leaking in wearing a real contract's key.

    Because those three look identical to a store, none of them may be applied
    silently. Accepting a correction is a decision, not a write: the differing
    evidence surfaces as a conflict, and an explicit reconciliation step -- one
    that does not exist yet -- decides what is true. Last-write-wins would make
    an ingestion bug indistinguishable from a legitimate revision, and would
    make a replay's result depend on the order in which writes happened to
    land.

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
