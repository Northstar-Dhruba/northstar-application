"""Application creation and freezing of one futures forward research decision.

A forward decision is what a strategy advises now -- "now" being an explicit
``as_of`` instant the caller supplies, never a clock. The decision is computed
from persisted daily history exactly as historical research computes one:
replay the contract's stored bars up to the boundary, analyse the latest
snapshot, and freeze the result. Reusing replay keeps one evidence boundary for
both research modes, so a forward decision can never see a bar that a
historical decision at the same instant could not.

One decision, not a back-fill
-----------------------------
Replay produces a snapshot for every stored bar, but forward testing records
only the latest one: the decision the strategy would take at ``as_of``. Earlier
snapshots describe decisions that were never taken forward and are ignored.
Fewer than twenty visible bars is ordinary warm-up; nothing is frozen, and no
placeholder is stored.

The decision instant is the completion of the latest stored bar at or before
``as_of``, not ``as_of`` itself. A boundary days after the latest bar therefore
produces the same decision -- the same natural key and, over the same stored
history, the same record -- so a daily run without new data is an idempotent
retry.

Freezing is the store's job
---------------------------
The use case never special-cases a retry. An equal record under an existing key
is the store's idempotent success; a different record under that key is the
store's conflict, and it propagates unchanged. That conflict is the point: if a
missing session is later back-filled inside the window, the recomputed decision
differs from the frozen one, and a frozen decision must not be rewritten.

BUY, HOLD and SELL are directional research classifications. Nothing here is an
order, a position, a margin requirement, a multiplier or a profit. Acquisition
happens separately; this reads persisted history only.
"""

from __future__ import annotations

from northstar_core.foundation.value_objects import PointInTime, Timeframe
from northstar_core.futures import FuturesContract
from northstar_core.strategy import FuturesAssetAnalysisGenerator, Strategy

from northstar_application.application_services.analyze_futures_replay_snapshot import (
    AnalyzeFuturesReplaySnapshotService,
)
from northstar_application.application_services.futures_forward_research_record import (
    FuturesForwardResearchRecord,
)
from northstar_application.application_services.record_forward_research_decision import (
    ForwardResearchContractViolationError,
)
from northstar_application.application_services.replay_futures_historical_market_data import (
    ReplayFuturesHistoricalMarketDataUseCase,
    UnsupportedFuturesReplayTimeframeError,
)
from northstar_application.ports import (
    FuturesForwardResearchRecordStore,
    FuturesHistoricalMarketDataQuery,
    FuturesHistoricalMarketDataRepository,
)

_SUPPORTED_TIMEFRAME = Timeframe("1d")


class FreezeFuturesForwardResearchDecisionUseCase:
    """Compute the as-of decision for one contract and strategy, and freeze it."""

    def __init__(
        self,
        repository: FuturesHistoricalMarketDataRepository,
        analysis_generator: FuturesAssetAnalysisGenerator,
        store: FuturesForwardResearchRecordStore,
    ) -> None:
        if not isinstance(repository, FuturesHistoricalMarketDataRepository):
            raise TypeError(
                "FreezeFuturesForwardResearchDecisionUseCase repository "
                "must be a FuturesHistoricalMarketDataRepository."
            )
        if not isinstance(analysis_generator, FuturesAssetAnalysisGenerator):
            raise TypeError(
                "FreezeFuturesForwardResearchDecisionUseCase analysis_generator "
                "must be a FuturesAssetAnalysisGenerator."
            )
        if not isinstance(store, FuturesForwardResearchRecordStore):
            raise TypeError(
                "FreezeFuturesForwardResearchDecisionUseCase store "
                "must be a FuturesForwardResearchRecordStore."
            )
        self._replay = ReplayFuturesHistoricalMarketDataUseCase(repository)
        self._analysis_generator = analysis_generator
        self._store = store

    def execute(
        self,
        contract: FuturesContract,
        timeframe: Timeframe,
        strategy: Strategy,
        as_of: PointInTime,
    ) -> FuturesForwardResearchRecord | None:
        """Freeze the decision visible at ``as_of``, or return None if there is none yet."""
        subject = "FreezeFuturesForwardResearchDecisionUseCase"
        if not isinstance(contract, FuturesContract):
            raise TypeError(f"{subject} contract must be a FuturesContract.")
        if not isinstance(timeframe, Timeframe):
            raise TypeError(f"{subject} timeframe must be a Timeframe.")
        if not isinstance(strategy, Strategy):
            raise TypeError(f"{subject} strategy must be a Strategy.")
        if not isinstance(as_of, PointInTime):
            raise TypeError(f"{subject} as-of must be a PointInTime.")
        if timeframe != _SUPPORTED_TIMEFRAME:
            raise UnsupportedFuturesReplayTimeframeError(
                f"{subject} supports session-daily research only; "
                f"timeframe {timeframe} is not {_SUPPORTED_TIMEFRAME}."
            )

        snapshots = self._replay.execute(
            FuturesHistoricalMarketDataQuery(
                contract=contract,
                timeframe=timeframe,
                start=None,
                end=as_of,
            )
        )
        if not snapshots:
            return None

        result = AnalyzeFuturesReplaySnapshotService(strategy, self._analysis_generator).execute(
            snapshots[-1]
        )
        if result is None:
            return None

        record = FuturesForwardResearchRecord(result)
        self._freeze(record)
        return record

    def _freeze(self, record: FuturesForwardResearchRecord) -> None:
        """Store one record, requiring the store to accept exactly it.

        A conflict is never caught: it means a different decision is already
        frozen under this key, which is evidence that must not be overwritten.
        """
        accepted = self._store.store((record,))

        if not isinstance(accepted, int) or isinstance(accepted, bool):
            raise ForwardResearchContractViolationError(
                "FuturesForwardResearchRecordStore.store() must return an integer count."
            )
        if accepted != 1:
            raise ForwardResearchContractViolationError(
                f"FuturesForwardResearchRecordStore.store() returned count {accepted}, "
                "expected 1 for a single frozen decision."
            )
