"""Application orchestration for deterministic futures historical replay.

This use case serves session-daily futures research and accepts only the
``1d`` timeframe. The restriction belongs to this research workflow, not to
FuturesReplaySnapshot or FuturesHistoricalMarketDataQuery, both of which remain
timeframe-general. Epic 9 research is built on canonical session-daily bars,
and the cumulative representation -- snapshot ``k`` holds ``k`` bars -- grows
quadratically with history length: harmless for a contract's few hundred
sessions, unworkable for a contract's lifetime of minute bars. Accepting any
timeframe would advertise an intraday replay this workflow does not safely
provide, so an unsupported one is refused before the repository is read.

The futures parallel of ReplayHistoricalMarketDataUseCase, kept separate rather
than genericized: the equity replay groups several listings' bars under one
completion instant, while a futures replay reads one concrete contract at one
timeframe, where an instant can hold at most one bar.

Replay reads persisted history only. It depends on the repository port and
nothing else -- no acquisition source, no calendar, no clock -- so the bars it
replays are exactly the evidence already stored, and a session in which nothing
traded simply has no bar and therefore no snapshot.

Repository output is refused rather than repaired
-------------------------------------------------
Each bar is checked against the query before any snapshot is built: it must be
a FuturesOHLCVBar, belong to the queried contract and timeframe, fall inside
the inclusive window, and follow its predecessor strictly by
PointInTime.compare(). Nothing is sorted, filtered or deduplicated. A bar out
of order, out of window or for another contract means the repository broke its
contract, and quietly correcting that would hide the defect while replaying
evidence the caller never asked for.

The snapshot re-checks several of these rules itself. They are checked here as
well, first, for attribution: a Core validation error from inside snapshot
construction would not say that the repository is at fault, nor which bar.
"""

from __future__ import annotations

from northstar_core.foundation.value_objects import Timeframe
from northstar_core.futures import FuturesReplaySnapshot

from northstar_application.application_services._futures_repository_output import (
    validate_futures_repository_bars,
)
from northstar_application.ports import (
    FuturesHistoricalMarketDataQuery,
    FuturesHistoricalMarketDataRepository,
)

_SUPPORTED_TIMEFRAME = Timeframe("1d")


class UnsupportedFuturesReplayTimeframeError(ValueError):
    """Raised when futures replay is requested at a timeframe other than session-daily.

    This is a caller error, not a repository one: the request is well formed
    but asks this research workflow for something it does not provide.
    """


class ReplayFuturesHistoricalMarketDataUseCase:
    """Build cumulative, look-ahead-safe daily replay snapshots of one futures contract.

    Session-daily futures research only; see the module documentation.
    """

    def __init__(self, repository: FuturesHistoricalMarketDataRepository) -> None:
        if not isinstance(repository, FuturesHistoricalMarketDataRepository):
            raise TypeError(
                "ReplayFuturesHistoricalMarketDataUseCase repository "
                "must be a FuturesHistoricalMarketDataRepository."
            )
        self._repository = repository

    def execute(self, query: FuturesHistoricalMarketDataQuery) -> tuple[FuturesReplaySnapshot, ...]:
        """Return one cumulative snapshot per stored bar, oldest to newest.

        Snapshot ``k`` holds bars ``1..k`` and is dated at bar ``k``'s
        completion instant, so no snapshot can see a bar that completed after
        it. A window holding no stored bar returns an empty tuple: there is no
        empty snapshot, because a replay boundary exists only where an
        observation completed.

        The window is the query's, passed through unchanged. Replay neither
        widens it for warm-up nor extends it past the last stored bar.
        """
        if not isinstance(query, FuturesHistoricalMarketDataQuery):
            raise TypeError(
                "ReplayFuturesHistoricalMarketDataUseCase query "
                "must be a FuturesHistoricalMarketDataQuery."
            )
        if query.timeframe != _SUPPORTED_TIMEFRAME:
            raise UnsupportedFuturesReplayTimeframeError(
                f"ReplayFuturesHistoricalMarketDataUseCase supports session-daily research "
                f"only; timeframe {query.timeframe} is not {_SUPPORTED_TIMEFRAME}."
            )

        bars = self._repository.get_bars(query)
        validate_futures_repository_bars(bars, query)

        return tuple(
            FuturesReplaySnapshot(
                contract=query.contract,
                timeframe=query.timeframe,
                replay_instant=bar.point_in_time,
                observations=bars[: index + 1],
            )
            for index, bar in enumerate(bars)
        )
