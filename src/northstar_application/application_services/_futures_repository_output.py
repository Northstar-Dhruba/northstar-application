"""Validation of futures repository output shared by research use cases.

Replay and outcome measurement both read one contract's stored bars through
FuturesHistoricalMarketDataRepository and both must refuse output that breaks
its contract rather than repair it. The rule lives here once so the two cannot
drift: a bar the replay would reject is a bar measurement rejects too.

Nothing is sorted, filtered or deduplicated. Each violation raises
FuturesHistoricalDataContractViolationError naming the repository and the
offending observation, because the caller controls the query and so a
malformed answer can only be the repository's fault.

This module is internal to the Application Services package.
"""

from __future__ import annotations

from northstar_core.futures import FuturesOHLCVBar

from northstar_application.application_services.acquire_futures_daily_history import (
    FuturesHistoricalDataContractViolationError,
)
from northstar_application.ports import FuturesHistoricalMarketDataQuery


def validate_futures_repository_bars(
    bars: object, query: FuturesHistoricalMarketDataQuery
) -> tuple[FuturesOHLCVBar, ...]:
    """Return ``bars`` unchanged if they honour ``query``; raise otherwise.

    Required: a tuple of FuturesOHLCVBar, each for the queried contract and
    timeframe, each inside the inclusive window, strictly increasing by
    PointInTime.compare() -- which also rejects a duplicated instant.
    """
    if not isinstance(bars, tuple):
        raise FuturesHistoricalDataContractViolationError(
            "FuturesHistoricalMarketDataRepository must return a tuple of FuturesOHLCVBar."
        )

    for index, bar in enumerate(bars):
        if not isinstance(bar, FuturesOHLCVBar):
            raise FuturesHistoricalDataContractViolationError(
                f"FuturesHistoricalMarketDataRepository observation {index} "
                "must be a FuturesOHLCVBar."
            )
        if bar.contract != query.contract:
            raise FuturesHistoricalDataContractViolationError(
                f"FuturesHistoricalMarketDataRepository observation {index} is for "
                f"{bar.contract}, not the queried contract {query.contract}."
            )
        if bar.timeframe != query.timeframe:
            raise FuturesHistoricalDataContractViolationError(
                f"FuturesHistoricalMarketDataRepository observation {index} timeframe "
                f"{bar.timeframe} does not match the queried timeframe {query.timeframe}."
            )
        if not query.covers(bar.point_in_time):
            raise FuturesHistoricalDataContractViolationError(
                f"FuturesHistoricalMarketDataRepository observation {index} at "
                f"{bar.point_in_time} is outside the queried window "
                f"[{query.start}, {query.end}]."
            )
        if index:
            order = bars[index - 1].point_in_time.compare(bar.point_in_time)
            if order == 0:
                raise FuturesHistoricalDataContractViolationError(
                    f"FuturesHistoricalMarketDataRepository observations {index - 1} and "
                    f"{index} share the instant {bar.point_in_time}; one contract and "
                    "timeframe hold at most one bar per instant."
                )
            if order > 0:
                raise FuturesHistoricalDataContractViolationError(
                    f"FuturesHistoricalMarketDataRepository observations must be ordered "
                    f"oldest to newest; observation {index} at {bar.point_in_time} "
                    f"precedes observation {index - 1} at {bars[index - 1].point_in_time}."
                )

    return bars
