"""Application simulation of filling one futures paper order from stored daily bars.

A futures paper order exists from its decision instant D. It is filled at the
OPEN of the first persisted daily bar strictly after D, and only once that bar
exists in canonical storage. Before then the order is pending, which this use
case reports as ``None``; there is no order-status value.

Session-daily only
------------------
Futures paper trading in 9.9 is 1d-only. The order deliberately carries no
timeframe, so the query here names ``1d`` explicitly. Intraday execution must
not inherit this silently: it needs a timeframe on the order or its evidence,
and a new selection rule.

Price versus instant
--------------------
The fill quote is the selected bar's OPEN: the simulated execution price
represents the opening of the first session after the decision. ``filled_at``
is that bar's PointInTime -- its completion boundary, the instant Northstar can
first establish the simulated fact from a completed canonical daily bar. It
does not claim the exchange executed anything at the session close.

Selection
---------
The query window is ``[D, available_through]``, inclusive, so the decision bar
itself may come back; it is never fillable. Only bars strictly after D are
candidates, compared with PointInTime.compare(), and the first one wins. Bars
are observations, not calendar days: a gap in the stored history is skipped
without a calendar, so a Monday decision with a Thursday next bar fills on
Thursday. Appending later bars cannot change an already selected fill.

Back-fill is accepted derived-simulation debt
---------------------------------------------
The fill is derived, not frozen. If a previously missing bar is later inserted
between D and the selected bar, re-simulation selects the inserted bar and
produces a different fill for the same order. Persistence rejects that as a
same-order conflict; resolving it is out of scope here.

The use case reads no clock, no provider, no calendar and persists nothing.
"""

from __future__ import annotations

from northstar_core.foundation.value_objects import PointInTime, Timeframe
from northstar_core.paper_trading import FuturesPaperFill, FuturesPaperOrder

from northstar_application.application_services._futures_repository_output import (
    validate_futures_repository_bars,
)
from northstar_application.application_services.futures_paper_execution_identities import (
    FuturesPaperExecutionIdentityService,
)
from northstar_application.ports import (
    FuturesHistoricalMarketDataQuery,
    FuturesHistoricalMarketDataRepository,
)

_PAPER_TRADING_TIMEFRAME = Timeframe("1d")


class SimulateFuturesPaperOrderFillUseCase:
    """Fill one futures paper order at the next stored daily open, if it exists."""

    def __init__(self, repository: FuturesHistoricalMarketDataRepository) -> None:
        if not isinstance(repository, FuturesHistoricalMarketDataRepository):
            raise TypeError(
                "SimulateFuturesPaperOrderFillUseCase repository must be a "
                "FuturesHistoricalMarketDataRepository."
            )
        self._repository = repository
        self._identities = FuturesPaperExecutionIdentityService()

    def execute(
        self,
        order: FuturesPaperOrder,
        available_through: PointInTime,
    ) -> FuturesPaperFill | None:
        """Return the order's full fill, or None while it remains pending.

        ``available_through`` is the last instant whose stored evidence may be
        read. It may equal the decision instant, in which case no strictly
        later bar can exist and the order is pending.
        """
        if order is None:
            raise TypeError("SimulateFuturesPaperOrderFillUseCase order cannot be None.")
        if not isinstance(order, FuturesPaperOrder):
            raise TypeError(
                "SimulateFuturesPaperOrderFillUseCase order must be a FuturesPaperOrder."
            )
        if available_through is None:
            raise TypeError(
                "SimulateFuturesPaperOrderFillUseCase available-through cannot be None."
            )
        if not isinstance(available_through, PointInTime):
            raise TypeError(
                "SimulateFuturesPaperOrderFillUseCase available-through must be a PointInTime."
            )

        intent = order.intent
        decided_at = intent.decided_at
        if available_through.compare(decided_at) < 0:
            raise ValueError(
                f"SimulateFuturesPaperOrderFillUseCase available-through {available_through} "
                f"precedes the decision instant {decided_at}."
            )

        query = FuturesHistoricalMarketDataQuery(
            contract=intent.contract,
            timeframe=_PAPER_TRADING_TIMEFRAME,
            start=decided_at,
            end=available_through,
        )
        bars = validate_futures_repository_bars(self._repository.get_bars(query), query)

        for bar in bars:
            if bar.point_in_time.compare(decided_at) > 0:
                return FuturesPaperFill(
                    identity=self._identities.fill_identity(order.identity),
                    order_identity=order.identity,
                    intent=intent,
                    contracts=intent.contracts,
                    fill_quote=bar.open,
                    filled_at=bar.point_in_time,
                )
        return None
