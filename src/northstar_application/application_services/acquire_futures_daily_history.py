"""Application use case for acquiring and persisting futures daily history.

The workflow is: enumerate the sessions a range names, acquire each session's
minute bars, fold each session into one daily bar, and persist it. Everything
that decides *which* instants belong to a session comes from the calendar and
the fold, so this use case reads no clock and computes no boundary itself.

Persistence is per session, deliberately
----------------------------------------
Each session is stored as it is produced rather than buffering the whole range
and writing once at the end. Sessions are independent evidence: a daily bar is
complete under its own natural key and no invariant spans two of them, so a
range-wide transaction would protect nothing while making a long backfill
all-or-nothing.

The consequence is accepted rather than worked around. A run that fails partway
leaves the sessions it already stored and returns no result. That partial
history is valid, and a rerun resumes safely because the store treats an
identical bar under an existing key as an idempotent success. Coordinating a
transaction across a sequence of provider calls would buy nothing and cost the
resumability.
"""

from __future__ import annotations

from dataclasses import dataclass

from northstar_core.futures import FuturesOHLCVBar

from northstar_application.application_services.aggregate_futures_daily_session_bar import (
    AggregateFuturesDailySessionBarUseCase,
    InvalidFuturesSessionAggregationError,
)
from northstar_application.ports import (
    FuturesDailyHistoricalAcquisitionQuery,
    FuturesHistoricalMarketDataSource,
    FuturesHistoricalMarketDataStore,
    FuturesTradingSession,
    FuturesTradingSessionResolver,
)


class FuturesHistoricalDataContractViolationError(ValueError):
    """Raised when a futures historical source or store violates the port contract."""


@dataclass(frozen=True, slots=True)
class FuturesDailyAcquisitionResult:
    """Outcome of one futures daily historical acquisition run.

    ``session_count`` is how many sessions the calendar found in the requested
    label range. It is normally smaller than the number of days requested,
    because weekends, holidays and closures are not sessions, and the gap
    between the two is where those went.

    ``daily_bar_count`` is how many of those sessions produced a daily bar that
    the store accepted. The gap between the two counts is the sessions in which
    nothing traded.

    It is deliberately not called a stored count. The store accepts an
    identical bar under an existing key idempotently and reports it as
    accepted, so a rerun reports the same number without inserting a row. The
    count answers "how many sessions are now persisted", not "how many rows
    were written", and naming it after insertion would invite the wrong
    reading.
    """

    query: FuturesDailyHistoricalAcquisitionQuery
    session_count: int
    daily_bar_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.query, FuturesDailyHistoricalAcquisitionQuery):
            raise TypeError(
                "FuturesDailyAcquisitionResult query "
                "must be a FuturesDailyHistoricalAcquisitionQuery."
            )
        _validate_count(self.session_count, "session_count")
        _validate_count(self.daily_bar_count, "daily_bar_count")
        if self.daily_bar_count > self.session_count:
            raise ValueError(
                "FuturesDailyAcquisitionResult daily_bar_count cannot exceed session_count."
            )


def _validate_count(value: int, field_name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"FuturesDailyAcquisitionResult {field_name} must be an integer.")
    if value < 0:
        raise ValueError(f"FuturesDailyAcquisitionResult {field_name} must be non-negative.")


class AcquireFuturesDailyHistoryUseCase:
    """Acquire, fold and persist one contract's daily history over a session range."""

    def __init__(
        self,
        session_resolver: FuturesTradingSessionResolver,
        source: FuturesHistoricalMarketDataSource,
        aggregator: AggregateFuturesDailySessionBarUseCase,
        store: FuturesHistoricalMarketDataStore,
    ) -> None:
        if not isinstance(session_resolver, FuturesTradingSessionResolver):
            raise TypeError(
                "AcquireFuturesDailyHistoryUseCase session_resolver "
                "must be a FuturesTradingSessionResolver."
            )
        if not isinstance(source, FuturesHistoricalMarketDataSource):
            raise TypeError(
                "AcquireFuturesDailyHistoryUseCase source "
                "must be a FuturesHistoricalMarketDataSource."
            )
        if not isinstance(aggregator, AggregateFuturesDailySessionBarUseCase):
            raise TypeError(
                "AcquireFuturesDailyHistoryUseCase aggregator "
                "must be an AggregateFuturesDailySessionBarUseCase."
            )
        if not isinstance(store, FuturesHistoricalMarketDataStore):
            raise TypeError(
                "AcquireFuturesDailyHistoryUseCase store "
                "must be a FuturesHistoricalMarketDataStore."
            )

        self._session_resolver = session_resolver
        self._source = source
        self._aggregator = aggregator
        self._store = store

    def execute(
        self, query: FuturesDailyHistoricalAcquisitionQuery
    ) -> FuturesDailyAcquisitionResult:
        """Acquire every session in the range and persist the daily bars produced."""
        if not isinstance(query, FuturesDailyHistoricalAcquisitionQuery):
            raise TypeError(
                "AcquireFuturesDailyHistoryUseCase query "
                "must be a FuturesDailyHistoricalAcquisitionQuery."
            )

        sessions = self._session_resolver.sessions_in_range(
            query.contract.product,
            query.start_trading_date,
            query.end_trading_date,
        )

        daily_bar_count = 0
        for session in sessions:
            daily = self._acquire_session(query, session)
            if daily is None:
                continue
            self._persist(daily)
            daily_bar_count += 1

        return FuturesDailyAcquisitionResult(
            query=query,
            session_count=len(sessions),
            daily_bar_count=daily_bar_count,
        )

    def _acquire_session(
        self, query: FuturesDailyHistoricalAcquisitionQuery, session: FuturesTradingSession
    ) -> FuturesOHLCVBar | None:
        """Fetch and fold one session, attributing any contract breach to the source.

        The source's output contract is not re-checked here. Every rule it must
        honour -- tuple shape, member type, contract, timeframe, strict
        chronological order, session membership, integral volume -- is already
        enforced by the aggregation fold, and restating them would give one
        rule two homes to drift between. What this layer adds is attribution:
        the caller controls both the contract and the session, so a fold that
        refuses these bars is telling us the source misbehaved.
        """
        bars = self._source.fetch_session_bars(query.contract, session)

        try:
            return self._aggregator.execute(query.contract, session, bars)
        except InvalidFuturesSessionAggregationError as exc:
            raise FuturesHistoricalDataContractViolationError(
                f"FuturesHistoricalMarketDataSource returned bars violating the canonical "
                f"contract for {query.contract} session {session.trading_date.isoformat()}: "
                f"{exc}"
            ) from exc

    def _persist(self, daily: FuturesOHLCVBar) -> None:
        """Store one daily bar, requiring the store to accept exactly it.

        A conflict is never caught. It means stored evidence disagrees with
        what was just acquired, which is the one condition requiring an
        explicit reconciliation decision; reporting it as "acquisition failed"
        would bury the only signal that matters.
        """
        accepted = self._store.store((daily,))

        if not isinstance(accepted, int) or isinstance(accepted, bool):
            raise FuturesHistoricalDataContractViolationError(
                "FuturesHistoricalMarketDataStore.store() must return an integer count."
            )
        if accepted != 1:
            raise FuturesHistoricalDataContractViolationError(
                f"FuturesHistoricalMarketDataStore.store() returned count {accepted}, "
                f"expected 1 for a single daily bar."
            )
