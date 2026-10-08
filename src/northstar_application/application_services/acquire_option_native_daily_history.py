"""Application use case for acquiring and persisting native daily option bars.

The workflow is: resolve the option sessions the range names, fetch the source's
daily candles for the exact contract once, match each candle to exactly one
resolved session, stamp each canonical OptionOHLCVBar at its session's close
with the daily timeframe, and persist the range as one batch.

The calendar decides which sessions exist
-----------------------------------------
The option session resolver is the sole authority on which labels are sessions.
The source and the calendar must agree exactly:

- a resolved session with no candle is a gap, and fails;
- a candle for a label that is not a resolved session is unexplained data, and
  fails.

This is deliberately conservative. Whether an illiquid but listed strike
legitimately has no daily candle is not known yet, so a missing candle is not
silently read as "nothing traded". The rule can be revisited once provider
evidence shows how such strikes are published.

Finality is the caller's decision
---------------------------------
This use case reads no clock and makes no judgement about whether a candle is
final. Acquiring a candle records what the provider returned, nothing more.

Persistence is one batch
------------------------
The whole range is validated before anything is written and stored in one
all-or-nothing batch. A rerun of a stored range is an idempotent success, and a
differing value under an existing key is a conflict that propagates unwrapped.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from northstar_core.foundation.value_objects import Timeframe
from northstar_core.options import InvalidOptionOHLCVBarError, OptionContract, OptionOHLCVBar

from northstar_application.ports import (
    OptionDailyAcquisitionQuery,
    OptionHistoricalMarketDataStore,
    OptionNativeDailyMarketDataSource,
    OptionNativeDailyObservation,
    OptionTradingSession,
    OptionTradingSessionResolver,
)

_DAILY = Timeframe("1d")
_SUBJECT = "AcquireOptionNativeDailyHistoryUseCase"


class OptionHistoricalDataContractViolationError(ValueError):
    """Raised when an option source, resolver or store violates its port contract."""


class OptionDailySessionCoverageError(ValueError):
    """Raised when native daily option candles and the session calendar disagree.

    ``missing_trading_dates`` are resolved sessions with no candle and
    ``unexpected_trading_dates`` are candles for labels that are not resolved
    sessions. Both are ascending, and at least one is non-empty.
    """

    def __init__(
        self,
        contract: OptionContract,
        missing_trading_dates: tuple[date, ...],
        unexpected_trading_dates: tuple[date, ...],
    ) -> None:
        self.contract = contract
        self.missing_trading_dates = missing_trading_dates
        self.unexpected_trading_dates = unexpected_trading_dates
        parts = []
        if missing_trading_dates:
            parts.append(
                "no candle for sessions " + ", ".join(d.isoformat() for d in missing_trading_dates)
            )
        if unexpected_trading_dates:
            parts.append(
                "candles for non-sessions "
                + ", ".join(d.isoformat() for d in unexpected_trading_dates)
            )
        super().__init__(f"Option daily candles for {contract} disagree with the calendar: "
                         + "; ".join(parts) + ".")  # fmt: skip


@dataclass(frozen=True, slots=True)
class OptionDailyAcquisitionResult:
    """Outcome of one option daily acquisition run.

    ``session_count`` is how many option sessions the calendar found in the
    requested range. ``daily_bar_count`` is how many bars the store accepted;
    under strict coverage it always equals ``session_count`` on success. An
    idempotent re-store counts as accepted, so this is not a count of rows
    written.
    """

    query: OptionDailyAcquisitionQuery
    session_count: int
    daily_bar_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.query, OptionDailyAcquisitionQuery):
            raise TypeError(
                "OptionDailyAcquisitionResult query must be an OptionDailyAcquisitionQuery."
            )
        for name in ("session_count", "daily_bar_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"OptionDailyAcquisitionResult {name} must be an integer.")
            if value < 0:
                raise ValueError(f"OptionDailyAcquisitionResult {name} cannot be negative.")
        if self.daily_bar_count > self.session_count:
            raise ValueError(
                "OptionDailyAcquisitionResult daily_bar_count cannot exceed session_count."
            )


class AcquireOptionNativeDailyHistoryUseCase:
    """Acquire, session-stamp and persist one exact option contract's daily candles."""

    def __init__(
        self,
        session_resolver: OptionTradingSessionResolver,
        source: OptionNativeDailyMarketDataSource,
        store: OptionHistoricalMarketDataStore,
    ) -> None:
        for name, value, expected in (
            ("session_resolver", session_resolver, OptionTradingSessionResolver),
            ("source", source, OptionNativeDailyMarketDataSource),
            ("store", store, OptionHistoricalMarketDataStore),
        ):
            if not isinstance(value, expected):
                raise TypeError(f"{_SUBJECT} {name} must be an {expected.__name__}.")
        self._session_resolver = session_resolver
        self._source = source
        self._store = store

    def execute(self, query: OptionDailyAcquisitionQuery) -> OptionDailyAcquisitionResult:
        """Acquire every session in the range and persist one daily bar per session."""
        if not isinstance(query, OptionDailyAcquisitionQuery):
            raise TypeError(f"{_SUBJECT} query must be an OptionDailyAcquisitionQuery.")

        sessions = _validate_sessions(
            self._session_resolver.sessions_in_range(
                query.contract.product, query.start_trading_date, query.end_trading_date
            ),
            query,
        )
        observations = self._source.fetch_daily_observations(
            query.contract, query.start_trading_date, query.end_trading_date
        )

        by_date = _index_observations(query.contract, query, observations)
        _require_coverage(query.contract, sessions, by_date)

        bars = tuple(
            _daily_bar(query.contract, session, by_date[session.trading_date])
            for session in sessions
        )
        if bars:
            self._persist(bars)

        return OptionDailyAcquisitionResult(
            query=query, session_count=len(sessions), daily_bar_count=len(bars)
        )

    def _persist(self, bars: tuple[OptionOHLCVBar, ...]) -> None:
        """Store the range as one batch; a conflict is never caught."""
        accepted = self._store.store(bars)
        if isinstance(accepted, bool) or not isinstance(accepted, int):
            raise OptionHistoricalDataContractViolationError(
                "OptionHistoricalMarketDataStore.store() must return an integer count."
            )
        if accepted != len(bars):
            raise OptionHistoricalDataContractViolationError(
                f"OptionHistoricalMarketDataStore.store() returned count {accepted}, "
                f"expected {len(bars)} for the acquired daily bars."
            )


def _validate_sessions(
    sessions: object, query: OptionDailyAcquisitionQuery
) -> tuple[OptionTradingSession, ...]:
    """Require a resolver answer that is ascending, unique and inside the range."""
    if not isinstance(sessions, tuple) or not all(
        isinstance(session, OptionTradingSession) for session in sessions
    ):
        raise OptionHistoricalDataContractViolationError(
            "OptionTradingSessionResolver must return a tuple of OptionTradingSession."
        )
    labels = [session.trading_date for session in sessions]
    if labels != sorted(set(labels)):
        raise OptionHistoricalDataContractViolationError(
            "OptionTradingSessionResolver must return strictly ascending, unique sessions."
        )
    if labels and not (
        query.start_trading_date <= labels[0] and labels[-1] <= query.end_trading_date
    ):
        raise OptionHistoricalDataContractViolationError(
            "OptionTradingSessionResolver returned a session outside the requested range."
        )
    return sessions


def _index_observations(
    contract: OptionContract, query: OptionDailyAcquisitionQuery, observations: object
) -> dict[date, OptionNativeDailyObservation]:
    """Key the source's output by session label, refusing anything malformed."""
    if not isinstance(observations, tuple):
        raise OptionHistoricalDataContractViolationError(
            "OptionNativeDailyMarketDataSource must return a tuple of OptionNativeDailyObservation."
        )
    by_date: dict[date, OptionNativeDailyObservation] = {}
    for index, observation in enumerate(observations):
        if not isinstance(observation, OptionNativeDailyObservation):
            raise OptionHistoricalDataContractViolationError(
                f"OptionNativeDailyMarketDataSource observation {index} "
                "must be an OptionNativeDailyObservation."
            )
        if observation.contract != contract:
            raise OptionHistoricalDataContractViolationError(
                f"OptionNativeDailyMarketDataSource observation {index} is for "
                f"{observation.contract}, not the requested contract {contract}."
            )
        if not query.start_trading_date <= observation.trading_date <= query.end_trading_date:
            raise OptionHistoricalDataContractViolationError(
                f"OptionNativeDailyMarketDataSource observation {index} is labelled "
                f"{observation.trading_date.isoformat()}, outside the requested range."
            )
        if observation.trading_date in by_date:
            raise OptionHistoricalDataContractViolationError(
                f"OptionNativeDailyMarketDataSource returned more than one candle for "
                f"{contract} session {observation.trading_date.isoformat()}."
            )
        by_date[observation.trading_date] = observation
    return by_date


def _require_coverage(
    contract: OptionContract,
    sessions: tuple[OptionTradingSession, ...],
    by_date: dict[date, OptionNativeDailyObservation],
) -> None:
    """Require exactly one candle per resolved session and none elsewhere."""
    expected = {session.trading_date for session in sessions}
    missing = tuple(
        session.trading_date for session in sessions if session.trading_date not in by_date
    )
    unexpected = tuple(sorted(label for label in by_date if label not in expected))
    if missing or unexpected:
        raise OptionDailySessionCoverageError(contract, missing, unexpected)


def _daily_bar(
    contract: OptionContract,
    session: OptionTradingSession,
    observation: OptionNativeDailyObservation,
) -> OptionOHLCVBar:
    """Build the canonical daily bar, stamped at the resolved session close."""
    try:
        return OptionOHLCVBar(
            contract=contract,
            point_in_time=session.closes_at,
            timeframe=_DAILY,
            open=observation.open,
            high=observation.high,
            low=observation.low,
            close=observation.close,
            volume=observation.volume,
        )
    except InvalidOptionOHLCVBarError as exc:
        raise OptionHistoricalDataContractViolationError(
            f"OptionNativeDailyMarketDataSource returned an incoherent candle for "
            f"{contract} session {session.trading_date.isoformat()}: {exc}"
        ) from exc
