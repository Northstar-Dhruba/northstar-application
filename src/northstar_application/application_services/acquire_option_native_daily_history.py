"""Application use case for acquiring and persisting native daily option bars.

The workflow is: resolve the option sessions the range names, fetch the source's
daily candles for the exact contract once, match each candle to exactly one
resolved session, stamp each canonical OptionOHLCVBar at its session's close
with the daily timeframe, and persist the candles as one batch.

Observed-subset coverage (ADR-015)
----------------------------------
The option session resolver is the sole authority on which labels are sessions,
and the candles must be a subset of them:

- a candle for a label that is not a resolved session is unexplained data, and
  fails;
- a resolved session with no candle is a session without a provider candle. It
  is allowed and reported in ``missing_trading_dates``, and nothing is stored
  for it.

Listed option contracts often go a whole session without a trade, and providers
omit the daily candle for such a session rather than publish a zero-volume one.
Nothing is ever manufactured in its place: no zero bar, no carried-forward or
settlement-price bar, no synthetic close or volume. Nor is a missing candle
called a no-trade session: a provider master carries no historical listing
start, so the absence may also predate the contract's listing. A later
acquisition may store a candle the provider exposes afterwards.

This supersedes ADR-014's strict coverage for options only.

Finality is the caller's decision
---------------------------------
This use case reads no clock and makes no judgement about whether a candle is
final. Acquiring a candle records what the provider returned, nothing more.

Persistence is one batch
------------------------
The whole range is validated before anything is written, and the store is
called exactly once with every acquired bar -- an empty batch when the range
has no candle -- as one all-or-nothing batch. A rerun of a stored range is an
idempotent success, and a differing value under an existing key is a conflict
that propagates unwrapped.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

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

    ``unexpected_trading_dates`` are candles for labels that are not resolved
    sessions. ``missing_trading_dates`` are resolved sessions with no candle;
    under observed-subset coverage those are reported in the acquisition result
    instead, so the use case raises this error with it empty. Both are ascending,
    and at least one is non-empty.
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
    an idempotent re-store counts as accepted, so this is not a count of rows
    written. ``missing_trading_dates`` are the resolved sessions for which the
    source returned no candle -- sessions without a provider candle -- ascending
    and unique, and disjoint from the sessions the bars represent. Every session
    is one or the other:
    ``daily_bar_count + len(missing_trading_dates) == session_count``.
    """

    query: OptionDailyAcquisitionQuery
    session_count: int
    daily_bar_count: int
    missing_trading_dates: tuple[date, ...]

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
        missing = self.missing_trading_dates
        if not isinstance(missing, tuple) or not all(
            isinstance(day, date) and not isinstance(day, datetime) for day in missing
        ):
            raise TypeError(
                "OptionDailyAcquisitionResult missing_trading_dates must be a tuple of plain dates."
            )
        if list(missing) != sorted(set(missing)):
            raise ValueError(
                "OptionDailyAcquisitionResult missing_trading_dates must be strictly ascending."
            )
        if missing and not (
            self.query.start_trading_date <= missing[0]
            and missing[-1] <= self.query.end_trading_date
        ):
            raise ValueError(
                "OptionDailyAcquisitionResult missing_trading_dates must lie in the query range."
            )
        if self.daily_bar_count + len(missing) != self.session_count:
            raise ValueError(
                "OptionDailyAcquisitionResult daily_bar_count plus missing_trading_dates "
                "must equal session_count."
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
        """Acquire the range and persist one daily bar per session with a candle."""
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
        _require_observed_subset(query.contract, sessions, by_date)

        bars = tuple(
            _daily_bar(query.contract, session, by_date[session.trading_date])
            for session in sessions
            if session.trading_date in by_date
        )
        missing = tuple(
            session.trading_date for session in sessions if session.trading_date not in by_date
        )
        self._persist(bars)

        return OptionDailyAcquisitionResult(
            query=query,
            session_count=len(sessions),
            daily_bar_count=len(bars),
            missing_trading_dates=missing,
        )

    def _persist(self, bars: tuple[OptionOHLCVBar, ...]) -> None:
        """Store the bars as one batch, empty or not; a conflict is never caught."""
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


def _require_observed_subset(
    contract: OptionContract,
    sessions: tuple[OptionTradingSession, ...],
    by_date: dict[date, OptionNativeDailyObservation],
) -> None:
    """Require every candle to be for a resolved session; a session may lack one."""
    expected = {session.trading_date for session in sessions}
    unexpected = tuple(sorted(label for label in by_date if label not in expected))
    if unexpected:
        raise OptionDailySessionCoverageError(contract, (), unexpected)


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
