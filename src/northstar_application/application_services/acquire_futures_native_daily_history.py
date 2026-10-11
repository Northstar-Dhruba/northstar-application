"""Application use case for acquiring and persisting provider-native daily futures bars.

This runs beside AcquireFuturesDailyHistoryUseCase, not in place of it. That
use case folds minute bars into a daily bar. This one takes a daily candle a
provider already publishes and turns it into the same canonical
FuturesOHLCVBar, so stored history and every reader of it remain unchanged.

The workflow is: resolve the sessions the range names, fetch the source's daily
candles for the range once, match each candle to exactly one resolved session,
stamp each bar at its session's close, and persist.

The calendar decides which sessions exist
-----------------------------------------
The session resolver is the sole authority on which labels are sessions. The
source and the calendar must agree exactly, and any disagreement is refused
rather than reconciled:

- a resolved session with no candle is a gap, not a no-trade day
- a candle for a label that is not a resolved session is unexplained data

The minute path treats a session with no bars as "nothing traded". That reading
does not carry over. A native daily provider publishes a candle for every
session in which a contract is listed, so a missing candle means data that has
not been published, or was lost. Storing nothing and reporting success would
make that indistinguishable from a complete history.

Finality is the caller's decision
---------------------------------
This use case reads no clock and has no view on whether a session's candle is
final. It acquires exactly the range it is given. A caller that names a session
whose candle the provider has not yet published gets a coverage error, which is
the right outcome. The cutoff for a safe range belongs to whoever composes the
runtime, once provider finality has been established.

Persistence is one batch, deliberately
--------------------------------------
The minute path stores per session because each session is a separate provider
call. Here the whole range arrives in one call and is fully validated before
anything is written, so a single batch is all-or-nothing at no cost: an invalid
range stores nothing. A rerun of a stored range is an idempotent success under
the store's semantics, and a differing value under an existing key is a
conflict that propagates unwrapped.

Reviewed candles
----------------
A caller may require that every candle is the exact version an operator
reviewed, by passing FuturesReviewedCandleVerification. That object is the
explicit choice and always carries an approval set, so reviewed mode can
never run without one; an empty set approves nothing and refuses every
session. Two checks then run, and neither can be skipped:

- after the sessions are resolved and before the provider is asked, every
  resolved session must have an approved fingerprint
  (FuturesCandleApprovalMissingError names every one that does not);
- after the existing structure and coverage checks have built every bar, and
  before the one store call, each bar's fdc1 fingerprint must equal its
  approved one (FuturesCandleApprovalMismatchError names every one that
  differs).

The bars that are fingerprinted are the very tuple passed to the store, and
no provider is asked twice, so nothing can change between the check and the
write. Any failure stores nothing. A bar already stored is still checked,
because the fetched candle is verified before the store compares it with
what it holds.

Without that object the use case behaves exactly as before. That keeps the
existing callers unchanged until their own reviewed-mode integration; it is
not by itself a guarantee that no caller can acquire unreviewed candles.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from northstar_core.foundation.value_objects import Timeframe
from northstar_core.futures import FuturesContract, FuturesOHLCVBar, InvalidFuturesOHLCVBarError

from northstar_application.application_services.acquire_futures_daily_history import (
    FuturesDailyAcquisitionResult,
    FuturesHistoricalDataContractViolationError,
)
from northstar_application.application_services.futures_candle_approvals import (
    FuturesCandleApprovalSet,
    InvalidFuturesCandleApprovalSetError,
)
from northstar_application.application_services.futures_daily_candle_identity import (
    futures_daily_candle_fingerprint,
)
from northstar_application.ports import (
    FuturesDailyHistoricalAcquisitionQuery,
    FuturesHistoricalMarketDataStore,
    FuturesNativeDailyMarketDataSource,
    FuturesNativeDailyObservation,
    FuturesTradingSession,
    FuturesTradingSessionResolver,
)

_DAILY = Timeframe("1d")


class FuturesDailySessionCoverageError(ValueError):
    """Raised when native daily candles and the session calendar disagree.

    ``missing_trading_dates`` are resolved sessions with no candle.
    ``unexpected_trading_dates`` are candles for labels the calendar does not
    recognise as sessions. Both are ascending, and at least one is non-empty.

    It is kept separate from FuturesHistoricalDataContractViolationError
    because the fault cannot be attributed from here. A missing candle may be
    one the provider has not yet published. An unexpected one may be a special
    session the calendar does not know about. Either way, the remedy is a
    decision outside this use case.
    """

    def __init__(
        self,
        contract: FuturesContract,
        missing_trading_dates: tuple[date, ...],
        unexpected_trading_dates: tuple[date, ...],
    ) -> None:
        self.contract = contract
        self.missing_trading_dates = missing_trading_dates
        self.unexpected_trading_dates = unexpected_trading_dates
        parts = []
        if missing_trading_dates:
            parts.append(
                "expected sessions with no candle: "
                + ", ".join(d.isoformat() for d in missing_trading_dates)
            )
        if unexpected_trading_dates:
            parts.append(
                "candles for non-session dates: "
                + ", ".join(d.isoformat() for d in unexpected_trading_dates)
            )
        super().__init__(
            f"Native daily candles for {contract} do not match the session calendar; "
            + "; ".join(parts)
            + "."
        )


@dataclass(frozen=True, slots=True)
class FuturesReviewedCandleVerification:
    """The explicit choice to accept only operator-reviewed candle versions.

    It cannot exist without an approval set, so selecting reviewed mode always
    supplies one. An empty set is still a set: it approves nothing.
    """

    approvals: FuturesCandleApprovalSet

    def __post_init__(self) -> None:
        if not isinstance(self.approvals, FuturesCandleApprovalSet):
            raise InvalidFuturesCandleApprovalSetError(
                "Reviewed candle verification requires a FuturesCandleApprovalSet."
            )


class AcquireFuturesNativeDailyHistoryUseCase:
    """Acquire, session-stamp and persist one contract's native daily candles."""

    def __init__(
        self,
        session_resolver: FuturesTradingSessionResolver,
        source: FuturesNativeDailyMarketDataSource,
        store: FuturesHistoricalMarketDataStore,
        *,
        reviewed_candles: FuturesReviewedCandleVerification | None = None,
    ) -> None:
        if not isinstance(session_resolver, FuturesTradingSessionResolver):
            raise TypeError(
                "AcquireFuturesNativeDailyHistoryUseCase session_resolver "
                "must be a FuturesTradingSessionResolver."
            )
        if not isinstance(source, FuturesNativeDailyMarketDataSource):
            raise TypeError(
                "AcquireFuturesNativeDailyHistoryUseCase source "
                "must be a FuturesNativeDailyMarketDataSource."
            )
        if not isinstance(store, FuturesHistoricalMarketDataStore):
            raise TypeError(
                "AcquireFuturesNativeDailyHistoryUseCase store "
                "must be a FuturesHistoricalMarketDataStore."
            )
        if reviewed_candles is not None and not isinstance(
            reviewed_candles, FuturesReviewedCandleVerification
        ):
            # A bare approval set is refused: enforcement is never implied.
            raise TypeError(
                "AcquireFuturesNativeDailyHistoryUseCase reviewed_candles "
                "must be a FuturesReviewedCandleVerification or None."
            )

        self._session_resolver = session_resolver
        self._source = source
        self._store = store
        self._reviewed = reviewed_candles

    def execute(
        self, query: FuturesDailyHistoricalAcquisitionQuery
    ) -> FuturesDailyAcquisitionResult:
        """Acquire every session in the range and persist one daily bar per session.

        Every resolved session yields a bar or the run fails, so on success
        ``daily_bar_count`` always equals ``session_count``.
        """
        if not isinstance(query, FuturesDailyHistoricalAcquisitionQuery):
            raise TypeError(
                "AcquireFuturesNativeDailyHistoryUseCase query "
                "must be a FuturesDailyHistoricalAcquisitionQuery."
            )

        sessions = self._session_resolver.sessions_in_range(
            query.contract.product,
            query.start_trading_date,
            query.end_trading_date,
        )
        if self._reviewed is not None:
            _require_approvals(self._reviewed.approvals, query.contract, sessions)

        observations = self._source.fetch_daily_observations(
            query.contract,
            query.start_trading_date,
            query.end_trading_date,
        )

        by_date = _index_observations(query.contract, observations)
        _require_coverage(query.contract, sessions, by_date)

        # Calendar order, not source order: each bar is placed by looking up
        # its session's label, so how the source ordered its output is
        # irrelevant to what is persisted.
        bars = tuple(
            _daily_bar(query.contract, session, by_date[session.trading_date])
            for session in sessions
        )

        if self._reviewed is not None and bars:
            # The same tuple is verified and then stored: nothing in between.
            _verify_bars(self._reviewed.approvals, query.contract, sessions, bars)

        if bars:
            self._persist(bars)

        return FuturesDailyAcquisitionResult(
            query=query,
            session_count=len(sessions),
            daily_bar_count=len(bars),
        )

    def _persist(self, bars: tuple[FuturesOHLCVBar, ...]) -> None:
        """Store the range as one batch, requiring the store to accept all of it.

        A conflict is never caught. It means stored evidence disagrees with
        what was just acquired, which needs an explicit reconciliation decision.
        """
        accepted = self._store.store(bars)

        if not isinstance(accepted, int) or isinstance(accepted, bool):
            raise FuturesHistoricalDataContractViolationError(
                "FuturesHistoricalMarketDataStore.store() must return an integer count."
            )
        if accepted != len(bars):
            raise FuturesHistoricalDataContractViolationError(
                f"FuturesHistoricalMarketDataStore.store() returned count {accepted}, "
                f"expected {len(bars)} for the acquired daily bars."
            )


def _index_observations(
    contract: FuturesContract, observations: object
) -> dict[date, FuturesNativeDailyObservation]:
    """Key the source's output by session label, refusing anything malformed.

    The caller controls the contract, so a mismatch is the source's fault. A
    duplicated label is refused even when both candles are equal: a source
    that repeats a candle has a defect, and picking one would hide which
    candle it meant.
    """
    if not isinstance(observations, tuple):
        raise FuturesHistoricalDataContractViolationError(
            "FuturesNativeDailyMarketDataSource must return a tuple of "
            "FuturesNativeDailyObservation."
        )

    by_date: dict[date, FuturesNativeDailyObservation] = {}
    for index, observation in enumerate(observations):
        if not isinstance(observation, FuturesNativeDailyObservation):
            raise FuturesHistoricalDataContractViolationError(
                f"FuturesNativeDailyMarketDataSource observation {index} "
                "must be a FuturesNativeDailyObservation."
            )
        if observation.contract != contract:
            raise FuturesHistoricalDataContractViolationError(
                f"FuturesNativeDailyMarketDataSource observation {index} is for "
                f"{observation.contract}, not the requested contract {contract}."
            )
        if observation.trading_date in by_date:
            raise FuturesHistoricalDataContractViolationError(
                f"FuturesNativeDailyMarketDataSource returned more than one candle for "
                f"{contract} session {observation.trading_date.isoformat()}."
            )
        by_date[observation.trading_date] = observation

    return by_date


def _require_coverage(
    contract: FuturesContract,
    sessions: tuple[FuturesTradingSession, ...],
    by_date: dict[date, FuturesNativeDailyObservation],
) -> None:
    """Require exactly one candle per resolved session and none elsewhere."""
    expected = {session.trading_date for session in sessions}
    missing = tuple(
        session.trading_date for session in sessions if session.trading_date not in by_date
    )
    unexpected = tuple(sorted(label for label in by_date if label not in expected))

    if missing or unexpected:
        raise FuturesDailySessionCoverageError(contract, missing, unexpected)


def _daily_bar(
    contract: FuturesContract,
    session: FuturesTradingSession,
    observation: FuturesNativeDailyObservation,
) -> FuturesOHLCVBar:
    """Build the canonical daily bar, stamped at the resolved session close.

    The contract is the requested one rather than the observation's echo.
    They are already known to be equal, and identity comes from the caller.
    """
    try:
        return FuturesOHLCVBar(
            contract=contract,
            point_in_time=session.closes_at,
            timeframe=_DAILY,
            open=observation.open,
            high=observation.high,
            low=observation.low,
            close=observation.close,
            volume=observation.volume,
        )
    except InvalidFuturesOHLCVBarError as exc:
        raise FuturesHistoricalDataContractViolationError(
            f"FuturesNativeDailyMarketDataSource returned an incoherent candle for "
            f"{contract} session {session.trading_date.isoformat()}: {exc}"
        ) from exc


def _require_approvals(
    approvals: FuturesCandleApprovalSet,
    contract: FuturesContract,
    sessions: tuple[FuturesTradingSession, ...],
) -> None:
    """Require an approval for every resolved session before any provider request."""
    if approvals.product != contract.product:
        raise InvalidFuturesCandleApprovalSetError(
            f"The candle approval set is for {approvals.product}, not {contract.product}."
        )
    if sessions:
        approvals.require_approvals(contract, (session.trading_date for session in sessions))


def _verify_bars(
    approvals: FuturesCandleApprovalSet,
    contract: FuturesContract,
    sessions: tuple[FuturesTradingSession, ...],
    bars: tuple[FuturesOHLCVBar, ...],
) -> None:
    """Require every bar about to be stored to be its session's approved version."""
    approvals.verify_fingerprints(
        contract,
        {
            session.trading_date: futures_daily_candle_fingerprint(
                contract, session.trading_date, bar
            )
            for session, bar in zip(sessions, bars, strict=True)
        },
    )
