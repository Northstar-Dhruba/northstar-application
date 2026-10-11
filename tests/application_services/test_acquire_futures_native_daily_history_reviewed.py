"""Reviewed-candle verification inside native daily futures acquisition.

Every collaborator is a recording fake and every value is synthetic. Approved
fingerprints are produced by the merged fdc1 implementation over synthetic
candles and published through the merged manifest parser, so these tests
exercise exactly the APIs the use case relies on.

The fake store mirrors the real store's contract: an absent key is inserted,
an equal stored bar is accepted, a differing one raises a conflict and the
whole batch stores nothing.
"""

from __future__ import annotations

import ast
import inspect
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest
from northstar_core.derivatives import ExpirationDate, QuoteValue
from northstar_core.foundation.value_objects import (
    ExchangeCode,
    PointInTime,
    Quantity,
    Symbol,
    Timeframe,
)
from northstar_core.futures import FuturesContract, FuturesOHLCVBar, FuturesProductReference

import northstar_application.application_services.acquire_futures_native_daily_history as module
from northstar_application.application_services import (
    AcquireFuturesNativeDailyHistoryUseCase,
    FuturesCandleApprovalMismatchError,
    FuturesCandleApprovalMissingError,
    FuturesCandleApprovalSet,
    FuturesDailySessionCoverageError,
    FuturesHistoricalDataContractViolationError,
    FuturesReviewedCandleVerification,
    InvalidFuturesCandleApprovalSetError,
    InvalidFuturesDailyCandleIdentityError,
    futures_daily_candle_fingerprint,
    parse_futures_candle_approval_manifest,
)
from northstar_application.ports import (
    FuturesDailyHistoricalAcquisitionQuery,
    FuturesHistoricalMarketDataConflictError,
    FuturesHistoricalMarketDataStore,
    FuturesNativeDailyMarketDataSource,
    FuturesNativeDailyObservation,
    FuturesTradingSession,
    FuturesTradingSessionResolver,
)

_NIFTY = FuturesProductReference(Symbol("NIFTY"), ExchangeCode("NSE"))
_BANKNIFTY = FuturesProductReference(Symbol("BANKNIFTY"), ExchangeCode("NSE"))
_OCT = FuturesContract(_NIFTY, ExpirationDate("2026-10-27"))
_NOV = FuturesContract(_NIFTY, ExpirationDate("2026-11-24"))
_BANK_OCT = FuturesContract(_BANKNIFTY, ExpirationDate("2026-10-27"))

_D1, _D2, _D3 = date(2026, 9, 29), date(2026, 9, 30), date(2026, 10, 1)
_HOLIDAY = date(2026, 10, 2)
_SESSIONS = (_D1, _D2, _D3)
_CLOSES = {_D1: "100", _D2: "100.5", _D3: "101"}


def _session(label: date) -> FuturesTradingSession:
    day = label.isoformat()
    return FuturesTradingSession(
        label, PointInTime(f"{day}T03:45:00Z"), PointInTime(f"{day}T10:10:00Z")
    )


def _obs(
    label: date,
    *,
    contract: FuturesContract = _OCT,
    open: str = "100.25",
    high: str = "102.5",
    low: str = "99.5",
    close: str | None = None,
    volume: str = "42",
) -> FuturesNativeDailyObservation:
    return FuturesNativeDailyObservation(
        contract=contract,
        trading_date=label,
        open=QuoteValue(Decimal(open)),
        high=QuoteValue(Decimal(high)),
        low=QuoteValue(Decimal(low)),
        close=QuoteValue(Decimal(_CLOSES[label] if close is None else close)),
        volume=Quantity(Decimal(volume)),
    )


def _bar(observation: FuturesNativeDailyObservation) -> FuturesOHLCVBar:
    """The canonical bar the use case builds for an observation."""
    return FuturesOHLCVBar(
        contract=observation.contract,
        point_in_time=_session(observation.trading_date).closes_at,
        timeframe=Timeframe("1d"),
        open=observation.open,
        high=observation.high,
        low=observation.low,
        close=observation.close,
        volume=observation.volume,
    )


def _fingerprint(observation: FuturesNativeDailyObservation) -> str:
    return futures_daily_candle_fingerprint(
        observation.contract, observation.trading_date, _bar(observation)
    )


def _manifest(
    entries: list[tuple[str, date, str]], product: FuturesProductReference = _NIFTY
) -> FuturesCandleApprovalSet:
    text = ",".join(f"{expiry}/{label.isoformat()}={fp}" for expiry, label, fp in entries)
    return parse_futures_candle_approval_manifest(text, product)


def _approve(*observations: FuturesNativeDailyObservation) -> FuturesCandleApprovalSet:
    """Approve exactly these synthetic candles, as an operator's manifest would."""
    return _manifest(
        [(o.contract.expiration_date.value, o.trading_date, _fingerprint(o)) for o in observations]
    )


def _week() -> tuple[FuturesNativeDailyObservation, ...]:
    return tuple(_obs(label) for label in _SESSIONS)


_APPROVED_WEEK = _approve(*_week())


class Resolver(FuturesTradingSessionResolver):
    def __init__(self, labels: tuple[date, ...] = _SESSIONS) -> None:
        self.labels = labels
        self.calls = 0

    def resolve(self, product, trading_date):  # pragma: no cover - unused
        raise NotImplementedError

    def sessions_in_range(self, product, start_date, end_date):
        self.calls += 1
        return tuple(_session(label) for label in self.labels if start_date <= label <= end_date)


class Source(FuturesNativeDailyMarketDataSource):
    def __init__(self, observations: tuple[FuturesNativeDailyObservation, ...]) -> None:
        self.observations = observations
        self.calls: list[tuple[FuturesContract, date, date]] = []

    def fetch_daily_observations(self, contract, start_trading_date, end_trading_date):
        self.calls.append((contract, start_trading_date, end_trading_date))
        return self.observations


class Store(FuturesHistoricalMarketDataStore):
    """Atomic batch: insert absent keys, accept equal ones, refuse differing ones."""

    def __init__(self, existing: tuple[FuturesOHLCVBar, ...] = ()) -> None:
        self.by_key = {bar.natural_key: bar for bar in existing}
        self.calls: list[tuple[FuturesOHLCVBar, ...]] = []

    def store(self, bars: tuple[FuturesOHLCVBar, ...]) -> int:
        self.calls.append(bars)
        for bar in bars:
            stored = self.by_key.get(bar.natural_key)
            if stored is not None and stored != bar:
                raise FuturesHistoricalMarketDataConflictError(
                    "A different futures bar is already stored under this natural key."
                )
        for bar in bars:
            self.by_key.setdefault(bar.natural_key, bar)
        return len(bars)

    def snapshot(self) -> dict:
        return dict(self.by_key)


def _run(
    approvals: FuturesCandleApprovalSet,
    observations: tuple[FuturesNativeDailyObservation, ...],
    *,
    store: Store | None = None,
    start: date = _D1,
    end: date = _D3,
    contract: FuturesContract = _OCT,
    resolver: Resolver | None = None,
):
    source = Source(observations)
    store = Store() if store is None else store
    use_case = AcquireFuturesNativeDailyHistoryUseCase(
        resolver or Resolver(),
        source,
        store,
        reviewed_candles=FuturesReviewedCandleVerification(approvals),
    )
    query = FuturesDailyHistoricalAcquisitionQuery(contract, start, end)
    return use_case, query, source, store


def _refuse(error_type, approvals, observations, *, store=None, **kwargs):
    """Run and return the error, asserting the store's contents never changed."""
    use_case, query, source, store = _run(approvals, observations, store=store, **kwargs)
    before = store.snapshot()
    with pytest.raises(error_type) as raised:
        use_case.execute(query)
    assert store.snapshot() == before
    return raised.value, source, store


# ---------------------------------------------------------------------------
# Approved candles are acquired
# ---------------------------------------------------------------------------


def test_one_approved_matching_session_is_stored() -> None:
    observation = _obs(_D1)
    use_case, query, source, store = _run(_approve(observation), (observation,), end=_D1)

    result = use_case.execute(query)

    assert (result.session_count, result.daily_bar_count) == (1, 1)
    assert len(source.calls) == 1
    assert store.calls == [(_bar(observation),)]


def test_several_approved_matching_sessions_are_stored_in_one_batch() -> None:
    use_case, query, source, store = _run(_APPROVED_WEEK, _week())

    result = use_case.execute(query)

    assert (result.session_count, result.daily_bar_count) == (3, 3)
    assert store.calls == [tuple(_bar(o) for o in _week())]


@pytest.mark.parametrize(
    "spelling",
    [
        {"open": "100.250", "high": "102.50", "low": "99.50", "close": "100.00"},
        {"open": "1.0025E+2", "high": "1.025E2", "low": "995E-1", "close": "1E+2"},
        {"volume": "42.000"},
    ],
)
def test_a_decimal_equivalent_provider_spelling_matches(spelling: dict[str, str]) -> None:
    approvals = _approve(_obs(_D1))
    use_case, query, _, store = _run(approvals, (_obs(_D1, **spelling),), end=_D1)

    assert use_case.execute(query).daily_bar_count == 1
    assert len(store.calls) == 1


# ---------------------------------------------------------------------------
# Before the provider request: every session must be approved
# ---------------------------------------------------------------------------


def test_one_session_without_approval_is_refused_before_the_provider() -> None:
    error, source, store = _refuse(
        FuturesCandleApprovalMissingError, _manifest([]), (_obs(_D1),), end=_D1
    )

    assert (error.contract, error.trading_dates) == (_OCT, (_D1,))
    assert source.calls == [] and store.calls == []


@pytest.mark.parametrize("unapproved", _SESSIONS, ids=["first", "middle", "final"])
def test_one_missing_approval_in_a_range_refuses_the_whole_range(unapproved: date) -> None:
    approvals = _approve(*(o for o in _week() if o.trading_date != unapproved))

    error, source, store = _refuse(FuturesCandleApprovalMissingError, approvals, _week())

    assert (error.contract, error.trading_dates) == (_OCT, (unapproved,))
    assert unapproved.isoformat() in str(error)
    assert source.calls == [] and store.calls == []


def test_every_missing_session_is_named() -> None:
    error, source, _ = _refuse(FuturesCandleApprovalMissingError, _approve(_obs(_D2)), _week())

    assert error.trading_dates == (_D1, _D3)
    assert source.calls == []


def test_an_empty_approval_set_still_verifies_and_refuses_everything() -> None:
    error, source, store = _refuse(FuturesCandleApprovalMissingError, _manifest([]), _week())

    assert error.trading_dates == _SESSIONS
    assert source.calls == [] and store.calls == []


def test_approving_the_final_date_does_not_approve_earlier_sessions() -> None:
    error, source, _ = _refuse(FuturesCandleApprovalMissingError, _approve(_obs(_D3)), _week())

    assert error.trading_dates == (_D1, _D2)
    assert source.calls == []


def test_an_approval_for_another_expiry_does_not_count() -> None:
    november = _manifest([("2026-11-24", _D1, _fingerprint(_obs(_D1, contract=_NOV)))])

    error, source, _ = _refuse(FuturesCandleApprovalMissingError, november, (_obs(_D1),), end=_D1)

    assert (error.contract, error.trading_dates) == (_OCT, (_D1,))
    assert source.calls == []


def test_an_approval_for_a_neighbouring_date_does_not_count() -> None:
    error, source, _ = _refuse(
        FuturesCandleApprovalMissingError, _approve(_obs(_D2)), (_obs(_D1),), end=_D1
    )

    assert error.trading_dates == (_D1,)
    assert source.calls == []


def test_an_approval_set_for_another_product_is_refused_as_configuration() -> None:
    banknifty = _manifest([], product=_BANKNIFTY)

    error, source, store = _refuse(
        InvalidFuturesCandleApprovalSetError, banknifty, (_obs(_D1),), end=_D1
    )

    assert str(error) == "The candle approval set is for BANKNIFTY@NSE, not NIFTY@NSE."
    assert source.calls == [] and store.calls == []


# ---------------------------------------------------------------------------
# After the fetch, before the write: every bar must be the approved version
# ---------------------------------------------------------------------------


def test_one_mismatched_session_is_refused_before_the_store() -> None:
    error, source, store = _refuse(
        FuturesCandleApprovalMismatchError,
        _approve(_obs(_D1)),
        (_obs(_D1, close="100.05"),),
        end=_D1,
    )

    assert (error.contract, error.trading_dates) == (_OCT, (_D1,))
    assert len(source.calls) == 1
    assert store.calls == []


@pytest.mark.parametrize("changed", _SESSIONS, ids=["first", "middle", "final"])
def test_one_mismatch_anywhere_stores_no_part_of_the_batch(changed: date) -> None:
    served = tuple(
        _obs(label, close="99.75") if label == changed else _obs(label) for label in _SESSIONS
    )
    existing = Store()

    error, _, store = _refuse(
        FuturesCandleApprovalMismatchError, _APPROVED_WEEK, served, store=existing
    )

    assert error.trading_dates == (changed,)
    assert store.calls == []
    assert store.by_key == {}


@pytest.mark.parametrize(
    "change",
    [
        {"open": "100.3"},
        {"high": "102.55"},
        {"low": "99.45"},
        {"close": "100.01"},
        {"volume": "43"},
        {"close": "100.0000000000000000000001"},
    ],
    ids=["open", "high", "low", "close", "volume", "deep-decimal"],
)
def test_each_economic_change_is_a_mismatch(change: dict[str, str]) -> None:
    error, source, store = _refuse(
        FuturesCandleApprovalMismatchError, _approve(_obs(_D1)), (_obs(_D1, **change),), end=_D1
    )

    assert error.trading_dates == (_D1,)
    assert len(source.calls) == 1 and store.calls == []


def test_every_mismatched_session_is_named() -> None:
    served = (_obs(_D1, close="99.75"), _obs(_D2), _obs(_D3, volume="41"))

    error, _, store = _refuse(FuturesCandleApprovalMismatchError, _APPROVED_WEEK, served)

    assert error.trading_dates == (_D1, _D3)
    assert store.calls == []


def test_a_fingerprint_of_another_contracts_candle_is_a_mismatch() -> None:
    for other in (_BANK_OCT, _NOV):
        approvals = _manifest([("2026-10-27", _D1, _fingerprint(_obs(_D1, contract=other)))])

        error, _, store = _refuse(
            FuturesCandleApprovalMismatchError, approvals, (_obs(_D1),), end=_D1
        )

        assert (error.contract, error.trading_dates) == (_OCT, (_D1,))
        assert store.calls == []


def test_a_fingerprint_of_another_sessions_candle_is_a_mismatch() -> None:
    # The values are identical; only the session the fingerprint was made for differs.
    same_values_other_day = _obs(_D2, close=_CLOSES[_D1])
    approvals = _manifest([("2026-10-27", _D1, _fingerprint(same_values_other_day))])

    error, _, store = _refuse(FuturesCandleApprovalMismatchError, approvals, (_obs(_D1),), end=_D1)

    assert error.trading_dates == (_D1,)
    assert store.calls == []


def test_no_message_repeats_a_fingerprint() -> None:
    approved = _fingerprint(_obs(_D1))
    error, _, _ = _refuse(
        FuturesCandleApprovalMismatchError,
        _approve(_obs(_D1)),
        (_obs(_D1, close="100.05"),),
        end=_D1,
    )

    assert approved not in str(error)
    assert _fingerprint(_obs(_D1, close="100.05")) not in str(error)


def test_a_candle_without_an_fdc1_identity_is_refused_not_stored() -> None:
    tiny = _obs(_D1, open="0.0000001", low="0.0000001", high="1", close="1")
    approvals = _manifest([("2026-10-27", _D1, "fdc1-sha256:" + "a" * 64)])

    _, _, store = _refuse(InvalidFuturesDailyCandleIdentityError, approvals, (tiny,), end=_D1)

    assert store.calls == []


# ---------------------------------------------------------------------------
# Existing structure, coverage and storage rules are unchanged and run first
# ---------------------------------------------------------------------------


def test_a_missing_provider_session_keeps_the_coverage_error() -> None:
    error, _, store = _refuse(FuturesDailySessionCoverageError, _APPROVED_WEEK, _week()[:2])

    assert error.missing_trading_dates == (_D3,)
    assert store.calls == []


def test_a_duplicate_provider_session_keeps_the_contract_violation() -> None:
    _, _, store = _refuse(
        FuturesHistoricalDataContractViolationError,
        _APPROVED_WEEK,
        (*_week(), _obs(_D1)),
    )

    assert store.calls == []


def test_an_unexpected_provider_session_keeps_the_coverage_error() -> None:
    holiday = FuturesNativeDailyObservation(
        _OCT, _HOLIDAY, *(QuoteValue(Decimal("100")),) * 4, Quantity(Decimal("1"))
    )

    error, _, store = _refuse(FuturesDailySessionCoverageError, _APPROVED_WEEK, (*_week(), holiday))

    assert error.unexpected_trading_dates == (_HOLIDAY,)
    assert store.calls == []


def test_a_candle_for_another_contract_keeps_the_contract_violation() -> None:
    _, _, store = _refuse(
        FuturesHistoricalDataContractViolationError,
        _approve(_obs(_D1)),
        (_obs(_D1, contract=_NOV),),
        end=_D1,
    )

    assert store.calls == []


def test_an_identical_stored_bar_is_accepted_and_still_verified(monkeypatch) -> None:
    seen = _spy_fingerprints(monkeypatch)
    observation = _obs(_D1)
    existing = Store((_bar(observation),))
    use_case, query, _, store = _run(_approve(observation), (observation,), end=_D1, store=existing)

    assert use_case.execute(query).daily_bar_count == 1
    assert len(seen) == 1
    assert list(store.by_key.values()) == [_bar(observation)]


def test_a_conflicting_stored_bar_keeps_the_store_conflict() -> None:
    approved = _obs(_D1)
    existing = Store((_bar(_obs(_D1, close="99.75")),))

    _refuse(
        FuturesHistoricalMarketDataConflictError,
        _approve(approved),
        (approved,),
        end=_D1,
        store=existing,
    )

    assert list(existing.by_key.values()) == [_bar(_obs(_D1, close="99.75"))]


def test_a_changed_candle_over_a_stored_bar_is_a_mismatch_not_a_conflict() -> None:
    approved = _obs(_D1)
    existing = Store((_bar(approved),))

    error, _, store = _refuse(
        FuturesCandleApprovalMismatchError,
        _approve(approved),
        (_obs(_D1, close="100.05"),),
        end=_D1,
        store=existing,
    )

    assert error.trading_dates == (_D1,)
    assert store.calls == []
    assert list(store.by_key.values()) == [_bar(approved)]


def test_a_repeated_acquisition_is_verified_every_time(monkeypatch) -> None:
    seen = _spy_fingerprints(monkeypatch)
    store = Store()
    for _ in range(2):
        use_case, query, _, store = _run(_APPROVED_WEEK, _week(), store=store)
        assert use_case.execute(query).daily_bar_count == 3

    assert len(seen) == 6
    assert len(store.calls) == 2 and len(store.by_key) == 3


# ---------------------------------------------------------------------------
# The verified bars are exactly the stored bars
# ---------------------------------------------------------------------------


def _spy_fingerprints(monkeypatch) -> list[FuturesOHLCVBar]:
    seen: list[FuturesOHLCVBar] = []
    real = module.futures_daily_candle_fingerprint

    def spy(contract, trading_date, bar):
        seen.append(bar)
        return real(contract, trading_date, bar)

    monkeypatch.setattr(module, "futures_daily_candle_fingerprint", spy)
    return seen


def test_the_fingerprinted_bars_are_the_very_objects_stored(monkeypatch) -> None:
    seen = _spy_fingerprints(monkeypatch)
    use_case, query, source, store = _run(_APPROVED_WEEK, _week())

    use_case.execute(query)

    assert len(store.calls) == 1
    assert len(seen) == len(store.calls[0]) == 3
    assert all(stored is verified for stored, verified in zip(store.calls[0], seen, strict=True))
    assert len(source.calls) == 1  # one provider response, verified and stored


def test_without_reviewed_verification_nothing_is_fingerprinted(monkeypatch) -> None:
    seen = _spy_fingerprints(monkeypatch)
    store = Store()
    use_case = AcquireFuturesNativeDailyHistoryUseCase(Resolver(), Source(_week()), store)

    assert (
        use_case.execute(FuturesDailyHistoricalAcquisitionQuery(_OCT, _D1, _D3)).daily_bar_count
        == 3
    )
    assert seen == []


# ---------------------------------------------------------------------------
# The enforcement contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("missing", [None, "", (), "fdc1-sha256:" + "a" * 64])
def test_reviewed_mode_cannot_be_selected_without_an_approval_set(missing: object) -> None:
    with pytest.raises(
        InvalidFuturesCandleApprovalSetError,
        match="^Reviewed candle verification requires a FuturesCandleApprovalSet.$",
    ):
        FuturesReviewedCandleVerification(missing)  # type: ignore[arg-type]


def test_a_bare_approval_set_never_implies_enforcement() -> None:
    source, store = Source(_week()), Store()

    with pytest.raises(TypeError, match="must be a FuturesReviewedCandleVerification or None"):
        AcquireFuturesNativeDailyHistoryUseCase(
            Resolver(),
            source,
            store,
            reviewed_candles=_APPROVED_WEEK,  # type: ignore[arg-type]
        )
    assert source.calls == [] and store.calls == []


def test_the_enforcement_option_is_keyword_only_and_immutable() -> None:
    parameter = inspect.signature(AcquireFuturesNativeDailyHistoryUseCase).parameters[
        "reviewed_candles"
    ]
    verification = FuturesReviewedCandleVerification(_APPROVED_WEEK)

    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is None
    with pytest.raises(AttributeError):
        verification.approvals = _manifest([])  # type: ignore[misc]


def test_acquisition_never_touches_paper_trading_state() -> None:
    """The use case has no decision, order or fill collaborator to change."""
    parameters = set(inspect.signature(AcquireFuturesNativeDailyHistoryUseCase).parameters)
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    names = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }

    assert parameters == {"session_resolver", "source", "store", "reviewed_candles"}
    assert not any(
        word in name for name in names for word in ("Paper", "Order", "Fill", "Forward", "Decision")
    )
    assert not {
        str(node.module).split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    } & {"northstar_infrastructure", "northstar_api", "sqlite3", "os"}


def test_the_public_surface_is_exported() -> None:
    import northstar_application.application_services as services

    assert "FuturesReviewedCandleVerification" in services.__all__
