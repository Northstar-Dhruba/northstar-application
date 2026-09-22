"""Contract tests for the FuturesHistoricalMarketDataStore Application port.

The port cannot enforce its own semantics, so the obligations are pinned here
against a conforming in-memory reference implementation. An adapter that fails
these behaviours is not a FuturesHistoricalMarketDataStore, whatever its
storage engine.

The central obligation is that evidence is never silently rewritten, which is
where this port deliberately parts company with HistoricalMarketDataStore.
"""

from __future__ import annotations

from decimal import Decimal, localcontext

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

from northstar_application.ports import (
    FuturesHistoricalMarketDataConflictError,
    FuturesHistoricalMarketDataStore,
    HistoricalMarketDataStore,
)

_CME = ExchangeCode("CME")
_NYMEX = ExchangeCode("NYMEX")

_ES = FuturesProductReference(Symbol("ES"), _CME)
_MES = FuturesProductReference(Symbol("MES"), _CME)
_CL = FuturesProductReference(Symbol("CL"), _NYMEX)

_MARCH = ExpirationDate("2026-03-20")
_JUNE = ExpirationDate("2026-06-19")

_ES_MARCH = FuturesContract(_ES, _MARCH)
_MES_MARCH = FuturesContract(_MES, _MARCH)
_ES_JUNE = FuturesContract(_ES, _JUNE)
_CL_MAY = FuturesContract(_CL, ExpirationDate("2020-05-19"))

_DAILY = Timeframe("1d")
_HOURLY = Timeframe("1h")


def _instant(day: int, suffix: str = "") -> PointInTime:
    return PointInTime(f"2026-01-{day:02d}T21:00:00{suffix}Z")


def _bar(
    contract: FuturesContract = _ES_MARCH,
    *,
    day: int = 15,
    point_in_time: PointInTime | None = None,
    timeframe: Timeframe = _DAILY,
    open_quote: str = "5430.00",
    high: str = "5450.25",
    low: str = "5425.50",
    close: str = "5442.75",
    volume: str = "1250000",
) -> FuturesOHLCVBar:
    return FuturesOHLCVBar(
        contract=contract,
        point_in_time=point_in_time if point_in_time is not None else _instant(day),
        timeframe=timeframe,
        open=QuoteValue(Decimal(open_quote)),
        high=QuoteValue(Decimal(high)),
        low=QuoteValue(Decimal(low)),
        close=QuoteValue(Decimal(close)),
        volume=Quantity(Decimal(volume)),
    )


class ReferenceFuturesHistoricalMarketDataStore(FuturesHistoricalMarketDataStore):
    """A minimal store honouring every documented obligation."""

    def __init__(self) -> None:
        self.by_key: dict[tuple[FuturesContract, PointInTime, Timeframe], FuturesOHLCVBar] = {}

    def store(self, bars: tuple[FuturesOHLCVBar, ...]) -> int:
        if not bars:
            return 0

        batch_keys: set[tuple[FuturesContract, PointInTime, Timeframe]] = set()
        for bar in bars:
            if bar.natural_key in batch_keys:
                raise FuturesHistoricalMarketDataConflictError(
                    "Batch contains two bars sharing one natural key."
                )
            batch_keys.add(bar.natural_key)

        # Stage everything, so a failure late in the batch persists nothing.
        staged: dict[tuple[FuturesContract, PointInTime, Timeframe], FuturesOHLCVBar] = {}
        for bar in bars:
            stored = self.by_key.get(bar.natural_key)
            if stored is not None and stored != bar:
                raise FuturesHistoricalMarketDataConflictError(
                    "A different bar is already stored under this natural key."
                )
            if stored is None:
                staged[bar.natural_key] = bar

        self.by_key.update(staged)
        return len(bars)


# ---------------------------------------------------------------------------
# Accepting new evidence
# ---------------------------------------------------------------------------


def test_storing_a_new_bar_returns_the_batch_size() -> None:
    store = ReferenceFuturesHistoricalMarketDataStore()

    assert store.store((_bar(),)) == 1


def test_storing_a_batch_returns_the_whole_batch_size() -> None:
    store = ReferenceFuturesHistoricalMarketDataStore()

    assert store.store(tuple(_bar(day=day) for day in (14, 15, 16))) == 3
    assert len(store.by_key) == 3


def test_storing_an_empty_tuple_is_a_safe_no_op_returning_zero() -> None:
    """The house convention, shared with every other store port."""
    store = ReferenceFuturesHistoricalMarketDataStore()

    assert store.store(()) == 0
    assert store.by_key == {}


def test_bars_are_keyed_by_contract_instant_and_timeframe() -> None:
    store = ReferenceFuturesHistoricalMarketDataStore()

    store.store((_bar(),))

    assert list(store.by_key) == [(_ES_MARCH, _instant(15), _DAILY)]


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


def test_re_storing_an_identical_bar_is_idempotent() -> None:
    store = ReferenceFuturesHistoricalMarketDataStore()
    bar = _bar()

    assert store.store((bar,)) == 1
    assert store.store((bar,)) == 1
    assert len(store.by_key) == 1


def test_a_retry_with_an_equal_but_distinct_bar_is_idempotent() -> None:
    """A retried ingestion rebuilds its bars; equal by value is the same evidence."""
    store = ReferenceFuturesHistoricalMarketDataStore()
    first, second = _bar(), _bar()

    assert first is not second
    store.store((first,))

    assert store.store((second,)) == 1
    assert len(store.by_key) == 1


def test_an_idempotent_re_store_counts_as_accepted() -> None:
    store = ReferenceFuturesHistoricalMarketDataStore()
    bars = tuple(_bar(day=day) for day in (14, 15, 16))
    store.store(bars)

    assert store.store(bars) == 3


def test_a_partially_overlapping_batch_is_accepted_whole() -> None:
    store = ReferenceFuturesHistoricalMarketDataStore()
    store.store((_bar(day=14), _bar(day=15)))

    assert store.store((_bar(day=15), _bar(day=16))) == 2
    assert len(store.by_key) == 3


def test_equivalent_quote_spellings_are_the_same_evidence() -> None:
    """Canonicalization means 5430 and 5430.0000 are one value, not a conflict."""
    store = ReferenceFuturesHistoricalMarketDataStore()
    store.store((_bar(open_quote="5430.00"),))

    assert store.store((_bar(open_quote="5430.0000"),)) == 1
    assert len(store.by_key) == 1


def test_an_offset_equivalent_instant_is_the_same_key() -> None:
    store = ReferenceFuturesHistoricalMarketDataStore()
    store.store((_bar(point_in_time=PointInTime("2026-01-15T21:00:00Z")),))

    assert store.store((_bar(point_in_time=PointInTime("2026-01-16T02:30:00+05:30")),)) == 1
    assert len(store.by_key) == 1


# ---------------------------------------------------------------------------
# Conflict: evidence is never silently rewritten
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("open_quote", "5431.00"),
        ("high", "5460.25"),
        ("low", "5420.50"),
        ("close", "5443.75"),
        ("volume", "1250001"),
    ],
)
def test_a_differing_bar_under_one_key_raises(field: str, value: str) -> None:
    store = ReferenceFuturesHistoricalMarketDataStore()
    store.store((_bar(),))

    with pytest.raises(FuturesHistoricalMarketDataConflictError):
        store.store((_bar(**{field: value}),))


def test_a_conflicting_store_does_not_overwrite_the_stored_bar() -> None:
    """The point of the rule: the original evidence survives the attempt."""
    store = ReferenceFuturesHistoricalMarketDataStore()
    original = _bar(close="5442.75")
    store.store((original,))

    with pytest.raises(FuturesHistoricalMarketDataConflictError):
        store.store((_bar(close="5430.00"),))

    assert store.by_key[original.natural_key] == original
    assert store.by_key[original.natural_key].close == QuoteValue(Decimal("5442.75"))
    assert store.by_key[original.natural_key].close != QuoteValue(Decimal("5430.00"))


def test_a_conflict_late_in_a_batch_persists_nothing_from_that_batch() -> None:
    """Partial batch success is not permitted."""
    store = ReferenceFuturesHistoricalMarketDataStore()
    store.store((_bar(day=15),))

    with pytest.raises(FuturesHistoricalMarketDataConflictError):
        store.store((_bar(day=16), _bar(day=17), _bar(day=15, close="5430.00")))

    assert set(store.by_key) == {(_ES_MARCH, _instant(15), _DAILY)}


def test_a_batch_sharing_one_natural_key_raises() -> None:
    store = ReferenceFuturesHistoricalMarketDataStore()

    with pytest.raises(FuturesHistoricalMarketDataConflictError, match="Batch contains"):
        store.store((_bar(close="5442.75"), _bar(close="5430.00")))

    assert store.by_key == {}


def test_a_batch_repeating_one_identical_bar_raises() -> None:
    """Even equal duplicates surface, because a duplicated element is a caller defect."""
    store = ReferenceFuturesHistoricalMarketDataStore()
    bar = _bar()

    with pytest.raises(FuturesHistoricalMarketDataConflictError, match="Batch contains"):
        store.store((bar, bar))


def test_the_conflict_error_is_a_value_error() -> None:
    assert issubclass(FuturesHistoricalMarketDataConflictError, ValueError)


# ---------------------------------------------------------------------------
# Key isolation
# ---------------------------------------------------------------------------


def test_standard_and_micro_contracts_occupy_different_keys() -> None:
    """ES and MES share venue, underlying and expiry; storing both is not a conflict."""
    store = ReferenceFuturesHistoricalMarketDataStore()

    assert store.store((_bar(_ES_MARCH), _bar(_MES_MARCH))) == 2
    assert len(store.by_key) == 2


def test_two_expiries_of_one_product_occupy_different_keys() -> None:
    store = ReferenceFuturesHistoricalMarketDataStore()

    assert store.store((_bar(_ES_MARCH), _bar(_ES_JUNE))) == 2
    assert len(store.by_key) == 2


def test_two_timeframes_at_one_instant_occupy_different_keys() -> None:
    store = ReferenceFuturesHistoricalMarketDataStore()

    assert store.store((_bar(timeframe=_DAILY), _bar(timeframe=_HOURLY))) == 2
    assert len(store.by_key) == 2


def test_two_instants_of_one_contract_occupy_different_keys() -> None:
    store = ReferenceFuturesHistoricalMarketDataStore()

    assert store.store((_bar(day=15), _bar(day=16))) == 2
    assert len(store.by_key) == 2


def test_a_differing_micro_bar_does_not_conflict_with_the_standard_bar() -> None:
    store = ReferenceFuturesHistoricalMarketDataStore()
    store.store((_bar(_ES_MARCH, close="5442.75"),))

    assert store.store((_bar(_MES_MARCH, close="5430.00"),)) == 1


# ---------------------------------------------------------------------------
# Values survive storage unchanged
# ---------------------------------------------------------------------------


def test_negative_quotes_are_stored_unchanged() -> None:
    store = ReferenceFuturesHistoricalMarketDataStore()
    bar = _bar(
        _CL_MAY,
        point_in_time=PointInTime("2020-04-20T18:30:00Z"),
        open_quote="-14.00",
        high="-10.50",
        low="-40.32",
        close="-37.63",
        volume="248000",
    )

    assert store.store((bar,)) == 1
    assert store.by_key[bar.natural_key].close == QuoteValue(Decimal("-37.63"))


def test_a_negative_bar_re_stored_identically_is_idempotent() -> None:
    store = ReferenceFuturesHistoricalMarketDataStore()
    members = {
        "point_in_time": PointInTime("2020-04-20T18:30:00Z"),
        "open_quote": "-14.00",
        "high": "-10.50",
        "low": "-40.32",
        "close": "-37.63",
    }
    store.store((_bar(_CL_MAY, **members),))

    assert store.store((_bar(_CL_MAY, **members),)) == 1
    assert len(store.by_key) == 1


@pytest.mark.parametrize("precision", [6, 28, 50])
def test_high_precision_values_survive_storage(precision: int) -> None:
    high_precision = "5430.12345678901234567890123456789012345678901234567891"
    volume = "1234567890123456789012345678901234567890"

    with localcontext() as context:
        context.prec = precision
        store = ReferenceFuturesHistoricalMarketDataStore()
        bar = _bar(
            open_quote=high_precision, high="9999", low="0", close=high_precision, volume=volume
        )
        store.store((bar,))

    stored = store.by_key[bar.natural_key]
    assert str(stored.open.value) == high_precision
    assert str(stored.volume.value) == volume


def test_a_high_precision_bar_re_stored_under_another_precision_is_idempotent() -> None:
    """Context-independent canonicalization is what makes this safe."""
    high_precision = "5430.12345678901234567890123456789012345678901234567891"
    store = ReferenceFuturesHistoricalMarketDataStore()

    for precision in (6, 28, 50):
        with localcontext() as context:
            context.prec = precision
            assert (
                store.store(
                    (_bar(open_quote=high_precision, high="9999", low="0", close=high_precision),)
                )
                == 1
            )

    assert len(store.by_key) == 1


def test_a_truncated_high_precision_bar_is_a_conflict_not_a_silent_merge() -> None:
    """Two genuinely different values must never be reconciled by rounding."""
    store = ReferenceFuturesHistoricalMarketDataStore()
    store.store((_bar(open_quote="5430.123456789012345678901234567890", high="9999", low="0"),))

    with pytest.raises(FuturesHistoricalMarketDataConflictError):
        store.store((_bar(open_quote="5430.123456", high="9999", low="0"),))


# ---------------------------------------------------------------------------
# Port shape, and the divergence from the equity store
# ---------------------------------------------------------------------------


def test_the_port_is_abstract() -> None:
    with pytest.raises(TypeError):
        FuturesHistoricalMarketDataStore()  # type: ignore[abstract]


def test_the_port_exposes_only_store() -> None:
    assert FuturesHistoricalMarketDataStore.__abstractmethods__ == frozenset({"store"})


def test_the_futures_store_is_not_an_equity_store() -> None:
    """Separate contracts, so the equity overwrite rule cannot be inherited."""
    assert not issubclass(FuturesHistoricalMarketDataStore, HistoricalMarketDataStore)
    assert not issubclass(HistoricalMarketDataStore, FuturesHistoricalMarketDataStore)


def test_the_equity_store_still_documents_replacement() -> None:
    """Pins the difference this port was created to avoid inheriting.

    If the equity contract is ever tightened, this test should fail and the
    divergence note on FuturesHistoricalMarketDataStore should be revisited.
    """
    equity = HistoricalMarketDataStore.__doc__ or ""
    futures = FuturesHistoricalMarketDataStore.__doc__ or ""

    assert "replaces or updates the stored state" in equity
    assert "must NOT overwrite" in futures
