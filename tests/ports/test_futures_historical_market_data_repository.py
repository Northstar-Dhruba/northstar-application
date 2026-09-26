"""Contract tests for the FuturesHistoricalMarketDataRepository Application port.

The port cannot enforce its own semantics, so the obligations are pinned here
against a conforming in-memory reference implementation. An adapter that fails
these behaviours is not a FuturesHistoricalMarketDataRepository, whatever its
storage engine.
"""

from __future__ import annotations

from decimal import Decimal, localcontext
from functools import cmp_to_key

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
    FuturesHistoricalMarketDataQuery,
    FuturesHistoricalMarketDataRepository,
    InvalidFuturesHistoricalMarketDataQueryError,
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


class ReferenceFuturesHistoricalMarketDataRepository(FuturesHistoricalMarketDataRepository):
    """A minimal repository honouring every documented obligation."""

    def __init__(self, bars: tuple[FuturesOHLCVBar, ...] = ()) -> None:
        self.bars = bars

    def get_bars(self, query: FuturesHistoricalMarketDataQuery) -> tuple[FuturesOHLCVBar, ...]:
        matching = [
            bar
            for bar in self.bars
            if bar.contract == query.contract
            and bar.timeframe == query.timeframe
            and query.covers(bar.point_in_time)
        ]
        ordered = sorted(
            matching,
            key=cmp_to_key(lambda left, right: left.point_in_time.compare(right.point_in_time)),
        )
        return tuple(ordered)


def _query(**kwargs: object) -> FuturesHistoricalMarketDataQuery:
    members: dict[str, object] = {"contract": _ES_MARCH, "timeframe": _DAILY}
    members.update(kwargs)
    return FuturesHistoricalMarketDataQuery(**members)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Query construction and validation
# ---------------------------------------------------------------------------


def test_a_query_preserves_its_members() -> None:
    query = _query(start=_instant(1), end=_instant(31))

    assert query.contract == _ES_MARCH
    assert query.timeframe == _DAILY
    assert query.start == _instant(1)
    assert query.end == _instant(31)


def test_both_bounds_are_optional() -> None:
    query = _query()

    assert query.start is None
    assert query.end is None


@pytest.mark.parametrize(
    ("start", "end"),
    [(None, None), (_instant(1), None), (None, _instant(31)), (_instant(1), _instant(31))],
    ids=["open", "start_only", "end_only", "closed"],
)
def test_every_combination_of_bounds_is_constructible(
    start: PointInTime | None, end: PointInTime | None
) -> None:
    assert _query(start=start, end=end).covers(_instant(15))


def test_equal_endpoints_are_valid() -> None:
    """Matches HistoricalMarketDataQuery, where equal endpoints select one instant."""
    query = _query(start=_instant(15), end=_instant(15))

    assert query.covers(_instant(15))


def test_a_start_after_end_is_rejected() -> None:
    with pytest.raises(
        InvalidFuturesHistoricalMarketDataQueryError, match="start must be before or equal to end"
    ):
        _query(start=_instant(31), end=_instant(1))


def test_an_inverted_window_is_detected_semantically_not_textually() -> None:
    """Sub-second text ordering would call this window valid."""
    later = _instant(15, ".5")
    earlier = _instant(15)

    assert later.value < earlier.value  # the text trap
    with pytest.raises(
        InvalidFuturesHistoricalMarketDataQueryError, match="start must be before or equal to end"
    ):
        _query(start=later, end=earlier)


def test_offset_equivalent_endpoints_are_not_an_inverted_window() -> None:
    utc = PointInTime("2026-01-15T21:00:00Z")
    offset = PointInTime("2026-01-16T02:30:00+05:30")

    assert _query(start=utc, end=offset).covers(utc)
    assert _query(start=offset, end=utc).covers(offset)


@pytest.mark.parametrize(
    ("member", "value", "expected"),
    [
        ("contract", _ES, "contract must be a FuturesContract value"),
        ("contract", "ES@CME", "contract must be a FuturesContract value"),
        ("timeframe", "1d", "timeframe must be a Timeframe value"),
        ("start", "2026-01-15T21:00:00Z", "start must be a PointInTime value"),
        ("end", "2026-01-15T21:00:00Z", "end must be a PointInTime value"),
    ],
)
def test_wrong_member_types_are_rejected(member: str, value: object, expected: str) -> None:
    with pytest.raises(InvalidFuturesHistoricalMarketDataQueryError, match=expected):
        _query(**{member: value})


def test_the_query_cannot_be_made_by_a_broader_identity() -> None:
    """A product, an underlying or a listing cannot name one expiring contract."""
    query = _query()

    for absent in ("underlying", "product", "symbol", "listing_reference", "provider_symbol"):
        assert not hasattr(query, absent)
    assert set(FuturesHistoricalMarketDataQuery.__slots__) == {
        "contract",
        "timeframe",
        "start",
        "end",
    }


def test_the_query_is_immutable() -> None:
    query = _query()

    with pytest.raises(AttributeError):
        query.contract = _MES_MARCH


# ---------------------------------------------------------------------------
# Inclusive window semantics
# ---------------------------------------------------------------------------


def test_both_bounds_are_inclusive() -> None:
    """Matches HistoricalMarketDataQuery: start and end are both inclusive."""
    repository = ReferenceFuturesHistoricalMarketDataRepository(
        tuple(_bar(day=day) for day in (14, 15, 16, 17, 18))
    )

    result = repository.get_bars(_query(start=_instant(15), end=_instant(17)))

    assert [bar.point_in_time for bar in result] == [_instant(15), _instant(16), _instant(17)]


def test_an_equal_endpoint_window_selects_exactly_one_bar() -> None:
    repository = ReferenceFuturesHistoricalMarketDataRepository(
        tuple(_bar(day=day) for day in (14, 15, 16))
    )

    result = repository.get_bars(_query(start=_instant(15), end=_instant(15)))

    assert len(result) == 1
    assert result[0].point_in_time == _instant(15)


def test_an_open_window_returns_the_whole_contract_history() -> None:
    repository = ReferenceFuturesHistoricalMarketDataRepository(
        tuple(_bar(day=day) for day in (14, 15, 16))
    )

    assert len(repository.get_bars(_query())) == 3


def test_a_start_only_window_is_bounded_below_and_open_above() -> None:
    repository = ReferenceFuturesHistoricalMarketDataRepository(
        tuple(_bar(day=day) for day in (14, 15, 16))
    )

    result = repository.get_bars(_query(start=_instant(15)))

    assert [bar.point_in_time for bar in result] == [_instant(15), _instant(16)]


def test_an_end_only_window_is_open_below_and_bounded_above() -> None:
    repository = ReferenceFuturesHistoricalMarketDataRepository(
        tuple(_bar(day=day) for day in (14, 15, 16))
    )

    result = repository.get_bars(_query(end=_instant(15)))

    assert [bar.point_in_time for bar in result] == [_instant(14), _instant(15)]


def test_bars_outside_the_window_are_excluded() -> None:
    repository = ReferenceFuturesHistoricalMarketDataRepository(
        tuple(_bar(day=day) for day in (1, 15, 31))
    )

    result = repository.get_bars(_query(start=_instant(10), end=_instant(20)))

    assert [bar.point_in_time for bar in result] == [_instant(15)]


def test_an_offset_equivalent_bound_includes_the_matching_instant() -> None:
    """A bound written at a different UTC offset is the same instant."""
    bar = _bar(point_in_time=PointInTime("2026-01-15T21:00:00Z"))
    repository = ReferenceFuturesHistoricalMarketDataRepository((bar,))
    offset_bound = PointInTime("2026-01-16T02:30:00+05:30")

    result = repository.get_bars(_query(start=offset_bound, end=offset_bound))

    assert result == (bar,)


def test_a_sub_second_bound_is_applied_semantically() -> None:
    """Text comparison would exclude the whole second from this window."""
    whole = _bar(point_in_time=_instant(15))
    repository = ReferenceFuturesHistoricalMarketDataRepository((whole,))

    assert repository.get_bars(_query(start=_instant(15, ".5"))) == ()
    assert repository.get_bars(_query(end=_instant(15, ".5"))) == (whole,)


# ---------------------------------------------------------------------------
# Contract and timeframe isolation
# ---------------------------------------------------------------------------


def test_standard_and_micro_contracts_stay_isolated() -> None:
    """ES and MES share venue, underlying and expiry; they are not one series."""
    standard = _bar(_ES_MARCH, open_quote="5430", high="5450", low="5425", close="5442")
    micro = _bar(_MES_MARCH, open_quote="5430", high="5450", low="5425", close="5442")
    repository = ReferenceFuturesHistoricalMarketDataRepository((standard, micro))

    assert repository.get_bars(_query(contract=_ES_MARCH)) == (standard,)
    assert repository.get_bars(_query(contract=_MES_MARCH)) == (micro,)


def test_two_expiries_of_one_product_stay_isolated() -> None:
    march = _bar(_ES_MARCH)
    june = _bar(_ES_JUNE)
    repository = ReferenceFuturesHistoricalMarketDataRepository((march, june))

    assert repository.get_bars(_query(contract=_ES_MARCH)) == (march,)
    assert repository.get_bars(_query(contract=_ES_JUNE)) == (june,)


def test_different_products_on_different_venues_stay_isolated() -> None:
    equity_index = _bar(_ES_MARCH)
    crude = _bar(_CL_MAY, open_quote="20", high="21", low="19", close="20.5")
    repository = ReferenceFuturesHistoricalMarketDataRepository((equity_index, crude))

    assert repository.get_bars(_query(contract=_CL_MAY)) == (crude,)


def test_timeframes_stay_isolated() -> None:
    daily = _bar(timeframe=_DAILY)
    hourly = _bar(timeframe=_HOURLY)
    repository = ReferenceFuturesHistoricalMarketDataRepository((daily, hourly))

    assert repository.get_bars(_query(timeframe=_DAILY)) == (daily,)
    assert repository.get_bars(_query(timeframe=_HOURLY)) == (hourly,)


def test_matching_is_by_value_not_identity() -> None:
    """A rebuilt contract equal to the stored one must still match."""
    repository = ReferenceFuturesHistoricalMarketDataRepository((_bar(_ES_MARCH),))
    rebuilt = FuturesContract(
        FuturesProductReference(Symbol("ES"), ExchangeCode("CME")),
        ExpirationDate("2026-03-20"),
    )

    assert rebuilt is not _ES_MARCH
    assert len(repository.get_bars(_query(contract=rebuilt))) == 1


# ---------------------------------------------------------------------------
# Ordering
# ---------------------------------------------------------------------------


def test_results_are_returned_oldest_to_newest() -> None:
    repository = ReferenceFuturesHistoricalMarketDataRepository(
        tuple(_bar(day=day) for day in (17, 14, 16, 15))
    )

    result = repository.get_bars(_query())

    assert [bar.point_in_time for bar in result] == [_instant(day) for day in (14, 15, 16, 17)]


def test_ordering_survives_the_sub_second_text_trap() -> None:
    """A canonical instant drops zero fractions, so '.5Z' sorts before 'Z' as text."""
    whole = _bar(point_in_time=_instant(15))
    fractional = _bar(point_in_time=_instant(15, ".5"))
    repository = ReferenceFuturesHistoricalMarketDataRepository((fractional, whole))

    result = repository.get_bars(_query())

    assert fractional.point_in_time.value < whole.point_in_time.value  # the trap
    assert result == (whole, fractional)


def test_ordering_is_stable_regardless_of_stored_order() -> None:
    days = (14, 15, 16, 17, 18)
    forwards = ReferenceFuturesHistoricalMarketDataRepository(tuple(_bar(day=day) for day in days))
    backwards = ReferenceFuturesHistoricalMarketDataRepository(
        tuple(_bar(day=day) for day in reversed(days))
    )

    assert forwards.get_bars(_query()) == backwards.get_bars(_query())


def test_ordering_places_offset_equivalent_instants_together() -> None:
    utc = _bar(point_in_time=PointInTime("2026-01-15T21:00:00Z"))
    repository = ReferenceFuturesHistoricalMarketDataRepository((_bar(day=16), utc, _bar(day=14)))

    result = repository.get_bars(_query())

    assert result[1] is utc


# ---------------------------------------------------------------------------
# Empty results
# ---------------------------------------------------------------------------


def test_an_empty_repository_returns_an_empty_tuple() -> None:
    assert ReferenceFuturesHistoricalMarketDataRepository().get_bars(_query()) == ()


def test_a_window_matching_nothing_returns_an_empty_tuple() -> None:
    repository = ReferenceFuturesHistoricalMarketDataRepository((_bar(day=15),))

    assert repository.get_bars(_query(start=_instant(20), end=_instant(25))) == ()


def test_an_unknown_contract_returns_an_empty_tuple() -> None:
    repository = ReferenceFuturesHistoricalMarketDataRepository((_bar(_ES_MARCH),))

    assert repository.get_bars(_query(contract=_MES_MARCH)) == ()


def test_the_result_is_an_immutable_tuple() -> None:
    result = ReferenceFuturesHistoricalMarketDataRepository((_bar(),)).get_bars(_query())

    assert isinstance(result, tuple)


# ---------------------------------------------------------------------------
# Values survive the round trip unchanged
# ---------------------------------------------------------------------------


def test_negative_quotes_pass_through_unchanged() -> None:
    """The session a Price-based bar could not have recorded at all."""
    bar = _bar(
        _CL_MAY,
        point_in_time=PointInTime("2020-04-20T18:30:00Z"),
        open_quote="-14.00",
        high="-10.50",
        low="-40.32",
        close="-37.63",
        volume="248000",
    )
    repository = ReferenceFuturesHistoricalMarketDataRepository((bar,))

    (retrieved,) = repository.get_bars(_query(contract=_CL_MAY))

    assert retrieved.close == QuoteValue(Decimal("-37.63"))
    assert retrieved.low == QuoteValue(Decimal("-40.32"))
    assert retrieved == bar


@pytest.mark.parametrize("precision", [6, 28, 50])
def test_high_precision_values_survive_the_port(precision: int) -> None:
    """Foundation canonicalization is context-independent; the port must not undo that."""
    high_precision = "5430.12345678901234567890123456789012345678901234567891"
    volume = "1234567890123456789012345678901234567890"

    with localcontext() as context:
        context.prec = precision
        bar = _bar(
            open_quote=high_precision, high="9999", low="0", close=high_precision, volume=volume
        )
        repository = ReferenceFuturesHistoricalMarketDataRepository((bar,))
        (retrieved,) = repository.get_bars(_query())

    assert str(retrieved.open.value) == high_precision
    assert str(retrieved.close.value) == high_precision
    assert str(retrieved.volume.value) == volume


def test_zero_volume_passes_through_unchanged() -> None:
    bar = _bar(volume="0")
    repository = ReferenceFuturesHistoricalMarketDataRepository((bar,))

    assert repository.get_bars(_query())[0].volume == Quantity(Decimal("0"))


# ---------------------------------------------------------------------------
# Port shape
# ---------------------------------------------------------------------------


def test_the_port_is_abstract() -> None:
    with pytest.raises(TypeError):
        FuturesHistoricalMarketDataRepository()  # type: ignore[abstract]


def test_the_port_exposes_only_get_bars() -> None:
    assert FuturesHistoricalMarketDataRepository.__abstractmethods__ == frozenset({"get_bars"})


def test_the_port_carries_no_acquisition_or_ingestion_method() -> None:
    for absent in ("fetch_history", "fetch_bars", "ingest", "store", "resolve_symbol"):
        assert not hasattr(FuturesHistoricalMarketDataRepository, absent)
