"""Tests for folding one futures session's minute bars into a daily bar.

The session windows and boundary minutes below are the real CME shapes
confirmed during the Epic 9.6a live probe: a session opening on the previous
civil day, an early close ending five hours sooner, and the 16:59 interval
whose completion lands exactly on that close.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal, getcontext, localcontext

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

from northstar_application.application_services import (
    AggregateFuturesDailySessionBarUseCase,
    InvalidFuturesSessionAggregationError,
)
from northstar_application.ports import FuturesTradingSession

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

_MINUTE = Timeframe("1m")
_DAILY = Timeframe("1d")

# The CME session labelled 2026-09-15: opens on the previous civil day.
_SESSION = FuturesTradingSession(
    date(2026, 9, 15),
    PointInTime("2026-09-14T22:00:00Z"),
    PointInTime("2026-09-15T22:00:00Z"),
)

# The early close: nineteen hours, ending at 17:00Z.
_EARLY_SESSION = FuturesTradingSession(
    date(2026, 7, 3),
    PointInTime("2026-07-02T22:00:00Z"),
    PointInTime("2026-07-03T17:00:00Z"),
)


@pytest.fixture
def aggregate() -> AggregateFuturesDailySessionBarUseCase:
    return AggregateFuturesDailySessionBarUseCase()


def _bar(
    completion: str,
    *,
    contract: FuturesContract = _ES_MARCH,
    timeframe: Timeframe = _MINUTE,
    open_quote: str = "7660.00",
    high: str = "7700.00",
    low: str = "7500.00",
    close: str = "7663.00",
    volume: str = "100",
) -> FuturesOHLCVBar:
    return FuturesOHLCVBar(
        contract=contract,
        point_in_time=PointInTime(completion),
        timeframe=timeframe,
        open=QuoteValue(Decimal(open_quote)),
        high=QuoteValue(Decimal(high)),
        low=QuoteValue(Decimal(low)),
        close=QuoteValue(Decimal(close)),
        volume=Quantity(Decimal(volume)),
    )


def _minute(hhmm: str, day: str = "2026-09-15") -> str:
    return f"{day}T{hhmm}:00Z"


# ---------------------------------------------------------------------------
# Empty session
# ---------------------------------------------------------------------------


def test_an_empty_session_yields_no_bar(
    aggregate: AggregateFuturesDailySessionBarUseCase,
) -> None:
    """No synthetic zero-volume bar: a fabricated OHLC never existed."""
    assert aggregate.execute(_ES_MARCH, _SESSION, ()) is None


# ---------------------------------------------------------------------------
# Single bar
# ---------------------------------------------------------------------------


def test_a_single_bar_carries_through_exactly(
    aggregate: AggregateFuturesDailySessionBarUseCase,
) -> None:
    source = _bar(_minute("14:30"), open_quote="7660", high="7665", low="7655", close="7663")

    daily = aggregate.execute(_ES_MARCH, _SESSION, (source,))

    assert daily is not None
    assert daily.open == QuoteValue(Decimal("7660"))
    assert daily.high == QuoteValue(Decimal("7665"))
    assert daily.low == QuoteValue(Decimal("7655"))
    assert daily.close == QuoteValue(Decimal("7663"))
    assert daily.volume == Quantity(Decimal("100"))


def test_the_daily_bar_is_stamped_at_the_session_close(
    aggregate: AggregateFuturesDailySessionBarUseCase,
) -> None:
    daily = aggregate.execute(_ES_MARCH, _SESSION, (_bar(_minute("14:30")),))

    assert daily is not None
    assert daily.point_in_time == _SESSION.closes_at
    assert daily.point_in_time == PointInTime("2026-09-15T22:00:00Z")


def test_the_daily_bar_carries_the_daily_timeframe_and_contract(
    aggregate: AggregateFuturesDailySessionBarUseCase,
) -> None:
    daily = aggregate.execute(_ES_MARCH, _SESSION, (_bar(_minute("14:30")),))

    assert daily is not None
    assert daily.timeframe == _DAILY
    assert daily.contract == _ES_MARCH
    assert daily.natural_key == (_ES_MARCH, _SESSION.closes_at, _DAILY)


# ---------------------------------------------------------------------------
# Multiple bars
# ---------------------------------------------------------------------------


def _session_bars() -> tuple[FuturesOHLCVBar, ...]:
    return (
        _bar(
            _minute("14:30"), open_quote="7660", high="7670", low="7658", close="7665", volume="100"
        ),
        _bar(
            _minute("14:31"), open_quote="7665", high="7699", low="7661", close="7680", volume="250"
        ),
        _bar(
            _minute("14:32"), open_quote="7680", high="7685", low="7575", close="7600", volume="75"
        ),
        _bar(
            _minute("14:33"), open_quote="7600", high="7620", low="7595", close="7612", volume="325"
        ),
    )


def test_the_fold_takes_first_open_max_high_min_low_last_close(
    aggregate: AggregateFuturesDailySessionBarUseCase,
) -> None:
    daily = aggregate.execute(_ES_MARCH, _SESSION, _session_bars())

    assert daily is not None
    assert daily.open == QuoteValue(Decimal("7660"))
    assert daily.high == QuoteValue(Decimal("7699"))
    assert daily.low == QuoteValue(Decimal("7575"))
    assert daily.close == QuoteValue(Decimal("7612"))


def test_volume_is_the_exact_sum(
    aggregate: AggregateFuturesDailySessionBarUseCase,
) -> None:
    daily = aggregate.execute(_ES_MARCH, _SESSION, _session_bars())

    assert daily is not None
    assert daily.volume == Quantity(Decimal("750"))


def test_the_extremes_are_not_the_first_or_last_bar(
    aggregate: AggregateFuturesDailySessionBarUseCase,
) -> None:
    """Guards the fold from passing by accident on a monotonic fixture."""
    bars = _session_bars()
    daily = aggregate.execute(_ES_MARCH, _SESSION, bars)

    assert daily is not None
    assert daily.high != bars[0].high
    assert daily.high != bars[-1].high
    assert daily.low != bars[0].low
    assert daily.low != bars[-1].low


# ---------------------------------------------------------------------------
# Sparse minutes
# ---------------------------------------------------------------------------


def test_large_gaps_between_minutes_are_valid(
    aggregate: AggregateFuturesDailySessionBarUseCase,
) -> None:
    """Databento emits no bar for a minute with no trades."""
    bars = (
        _bar(_minute("23:05", "2026-09-14"), close="7600", volume="5"),
        _bar(_minute("08:17"), close="7640", volume="12"),
        _bar(_minute("20:59"), close="7663", volume="349"),
    )

    daily = aggregate.execute(_ES_MARCH, _SESSION, bars)

    assert daily is not None
    assert daily.close == QuoteValue(Decimal("7663"))
    assert daily.volume == Quantity(Decimal("366"))


def test_a_two_bar_session_hours_apart_is_valid(
    aggregate: AggregateFuturesDailySessionBarUseCase,
) -> None:
    """The real 2026-07-03 shape: two bars in a forty-minute window."""
    bars = (
        _bar(_minute("16:44", "2026-07-03"), close="7624.50", volume="1"),
        _bar(_minute("17:00", "2026-07-03"), close="7623.75", volume="2"),
    )

    daily = aggregate.execute(_ES_MARCH, _EARLY_SESSION, bars)

    assert daily is not None
    assert daily.volume == Quantity(Decimal("3"))


def test_no_minimum_bar_count_is_required(
    aggregate: AggregateFuturesDailySessionBarUseCase,
) -> None:
    assert aggregate.execute(_ES_MARCH, _SESSION, (_bar(_minute("14:30")),)) is not None


# ---------------------------------------------------------------------------
# Ordering
# ---------------------------------------------------------------------------


def test_reversed_bars_are_rejected_not_sorted(
    aggregate: AggregateFuturesDailySessionBarUseCase,
) -> None:
    bars = (_bar(_minute("14:31")), _bar(_minute("14:30")))

    with pytest.raises(InvalidFuturesSessionAggregationError, match="strictly ordered"):
        aggregate.execute(_ES_MARCH, _SESSION, bars)


def test_equal_instants_are_rejected_as_duplicates(
    aggregate: AggregateFuturesDailySessionBarUseCase,
) -> None:
    """Contract and timeframe are pinned, so one instant is one natural key."""
    bars = (_bar(_minute("14:30"), close="7663"), _bar(_minute("14:30"), close="7664"))

    with pytest.raises(InvalidFuturesSessionAggregationError, match="strictly ordered"):
        aggregate.execute(_ES_MARCH, _SESSION, bars)


def test_an_identical_repeated_bar_is_also_rejected(
    aggregate: AggregateFuturesDailySessionBarUseCase,
) -> None:
    source = _bar(_minute("14:30"))

    with pytest.raises(InvalidFuturesSessionAggregationError, match="strictly ordered"):
        aggregate.execute(_ES_MARCH, _SESSION, (source, source))


def test_ordering_is_semantic_not_textual(
    aggregate: AggregateFuturesDailySessionBarUseCase,
) -> None:
    """'.5Z' sorts before 'Z' as text while being the later instant."""
    later = "2026-09-15T14:30:00.5Z"
    earlier = "2026-09-15T14:30:00Z"

    assert later < earlier  # the text trap
    assert PointInTime(later).compare(PointInTime(earlier)) > 0

    with pytest.raises(InvalidFuturesSessionAggregationError, match="strictly ordered"):
        aggregate.execute(_ES_MARCH, _SESSION, (_bar(later), _bar(earlier)))


def test_sub_second_bars_in_true_order_are_accepted(
    aggregate: AggregateFuturesDailySessionBarUseCase,
) -> None:
    bars = (_bar("2026-09-15T14:30:00Z"), _bar("2026-09-15T14:30:00.5Z"))

    assert aggregate.execute(_ES_MARCH, _SESSION, bars) is not None


# ---------------------------------------------------------------------------
# Session membership
# ---------------------------------------------------------------------------


def test_a_bar_completing_exactly_at_the_open_is_outside(
    aggregate: AggregateFuturesDailySessionBarUseCase,
) -> None:
    """Its interval began one minute before the session started."""
    with pytest.raises(InvalidFuturesSessionAggregationError, match="at or before the session"):
        aggregate.execute(_ES_MARCH, _SESSION, (_bar("2026-09-14T22:00:00Z"),))


def test_the_first_valid_completion_after_the_open_is_inside(
    aggregate: AggregateFuturesDailySessionBarUseCase,
) -> None:
    assert aggregate.execute(_ES_MARCH, _SESSION, (_bar("2026-09-14T22:01:00Z"),)) is not None


def test_a_bar_completing_exactly_at_the_close_is_inside(
    aggregate: AggregateFuturesDailySessionBarUseCase,
) -> None:
    assert aggregate.execute(_ES_MARCH, _SESSION, (_bar("2026-09-15T22:00:00Z"),)) is not None


def test_a_bar_completing_after_the_close_is_outside(
    aggregate: AggregateFuturesDailySessionBarUseCase,
) -> None:
    with pytest.raises(InvalidFuturesSessionAggregationError, match="after the session close"):
        aggregate.execute(_ES_MARCH, _SESSION, (_bar("2026-09-15T22:01:00Z"),))


def test_the_early_close_final_minute_is_included(
    aggregate: AggregateFuturesDailySessionBarUseCase,
) -> None:
    """Interval opens 16:59, completes 17:00, session closes 17:00."""
    daily = aggregate.execute(
        _ES_MARCH, _EARLY_SESSION, (_bar("2026-07-03T17:00:00Z", close="7623.75"),)
    )

    assert daily is not None
    assert daily.close == QuoteValue(Decimal("7623.75"))
    assert daily.point_in_time == PointInTime("2026-07-03T17:00:00Z")


def test_the_minute_after_an_early_close_is_excluded(
    aggregate: AggregateFuturesDailySessionBarUseCase,
) -> None:
    with pytest.raises(InvalidFuturesSessionAggregationError, match="after the session close"):
        aggregate.execute(_ES_MARCH, _EARLY_SESSION, (_bar("2026-07-03T17:01:00Z"),))


def test_a_bar_from_the_previous_session_is_rejected(
    aggregate: AggregateFuturesDailySessionBarUseCase,
) -> None:
    with pytest.raises(InvalidFuturesSessionAggregationError, match="at or before the session"):
        aggregate.execute(_ES_MARCH, _SESSION, (_bar("2026-09-14T20:59:00Z"),))


def test_a_bar_from_the_next_session_is_rejected(
    aggregate: AggregateFuturesDailySessionBarUseCase,
) -> None:
    """22:00Z bars are the opening trades of the following session."""
    with pytest.raises(InvalidFuturesSessionAggregationError, match="after the session close"):
        aggregate.execute(_ES_MARCH, _SESSION, (_bar("2026-09-15T22:01:00Z"),))


def test_one_out_of_window_bar_rejects_the_whole_session(
    aggregate: AggregateFuturesDailySessionBarUseCase,
) -> None:
    bars = (_bar(_minute("14:30")), _bar(_minute("14:31")), _bar("2026-09-15T22:01:00Z"))

    with pytest.raises(InvalidFuturesSessionAggregationError):
        aggregate.execute(_ES_MARCH, _SESSION, bars)


def test_an_offset_equivalent_boundary_is_compared_semantically(
    aggregate: AggregateFuturesDailySessionBarUseCase,
) -> None:
    """The close written at another offset is the same instant, still inside."""
    assert aggregate.execute(_ES_MARCH, _SESSION, (_bar("2026-09-16T03:30:00+05:30"),)) is not None


# ---------------------------------------------------------------------------
# Contract identity
# ---------------------------------------------------------------------------


def test_a_foreign_contract_is_rejected(
    aggregate: AggregateFuturesDailySessionBarUseCase,
) -> None:
    with pytest.raises(InvalidFuturesSessionAggregationError, match="expected"):
        aggregate.execute(_ES_MARCH, _SESSION, (_bar(_minute("14:30"), contract=_CL_MAY),))


def test_es_and_mes_cannot_be_mixed(
    aggregate: AggregateFuturesDailySessionBarUseCase,
) -> None:
    """One venue, one underlying, one expiry, two instruments."""
    bars = (
        _bar(_minute("14:30"), contract=_ES_MARCH),
        _bar(_minute("14:31"), contract=_MES_MARCH),
    )

    with pytest.raises(InvalidFuturesSessionAggregationError, match="expected"):
        aggregate.execute(_ES_MARCH, _SESSION, bars)


def test_march_and_june_cannot_be_mixed(
    aggregate: AggregateFuturesDailySessionBarUseCase,
) -> None:
    bars = (
        _bar(_minute("14:30"), contract=_ES_MARCH),
        _bar(_minute("14:31"), contract=_ES_JUNE),
    )

    with pytest.raises(InvalidFuturesSessionAggregationError, match="expected"):
        aggregate.execute(_ES_MARCH, _SESSION, bars)


def test_a_rebuilt_equal_contract_is_accepted(
    aggregate: AggregateFuturesDailySessionBarUseCase,
) -> None:
    """Matching is by value, not identity."""
    rebuilt = FuturesContract(
        FuturesProductReference(Symbol("ES"), ExchangeCode("CME")),
        ExpirationDate("2026-03-20"),
    )

    assert rebuilt is not _ES_MARCH
    assert aggregate.execute(rebuilt, _SESSION, (_bar(_minute("14:30")),)) is not None


# ---------------------------------------------------------------------------
# Timeframe
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("timeframe", ["5m", "15m", "1h", "1d"])
def test_a_non_minute_timeframe_is_rejected(
    aggregate: AggregateFuturesDailySessionBarUseCase, timeframe: str
) -> None:
    source = _bar(_minute("14:30"), timeframe=Timeframe(timeframe))

    with pytest.raises(InvalidFuturesSessionAggregationError, match="requires 1m bars"):
        aggregate.execute(_ES_MARCH, _SESSION, (source,))


def test_mixed_timeframes_are_rejected(
    aggregate: AggregateFuturesDailySessionBarUseCase,
) -> None:
    bars = (_bar(_minute("14:30")), _bar(_minute("14:31"), timeframe=Timeframe("1h")))

    with pytest.raises(InvalidFuturesSessionAggregationError, match="requires 1m bars"):
        aggregate.execute(_ES_MARCH, _SESSION, bars)


# ---------------------------------------------------------------------------
# Volume
# ---------------------------------------------------------------------------


def test_zero_volume_bars_are_valid(
    aggregate: AggregateFuturesDailySessionBarUseCase,
) -> None:
    bars = (_bar(_minute("14:30"), volume="0"), _bar(_minute("14:31"), volume="0"))

    daily = aggregate.execute(_ES_MARCH, _SESSION, bars)

    assert daily is not None
    assert daily.volume == Quantity(Decimal("0"))


@pytest.mark.parametrize("volume", ["1.5", "0.25", "100.0001"])
def test_a_fractional_volume_is_rejected(
    aggregate: AggregateFuturesDailySessionBarUseCase, volume: str
) -> None:
    """A contract count is integral; rounding one would invent evidence."""
    with pytest.raises(InvalidFuturesSessionAggregationError, match="integral volumes"):
        aggregate.execute(_ES_MARCH, _SESSION, (_bar(_minute("14:30"), volume=volume),))


def test_an_integral_volume_written_with_a_fraction_is_accepted(
    aggregate: AggregateFuturesDailySessionBarUseCase,
) -> None:
    """100.00 is integral; canonicalization already reduced it to 100."""
    daily = aggregate.execute(_ES_MARCH, _SESSION, (_bar(_minute("14:30"), volume="100.00"),))

    assert daily is not None
    assert daily.volume == Quantity(Decimal("100"))


def test_a_large_volume_magnitude_is_preserved_exactly(
    aggregate: AggregateFuturesDailySessionBarUseCase,
) -> None:
    big = "1234567890123456789012345678901234567890"
    bars = (_bar(_minute("14:30"), volume=big), _bar(_minute("14:31"), volume="1"))

    daily = aggregate.execute(_ES_MARCH, _SESSION, bars)

    assert daily is not None
    assert str(daily.volume.value) == "1234567890123456789012345678901234567891"


@pytest.mark.parametrize("precision", [6, 28, 50])
def test_volume_is_identical_under_any_ambient_precision(
    aggregate: AggregateFuturesDailySessionBarUseCase, precision: int
) -> None:
    """The total must not depend on the caller's decimal context."""
    big = "1234567890123456789012345678901234567890"
    bars = (_bar(_minute("14:30"), volume=big), _bar(_minute("14:31"), volume="1"))

    with localcontext() as context:
        context.prec = precision
        daily = aggregate.execute(_ES_MARCH, _SESSION, bars)

    assert daily is not None
    assert str(daily.volume.value) == "1234567890123456789012345678901234567891"


def test_aggregation_leaves_the_callers_context_untouched(
    aggregate: AggregateFuturesDailySessionBarUseCase,
) -> None:
    before = getcontext()
    precision, rounding, flags = before.prec, before.rounding, dict(before.flags)

    aggregate.execute(_ES_MARCH, _SESSION, _session_bars())

    after = getcontext()
    assert (after.prec, after.rounding) == (precision, rounding)
    assert dict(after.flags) == flags


def test_no_context_flag_is_raised_by_the_volume_sum(
    aggregate: AggregateFuturesDailySessionBarUseCase,
) -> None:
    """A raised flag would prove some step consulted the decimal context."""
    big = "1234567890123456789012345678901234567890"
    bars = (_bar(_minute("14:30"), volume=big), _bar(_minute("14:31"), volume="1"))

    with localcontext() as context:
        context.prec = 6
        context.clear_flags()

        aggregate.execute(_ES_MARCH, _SESSION, bars)

        assert not any(context.flags.values())


# ---------------------------------------------------------------------------
# Quotes
# ---------------------------------------------------------------------------


def test_an_entirely_negative_session_aggregates(
    aggregate: AggregateFuturesDailySessionBarUseCase,
) -> None:
    """The 2020-04-20 crude session a Price-based bar could not have held."""
    bars = (
        _bar(
            _minute("14:30"),
            open_quote="-14.00",
            high="-10.50",
            low="-20.00",
            close="-18.00",
            volume="100",
        ),
        _bar(
            _minute("14:31"),
            open_quote="-18.00",
            high="-15.00",
            low="-40.32",
            close="-37.63",
            volume="200",
        ),
    )

    daily = aggregate.execute(_ES_MARCH, _SESSION, bars)

    assert daily is not None
    assert daily.open == QuoteValue(Decimal("-14.00"))
    assert daily.high == QuoteValue(Decimal("-10.50"))
    assert daily.low == QuoteValue(Decimal("-40.32"))
    assert daily.close == QuoteValue(Decimal("-37.63"))


def test_a_session_crossing_zero_aggregates(
    aggregate: AggregateFuturesDailySessionBarUseCase,
) -> None:
    bars = (
        _bar(
            _minute("14:30"),
            open_quote="10.50",
            high="11.00",
            low="5.00",
            close="6.00",
            volume="10",
        ),
        _bar(
            _minute("14:31"),
            open_quote="6.00",
            high="6.50",
            low="-5.25",
            close="-2.00",
            volume="20",
        ),
    )

    daily = aggregate.execute(_ES_MARCH, _SESSION, bars)

    assert daily is not None
    assert daily.high == QuoteValue(Decimal("11.00"))
    assert daily.low == QuoteValue(Decimal("-5.25"))
    assert daily.low.value < 0 < daily.high.value


def test_negative_extremes_are_ordered_numerically_not_lexically(
    aggregate: AggregateFuturesDailySessionBarUseCase,
) -> None:
    """-40.32 is below -5.25; a text comparison would say the opposite."""
    bars = (
        _bar(
            _minute("14:30"),
            open_quote="-5.25",
            high="-5.25",
            low="-5.25",
            close="-5.25",
            volume="1",
        ),
        _bar(
            _minute("14:31"),
            open_quote="-40.32",
            high="-40.32",
            low="-40.32",
            close="-40.32",
            volume="1",
        ),
    )

    daily = aggregate.execute(_ES_MARCH, _SESSION, bars)

    assert daily is not None
    assert daily.low == QuoteValue(Decimal("-40.32"))
    assert daily.high == QuoteValue(Decimal("-5.25"))


def test_high_precision_quotes_survive_the_fold(
    aggregate: AggregateFuturesDailySessionBarUseCase,
) -> None:
    precise = "7660.12345678901234567890123456789012345678901234567891"
    source = _bar(_minute("14:30"), open_quote=precise, high="9999", low="0", close=precise)

    daily = aggregate.execute(_ES_MARCH, _SESSION, (source,))

    assert daily is not None
    assert str(daily.open.value) == precise
    assert str(daily.close.value) == precise


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_wrong_input_types_are_rejected(
    aggregate: AggregateFuturesDailySessionBarUseCase,
) -> None:
    with pytest.raises(InvalidFuturesSessionAggregationError, match="must be a FuturesContract"):
        aggregate.execute(_ES, _SESSION, ())
    with pytest.raises(
        InvalidFuturesSessionAggregationError, match="must be a FuturesTradingSession"
    ):
        aggregate.execute(_ES_MARCH, "2026-09-15", ())
    with pytest.raises(InvalidFuturesSessionAggregationError, match="must be a tuple"):
        aggregate.execute(_ES_MARCH, _SESSION, [_bar(_minute("14:30"))])
    with pytest.raises(InvalidFuturesSessionAggregationError, match="must all be FuturesOHLCVBar"):
        aggregate.execute(_ES_MARCH, _SESSION, ("not a bar",))


def test_the_error_is_a_value_error() -> None:
    assert issubclass(InvalidFuturesSessionAggregationError, ValueError)


# ---------------------------------------------------------------------------
# Determinism and purity
# ---------------------------------------------------------------------------


def test_repeated_aggregation_returns_an_equal_result(
    aggregate: AggregateFuturesDailySessionBarUseCase,
) -> None:
    bars = _session_bars()

    first = aggregate.execute(_ES_MARCH, _SESSION, bars)
    second = aggregate.execute(_ES_MARCH, _SESSION, bars)

    assert first == second
    assert hash(first) == hash(second)


def test_a_fresh_use_case_produces_the_same_result(
    aggregate: AggregateFuturesDailySessionBarUseCase,
) -> None:
    """The fold holds no state between calls."""
    bars = _session_bars()

    assert aggregate.execute(_ES_MARCH, _SESSION, bars) == (
        AggregateFuturesDailySessionBarUseCase().execute(_ES_MARCH, _SESSION, bars)
    )


def test_the_input_bars_are_not_mutated(
    aggregate: AggregateFuturesDailySessionBarUseCase,
) -> None:
    bars = _session_bars()
    snapshot = tuple(bars)

    aggregate.execute(_ES_MARCH, _SESSION, bars)

    assert bars == snapshot


def test_the_use_case_reads_no_clock_and_holds_no_dependency() -> None:
    import ast
    from pathlib import Path

    import northstar_application.application_services.aggregate_futures_daily_session_bar as module

    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))

    modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    for forbidden in ("exchange_calendars", "databento", "sqlite3", "random", "time", "os"):
        assert not any(m.split(".")[0] == forbidden for m in modules)

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert node.func.attr not in {"now", "utcnow", "today", "monotonic"}

    assert AggregateFuturesDailySessionBarUseCase().__dict__ == {}
