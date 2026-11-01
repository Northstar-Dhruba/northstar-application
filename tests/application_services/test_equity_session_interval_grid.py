"""Tests for the deterministic 5-minute interval grid of one equity trading session.

The session used here is invented: a morning phase 10:00-12:00 IST and an
afternoon phase 13:00-14:30 IST with a break between them, on a 2031 label.
These are not exchange hours. They exist to exercise phase alignment, the
break, and point-in-time completion.

A grid interval that has ended is a time fact only. Nothing here asserts that a
candle for it exists, is complete or is final.
"""

from __future__ import annotations

import ast
import inspect
from datetime import UTC, date, datetime, timedelta, timezone

import pytest
from northstar_core.foundation.value_objects import ExchangeCode, PointInTime, Timeframe

from northstar_application import application_services
from northstar_application.application_services import (
    EquitySessionInterval,
    EquitySessionIntervalGrid,
    InvalidEquitySessionIntervalGridError,
)
from northstar_application.application_services import (
    equity_session_interval_grid as grid_module,
)
from northstar_application.ports import (
    EquitySessionKind,
    EquityTradingPhase,
    EquityTradingSession,
)

# Application carries no timezone database dependency; the named Asia/Kolkata
# conversion is exercised by the Infrastructure session adapter tests. India
# observes no daylight saving, so a fixed +05:30 view is exact for display here.
_IST = timezone(timedelta(hours=5, minutes=30))
_FIVE_MINUTES = Timeframe("5m")
_LABEL = date(2031, 3, 4)


def _instant(text: str) -> PointInTime:
    return PointInTime(text)


def _phase(opens: str, closes: str) -> EquityTradingPhase:
    return EquityTradingPhase(opens_at=_instant(opens), closes_at=_instant(closes))


# 10:00-12:00 IST and 13:00-14:30 IST, in UTC.
_MORNING = _phase("2031-03-04T04:30:00Z", "2031-03-04T06:30:00Z")
_AFTERNOON = _phase("2031-03-04T07:30:00Z", "2031-03-04T09:00:00Z")


def _session(
    phases: tuple[EquityTradingPhase, ...] = (_MORNING, _AFTERNOON),
    kind: EquitySessionKind = EquitySessionKind.NORMAL,
) -> EquityTradingSession:
    return EquityTradingSession(
        exchange_code=ExchangeCode("NSE"),
        trading_date=_LABEL,
        kind=kind,
        phases=phases,
        reference="synthetic-reference grid-tests",
    )


def _grid(session: EquityTradingSession | None = None) -> EquitySessionIntervalGrid:
    return EquitySessionIntervalGrid(session=session or _session(), timeframe=_FIVE_MINUTES)


def _utc(instant: PointInTime) -> datetime:
    return datetime.fromisoformat(instant.value.replace("Z", "+00:00"))


def _ends(intervals: tuple[EquitySessionInterval, ...]) -> list[str]:
    return [interval.closes_at.value for interval in intervals]


# ----------------------------------------------------------------------
# A. Expected intervals
# ----------------------------------------------------------------------


def test_grid_lays_out_every_five_minute_interval_of_each_phase() -> None:
    intervals = _grid().intervals

    # 120 minutes and 90 minutes of continuous trading.
    assert len(intervals) == 24 + 18
    assert intervals[0] == EquitySessionInterval(
        opens_at=_instant("2031-03-04T04:30:00Z"), closes_at=_instant("2031-03-04T04:35:00Z")
    )
    assert intervals[23] == EquitySessionInterval(
        opens_at=_instant("2031-03-04T06:25:00Z"), closes_at=_instant("2031-03-04T06:30:00Z")
    )
    assert intervals[24] == EquitySessionInterval(
        opens_at=_instant("2031-03-04T07:30:00Z"), closes_at=_instant("2031-03-04T07:35:00Z")
    )
    assert intervals[-1] == EquitySessionInterval(
        opens_at=_instant("2031-03-04T08:55:00Z"), closes_at=_instant("2031-03-04T09:00:00Z")
    )


def test_every_interval_is_exactly_five_minutes_and_contiguous_within_its_phase() -> None:
    intervals = _grid().intervals

    for interval in intervals:
        assert _utc(interval.closes_at) - _utc(interval.opens_at) == timedelta(minutes=5)
    for earlier, later in zip(intervals, intervals[1:], strict=False):
        if earlier.closes_at == _MORNING.closes_at:
            assert later.opens_at == _AFTERNOON.opens_at
        else:
            assert later.opens_at == earlier.closes_at


def test_no_interval_falls_in_or_crosses_the_break() -> None:
    break_start = _utc(_MORNING.closes_at)
    break_end = _utc(_AFTERNOON.opens_at)

    for interval in _grid().intervals:
        opens, closes = _utc(interval.opens_at), _utc(interval.closes_at)
        assert closes <= break_start or opens >= break_end


def test_no_interval_lies_outside_the_session_phases() -> None:
    session = _session()

    for interval in _grid(session).intervals:
        assert interval.opens_at.compare(session.opens_at) >= 0
        assert interval.closes_at.compare(session.closes_at) <= 0


def test_interval_is_stamped_at_its_end_in_canonical_utc() -> None:
    for interval in _grid().intervals:
        assert interval.point_in_time == interval.closes_at
        assert interval.closes_at.value.endswith("Z")
        assert interval.opens_at.value.endswith("Z")


def test_interval_ends_present_in_ist() -> None:
    intervals = _grid().intervals

    first_end = _utc(intervals[0].point_in_time).astimezone(_IST)
    last_end = _utc(intervals[-1].point_in_time).astimezone(_IST)

    assert first_end.isoformat() == "2031-03-04T10:05:00+05:30"
    assert last_end.isoformat() == "2031-03-04T14:30:00+05:30"


def test_interval_ending_at_identifies_only_expected_interval_ends() -> None:
    grid = _grid()

    expected = grid.interval_ending_at(_instant("2031-03-04T06:30:00Z"))
    assert expected == grid.intervals[23]
    # A phase opening, an instant inside the break and an off-grid instant are
    # not interval ends.
    assert grid.interval_ending_at(_instant("2031-03-04T04:30:00Z")) is None
    assert grid.interval_ending_at(_instant("2031-03-04T07:00:00Z")) is None
    assert grid.interval_ending_at(_instant("2031-03-04T04:33:00Z")) is None
    assert grid.interval_ending_at(_instant("2031-03-04T09:05:00Z")) is None


def test_special_short_session_has_only_its_own_intervals() -> None:
    # An invented one-hour special session, 18:00-19:00 IST.
    special = _session(
        (_phase("2031-03-04T12:30:00Z", "2031-03-04T13:30:00Z"),), EquitySessionKind.SPECIAL
    )

    intervals = _grid(special).intervals

    assert len(intervals) == 12
    assert intervals[0].opens_at == _instant("2031-03-04T12:30:00Z")
    assert intervals[-1].closes_at == _instant("2031-03-04T13:30:00Z")


# ----------------------------------------------------------------------
# B. Intervals ended at an explicit instant
# ----------------------------------------------------------------------


def test_interval_ending_exactly_at_as_of_has_ended() -> None:
    ended = _grid().ended_as_of(_instant("2031-03-04T04:35:00Z"))

    assert _ends(ended) == ["2031-03-04T04:35:00Z"]


def test_interval_one_microsecond_before_its_end_has_not_ended() -> None:
    ended = _grid().ended_as_of(_instant("2031-03-04T04:34:59.999999Z"))

    assert ended == ()


def test_interval_still_forming_is_excluded() -> None:
    ended = _grid().ended_as_of(_instant("2031-03-04T04:47:30Z"))

    assert _ends(ended) == ["2031-03-04T04:35:00Z", "2031-03-04T04:40:00Z", "2031-03-04T04:45:00Z"]


def test_nothing_has_ended_before_the_session_or_at_its_open() -> None:
    grid = _grid()

    assert grid.ended_as_of(_instant("2031-03-04T00:00:00Z")) == ()
    assert grid.ended_as_of(_instant("2031-03-04T04:30:00Z")) == ()


def test_during_the_break_only_the_first_phase_has_ended() -> None:
    ended = _grid().ended_as_of(_instant("2031-03-04T07:00:00Z"))

    assert len(ended) == 24
    assert ended[-1].closes_at == _MORNING.closes_at


def test_everything_has_ended_after_the_session() -> None:
    grid = _grid()

    assert grid.ended_as_of(_instant("2031-03-04T09:00:00Z")) == grid.intervals
    assert grid.ended_as_of(_instant("2031-03-05T09:00:00Z")) == grid.intervals


def test_as_of_with_an_ist_offset_is_compared_chronologically() -> None:
    ended = _grid().ended_as_of(_instant("2031-03-04T10:10:00+05:30"))

    assert _ends(ended) == ["2031-03-04T04:35:00Z", "2031-03-04T04:40:00Z"]


# ----------------------------------------------------------------------
# C. Point-in-time safety
# ----------------------------------------------------------------------


def test_no_interval_ending_after_as_of_is_ever_reported_ended() -> None:
    grid = _grid()
    start = datetime(2031, 3, 4, 4, 0, tzinfo=UTC)

    for minute in range(0, 6 * 60, 1):
        as_of_value = start + timedelta(minutes=minute, seconds=30)
        as_of = _instant(as_of_value.strftime("%Y-%m-%dT%H:%M:%SZ"))
        for interval in grid.ended_as_of(as_of):
            assert _utc(interval.closes_at) <= as_of_value


def test_ended_intervals_grow_monotonically_with_as_of() -> None:
    grid = _grid()
    previous: tuple[EquitySessionInterval, ...] = ()

    for interval in grid.intervals:
        ended = grid.ended_as_of(interval.closes_at)
        assert ended[: len(previous)] == previous
        assert ended[-1] == interval
        previous = ended


def test_identical_inputs_give_identical_grids_and_answers() -> None:
    first, second = _grid(), _grid()
    as_of = _instant("2031-03-04T08:12:00Z")

    assert first == second
    assert first.intervals == second.intervals
    assert first.ended_as_of(as_of) == second.ended_as_of(as_of)


def test_grid_module_reads_no_clock() -> None:
    tree = ast.parse(inspect.getsource(grid_module))
    called = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }

    assert not called & {"now", "today", "utcnow", "time", "monotonic", "localtime"}


# ----------------------------------------------------------------------
# Fail-closed construction
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("opens", "closes"),
    [
        ("2031-03-04T04:32:00Z", "2031-03-04T06:30:00Z"),
        ("2031-03-04T04:30:00Z", "2031-03-04T06:33:00Z"),
        ("2031-03-04T04:30:30Z", "2031-03-04T06:30:00Z"),
        ("2031-03-04T04:30:00.5Z", "2031-03-04T06:30:00Z"),
    ],
)
def test_unaligned_phase_is_refused_not_rounded(opens: str, closes: str) -> None:
    session = _session((_phase(opens, closes),))

    with pytest.raises(InvalidEquitySessionIntervalGridError, match="not aligned"):
        _grid(session)


def test_unaligned_later_phase_refuses_the_whole_grid() -> None:
    session = _session((_MORNING, _phase("2031-03-04T07:30:00Z", "2031-03-04T08:58:00Z")))

    with pytest.raises(InvalidEquitySessionIntervalGridError, match="not aligned"):
        _grid(session)


@pytest.mark.parametrize("label", ["1m", "15m", "1h", "1d"])
def test_timeframes_other_than_five_minutes_are_refused(label: str) -> None:
    with pytest.raises(InvalidEquitySessionIntervalGridError, match="does not support"):
        EquitySessionIntervalGrid(session=_session(), timeframe=Timeframe(label))


def test_grid_refuses_inputs_of_the_wrong_type() -> None:
    with pytest.raises(TypeError, match="EquityTradingSession"):
        EquitySessionIntervalGrid(session="session", timeframe=_FIVE_MINUTES)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="Timeframe"):
        EquitySessionIntervalGrid(session=_session(), timeframe="5m")  # type: ignore[arg-type]
    grid = _grid()
    with pytest.raises(TypeError, match="as_of"):
        grid.ended_as_of("2031-03-04T05:00:00Z")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="point_in_time"):
        grid.interval_ending_at(None)  # type: ignore[arg-type]


def test_intervals_cannot_be_supplied_by_the_caller() -> None:
    with pytest.raises(TypeError):
        EquitySessionIntervalGrid(  # type: ignore[call-arg]
            session=_session(), timeframe=_FIVE_MINUTES, intervals=()
        )


def test_application_services_export_the_grid() -> None:
    for name in (
        "EquitySessionInterval",
        "EquitySessionIntervalGrid",
        "InvalidEquitySessionIntervalGridError",
    ):
        assert name in application_services.__all__
