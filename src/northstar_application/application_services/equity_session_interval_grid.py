"""Deterministic intraday interval grid of one resolved equity trading session.

The grid answers two questions and only two:

A. Which intervals does the session reference expect? One interval per
   timeframe step inside each continuous-trading phase, and none inside a
   break, before the first phase or after the last.
B. Which of those intervals had ended at an explicit observation instant? An
   interval has ended when its end is at or before the instant. The interval
   still forming is never included, not even partially.

It does not answer whether a candle for an interval exists, is complete or is
final. A grid interval that has ended is a time fact, not an approved candle;
candle acquisition, finality and correction are separate concerns.

Every interval is stamped at its END, which is the instant a completed bar for
it carries: the interval ``(opens_at, closes_at]`` is the bar stamped
``closes_at``. All instants are canonical UTC PointInTime values; a local
exchange representation is a presentation concern of the caller.

Phase boundaries must lie on the timeframe's grid in UTC. A phase whose open or
close is not aligned is refused rather than rounded, shortened or extended, so
no partial interval can ever be invented. No clock is read: the observation
instant is always supplied.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from northstar_core.foundation.value_objects import PointInTime, Timeframe

from northstar_application.ports.equity_trading_session_resolver import EquityTradingSession

# Only the intraday step EI-1A needs. Timeframe is a label without a duration,
# so the step is stated here rather than parsed from the label.
_INTERVAL_STEPS: dict[Timeframe, timedelta] = {Timeframe("5m"): timedelta(minutes=5)}

_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


class InvalidEquitySessionIntervalGridError(ValueError):
    """Raised when an interval grid cannot be built exactly from a session."""


def _to_datetime(instant: PointInTime) -> datetime:
    return datetime.fromisoformat(instant.value.replace("Z", "+00:00")).astimezone(UTC)


def _to_point_in_time(value: datetime) -> PointInTime:
    return PointInTime(value.strftime("%Y-%m-%dT%H:%M:%SZ"))


def _is_aligned(value: datetime, step: timedelta) -> bool:
    return (value - _EPOCH) % step == timedelta(0)


@dataclass(frozen=True, slots=True)
class EquitySessionInterval:
    """One expected interval, half-open on the left and closed on the right."""

    opens_at: PointInTime
    closes_at: PointInTime

    @property
    def point_in_time(self) -> PointInTime:
        """Return the instant a completed bar for this interval is stamped at: its end."""
        return self.closes_at


@dataclass(frozen=True, slots=True)
class EquitySessionIntervalGrid:
    """The expected intervals of one session at one intraday timeframe, ascending."""

    session: EquityTradingSession
    timeframe: Timeframe
    intervals: tuple[EquitySessionInterval, ...] = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.session, EquityTradingSession):
            raise TypeError("EquitySessionIntervalGrid session must be an EquityTradingSession.")
        if not isinstance(self.timeframe, Timeframe):
            raise TypeError("EquitySessionIntervalGrid timeframe must be a Timeframe.")
        step = _INTERVAL_STEPS.get(self.timeframe)
        if step is None:
            raise InvalidEquitySessionIntervalGridError(
                f"EquitySessionIntervalGrid does not support timeframe {self.timeframe}. "
                f"Supported timeframes are {sorted(str(key) for key in _INTERVAL_STEPS)}."
            )

        intervals: list[EquitySessionInterval] = []
        for phase in self.session.phases:
            opens = _to_datetime(phase.opens_at)
            closes = _to_datetime(phase.closes_at)
            if not _is_aligned(opens, step) or not _is_aligned(closes, step):
                raise InvalidEquitySessionIntervalGridError(
                    f"Equity session phase {phase} is not aligned to the {self.timeframe} grid."
                )
            start = opens
            while start < closes:
                end = start + step
                intervals.append(
                    EquitySessionInterval(
                        opens_at=_to_point_in_time(start), closes_at=_to_point_in_time(end)
                    )
                )
                start = end
        object.__setattr__(self, "intervals", tuple(intervals))

    def ended_as_of(self, as_of: PointInTime) -> tuple[EquitySessionInterval, ...]:
        """Return the expected intervals whose end is at or before ``as_of``, ascending.

        An interval ending exactly at ``as_of`` has ended. The interval still
        forming at ``as_of`` is excluded. Having ended does not make a candle
        for the interval available, complete or final.
        """
        if not isinstance(as_of, PointInTime):
            raise TypeError("EquitySessionIntervalGrid as_of must be a PointInTime.")
        return tuple(
            interval for interval in self.intervals if interval.closes_at.compare(as_of) <= 0
        )

    def interval_ending_at(self, point_in_time: PointInTime) -> EquitySessionInterval | None:
        """Return the expected interval stamped ``point_in_time``, or None if none is expected."""
        if not isinstance(point_in_time, PointInTime):
            raise TypeError("EquitySessionIntervalGrid point_in_time must be a PointInTime.")
        for interval in self.intervals:
            if interval.closes_at == point_in_time:
                return interval
        return None
