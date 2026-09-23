"""Application folding of one futures trading session's minute bars into a daily bar.

No provider supplies a session-aligned daily futures bar that Northstar has
evidence for: the one schema that claimed to is unavailable for CME data, and
the daily schema that does exist aggregates UTC calendar days, whose window
straddles two trading sessions. A daily bar is therefore folded here, from
minute bars the session itself selects.

The fold is a pure function. It reads no clock, holds no dependency, performs
no I/O and persists nothing, so the same minutes always produce the same daily
bar.

Completion instants and session membership
------------------------------------------
A Northstar bar is stamped at the instant its interval *completes*, while a
provider stamps the interval's *open*. Substituting one for the other inverts
the membership rule. The provider's half-open window

    session_open <= interval_open < session_close

becomes, in completion terms

    session.opens_at < point_in_time <= session.closes_at

half-open on the left and closed on the right. The right-closed end is what
admits a session's final minute: on an early close the last interval opens at
16:59 and completes at 17:00, exactly when the session closes.

What this fold does not do
--------------------------
It applies no settlement price. A settlement is a computed mark published on
its own schedule, not the close of a traded interval, and mixing the two would
produce a bar whose timestamp and value described different events.
"""

from __future__ import annotations

from decimal import Decimal

from northstar_core.derivatives import QuoteValue
from northstar_core.foundation.value_objects import Quantity, Timeframe
from northstar_core.futures import FuturesContract, FuturesOHLCVBar

from northstar_application.ports import FuturesTradingSession

_MINUTE = Timeframe("1m")
_DAILY = Timeframe("1d")


class InvalidFuturesSessionAggregationError(ValueError):
    """Raised when minute bars cannot produce a coherent daily session bar."""


class AggregateFuturesDailySessionBarUseCase:
    """Fold one session's minute bars into a single daily bar.

    The use case holds no dependencies. The session window is supplied by the
    caller, which is what keeps this fold independent of any calendar.
    """

    def execute(
        self,
        contract: FuturesContract,
        session: FuturesTradingSession,
        bars: tuple[FuturesOHLCVBar, ...],
    ) -> FuturesOHLCVBar | None:
        """Return the session's daily bar, or None when it contains no bars.

        A session with no bars yields nothing rather than a synthetic
        zero-volume bar. A fabricated bar would need open, high, low and close
        values that never existed, and would occupy a real natural key in a
        store that refuses to correct it afterwards.
        """
        self._validate_inputs(contract, session, bars)

        if not bars:
            return None

        self._validate_bars(contract, session, bars)

        return FuturesOHLCVBar(
            contract=contract,
            point_in_time=session.closes_at,
            timeframe=_DAILY,
            open=bars[0].open,
            high=max((bar.high for bar in bars), key=_quote_key),
            low=min((bar.low for bar in bars), key=_quote_key),
            close=bars[-1].close,
            volume=_total_volume(bars),
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_inputs(
        contract: FuturesContract,
        session: FuturesTradingSession,
        bars: tuple[FuturesOHLCVBar, ...],
    ) -> None:
        if not isinstance(contract, FuturesContract):
            raise InvalidFuturesSessionAggregationError(
                "Futures session aggregation contract must be a FuturesContract value."
            )
        if not isinstance(session, FuturesTradingSession):
            raise InvalidFuturesSessionAggregationError(
                "Futures session aggregation session must be a FuturesTradingSession value."
            )
        if not isinstance(bars, tuple):
            raise InvalidFuturesSessionAggregationError(
                "Futures session aggregation bars must be a tuple."
            )
        for bar in bars:
            if not isinstance(bar, FuturesOHLCVBar):
                raise InvalidFuturesSessionAggregationError(
                    "Futures session aggregation bars must all be FuturesOHLCVBar values."
                )

    @staticmethod
    def _validate_bars(
        contract: FuturesContract,
        session: FuturesTradingSession,
        bars: tuple[FuturesOHLCVBar, ...],
    ) -> None:
        for bar in bars:
            if bar.contract != contract:
                raise InvalidFuturesSessionAggregationError(
                    f"Futures session aggregation received a bar for {bar.contract}, "
                    f"expected {contract}."
                )
            if bar.timeframe != _MINUTE:
                raise InvalidFuturesSessionAggregationError(
                    f"Futures session aggregation requires {_MINUTE} bars, "
                    f"received {bar.timeframe}."
                )
            # Half-open left, closed right. Compared semantically: a canonical
            # instant drops zero fractional seconds, so text ordering and
            # chronology disagree.
            if session.opens_at.compare(bar.point_in_time) >= 0:
                raise InvalidFuturesSessionAggregationError(
                    f"Futures session aggregation received a bar completing at "
                    f"{bar.point_in_time}, at or before the session open "
                    f"{session.opens_at}."
                )
            if bar.point_in_time.compare(session.closes_at) > 0:
                raise InvalidFuturesSessionAggregationError(
                    f"Futures session aggregation received a bar completing at "
                    f"{bar.point_in_time}, after the session close {session.closes_at}."
                )

        # Strictly increasing. This also rejects duplicates: two bars sharing
        # one instant are the same natural key, since contract and timeframe
        # are already pinned. Malformed order is refused, never sorted --
        # reordering a source's output would hide the defect that produced it.
        for earlier, later in zip(bars, bars[1:], strict=False):
            if earlier.point_in_time.compare(later.point_in_time) >= 0:
                raise InvalidFuturesSessionAggregationError(
                    "Futures session aggregation bars must be strictly ordered "
                    f"oldest to newest; {earlier.point_in_time} precedes "
                    f"{later.point_in_time}."
                )


# ---------------------------------------------------------------------------
# Folding helpers
# ---------------------------------------------------------------------------


def _quote_key(quote: QuoteValue) -> QuoteValue:
    """Return the ordering key for a quotation.

    QuoteValue is orderable, so the extremes are selected by comparison alone.
    No arithmetic is performed on a quotation: a quotation is a number in its
    product's own convention, and adding or averaging two of them would assert
    a relationship the convention does not guarantee. Comparison works
    unchanged for negative quotations and for a session that crosses zero.
    """
    return quote


def _total_volume(bars: tuple[FuturesOHLCVBar, ...]) -> Quantity:
    """Sum session volume exactly, without consulting the decimal context.

    Decimal arithmetic rounds to the ambient decimal precision, so summing
    through it would make the total depend on whoever holds the context. A
    volume is a count of contracts, so the sum is taken in Python's unbounded
    int, which is exact by construction and consults no context at all. That
    keeps this fold correct regardless of what the surrounding context is set
    to, and independent of how Quantity chooses to implement arithmetic.

    A non-integral volume is refused rather than truncated. Rounding a
    fractional volume into a contract count would invent evidence, and the
    value is meaningless for futures in any case.
    """
    total = 0
    for bar in bars:
        value = bar.volume.value
        if value != value.to_integral_value():
            raise InvalidFuturesSessionAggregationError(
                f"Futures session aggregation requires integral volumes; "
                f"received {value} at {bar.point_in_time}."
            )
        total += int(value)
    return Quantity(Decimal(total))
