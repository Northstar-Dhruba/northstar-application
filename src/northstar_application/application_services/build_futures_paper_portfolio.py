"""Application folding of futures paper fills into a portfolio snapshot.

Futures holdings are derived, never stored. A portfolio is a pure function of
the fills that produced it and the instant it is asked about.

Each concrete FuturesContract holds one signed net position; different
products, exchanges and expiries never pool, and nothing rolls. A fill moves
exposure by ``+contracts`` for BUY and ``-contracts`` for SELL, in plain
integers. With current net ``p`` and trade delta ``d`` at quote ``q``:

    flat                      -> net d, average q
    same direction            -> net p + d, contract-weighted average
    opposite, |d| <  |p|      -> net p + d, average unchanged
    opposite, |d| == |p|      -> position removed (flat is absence)
    opposite, |d| >  |p|      -> net p + d, average q (old basis discarded)

The weighted average is linear in the quote, so zero and negative quotes need
no special case, and it is computed under one explicit Decimal context so the
caller's context can never change a basis.

The fold changes exposure and entry basis only. It calculates no realised or
unrealised profit and loss, notional, margin or multiplier, reads no market
data and no clock, and persists nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_HALF_EVEN, Context, localcontext

from northstar_core.derivatives import QuoteValue
from northstar_core.foundation.value_objects import PointInTime
from northstar_core.futures import FuturesContract
from northstar_core.paper_trading import (
    FuturesPaperFill,
    FuturesPaperPortfolio,
    FuturesPosition,
    OrderSide,
    PaperPortfolioIdentity,
)
from northstar_core.strategy import StrategyIdentity

_BASIS_CONTEXT = Context(prec=28, rounding=ROUND_HALF_EVEN)


class InvalidFuturesPaperFillHistoryError(ValueError):
    """Raised when a futures paper fill history cannot produce a coherent portfolio."""


@dataclass(frozen=True, slots=True)
class _Transition:
    """What one fill did to one contract's position.

    ``contracts_closed`` is how much of the prior position the fill closed, and
    ``closing_average_entry`` the basis it closed against; both are zero/None
    when the fill only opened or added to exposure.
    """

    position: FuturesPosition | None
    contracts_closed: int
    closing_average_entry: QuoteValue | None


def _transition(position: FuturesPosition | None, fill: FuturesPaperFill) -> _Transition:
    """Apply one fill to one contract's position; the single futures fold rule."""
    count = fill.contracts.value
    delta = count if fill.side is OrderSide.BUY else -count
    quote = fill.fill_quote

    if position is None:
        return _Transition(FuturesPosition(fill.contract, delta, quote), 0, None)

    current = position.net_contracts
    net = current + delta
    if (current > 0) == (delta > 0):
        held = abs(current)
        with localcontext(_BASIS_CONTEXT):
            average = (held * position.average_entry.value + count * quote.value) / (held + count)
        return _Transition(FuturesPosition(fill.contract, net, QuoteValue(average)), 0, None)

    closed = min(abs(delta), abs(current))
    if net == 0:
        resulting = None
    elif abs(delta) < abs(current):
        resulting = FuturesPosition(fill.contract, net, position.average_entry)
    else:
        resulting = FuturesPosition(fill.contract, net, quote)
    return _Transition(resulting, closed, position.average_entry)


def _validate_fill_history(
    subject: str,
    portfolio_identity: PaperPortfolioIdentity,
    strategy_identity: StrategyIdentity,
    fills: tuple[FuturesPaperFill, ...],
) -> None:
    """Require one owned, duplicate-free, canonically ordered history.

    Every supplied fill is checked, including those after an as-of cutoff: a
    foreign or duplicated fill is a defect in the history, not a matter of
    timing.
    """
    seen_fills: set[str] = set()
    seen_orders: set[str] = set()
    previous: FuturesPaperFill | None = None
    for fill in fills:
        if fill.portfolio_identity != portfolio_identity:
            raise InvalidFuturesPaperFillHistoryError(
                f"{subject} fills must all belong to the requested portfolio."
            )
        if fill.strategy_identity != strategy_identity:
            raise InvalidFuturesPaperFillHistoryError(
                f"{subject} fills must all belong to the portfolio's strategy."
            )
        if fill.identity.identity in seen_fills:
            raise InvalidFuturesPaperFillHistoryError(
                f"{subject} fills cannot contain duplicate fill identities."
            )
        seen_fills.add(fill.identity.identity)
        if fill.order_identity.identity in seen_orders:
            raise InvalidFuturesPaperFillHistoryError(
                f"{subject} fills cannot contain two fills for one order."
            )
        seen_orders.add(fill.order_identity.identity)

        if previous is not None:
            instant = previous.filled_at.compare(fill.filled_at)
            if instant > 0 or (
                instant == 0 and previous.order_identity.identity > fill.order_identity.identity
            ):
                raise InvalidFuturesPaperFillHistoryError(
                    f"{subject} fills must be ordered by fill instant, then order identity."
                )
        previous = fill


class BuildFuturesPaperPortfolioUseCase:
    """Derive one futures paper portfolio snapshot from its fills.

    Fills are expected in canonical order -- ``filled_at`` ascending by
    PointInTime.compare(), then ``order_identity`` -- which is exactly what
    FuturesPaperFillRepository returns. Malformed order is rejected, never
    repaired. Fills after ``as_of`` are valid history and are ignored.
    """

    def execute(
        self,
        portfolio_identity: PaperPortfolioIdentity,
        strategy_identity: StrategyIdentity,
        fills: tuple[FuturesPaperFill, ...],
        as_of: PointInTime,
    ) -> FuturesPaperPortfolio:
        """Fold every fill visible at ``as_of`` into a portfolio snapshot."""
        self._validate_inputs(portfolio_identity, strategy_identity, fills, as_of)
        _validate_fill_history(
            "BuildFuturesPaperPortfolioUseCase", portfolio_identity, strategy_identity, fills
        )

        held: dict[FuturesContract, FuturesPosition] = {}
        for fill in fills:
            if fill.filled_at.compare(as_of) > 0:
                continue
            updated = _transition(held.get(fill.contract), fill).position
            if updated is None:
                del held[fill.contract]
            else:
                held[fill.contract] = updated

        positions = tuple(
            held[contract] for contract in sorted(held, key=lambda item: item.natural_key)
        )
        return FuturesPaperPortfolio(
            identity=portfolio_identity,
            strategy_identity=strategy_identity,
            positions=positions,
            as_of=as_of,
        )

    @staticmethod
    def _validate_inputs(
        portfolio_identity: PaperPortfolioIdentity,
        strategy_identity: StrategyIdentity,
        fills: tuple[FuturesPaperFill, ...],
        as_of: PointInTime,
    ) -> None:
        if portfolio_identity is None:
            raise TypeError("BuildFuturesPaperPortfolioUseCase portfolio identity cannot be None.")
        if not isinstance(portfolio_identity, PaperPortfolioIdentity):
            raise TypeError(
                "BuildFuturesPaperPortfolioUseCase portfolio identity must be a "
                "PaperPortfolioIdentity."
            )
        if strategy_identity is None:
            raise TypeError("BuildFuturesPaperPortfolioUseCase strategy identity cannot be None.")
        if not isinstance(strategy_identity, StrategyIdentity):
            raise TypeError(
                "BuildFuturesPaperPortfolioUseCase strategy identity must be a StrategyIdentity."
            )
        if fills is None:
            raise TypeError("BuildFuturesPaperPortfolioUseCase fills cannot be None.")
        if not isinstance(fills, tuple):
            raise TypeError("BuildFuturesPaperPortfolioUseCase fills must be a tuple.")
        if not all(isinstance(fill, FuturesPaperFill) for fill in fills):
            raise TypeError(
                "BuildFuturesPaperPortfolioUseCase fills must contain FuturesPaperFill values."
            )
        if as_of is None:
            raise TypeError("BuildFuturesPaperPortfolioUseCase as-of instant cannot be None.")
        if not isinstance(as_of, PointInTime):
            raise TypeError(
                "BuildFuturesPaperPortfolioUseCase as-of instant must be a PointInTime."
            )
