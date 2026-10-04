"""Application derivation of gross realized futures profit and loss.

Realized P&L is reconstructed from immutable fills and contract economics; it
is never stored. Each concrete contract is replayed through the same private
transition the portfolio fold uses, so a position's basis here is exactly the
basis the portfolio reports -- including its 28-digit rounded weighted average.
There is no hidden exact cost basis and no lot tracking.

When a fill closes ``c`` contracts of a prior signed position ``p`` held at
average entry ``a``, at fill quote ``q``, it realizes

    (q - a) * sign(p) * c        quote points

Opening or adding to a position realizes nothing, and a reversal realizes only
the ``abs(p)`` contracts that close the old position; the excess opens the new
one at ``q``. The formula is linear through zero, so negative and zero quotes
need no special case.

Each contract accumulates quote points, then converts once:

    realized = total quote points * point_value.amount    in point_value.currency

The point value is the contract's own: economics are resolved by the complete
FuturesContract, never by its product, so one expiry's points are never
converted at another expiry's point value.

All arithmetic runs on Decimals under one explicit context, and Money is only
constructed from the finished amount: Money's own operators observe the
caller's context and are never used here.

This is gross simulated P&L: no fees, commission, slippage, margin, notional or
settlement. It reads no market data, no clock and persists nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_HALF_EVEN, Context, Decimal, localcontext

from northstar_core.foundation.value_objects import Money, PointInTime
from northstar_core.futures import FuturesContract, FuturesContractEconomics
from northstar_core.paper_trading import (
    FuturesPaperFill,
    FuturesPosition,
    PaperPortfolioIdentity,
)
from northstar_core.strategy import StrategyIdentity

from northstar_application.application_services.build_futures_paper_portfolio import (
    _transition,
    _validate_fill_history,
)

_PNL_CONTEXT = Context(prec=28, rounding=ROUND_HALF_EVEN)
_SUBJECT = "CalculateFuturesRealizedPnlUseCase"


class InvalidFuturesContractEconomicsInputError(ValueError):
    """Raised when supplied contract economics are duplicated or missing for a fill."""


@dataclass(frozen=True, slots=True)
class FuturesContractRealizedPnl:
    """Gross realized P&L of one concrete futures contract, in its settlement currency.

    A contract with visible fills but nothing closed yet carries zero, which is
    different from a contract absent from the calculation.
    """

    contract: FuturesContract
    realized_pnl: Money

    def __post_init__(self) -> None:
        if not isinstance(self.contract, FuturesContract):
            raise TypeError("FuturesContractRealizedPnl contract must be a FuturesContract.")
        if not isinstance(self.realized_pnl, Money):
            raise TypeError("FuturesContractRealizedPnl realized P&L must be a Money value.")


class CalculateFuturesRealizedPnlUseCase:
    """Reconstruct gross realized P&L per contract from fills and economics.

    Fill history follows BuildFuturesPaperPortfolioUseCase exactly: every
    supplied fill must be owned, unique and canonically ordered, and fills after
    ``as_of`` are valid history that realize nothing yet.
    """

    def execute(
        self,
        portfolio_identity: PaperPortfolioIdentity,
        strategy_identity: StrategyIdentity,
        fills: tuple[FuturesPaperFill, ...],
        economics: tuple[FuturesContractEconomics, ...],
        as_of: PointInTime,
    ) -> tuple[FuturesContractRealizedPnl, ...]:
        """Return one realized result per contract with a fill visible at ``as_of``."""
        self._validate_inputs(portfolio_identity, strategy_identity, fills, economics, as_of)
        _validate_fill_history(_SUBJECT, portfolio_identity, strategy_identity, fills)
        lookup = self._economics_lookup(economics)

        held: dict[FuturesContract, FuturesPosition] = {}
        points: dict[FuturesContract, Decimal] = {}
        for fill in fills:
            if fill.filled_at.compare(as_of) > 0:
                continue
            if fill.contract not in lookup:
                raise InvalidFuturesContractEconomicsInputError(
                    f"{_SUBJECT} has no contract economics for {fill.contract}."
                )

            prior = held.get(fill.contract)
            transition = _transition(prior, fill)
            if transition.position is None:
                held.pop(fill.contract, None)
            else:
                held[fill.contract] = transition.position

            accumulated = points.setdefault(fill.contract, Decimal(0))
            if transition.contracts_closed:
                direction = 1 if prior.net_contracts > 0 else -1
                with localcontext(_PNL_CONTEXT):
                    points[fill.contract] = accumulated + (
                        (fill.fill_quote.value - transition.closing_average_entry.value)
                        * direction
                        * transition.contracts_closed
                    )

        results: list[FuturesContractRealizedPnl] = []
        for contract in sorted(points, key=lambda item: item.natural_key):
            point_value = lookup[contract].point_value
            with localcontext(_PNL_CONTEXT):
                amount = points[contract] * point_value.amount
            results.append(
                FuturesContractRealizedPnl(contract, Money(amount, point_value.currency))
            )
        return tuple(results)

    @staticmethod
    def _economics_lookup(
        economics: tuple[FuturesContractEconomics, ...],
    ) -> dict[FuturesContract, FuturesContractEconomics]:
        lookup: dict[FuturesContract, FuturesContractEconomics] = {}
        for entry in economics:
            if entry.contract in lookup:
                raise InvalidFuturesContractEconomicsInputError(
                    f"{_SUBJECT} economics contain {entry.contract} more than once."
                )
            lookup[entry.contract] = entry
        return lookup

    @staticmethod
    def _validate_inputs(
        portfolio_identity: PaperPortfolioIdentity,
        strategy_identity: StrategyIdentity,
        fills: tuple[FuturesPaperFill, ...],
        economics: tuple[FuturesContractEconomics, ...],
        as_of: PointInTime,
    ) -> None:
        if not isinstance(portfolio_identity, PaperPortfolioIdentity):
            raise TypeError(f"{_SUBJECT} portfolio identity must be a PaperPortfolioIdentity.")
        if not isinstance(strategy_identity, StrategyIdentity):
            raise TypeError(f"{_SUBJECT} strategy identity must be a StrategyIdentity.")
        if not isinstance(fills, tuple):
            raise TypeError(f"{_SUBJECT} fills must be a tuple.")
        if not all(isinstance(fill, FuturesPaperFill) for fill in fills):
            raise TypeError(f"{_SUBJECT} fills must contain FuturesPaperFill values.")
        if not isinstance(economics, tuple):
            raise TypeError(f"{_SUBJECT} economics must be a tuple.")
        if not all(isinstance(entry, FuturesContractEconomics) for entry in economics):
            raise TypeError(f"{_SUBJECT} economics must contain FuturesContractEconomics values.")
        if not isinstance(as_of, PointInTime):
            raise TypeError(f"{_SUBJECT} as-of instant must be a PointInTime.")
