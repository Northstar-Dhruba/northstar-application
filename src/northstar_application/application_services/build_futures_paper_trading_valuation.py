"""Application valuation of one futures paper portfolio at one explicit cutoff.

A valuation pairs, per concrete FuturesContract, the gross simulated realized
P&L reconstructed from the fills visible at the cutoff with the gross simulated
unrealized P&L of the position still open there. "Gross" means commissions,
fees, slippage, financing and taxes are all excluded; nothing here is net P&L.

It composes the existing rules rather than restating them:

- the paper history is loaded and validated exactly as a paper-trading run
  validates it -- the complete history, including facts after the cutoff;
- BuildFuturesPaperPortfolioUseCase folds the portfolio at the cutoff;
- CalculateFuturesRealizedPnlUseCase realizes P&L from the complete fills and
  decides which contracts are covered: every contract with a visible fill;
- ValueFuturesPaperPortfolioUseCase marks the open positions.

One cutoff governs all three, so no fill or bar after it can move the result.
Nothing is persisted, no clock is read, and currencies are never summed or
converted: a valuation deliberately has no portfolio total.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from northstar_core.derivatives import QuoteValue
from northstar_core.foundation.value_objects import Currency, Money, PointInTime
from northstar_core.futures import FuturesContract, FuturesProductEconomics, FuturesProductReference
from northstar_core.paper_trading import (
    FuturesPaperPortfolio,
    FuturesPosition,
    PaperPortfolioIdentity,
)
from northstar_core.strategy import StrategyIdentity

from northstar_application.application_services.build_futures_paper_portfolio import (
    BuildFuturesPaperPortfolioUseCase,
)
from northstar_application.application_services.calculate_futures_realized_pnl import (
    CalculateFuturesRealizedPnlUseCase,
    FuturesContractRealizedPnl,
)
from northstar_application.application_services.run_futures_paper_trading_decision import (
    _load_history,
)
from northstar_application.application_services.value_futures_paper_portfolio import (
    FuturesContractUnrealizedPnl,
    ValueFuturesPaperPortfolioUseCase,
    _economics_for,
)
from northstar_application.ports import (
    FuturesHistoricalMarketDataRepository,
    FuturesPaperFillRepository,
    FuturesPaperOrderRepository,
    FuturesProductEconomicsRepository,
)

_SUBJECT = "BuildFuturesPaperTradingValuationUseCase"


@dataclass(frozen=True, slots=True)
class FuturesContractPnl:
    """Gross simulated realized and unrealized P&L of one concrete futures contract.

    Exactly one of three states holds:

    - flat: no ``position`` and no mark; ``unrealized_pnl`` is exactly zero,
      because no exposure remains;
    - open and marked: ``position``, ``mark_quote``, ``mark_instant`` and
      ``unrealized_pnl`` are all present;
    - open and unmarked: ``position`` is present but no stored daily close was
      observable by the cutoff, so every mark field and ``unrealized_pnl`` are
      None. None is never zero: an unmarked position has no known value.

    ``realized_pnl`` is always known. Both amounts are in the settlement
    currency, ``realized_pnl.currency``.
    """

    contract: FuturesContract
    realized_pnl: Money
    position: FuturesPosition | None
    mark_quote: QuoteValue | None
    mark_instant: PointInTime | None
    unrealized_pnl: Money | None

    def __post_init__(self) -> None:
        self._validate_types()
        subject = "FuturesContractPnl"
        currency = self.realized_pnl.currency
        marks = (self.mark_quote, self.mark_instant, self.unrealized_pnl)

        if self.position is None:
            if self.mark_quote is not None or self.mark_instant is not None:
                raise ValueError(f"{subject} for a flat contract cannot carry a mark.")
            if self.unrealized_pnl != Money(Decimal(0), currency):
                raise ValueError(
                    f"{subject} for a flat contract must carry exactly zero unrealized P&L "
                    "in its settlement currency."
                )
            return

        if self.position.contract != self.contract:
            raise ValueError(f"{subject} position must be for its contract.")
        if any(value is None for value in marks) and any(value is not None for value in marks):
            raise ValueError(
                f"{subject} for an open position must carry a mark quote, mark instant and "
                "unrealized P&L together, or none of them."
            )
        if self.unrealized_pnl is not None and self.unrealized_pnl.currency != currency:
            raise ValueError(
                f"{subject} realized and unrealized P&L must share one settlement currency."
            )

    def _validate_types(self) -> None:
        subject = "FuturesContractPnl"
        if not isinstance(self.contract, FuturesContract):
            raise TypeError(f"{subject} contract must be a FuturesContract.")
        if not isinstance(self.realized_pnl, Money):
            raise TypeError(f"{subject} realized P&L must be a Money value.")
        for name, expected in (
            ("position", FuturesPosition),
            ("mark_quote", QuoteValue),
            ("mark_instant", PointInTime),
            ("unrealized_pnl", Money),
        ):
            value = getattr(self, name)
            if value is not None and not isinstance(value, expected):
                raise TypeError(f"{subject} {name} must be a {expected.__name__} or None.")

    @property
    def settlement_currency(self) -> Currency:
        """Return the currency both P&L amounts are expressed in."""
        return self.realized_pnl.currency

    @property
    def is_open(self) -> bool:
        """Return whether a position in this contract remains open at the cutoff."""
        return self.position is not None


@dataclass(frozen=True, slots=True)
class FuturesPaperTradingValuation:
    """Gross simulated P&L of one futures paper portfolio at one cutoff.

    ``contracts`` holds one row per concrete contract with a fill visible at
    ``available_through``, ordered by FuturesContract.natural_key. Every open
    position of ``portfolio`` has exactly one row carrying that position; rows
    without a position are contracts traded earlier and flat at the cutoff.
    There is no total: currencies may differ and an unmarked position has no
    known unrealized amount.
    """

    portfolio_identity: PaperPortfolioIdentity
    strategy_identity: StrategyIdentity
    available_through: PointInTime
    portfolio: FuturesPaperPortfolio
    contracts: tuple[FuturesContractPnl, ...]

    def __post_init__(self) -> None:
        subject = "FuturesPaperTradingValuation"
        for name, expected in (
            ("portfolio_identity", PaperPortfolioIdentity),
            ("strategy_identity", StrategyIdentity),
            ("available_through", PointInTime),
            ("portfolio", FuturesPaperPortfolio),
        ):
            if not isinstance(getattr(self, name), expected):
                raise TypeError(f"{subject} {name} must be a {expected.__name__}.")
        if not isinstance(self.contracts, tuple):
            raise TypeError(f"{subject} contracts must be a tuple.")
        if not all(isinstance(row, FuturesContractPnl) for row in self.contracts):
            raise TypeError(f"{subject} contracts must contain FuturesContractPnl values.")

        portfolio = self.portfolio
        if portfolio.identity != self.portfolio_identity:
            raise ValueError(f"{subject} portfolio must be the valued portfolio.")
        if portfolio.strategy_identity != self.strategy_identity:
            raise ValueError(f"{subject} portfolio must belong to the valued strategy.")
        if portfolio.as_of.compare(self.available_through) != 0:
            raise ValueError(f"{subject} portfolio must be as of the valuation cutoff.")

        for previous, row in zip(self.contracts, self.contracts[1:], strict=False):
            if previous.contract.natural_key >= row.contract.natural_key:
                raise ValueError(
                    f"{subject} contracts must be unique and ordered by contract natural key."
                )
        open_positions = tuple(row.position for row in self.contracts if row.position is not None)
        if open_positions != portfolio.positions:
            raise ValueError(
                f"{subject} open contract rows must carry exactly the portfolio's positions."
            )

    @property
    def contract_count(self) -> int:
        """Return how many contracts have a fill visible at the cutoff."""
        return len(self.contracts)

    @property
    def open_position_count(self) -> int:
        """Return how many of those contracts hold an open position at the cutoff."""
        return len(self.portfolio.positions)


def _combine(
    realized: FuturesContractRealizedPnl, unrealized: FuturesContractUnrealizedPnl | None
) -> FuturesContractPnl:
    if unrealized is None:
        return FuturesContractPnl(
            contract=realized.contract,
            realized_pnl=realized.realized_pnl,
            position=None,
            mark_quote=None,
            mark_instant=None,
            unrealized_pnl=Money(Decimal(0), realized.realized_pnl.currency),
        )
    return FuturesContractPnl(
        contract=realized.contract,
        realized_pnl=realized.realized_pnl,
        position=unrealized.position,
        mark_quote=unrealized.mark_quote,
        mark_instant=unrealized.mark_instant,
        unrealized_pnl=unrealized.unrealized_pnl,
    )


class BuildFuturesPaperTradingValuationUseCase:
    """Value one futures paper portfolio's contracts at one explicit cutoff."""

    def __init__(
        self,
        order_repository: FuturesPaperOrderRepository,
        fill_repository: FuturesPaperFillRepository,
        market_repository: FuturesHistoricalMarketDataRepository,
        economics_repository: FuturesProductEconomicsRepository,
    ) -> None:
        for name, value, expected in (
            ("order_repository", order_repository, FuturesPaperOrderRepository),
            ("fill_repository", fill_repository, FuturesPaperFillRepository),
            ("market_repository", market_repository, FuturesHistoricalMarketDataRepository),
            ("economics_repository", economics_repository, FuturesProductEconomicsRepository),
        ):
            if not isinstance(value, expected):
                raise TypeError(f"{_SUBJECT} {name} must be a {expected.__name__}.")

        self._order_repository = order_repository
        self._fill_repository = fill_repository
        self._economics_repository = economics_repository
        self._build_portfolio = BuildFuturesPaperPortfolioUseCase()
        self._realize = CalculateFuturesRealizedPnlUseCase()
        self._mark = ValueFuturesPaperPortfolioUseCase(market_repository, economics_repository)

    def execute(
        self,
        portfolio_identity: PaperPortfolioIdentity,
        strategy_identity: StrategyIdentity,
        available_through: PointInTime,
    ) -> FuturesPaperTradingValuation:
        """Return the portfolio's gross simulated P&L per contract at the cutoff."""
        for name, value, expected in (
            ("portfolio identity", portfolio_identity, PaperPortfolioIdentity),
            ("strategy identity", strategy_identity, StrategyIdentity),
            ("available-through", available_through, PointInTime),
        ):
            if not isinstance(value, expected):
                raise TypeError(f"{_SUBJECT} {name} must be a {expected.__name__}.")

        _, fills = _load_history(
            self._order_repository, self._fill_repository, portfolio_identity, strategy_identity
        )
        portfolio = self._build_portfolio.execute(
            portfolio_identity, strategy_identity, fills, available_through
        )

        # Only products with a visible fill need economics; a future-only product does not.
        economics: dict[FuturesProductReference, FuturesProductEconomics] = {}
        for fill in fills:
            product = fill.contract.product
            if fill.filled_at.compare(available_through) <= 0 and product not in economics:
                economics[product] = _economics_for(_SUBJECT, self._economics_repository, product)

        realized = self._realize.execute(
            portfolio_identity,
            strategy_identity,
            fills,
            tuple(economics.values()),
            available_through,
        )
        unrealized = self._mark.execute(portfolio, available_through)
        if tuple(row.position for row in unrealized) != portfolio.positions:
            raise ValueError(
                f"{_SUBJECT} unrealized valuation must cover exactly the portfolio's positions."
            )

        marked = {row.contract: row for row in unrealized}
        return FuturesPaperTradingValuation(
            portfolio_identity=portfolio_identity,
            strategy_identity=strategy_identity,
            available_through=available_through,
            portfolio=portfolio,
            contracts=tuple(_combine(row, marked.get(row.contract)) for row in realized),
        )
