"""Application valuation of open futures paper positions at an explicit cutoff.

Each open position in a FuturesPaperPortfolio is marked at the last stored daily
close observable at ``available_through``: the latest persisted 1d bar whose
completion instant is at or before the cutoff, compared with
PointInTime.compare(). The mark is not a settlement price, a live provider
quote or a clock price. A mark older than the cutoff is still used and its
instant is reported, so a stale mark is visible rather than hidden.

For signed net contracts ``n`` held at average entry ``a`` and mark ``m``:

    unrealized = (m - a) * n * point_value.amount    in point_value.currency

One signed formula serves long and short, and it is linear through zero, so
negative and zero quotes need nothing special. Arithmetic is Decimal-only under
one explicit context; Money is constructed from the finished amount and its own
operators, which observe the caller's context, are never used.

A position with no stored bar at or before the cutoff is reported unavailable,
never valued at zero. Missing product economics are an error: a partially
valued portfolio must not look complete. Nothing is persisted and no clock is
read.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_HALF_EVEN, Context, localcontext

from northstar_core.derivatives import QuoteValue
from northstar_core.foundation.value_objects import Currency, Money, PointInTime, Timeframe
from northstar_core.futures import (
    FuturesContract,
    FuturesOHLCVBar,
    FuturesProductEconomics,
    FuturesProductReference,
)
from northstar_core.paper_trading import FuturesPaperPortfolio, FuturesPosition

from northstar_application.application_services._futures_repository_output import (
    validate_futures_repository_bars,
)
from northstar_application.ports import (
    FuturesHistoricalMarketDataQuery,
    FuturesHistoricalMarketDataRepository,
    FuturesProductEconomicsRepository,
)

_DAILY = Timeframe("1d")
_PNL_CONTEXT = Context(prec=28, rounding=ROUND_HALF_EVEN)
_SUBJECT = "ValueFuturesPaperPortfolioUseCase"


class FuturesProductEconomicsNotFoundError(ValueError):
    """Raised when a product that must be valued has no configured economics.

    ``reference`` names the product, so callers never parse the message.
    """

    def __init__(self, message: str, reference: FuturesProductReference | None = None) -> None:
        super().__init__(message)
        self.reference = reference


class FuturesProductEconomicsContractViolationError(ValueError):
    """Raised when a FuturesProductEconomicsRepository violates its contract."""


def _economics_for(
    subject: str,
    repository: FuturesProductEconomicsRepository,
    reference: FuturesProductReference,
) -> FuturesProductEconomics:
    """Look up one product's economics, failing on a missing or foreign answer."""
    found = repository.get_economics(reference)
    if found is None:
        raise FuturesProductEconomicsNotFoundError(
            f"{subject} has no product economics for {reference}.", reference
        )
    if not isinstance(found, FuturesProductEconomics):
        raise FuturesProductEconomicsContractViolationError(
            "FuturesProductEconomicsRepository must return FuturesProductEconomics or None."
        )
    if found.reference != reference:
        raise FuturesProductEconomicsContractViolationError(
            f"FuturesProductEconomicsRepository returned economics for {found.reference} "
            f"when asked for {reference}."
        )
    return found


@dataclass(frozen=True, slots=True)
class FuturesContractUnrealizedPnl:
    """Unrealized valuation attempted for one open futures position.

    Either every mark field is present -- the last stored daily close, its
    instant and the resulting P&L -- or none is. All three None means no stored
    daily bar was observable by the cutoff; it never means zero P&L, which is
    an available valuation of ``Money(0, currency)``.
    """

    position: FuturesPosition
    settlement_currency: Currency
    mark_quote: QuoteValue | None
    mark_instant: PointInTime | None
    unrealized_pnl: Money | None

    def __post_init__(self) -> None:
        subject = "FuturesContractUnrealizedPnl"
        if not isinstance(self.position, FuturesPosition):
            raise TypeError(f"{subject} position must be a FuturesPosition.")
        if not isinstance(self.settlement_currency, Currency):
            raise TypeError(f"{subject} settlement currency must be a Currency.")
        if self.mark_quote is not None and not isinstance(self.mark_quote, QuoteValue):
            raise TypeError(f"{subject} mark quote must be a QuoteValue or None.")
        if self.mark_instant is not None and not isinstance(self.mark_instant, PointInTime):
            raise TypeError(f"{subject} mark instant must be a PointInTime or None.")
        if self.unrealized_pnl is not None and not isinstance(self.unrealized_pnl, Money):
            raise TypeError(f"{subject} unrealized P&L must be a Money value or None.")

        present = [
            value is not None for value in (self.mark_quote, self.mark_instant, self.unrealized_pnl)
        ]
        if any(present) and not all(present):
            raise ValueError(
                f"{subject} must carry a mark quote, mark instant and unrealized P&L together, "
                "or none of them."
            )
        if self.unrealized_pnl is not None and (
            self.unrealized_pnl.currency != self.settlement_currency
        ):
            raise ValueError(f"{subject} unrealized P&L must be in the settlement currency.")

    @property
    def contract(self) -> FuturesContract:
        """Return the contract of the valued position."""
        return self.position.contract

    @property
    def is_available(self) -> bool:
        """Return whether a stored daily close was observable by the cutoff."""
        return self.unrealized_pnl is not None


class ValueFuturesPaperPortfolioUseCase:
    """Mark every open position of one futures paper portfolio at one cutoff."""

    def __init__(
        self,
        market_repository: FuturesHistoricalMarketDataRepository,
        economics_repository: FuturesProductEconomicsRepository,
    ) -> None:
        if not isinstance(market_repository, FuturesHistoricalMarketDataRepository):
            raise TypeError(
                f"{_SUBJECT} market_repository must be a FuturesHistoricalMarketDataRepository."
            )
        if not isinstance(economics_repository, FuturesProductEconomicsRepository):
            raise TypeError(
                f"{_SUBJECT} economics_repository must be a FuturesProductEconomicsRepository."
            )
        self._market_repository = market_repository
        self._economics_repository = economics_repository

    def execute(
        self, portfolio: FuturesPaperPortfolio, available_through: PointInTime
    ) -> tuple[FuturesContractUnrealizedPnl, ...]:
        """Return one valuation per open position, in the portfolio's position order."""
        if not isinstance(portfolio, FuturesPaperPortfolio):
            raise TypeError(f"{_SUBJECT} portfolio must be a FuturesPaperPortfolio.")
        if not isinstance(available_through, PointInTime):
            raise TypeError(f"{_SUBJECT} available-through must be a PointInTime.")
        if portfolio.as_of.compare(available_through) != 0:
            raise ValueError(
                f"{_SUBJECT} portfolio as of {portfolio.as_of} cannot be valued with marks "
                f"available through {available_through}."
            )

        # Resolve every product first so missing economics fail before any mark is read.
        economics: dict[FuturesProductReference, FuturesProductEconomics] = {}
        for position in portfolio.positions:
            product = position.contract.product
            if product not in economics:
                economics[product] = _economics_for(_SUBJECT, self._economics_repository, product)

        return tuple(
            self._value(position, economics[position.contract.product], available_through)
            for position in portfolio.positions
        )

    def _value(
        self,
        position: FuturesPosition,
        economics: FuturesProductEconomics,
        available_through: PointInTime,
    ) -> FuturesContractUnrealizedPnl:
        point_value = economics.point_value
        mark = self._last_close(position.contract, available_through)
        if mark is None:
            return FuturesContractUnrealizedPnl(
                position=position,
                settlement_currency=point_value.currency,
                mark_quote=None,
                mark_instant=None,
                unrealized_pnl=None,
            )

        with localcontext(_PNL_CONTEXT):
            amount = (
                (mark.close.value - position.average_entry.value)
                * position.net_contracts
                * point_value.amount
            )
        return FuturesContractUnrealizedPnl(
            position=position,
            settlement_currency=point_value.currency,
            mark_quote=mark.close,
            mark_instant=mark.point_in_time,
            unrealized_pnl=Money(amount, point_value.currency),
        )

    def _last_close(
        self, contract: FuturesContract, available_through: PointInTime
    ) -> FuturesOHLCVBar | None:
        """Return the latest stored daily bar at or before the cutoff, if any."""
        query = FuturesHistoricalMarketDataQuery(
            contract=contract, timeframe=_DAILY, start=None, end=available_through
        )
        bars = validate_futures_repository_bars(self._market_repository.get_bars(query), query)
        latest: FuturesOHLCVBar | None = None
        for bar in bars:
            if bar.point_in_time.compare(available_through) <= 0:
                latest = bar
        return latest
