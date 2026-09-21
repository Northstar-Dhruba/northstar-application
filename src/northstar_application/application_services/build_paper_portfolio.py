"""Application folding of one paper fill history into a portfolio snapshot.

Holdings are derived, never stored. A portfolio is a pure function of the fills
that produced it and the instant it is asked about, so the same history always
folds to the same snapshot and a snapshot can be rebuilt at any earlier instant
without keeping a separate record.

The fold accounts only for what was transacted: quantity held and the cost paid
for it. It calculates no realised or unrealised profit and loss, holds no cash
or balance, applies no commission or slippage, reads no clock and persists
nothing.
"""

from __future__ import annotations

from decimal import ROUND_HALF_EVEN, Context, localcontext

from northstar_core.domain.value_objects import ListingReference
from northstar_core.foundation.value_objects import PointInTime, Price, Quantity
from northstar_core.paper_trading import (
    OrderSide,
    PaperFill,
    PaperPortfolio,
    PaperPortfolioIdentity,
    Position,
)

_COST_CONTEXT = Context(prec=28, rounding=ROUND_HALF_EVEN)


class InvalidPaperFillHistoryError(ValueError):
    """Raised when a paper fill history cannot produce a coherent portfolio."""


def _listing_key(listing_reference: ListingReference) -> tuple[str, str]:
    """Return the canonical position order key: symbol, then exchange code."""
    return (listing_reference.symbol.value, listing_reference.exchange_code.value)


class BuildPaperPortfolioUseCase:
    """Derive one portfolio snapshot from the fills that produced it.

    The use case holds no dependencies and reads no clock: the snapshot instant
    is supplied by the caller, which is what lets the same history be rebuilt
    as of any earlier point.
    """

    def execute(
        self,
        portfolio_identity: PaperPortfolioIdentity,
        fills: tuple[PaperFill, ...],
        as_of: PointInTime,
    ) -> PaperPortfolio:
        """Fold every fill visible at ``as_of`` into a portfolio snapshot."""
        self._validate_inputs(portfolio_identity, fills, as_of)
        self._validate_history(portfolio_identity, fills)

        visible = tuple(fill for fill in fills if fill.filled_at.compare(as_of) <= 0)

        held: dict[tuple[str, str], Position] = {}
        for fill in visible:
            key = _listing_key(fill.listing_reference)
            if fill.side is OrderSide.BUY:
                held[key] = self._apply_buy(held.get(key), fill)
                continue
            reduced = self._apply_sell(held.get(key), fill)
            if reduced is None:
                del held[key]
            else:
                held[key] = reduced

        positions = tuple(held[key] for key in sorted(held))
        return PaperPortfolio(
            identity=portfolio_identity,
            positions=positions,
            as_of=as_of,
        )

    @staticmethod
    def _apply_buy(position: Position | None, fill: PaperFill) -> Position:
        """Open a holding, or add to one at its new weighted average cost."""
        if position is None:
            return Position(
                listing_reference=fill.listing_reference,
                quantity=fill.quantity,
                average_price=fill.price,
            )

        if fill.price.currency != position.average_price.currency:
            raise InvalidPaperFillHistoryError(
                "BuildPaperPortfolioUseCase cannot add a fill denominated in "
                f"{fill.price.currency} to a position held in "
                f"{position.average_price.currency}."
            )

        with localcontext(_COST_CONTEXT):
            held_quantity = position.quantity.value
            filled_quantity = fill.quantity.value
            total_quantity = held_quantity + filled_quantity
            total_cost = (
                held_quantity * position.average_price.amount + filled_quantity * fill.price.amount
            )
            return Position(
                listing_reference=position.listing_reference,
                quantity=Quantity(total_quantity),
                average_price=Price(total_cost / total_quantity, fill.price.currency),
            )

    @staticmethod
    def _apply_sell(position: Position | None, fill: PaperFill) -> Position | None:
        """Reduce a holding, returning None when it is closed out entirely."""
        if position is None:
            raise InvalidPaperFillHistoryError(
                "BuildPaperPortfolioUseCase cannot sell "
                f"{fill.listing_reference}, which is not held."
            )
        if fill.price.currency != position.average_price.currency:
            raise InvalidPaperFillHistoryError(
                "BuildPaperPortfolioUseCase cannot sell a position held in "
                f"{position.average_price.currency} at a price denominated in "
                f"{fill.price.currency}."
            )
        if fill.quantity.value > position.quantity.value:
            raise InvalidPaperFillHistoryError(
                "BuildPaperPortfolioUseCase cannot sell more than is held; "
                "paper trading does not permit short positions."
            )
        if fill.quantity.value == position.quantity.value:
            return None

        with localcontext(_COST_CONTEXT):
            remaining = position.quantity.value - fill.quantity.value
            return Position(
                listing_reference=position.listing_reference,
                quantity=Quantity(remaining),
                average_price=position.average_price,
            )

    @staticmethod
    def _validate_inputs(
        portfolio_identity: PaperPortfolioIdentity,
        fills: tuple[PaperFill, ...],
        as_of: PointInTime,
    ) -> None:
        if portfolio_identity is None:
            raise TypeError("BuildPaperPortfolioUseCase portfolio identity cannot be None.")
        if not isinstance(portfolio_identity, PaperPortfolioIdentity):
            raise TypeError(
                "BuildPaperPortfolioUseCase portfolio identity must be a PaperPortfolioIdentity."
            )
        if fills is None:
            raise TypeError("BuildPaperPortfolioUseCase fills cannot be None.")
        if not isinstance(fills, tuple):
            raise TypeError("BuildPaperPortfolioUseCase fills must be a tuple.")
        if not all(isinstance(fill, PaperFill) for fill in fills):
            raise TypeError("BuildPaperPortfolioUseCase fills must contain PaperFill values.")
        if as_of is None:
            raise TypeError("BuildPaperPortfolioUseCase as-of instant cannot be None.")
        if not isinstance(as_of, PointInTime):
            raise TypeError("BuildPaperPortfolioUseCase as-of instant must be a PointInTime.")

    @staticmethod
    def _validate_history(
        portfolio_identity: PaperPortfolioIdentity, fills: tuple[PaperFill, ...]
    ) -> None:
        """Require one coherent, deterministically ordered history.

        Malformed input is rejected rather than reordered: a history that does
        not already state its own sequence cannot be folded without guessing
        which trade happened first.
        """
        seen_fills: set[str] = set()
        seen_orders: set[str] = set()
        previous: PaperFill | None = None
        for fill in fills:
            if fill.portfolio_identity != portfolio_identity:
                raise InvalidPaperFillHistoryError(
                    "BuildPaperPortfolioUseCase fills must all belong to the requested portfolio."
                )
            if fill.identity.identity in seen_fills:
                raise InvalidPaperFillHistoryError(
                    "BuildPaperPortfolioUseCase fills cannot contain duplicate fill identities."
                )
            seen_fills.add(fill.identity.identity)
            if fill.order_identity.identity in seen_orders:
                raise InvalidPaperFillHistoryError(
                    "BuildPaperPortfolioUseCase fills cannot contain two fills for one order."
                )
            seen_orders.add(fill.order_identity.identity)

            if previous is not None:
                instant = previous.filled_at.compare(fill.filled_at)
                if instant > 0 or (
                    instant == 0 and previous.identity.identity > fill.identity.identity
                ):
                    raise InvalidPaperFillHistoryError(
                        "BuildPaperPortfolioUseCase fills must be ordered by fill instant, "
                        "then fill identity."
                    )
            previous = fill
