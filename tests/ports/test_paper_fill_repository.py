"""Contract tests for the PaperFillRepository Application port.

Ordering is the obligation that matters most here and the one an abstract class
cannot enforce, so it is pinned against a conforming reference implementation
whose fixture deliberately includes the sub-second instants that defeat text
ordering.
"""

from __future__ import annotations

from functools import cmp_to_key

import pytest
from northstar_core.domain.value_objects import ListingReference
from northstar_core.foundation.value_objects import (
    Currency,
    ExchangeCode,
    PointInTime,
    Price,
    Quantity,
    Symbol,
)
from northstar_core.paper_trading import (
    ExecutionIntent,
    OrderSide,
    PaperFill,
    PaperFillIdentity,
    PaperOrderIdentity,
    PaperPortfolioIdentity,
)
from northstar_core.strategy import StrategyIdentity

from northstar_application.application_services import BuildPaperPortfolioUseCase
from northstar_application.ports import (
    InvalidPaperFillQueryError,
    PaperFillQuery,
    PaperFillRepository,
)

_USD = Currency("USD")
_LISTING = ListingReference(Symbol("AAPL"), ExchangeCode("NASDAQ"))
_PORTFOLIO = PaperPortfolioIdentity("paper-1")
_OTHER_PORTFOLIO = PaperPortfolioIdentity("paper-2")
_STRATEGY = StrategyIdentity("alpha")

_WHOLE_SECOND = PointInTime("2026-01-20T16:00:00Z")
_SUB_SECOND = PointInTime("2026-01-20T16:00:00.1Z")
_NEXT_DAY = PointInTime("2026-01-21T16:00:00Z")


def _fill(
    fill_id: str,
    filled_at: PointInTime,
    *,
    order_id: str | None = None,
    portfolio: PaperPortfolioIdentity = _PORTFOLIO,
    quantity: str = "10",
) -> PaperFill:
    return PaperFill(
        identity=PaperFillIdentity(fill_id),
        order_identity=PaperOrderIdentity(order_id or f"order-{fill_id}"),
        intent=ExecutionIntent(
            portfolio_identity=portfolio,
            listing_reference=_LISTING,
            side=OrderSide.BUY,
            quantity=Quantity(quantity),
            strategy_identity=_STRATEGY,
            decided_at=filled_at,
        ),
        quantity=Quantity(quantity),
        price=Price("100", _USD),
        filled_at=filled_at,
    )


def _compare(left: PaperFill, right: PaperFill) -> int:
    instant = left.filled_at.compare(right.filled_at)
    if instant:
        return instant
    a, b = left.identity.identity, right.identity.identity
    return (a > b) - (a < b)


class ReferencePaperFillRepository(PaperFillRepository):
    """A minimal repository honouring the documented ordering obligation."""

    def __init__(self, fills: tuple[PaperFill, ...] = ()) -> None:
        self.fills = fills
        self.queries: list[PaperFillQuery] = []

    def get_fills(self, query: PaperFillQuery) -> tuple[PaperFill, ...]:
        self.queries.append(query)
        owned = [fill for fill in self.fills if fill.portfolio_identity == query.portfolio_identity]
        return tuple(sorted(owned, key=cmp_to_key(_compare)))


def _query(portfolio: PaperPortfolioIdentity = _PORTFOLIO) -> PaperFillQuery:
    return PaperFillQuery(portfolio_identity=portfolio)


# ---------------------------------------------------------------------------
# Query contract
# ---------------------------------------------------------------------------


def test_query_carries_only_the_portfolio_identity() -> None:
    query = _query()

    assert query.portfolio_identity == _PORTFOLIO
    assert PaperFillQuery.__slots__ == ("portfolio_identity",)


def test_query_carries_no_listing_status_or_as_of_filter() -> None:
    """Listing narrowing and as-of filtering belong to the fold, not the query."""
    query = _query()

    for absent in (
        "symbol",
        "exchange_code",
        "listing_reference",
        "as_of",
        "status",
        "state",
        "side",
        "start",
        "end",
        "timeframe",
    ):
        assert not hasattr(query, absent)


@pytest.mark.parametrize("value", [None, "paper-1", 1, PaperFillIdentity("paper-1")])
def test_query_validates_its_portfolio_identity(value: object) -> None:
    with pytest.raises(InvalidPaperFillQueryError, match="must be a PaperPortfolioIdentity"):
        PaperFillQuery(portfolio_identity=value)


def test_query_is_immutable_and_compares_by_value() -> None:
    assert _query() == _query()
    assert hash(_query()) == hash(_query())
    assert _query() != _query(_OTHER_PORTFOLIO)

    with pytest.raises(AttributeError):
        _query().portfolio_identity = _OTHER_PORTFOLIO


# ---------------------------------------------------------------------------
# Retrieval contract
# ---------------------------------------------------------------------------


def test_an_empty_repository_returns_an_empty_tuple() -> None:
    assert ReferencePaperFillRepository().get_fills(_query()) == ()


def test_a_query_matching_nothing_returns_an_empty_tuple() -> None:
    repository = ReferencePaperFillRepository((_fill("f-1", _WHOLE_SECOND),))

    assert repository.get_fills(_query(_OTHER_PORTFOLIO)) == ()


def test_every_fill_for_the_portfolio_is_returned() -> None:
    fills = (_fill("f-1", _WHOLE_SECOND), _fill("f-2", _NEXT_DAY))

    assert len(ReferencePaperFillRepository(fills).get_fills(_query())) == 2


def test_other_portfolios_are_excluded() -> None:
    fills = (
        _fill("f-1", _WHOLE_SECOND),
        _fill("f-2", _WHOLE_SECOND, portfolio=_OTHER_PORTFOLIO),
    )

    retrieved = ReferencePaperFillRepository(fills).get_fills(_query())

    assert [fill.identity.identity for fill in retrieved] == ["f-1"]


def test_the_result_is_an_immutable_tuple() -> None:
    repository = ReferencePaperFillRepository((_fill("f-1", _WHOLE_SECOND),))

    assert isinstance(repository.get_fills(_query()), tuple)


# ---------------------------------------------------------------------------
# Deterministic ordering
# ---------------------------------------------------------------------------


def test_fills_are_returned_oldest_to_newest() -> None:
    fills = (_fill("f-2", _NEXT_DAY), _fill("f-1", _WHOLE_SECOND))

    retrieved = ReferencePaperFillRepository(fills).get_fills(_query())

    assert [fill.identity.identity for fill in retrieved] == ["f-1", "f-2"]


def test_equal_instants_are_ordered_by_fill_identity() -> None:
    fills = (
        _fill("zeta", _WHOLE_SECOND),
        _fill("alpha", _WHOLE_SECOND),
    )

    retrieved = ReferencePaperFillRepository(fills).get_fills(_query())

    assert [fill.identity.identity for fill in retrieved] == ["alpha", "zeta"]


def test_ordering_is_chronological_not_textual() -> None:
    """'.1Z' sorts before 'Z' as text while being the later instant."""
    fills = (_fill("f-2", _SUB_SECOND), _fill("f-1", _WHOLE_SECOND))

    retrieved = ReferencePaperFillRepository(fills).get_fills(_query())

    assert _SUB_SECOND.value < _WHOLE_SECOND.value
    assert _SUB_SECOND.compare(_WHOLE_SECOND) > 0
    assert [fill.filled_at for fill in retrieved] == [_WHOLE_SECOND, _SUB_SECOND]


def test_a_text_ordered_result_would_violate_the_contract() -> None:
    """Proves the ordering obligation is load-bearing, not incidentally met."""
    fills = (_fill("f-1", _WHOLE_SECOND), _fill("f-2", _SUB_SECOND))
    text_ordered = sorted(fills, key=lambda fill: fill.filled_at.value)

    conforming = ReferencePaperFillRepository(fills).get_fills(_query())

    assert [fill.filled_at for fill in text_ordered] != [fill.filled_at for fill in conforming]


def test_offset_equivalent_instants_tie_break_by_identity() -> None:
    offset = PointInTime("2026-01-20T21:30:00+05:30")
    fills = (_fill("zeta", offset), _fill("alpha", _WHOLE_SECOND))

    retrieved = ReferencePaperFillRepository(fills).get_fills(_query())

    assert offset.compare(_WHOLE_SECOND) == 0
    assert [fill.identity.identity for fill in retrieved] == ["alpha", "zeta"]


def test_repository_output_satisfies_the_folds_ordering_requirement() -> None:
    """The fold refuses to reorder, so the repository must deliver order."""
    fills = (
        _fill("f-3", _NEXT_DAY),
        _fill("f-2", _SUB_SECOND),
        _fill("f-1", _WHOLE_SECOND),
    )
    repository = ReferencePaperFillRepository(fills)

    ordered = repository.get_fills(_query())
    portfolio = BuildPaperPortfolioUseCase().execute(_PORTFOLIO, ordered, _NEXT_DAY)

    assert portfolio.get_position(_LISTING).quantity == Quantity("30")


# ---------------------------------------------------------------------------
# Port surface
# ---------------------------------------------------------------------------


def test_the_port_is_abstract() -> None:
    with pytest.raises(TypeError):
        PaperFillRepository()


def test_the_port_exposes_only_get_fills() -> None:
    public = [name for name in vars(PaperFillRepository) if not name.startswith("_")]

    assert public == ["get_fills"]


def test_the_repository_is_queried_by_portfolio_alone() -> None:
    repository = ReferencePaperFillRepository()

    repository.get_fills(_query())

    assert repository.queries == [_query()]
