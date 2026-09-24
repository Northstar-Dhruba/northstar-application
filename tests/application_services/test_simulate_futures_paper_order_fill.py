"""Tests for filling one futures paper order at the next stored daily open."""

from __future__ import annotations

import ast
import importlib
from decimal import Decimal
from pathlib import Path

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
from northstar_core.paper_trading import (
    FuturesContractCount,
    FuturesExecutionIntent,
    FuturesPaperFill,
    FuturesPaperOrder,
    OrderSide,
    PaperOrderIdentity,
    PaperPortfolioIdentity,
)
from northstar_core.strategy import StrategyIdentity

from northstar_application.application_services import (
    FuturesHistoricalDataContractViolationError,
    FuturesPaperExecutionIdentityService,
    SimulateFuturesPaperOrderFillUseCase,
)
from northstar_application.ports import (
    FuturesHistoricalMarketDataQuery,
    FuturesHistoricalMarketDataRepository,
)


def _contract(product: str = "ES", expiry: str = "2026-12-18") -> FuturesContract:
    return FuturesContract(
        FuturesProductReference(Symbol(product), ExchangeCode("CME")), ExpirationDate(expiry)
    )


_ES_DEC = _contract()
_DAILY = Timeframe("1d")

_MON = "2026-09-14T21:00:00Z"  # decision
_TUE = "2026-09-15T21:00:00Z"
_WED = "2026-09-16T21:00:00Z"
_THU = "2026-09-17T21:00:00Z"
_FRI = "2026-09-18T21:00:00Z"
_D = PointInTime(_MON)


def _bar(
    instant: str,
    open_: str = "100",
    *,
    high: str | None = None,
    low: str | None = None,
    close: str | None = None,
    contract: FuturesContract = _ES_DEC,
    timeframe: Timeframe = _DAILY,
) -> FuturesOHLCVBar:
    base = Decimal(open_)
    return FuturesOHLCVBar(
        contract=contract,
        point_in_time=PointInTime(instant),
        timeframe=timeframe,
        open=QuoteValue(base),
        high=QuoteValue(Decimal(high) if high is not None else base + 10),
        low=QuoteValue(Decimal(low) if low is not None else base - 10),
        close=QuoteValue(Decimal(close) if close is not None else base + 5),
        volume=Quantity(Decimal("1000")),
    )


def _order(
    decided_at: str = _MON,
    *,
    order_id: str = "order-1",
    side: OrderSide = OrderSide.BUY,
    contracts: int = 2,
) -> FuturesPaperOrder:
    return FuturesPaperOrder(
        identity=PaperOrderIdentity(order_id),
        intent=FuturesExecutionIntent(
            portfolio_identity=PaperPortfolioIdentity("futures-paper-1"),
            contract=_ES_DEC,
            side=side,
            contracts=FuturesContractCount(contracts),
            strategy_identity=StrategyIdentity("futures-forward"),
            decided_at=PointInTime(decided_at),
        ),
    )


class ReferenceRepository(FuturesHistoricalMarketDataRepository):
    """Returns stored bars matching the query, in the order they are held."""

    def __init__(self, bars: tuple[FuturesOHLCVBar, ...] = ()) -> None:
        self.bars = bars
        self.queries: list[FuturesHistoricalMarketDataQuery] = []

    def get_bars(self, query: FuturesHistoricalMarketDataQuery) -> tuple[FuturesOHLCVBar, ...]:
        self.queries.append(query)
        return tuple(
            bar
            for bar in self.bars
            if bar.contract == query.contract
            and bar.timeframe == query.timeframe
            and query.covers(bar.point_in_time)
        )


class RawRepository(FuturesHistoricalMarketDataRepository):
    """Returns whatever it is given, to exercise output validation."""

    def __init__(self, output: object) -> None:
        self.output = output

    def get_bars(self, query: FuturesHistoricalMarketDataQuery) -> tuple[FuturesOHLCVBar, ...]:
        return self.output  # type: ignore[return-value]


def _simulate(
    bars: tuple[FuturesOHLCVBar, ...],
    available_through: str = _FRI,
    order: FuturesPaperOrder | None = None,
) -> FuturesPaperFill | None:
    return SimulateFuturesPaperOrderFillUseCase(ReferenceRepository(bars)).execute(
        order if order is not None else _order(), PointInTime(available_through)
    )


# ---------------------------------------------------------------------------
# Pending
# ---------------------------------------------------------------------------


def test_no_bars_is_pending() -> None:
    assert _simulate(()) is None


def test_only_the_decision_bar_is_pending() -> None:
    """The decision bar is evidence the decision saw, never a fill."""
    assert _simulate((_bar(_MON, "90"),)) is None


def test_a_cutoff_at_the_decision_is_a_valid_pending_request() -> None:
    assert _simulate((_bar(_MON), _bar(_TUE)), available_through=_MON) is None


def test_a_later_bar_beyond_the_cutoff_is_excluded() -> None:
    assert _simulate((_bar(_THU),), available_through=_WED) is None


def test_a_cutoff_before_the_decision_is_rejected() -> None:
    with pytest.raises(ValueError, match="precedes the decision instant"):
        _simulate((_bar(_TUE),), available_through="2026-09-14T20:59:59Z")


# ---------------------------------------------------------------------------
# Fill-bar selection
# ---------------------------------------------------------------------------


def test_the_first_later_bar_fills_the_order() -> None:
    fill = _simulate((_bar(_MON, "90"), _bar(_TUE, "101"), _bar(_WED, "102")))

    assert fill is not None
    assert fill.filled_at == PointInTime(_TUE)
    assert fill.fill_quote == QuoteValue(Decimal("101"))


def test_the_fill_uses_the_open_not_high_low_or_close() -> None:
    bar = _bar(_TUE, "100", high="150", low="50", close="120")

    fill = _simulate((_bar(_MON, "90", close="95"), bar))

    assert fill.fill_quote == bar.open == QuoteValue(Decimal("100"))
    assert fill.fill_quote not in (bar.high, bar.low, bar.close)


def test_sparse_history_fills_on_the_next_actual_bar() -> None:
    """Monday decision, next stored bar Thursday: no calendar, no synthetic sessions."""
    fill = _simulate((_bar(_MON), _bar(_THU, "104")))

    assert fill.filled_at == PointInTime(_THU)
    assert fill.fill_quote == QuoteValue(Decimal("104"))


def test_later_bars_are_ignored_for_selection() -> None:
    fill = _simulate((_bar(_TUE, "101"), _bar(_WED, "999"), _bar(_THU, "888")))

    assert fill.filled_at == PointInTime(_TUE)
    assert fill.fill_quote == QuoteValue(Decimal("101"))


def test_appending_later_bars_does_not_change_the_fill() -> None:
    first = _simulate((_bar(_MON), _bar(_TUE, "101")), available_through=_TUE)
    later = _simulate(
        (_bar(_MON), _bar(_TUE, "101"), _bar(_WED, "150"), _bar(_THU, "50")),
        available_through=_FRI,
    )

    assert first is not None
    assert later == first


def test_a_back_filled_earlier_bar_changes_the_derived_fill() -> None:
    """Accepted debt: the fill is derived, so back-fill can move it."""
    original = _simulate((_bar(_THU, "104"),))
    back_filled = _simulate((_bar(_TUE, "101"), _bar(_THU, "104")))

    assert original.filled_at == PointInTime(_THU)
    assert back_filled.filled_at == PointInTime(_TUE)
    assert original.identity == back_filled.identity
    assert original != back_filled


# ---------------------------------------------------------------------------
# Quote sign
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("open_", ["0", "-37.63"])
def test_zero_and_negative_opens_fill(open_: str) -> None:
    fill = _simulate((_bar(_TUE, open_),))

    assert fill.fill_quote == QuoteValue(Decimal(open_))


# ---------------------------------------------------------------------------
# Semantic timestamps
# ---------------------------------------------------------------------------


def test_an_offset_equivalent_cutoff_reads_the_same_evidence() -> None:
    offset = "2026-09-16T02:30:00+05:30"  # == Tuesday 21:00Z

    assert PointInTime(offset).compare(PointInTime(_TUE)) == 0
    assert _simulate((_bar(_TUE, "101"),), available_through=offset) == _simulate(
        (_bar(_TUE, "101"),), available_through=_TUE
    )


def test_an_offset_equivalent_decision_bar_is_still_the_decision_bar() -> None:
    order = _order("2026-09-15T02:30:00+05:30")  # == Monday 21:00Z

    assert _simulate((_bar(_MON),), order=order) is None


def test_a_bar_half_a_second_after_the_decision_fills() -> None:
    """'.5Z' sorts before 'Z' as text, yet is the later instant."""
    half = "2026-09-14T21:00:00.5Z"

    assert half < _MON
    fill = _simulate((_bar(_MON, "90"), _bar(half, "91")), available_through=_TUE)

    assert fill.filled_at == PointInTime(half)
    assert fill.fill_quote == QuoteValue(Decimal("91"))


def test_a_whole_second_bar_before_a_sub_second_decision_is_not_fill_evidence() -> None:
    order = _order("2026-09-14T21:00:00.5Z")

    repository = ReferenceRepository((_bar(_MON, "90"), _bar(_TUE, "101")))
    fill = SimulateFuturesPaperOrderFillUseCase(repository).execute(order, PointInTime(_TUE))

    assert fill.filled_at == PointInTime(_TUE)
    assert fill.fill_quote == QuoteValue(Decimal("101"))


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------


def test_the_query_is_the_orders_contract_daily_from_decision_to_cutoff() -> None:
    repository = ReferenceRepository()

    SimulateFuturesPaperOrderFillUseCase(repository).execute(_order(), PointInTime(_WED))

    assert repository.queries == [
        FuturesHistoricalMarketDataQuery(
            contract=_ES_DEC, timeframe=_DAILY, start=_D, end=PointInTime(_WED)
        )
    ]


def test_other_contracts_and_timeframes_are_never_fill_evidence() -> None:
    bars = (
        _bar(_TUE, "1", contract=_contract("MES")),
        _bar(_TUE, "2", contract=_contract(expiry="2027-03-19")),
        _bar(_TUE, "3", timeframe=Timeframe("1h")),
        _bar(_WED, "102"),
    )

    assert _simulate(bars).filled_at == PointInTime(_WED)


# ---------------------------------------------------------------------------
# Malformed repository output is rejected, never repaired
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("output", "message"),
    [
        ([_bar(_TUE)], "must return a tuple"),
        (("bar",), "must be a FuturesOHLCVBar"),
        ((_bar(_TUE, contract=_contract("MES")),), "not the queried contract"),
        ((_bar(_TUE, timeframe=Timeframe("1h")),), "does not match the queried timeframe"),
        ((_bar(_FRI),), "outside the queried window"),
        ((_bar("2026-09-13T21:00:00Z"),), "outside the queried window"),
        ((_bar(_WED), _bar(_TUE)), "ordered oldest to newest"),
        ((_bar(_TUE), _bar(_TUE)), "share the instant"),
        (
            (_bar("2026-09-14T21:00:00.5Z"), _bar(_MON)),
            "ordered oldest to newest",
        ),
    ],
    ids=[
        "list",
        "foreign",
        "contract",
        "timeframe",
        "after-cutoff",
        "before-decision",
        "unordered",
        "duplicate",
        "text-ordered",
    ],
)
def test_malformed_repository_output_is_rejected(output: object, message: str) -> None:
    use_case = SimulateFuturesPaperOrderFillUseCase(RawRepository(output))

    with pytest.raises(FuturesHistoricalDataContractViolationError, match=message):
        use_case.execute(_order(), PointInTime(_WED))


# ---------------------------------------------------------------------------
# Constructed fill
# ---------------------------------------------------------------------------


def test_the_fill_executes_the_orders_full_intent() -> None:
    order = _order(side=OrderSide.SELL, contracts=7)

    fill = _simulate((_bar(_TUE),), order=order)

    assert fill.order_identity == order.identity
    assert fill.intent == order.intent
    assert fill.contracts == FuturesContractCount(7)
    assert fill.side is OrderSide.SELL


def test_the_fill_identity_is_derived_from_the_order_identity_only() -> None:
    order = _order()
    expected = FuturesPaperExecutionIdentityService().fill_identity(order.identity)

    first = _simulate((_bar(_TUE, "101"),), order=order)
    other_payload = _simulate((_bar(_WED, "555"),), order=order)

    assert first.identity == expected
    assert other_payload.identity == expected


def test_simulation_is_deterministic() -> None:
    bars = (_bar(_MON), _bar(_TUE, "101"))

    assert _simulate(bars) == _simulate(bars)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("repository", [None, "repository", object()])
def test_the_repository_dependency_is_type_checked(repository: object) -> None:
    with pytest.raises(TypeError, match="FuturesHistoricalMarketDataRepository"):
        SimulateFuturesPaperOrderFillUseCase(repository)


@pytest.mark.parametrize(
    ("order", "available_through", "message"),
    [
        (None, PointInTime(_TUE), "order cannot be None"),
        ("order", PointInTime(_TUE), "order must be a FuturesPaperOrder"),
        ("valid", None, "available-through cannot be None"),
        ("valid", _TUE, "available-through must be a PointInTime"),
    ],
)
def test_inputs_are_type_checked(order: object, available_through: object, message: str) -> None:
    use_case = SimulateFuturesPaperOrderFillUseCase(ReferenceRepository())

    with pytest.raises(TypeError, match=message):
        use_case.execute(_order() if order == "valid" else order, available_through)


def test_an_invalid_cutoff_reads_no_repository() -> None:
    repository = ReferenceRepository()

    with pytest.raises(ValueError):
        SimulateFuturesPaperOrderFillUseCase(repository).execute(
            _order(), PointInTime("2026-09-13T21:00:00Z")
        )

    assert repository.queries == []


# ---------------------------------------------------------------------------
# Module boundaries
# ---------------------------------------------------------------------------

_MODULE_NAME = "northstar_application.application_services.simulate_futures_paper_order_fill"


def _tree() -> ast.Module:
    module = importlib.import_module(_MODULE_NAME)
    return ast.parse(Path(module.__file__).read_text(encoding="utf-8"))


def test_the_simulation_depends_only_on_the_historical_repository_port() -> None:
    modules: set[str] = set()
    names: set[str] = set()
    for node in ast.walk(_tree()):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.add(node.module)
            names.update(alias.name for alias in node.names)

    for module in modules:
        assert module.split(".")[0] not in {
            "northstar_infrastructure",
            "sqlite3",
            "databento",
            "exchange_calendars",
            "requests",
            "urllib",
            "socket",
            "time",
            "datetime",
            "random",
            "uuid",
        }
        assert "acquire" not in module
        assert "session" not in module
    port_names = {name for name in names if name.startswith("Futures") and "Repository" in name}
    assert port_names == {"FuturesHistoricalMarketDataRepository"}
    assert not names & {
        "FuturesHistoricalMarketDataSource",
        "FuturesTradingSessionResolver",
        "Money",
        "Price",
        "Currency",
        "PaperOrderStatus",
    }


def test_the_simulation_reads_no_clock() -> None:
    for node in ast.walk(_tree()):
        if isinstance(node, ast.Attribute):
            assert node.attr not in {"now", "today", "utcnow", "close", "high", "low"}


def test_the_use_case_is_exported() -> None:
    import northstar_application.application_services as services

    assert "SimulateFuturesPaperOrderFillUseCase" in services.__all__
    assert not hasattr(services, "_PAPER_TRADING_TIMEFRAME")
