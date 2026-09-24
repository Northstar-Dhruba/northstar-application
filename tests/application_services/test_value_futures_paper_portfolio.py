"""Tests for marking open futures paper positions at the last stored daily close."""

from __future__ import annotations

import ast
import importlib
from decimal import (
    ROUND_CEILING,
    ROUND_DOWN,
    ROUND_HALF_EVEN,
    ROUND_HALF_UP,
    Context,
    Decimal,
    localcontext,
)
from functools import cmp_to_key
from pathlib import Path

import pytest
from northstar_core.derivatives import ExpirationDate, QuoteValue
from northstar_core.foundation.value_objects import (
    Currency,
    ExchangeCode,
    Money,
    PointInTime,
    Quantity,
    Symbol,
    Timeframe,
)
from northstar_core.futures import (
    FuturesContract,
    FuturesOHLCVBar,
    FuturesPointValue,
    FuturesProductEconomics,
    FuturesProductReference,
)
from northstar_core.paper_trading import (
    FuturesPaperPortfolio,
    FuturesPosition,
    PaperPortfolioIdentity,
)
from northstar_core.strategy import StrategyIdentity

from northstar_application.application_services import (
    FuturesContractUnrealizedPnl,
    FuturesHistoricalDataContractViolationError,
    FuturesProductEconomicsContractViolationError,
    FuturesProductEconomicsNotFoundError,
    ValueFuturesPaperPortfolioUseCase,
)
from northstar_application.ports import (
    FuturesHistoricalMarketDataQuery,
    FuturesHistoricalMarketDataRepository,
    FuturesProductEconomicsRepository,
)

_USD = Currency("USD")
_EUR = Currency("EUR")
_DAILY = Timeframe("1d")
_ES = FuturesProductReference(Symbol("ES"), ExchangeCode("CME"))
_FESX = FuturesProductReference(Symbol("FESX"), ExchangeCode("EUREX"))
_ES_DEC = FuturesContract(_ES, ExpirationDate("2026-12-18"))
_ES_MAR = FuturesContract(_ES, ExpirationDate("2027-03-19"))
_FESX_DEC = FuturesContract(_FESX, ExpirationDate("2026-12-18"))
_ES_ECONOMICS = FuturesProductEconomics(_ES, FuturesPointValue(Decimal("50"), _USD))
_FESX_ECONOMICS = FuturesProductEconomics(_FESX, FuturesPointValue(Decimal("10"), _EUR))
_PORTFOLIO = PaperPortfolioIdentity("futures-paper-1")
_STRATEGY = StrategyIdentity("futures-forward")
_CONTEXT = Context(prec=28, rounding=ROUND_HALF_EVEN)

_MON = "2026-09-14T21:00:00Z"
_TUE = "2026-09-15T21:00:00Z"
_WED = "2026-09-16T21:00:00Z"
_THU = "2026-09-17T21:00:00Z"
_T = PointInTime(_WED)


def _quote(value: str) -> QuoteValue:
    return QuoteValue(Decimal(value))


def _bar(
    instant: str, close: str, *, open_: str = "999", contract: FuturesContract = _ES_DEC
) -> FuturesOHLCVBar:
    closing, opening = Decimal(close), Decimal(open_)
    return FuturesOHLCVBar(
        contract=contract,
        point_in_time=PointInTime(instant),
        timeframe=_DAILY,
        open=QuoteValue(opening),
        high=QuoteValue(max(opening, closing) + 10),
        low=QuoteValue(min(opening, closing) - 10),
        close=QuoteValue(closing),
        volume=Quantity(Decimal("1000")),
    )


def _position(net: int, entry: str, contract: FuturesContract = _ES_DEC) -> FuturesPosition:
    return FuturesPosition(contract, net, _quote(entry))


def _portfolio(*positions: FuturesPosition, as_of: PointInTime = _T) -> FuturesPaperPortfolio:
    return FuturesPaperPortfolio(_PORTFOLIO, _STRATEGY, tuple(positions), as_of)


class MarketRepository(FuturesHistoricalMarketDataRepository):
    """A conforming repository over a list of stored bars."""

    def __init__(self, bars: list[FuturesOHLCVBar] | None = None) -> None:
        self.bars = bars if bars is not None else []
        self.queries: list[FuturesHistoricalMarketDataQuery] = []

    def get_bars(self, query: FuturesHistoricalMarketDataQuery):
        self.queries.append(query)
        matching = [
            bar
            for bar in self.bars
            if bar.contract == query.contract
            and bar.timeframe == query.timeframe
            and query.covers(bar.point_in_time)
        ]
        return tuple(
            sorted(matching, key=cmp_to_key(lambda a, b: a.point_in_time.compare(b.point_in_time)))
        )


class RawMarket(FuturesHistoricalMarketDataRepository):
    def __init__(self, output: object) -> None:
        self.output = output

    def get_bars(self, query):
        return self.output


class EconomicsRepository(FuturesProductEconomicsRepository):
    def __init__(self, *economics: FuturesProductEconomics) -> None:
        self.economics = {entry.reference: entry for entry in economics}
        self.calls: list[FuturesProductReference] = []

    def get_economics(self, reference: FuturesProductReference):
        self.calls.append(reference)
        return self.economics.get(reference)


class RawEconomics(FuturesProductEconomicsRepository):
    def __init__(self, output: object) -> None:
        self.output = output

    def get_economics(self, reference):
        return self.output


def _value(
    portfolio: FuturesPaperPortfolio,
    bars: list[FuturesOHLCVBar] | None = None,
    *,
    economics: FuturesProductEconomicsRepository | None = None,
    market: FuturesHistoricalMarketDataRepository | None = None,
    through: PointInTime = _T,
) -> tuple[FuturesContractUnrealizedPnl, ...]:
    return ValueFuturesPaperPortfolioUseCase(
        market if market is not None else MarketRepository(bars),
        economics if economics is not None else EconomicsRepository(_ES_ECONOMICS, _FESX_ECONOMICS),
    ).execute(portfolio, through)


def _pnl(position: FuturesPosition, mark: str) -> Money:
    (result,) = _value(_portfolio(position), [_bar(_TUE, mark, contract=position.contract)])
    return result.unrealized_pnl


def _usd(amount: str) -> Money:
    return Money(Decimal(amount), _USD)


# ---------------------------------------------------------------------------
# Economics port
# ---------------------------------------------------------------------------


def test_the_economics_port_is_abstract_with_one_lookup() -> None:
    with pytest.raises(TypeError):
        FuturesProductEconomicsRepository()
    public = [name for name in vars(FuturesProductEconomicsRepository) if not name.startswith("_")]

    assert public == ["get_economics"]


def test_a_conforming_repository_returns_economics_or_none() -> None:
    repository = EconomicsRepository(_ES_ECONOMICS)

    assert repository.get_economics(_ES) == _ES_ECONOMICS
    assert repository.get_economics(_FESX) is None


# ---------------------------------------------------------------------------
# Basic valuation, point value 50 USD per point per contract
# ---------------------------------------------------------------------------


def test_an_empty_portfolio_reads_nothing() -> None:
    market, economics = MarketRepository(), EconomicsRepository(_ES_ECONOMICS)

    assert _value(_portfolio(), market=market, economics=economics) == ()
    assert (market.queries, economics.calls) == ([], [])


@pytest.mark.parametrize(
    ("net", "entry", "mark", "expected"),
    [
        (2, "100", "110", "1000"),
        (2, "100", "90", "-1000"),
        (-2, "100", "90", "1000"),
        (-2, "100", "110", "-1000"),
    ],
    ids=["long-profit", "long-loss", "short-profit", "short-loss"],
)
def test_the_signed_formula_values_long_and_short(net, entry, mark, expected) -> None:
    assert _pnl(_position(net, entry), mark) == _usd(expected)


@pytest.mark.parametrize(
    ("net", "entry", "mark", "expected"),
    [
        (1, "-20", "-10", "500"),
        (-1, "-20", "-30", "500"),
        (1, "10", "-5", "-750"),
        (-1, "10", "-5", "750"),
        (1, "-10", "5", "750"),
        (-1, "-10", "5", "-750"),
        (1, "10", "0", "-500"),
        (-1, "10", "0", "500"),
    ],
    ids=[
        "long-negative-profit",
        "short-negative-profit",
        "long-cross-loss",
        "short-cross-profit",
        "long-cross-up-profit",
        "short-cross-up-loss",
        "long-to-zero-loss",
        "short-to-zero-profit",
    ],
)
def test_negative_quotes_and_zero_crossings_are_linear(net, entry, mark, expected) -> None:
    assert _pnl(_position(net, entry), mark) == _usd(expected)


def test_an_available_zero_is_not_an_unavailable_mark() -> None:
    (zero,) = _value(_portfolio(_position(3, "100")), [_bar(_TUE, "100")])
    (missing,) = _value(_portfolio(_position(3, "100")), [])

    assert zero.is_available
    assert (zero.mark_quote, zero.mark_instant, zero.unrealized_pnl) == (
        _quote("100"),
        PointInTime(_TUE),
        _usd("0"),
    )
    assert not missing.is_available
    assert (missing.mark_quote, missing.mark_instant, missing.unrealized_pnl) == (None, None, None)
    assert missing.settlement_currency == _USD


def test_the_result_keeps_the_exact_position_and_currency() -> None:
    position = _position(-3, "4321.5")
    (result,) = _value(_portfolio(position), [_bar(_TUE, "4300")])

    assert result.position is position
    assert result.contract == _ES_DEC
    assert result.settlement_currency == _USD
    assert result.unrealized_pnl.currency == _USD


# ---------------------------------------------------------------------------
# Mark semantics
# ---------------------------------------------------------------------------


def test_the_mark_is_the_close_not_the_open() -> None:
    (result,) = _value(_portfolio(_position(1, "100")), [_bar(_TUE, "110", open_="130")])

    assert result.mark_quote == _quote("110")
    assert result.unrealized_pnl == _usd("500")


def test_the_latest_visible_bar_marks_not_the_oldest() -> None:
    bars = [_bar(_MON, "90"), _bar(_TUE, "105"), _bar(_WED, "120")]

    (result,) = _value(_portfolio(_position(1, "100")), bars)

    assert (result.mark_quote, result.mark_instant) == (_quote("120"), _T)


def test_a_bar_exactly_at_the_cutoff_is_visible() -> None:
    (result,) = _value(_portfolio(_position(1, "100")), [_bar(_WED, "101")])

    assert result.mark_instant == _T


def test_future_bars_never_move_an_earlier_valuation() -> None:
    portfolio = _portfolio(_position(1, "100"))
    bars = [_bar(_TUE, "105"), _bar(_THU, "999")]

    first = _value(portfolio, bars)
    bars.append(_bar("2026-09-18T21:00:00Z", "-999"))
    second = _value(portfolio, bars)

    assert first == second
    assert first[0].mark_quote == _quote("105")


def test_a_future_bar_returned_by_a_bad_repository_is_a_contract_violation() -> None:
    with pytest.raises(FuturesHistoricalDataContractViolationError, match="outside"):
        _value(_portfolio(_position(1, "100")), market=RawMarket((_bar(_THU, "999"),)))


def test_a_sparse_history_uses_the_older_close_and_says_so() -> None:
    (result,) = _value(_portfolio(_position(1, "100")), [_bar(_MON, "95")])

    assert result.mark_instant == PointInTime(_MON)
    assert result.mark_quote == _quote("95")


def test_no_visible_bar_is_unavailable_not_an_error() -> None:
    (result,) = _value(_portfolio(_position(1, "100")), [_bar(_THU, "110")])

    assert not result.is_available


def test_mark_selection_is_semantic_not_textual() -> None:
    whole, half = "2026-09-16T21:00:00Z", "2026-09-16T21:00:00.5Z"
    bars = [_bar(whole, "101"), _bar(half, "102")]
    cutoff = PointInTime(half)

    (result,) = _value(_portfolio(_position(1, "100"), as_of=cutoff), bars, through=cutoff)

    assert half < whole  # the text trap
    assert result.mark_instant == PointInTime(half)
    assert result.mark_quote == _quote("102")


def test_an_offset_spelled_cutoff_sees_the_bar_at_that_instant() -> None:
    offset = PointInTime("2026-09-17T02:30:00+05:30")  # == Wednesday 21:00Z

    (result,) = _value(
        _portfolio(_position(1, "100"), as_of=offset), [_bar(_WED, "101")], through=offset
    )

    assert result.mark_instant == _T


def test_the_market_query_is_daily_and_ends_at_the_cutoff() -> None:
    market = MarketRepository([_bar(_TUE, "101")])

    _value(_portfolio(_position(1, "100")), market=market)

    assert market.queries == [
        FuturesHistoricalMarketDataQuery(contract=_ES_DEC, timeframe=_DAILY, start=None, end=_T)
    ]


def test_an_expired_open_contract_is_valued_from_its_last_stored_close() -> None:
    expired = FuturesContract(_ES, ExpirationDate("2026-09-10"))
    position = _position(1, "100", expired)

    (valued,) = _value(_portfolio(position), [_bar(_MON, "104", contract=expired)])
    (unvalued,) = _value(_portfolio(position), [])

    assert valued.unrealized_pnl == _usd("200")
    assert not unvalued.is_available


# ---------------------------------------------------------------------------
# Contracts, products and currencies
# ---------------------------------------------------------------------------


def test_expiries_share_one_economics_lookup_and_are_valued_separately() -> None:
    economics = EconomicsRepository(_ES_ECONOMICS)
    market = MarketRepository(
        [_bar(_TUE, "110", contract=_ES_DEC), _bar(_TUE, "80", contract=_ES_MAR)]
    )
    portfolio = _portfolio(_position(1, "100", _ES_DEC), _position(-2, "100", _ES_MAR))

    results = _value(portfolio, economics=economics, market=market)

    assert economics.calls == [_ES]
    assert [query.contract for query in market.queries] == [_ES_DEC, _ES_MAR]
    assert [(r.contract, r.unrealized_pnl) for r in results] == [
        (_ES_DEC, _usd("500")),
        (_ES_MAR, _usd("2000")),
    ]


def test_currencies_stay_separate() -> None:
    economics = EconomicsRepository(_ES_ECONOMICS, _FESX_ECONOMICS)
    portfolio = _portfolio(_position(1, "100"), _position(2, "5000", _FESX_DEC))
    bars = [_bar(_TUE, "110"), _bar(_TUE, "5020", contract=_FESX_DEC)]

    results = _value(portfolio, bars, economics=economics)

    assert economics.calls == [_ES, _FESX]
    assert [r.unrealized_pnl for r in results] == [_usd("500"), Money(Decimal("400"), _EUR)]


def test_one_missing_mark_does_not_fail_the_other_contracts() -> None:
    portfolio = _portfolio(_position(1, "100", _ES_DEC), _position(1, "100", _ES_MAR))

    results = _value(portfolio, [_bar(_TUE, "110", contract=_ES_DEC)])

    assert [r.is_available for r in results] == [True, False]


def test_results_follow_the_portfolio_position_order() -> None:
    positions = (
        _position(1, "1", _ES_DEC),
        _position(1, "1", _ES_MAR),
        _position(1, "1", _FESX_DEC),
    )

    results = _value(_portfolio(*positions), [])

    assert tuple(r.position for r in results) == positions


# ---------------------------------------------------------------------------
# Economics behaviour
# ---------------------------------------------------------------------------


def test_missing_economics_fails_the_whole_valuation_before_any_mark_read() -> None:
    market = MarketRepository([_bar(_TUE, "110")])
    portfolio = _portfolio(_position(1, "100"), _position(1, "5000", _FESX_DEC))

    with pytest.raises(FuturesProductEconomicsNotFoundError, match="FESX@EUREX"):
        _value(portfolio, market=market, economics=EconomicsRepository(_ES_ECONOMICS))

    assert market.queries == []


@pytest.mark.parametrize(
    ("output", "message"),
    [
        ("economics", "must return FuturesProductEconomics or None"),
        (FuturesPointValue(Decimal("50"), _USD), "must return FuturesProductEconomics or None"),
        (_FESX_ECONOMICS, "returned economics for FESX@EUREX when asked for ES@CME"),
    ],
    ids=["string", "point-value", "wrong-reference"],
)
def test_a_misbehaving_economics_repository_is_a_contract_violation(output, message) -> None:
    with pytest.raises(FuturesProductEconomicsContractViolationError, match=message):
        _value(_portfolio(_position(1, "100")), [_bar(_TUE, "110")], economics=RawEconomics(output))


# ---------------------------------------------------------------------------
# Coherence and inputs
# ---------------------------------------------------------------------------


def test_a_portfolio_at_another_instant_is_rejected_before_any_read() -> None:
    market, economics = MarketRepository(), EconomicsRepository(_ES_ECONOMICS)

    with pytest.raises(ValueError, match="cannot be valued with marks"):
        _value(
            _portfolio(_position(1, "100"), as_of=PointInTime(_TUE)),
            market=market,
            economics=economics,
        )

    assert (market.queries, economics.calls) == ([], [])


@pytest.mark.parametrize(
    ("market", "economics", "message"),
    [
        (object(), EconomicsRepository(), "market_repository"),
        (MarketRepository(), object(), "economics_repository"),
    ],
)
def test_the_ports_are_type_checked(market, economics, message: str) -> None:
    with pytest.raises(TypeError, match=message):
        ValueFuturesPaperPortfolioUseCase(market, economics)


def test_execute_inputs_are_type_checked() -> None:
    use_case = ValueFuturesPaperPortfolioUseCase(MarketRepository(), EconomicsRepository())

    with pytest.raises(TypeError, match="portfolio must be a FuturesPaperPortfolio"):
        use_case.execute((_position(1, "1"),), _T)
    with pytest.raises(TypeError, match="available-through must be a PointInTime"):
        use_case.execute(_portfolio(), _WED)


@pytest.mark.parametrize(
    "fields",
    [
        {"mark_quote": _quote("1")},
        {"mark_quote": _quote("1"), "mark_instant": _T},
        {"unrealized_pnl": _usd("1")},
        {"mark_instant": _T, "unrealized_pnl": _usd("1")},
    ],
)
def test_partial_mark_states_are_rejected(fields: dict) -> None:
    values = {"mark_quote": None, "mark_instant": None, "unrealized_pnl": None, **fields}

    with pytest.raises(ValueError, match="together, or none"):
        FuturesContractUnrealizedPnl(_position(1, "1"), _USD, **values)


def test_the_pnl_currency_must_be_the_settlement_currency() -> None:
    with pytest.raises(ValueError, match="settlement currency"):
        FuturesContractUnrealizedPnl(_position(1, "1"), _EUR, _quote("1"), _T, _usd("0"))


def test_the_result_stores_no_duplicated_position_facts() -> None:
    assert FuturesContractUnrealizedPnl.__slots__ == (
        "position",
        "settlement_currency",
        "mark_quote",
        "mark_instant",
        "unrealized_pnl",
    )


# ---------------------------------------------------------------------------
# Decimal determinism
# ---------------------------------------------------------------------------

_PRECISE_ENTRY = "100.1234567890123456789012345"
_PRECISE_MARK = "101.9876543210987654321098765"
_PRECISE_POINT = Decimal("12.34567890123456789012345678")


@pytest.mark.parametrize("precision", [6, 28, 50])
@pytest.mark.parametrize("rounding", [ROUND_DOWN, ROUND_CEILING, ROUND_HALF_UP])
def test_valuation_ignores_the_callers_decimal_context(precision: int, rounding: str) -> None:
    economics = EconomicsRepository(
        FuturesProductEconomics(_ES, FuturesPointValue(_PRECISE_POINT, _USD))
    )
    portfolio = _portfolio(_position(-7, _PRECISE_ENTRY))
    bars = [_bar(_TUE, _PRECISE_MARK)]

    with localcontext() as ambient:
        ambient.prec = precision
        ambient.rounding = rounding
        (result,) = _value(portfolio, bars, economics=economics)
        assert (ambient.prec, ambient.rounding) == (precision, rounding)

    expected = _CONTEXT.multiply(
        _CONTEXT.multiply(
            _CONTEXT.subtract(Decimal(_PRECISE_MARK), Decimal(_PRECISE_ENTRY)), Decimal(-7)
        ),
        _PRECISE_POINT,
    )
    assert result.unrealized_pnl == Money(expected, _USD)
    assert expected != Context(prec=80).multiply(
        Context(prec=80).multiply(
            Context(prec=80).subtract(Decimal(_PRECISE_MARK), Decimal(_PRECISE_ENTRY)),
            Decimal(-7),
        ),
        _PRECISE_POINT,
    )


# ---------------------------------------------------------------------------
# Structure and boundaries
# ---------------------------------------------------------------------------

_MODULE = "northstar_application.application_services.value_futures_paper_portfolio"
_ARITHMETIC = (ast.Add, ast.Sub, ast.Mult, ast.Div)


def _tree() -> ast.Module:
    return ast.parse(Path(importlib.import_module(_MODULE).__file__).read_text(encoding="utf-8"))


def test_every_arithmetic_operation_runs_under_the_explicit_context() -> None:
    tree = _tree()
    guarded: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.With) and any(
            isinstance(item.context_expr, ast.Call)
            and getattr(item.context_expr.func, "id", None) == "localcontext"
            for item in node.items
        ):
            guarded.update(id(child) for child in ast.walk(node))
    arithmetic = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.BinOp) and isinstance(node.op, _ARITHMETIC)
    ]

    assert arithmetic
    assert all(id(node) in guarded for node in arithmetic)
    assert not [node for node in ast.walk(tree) if isinstance(node, ast.AugAssign)]
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "Money":
            assert not [
                a for a in node.args for inner in ast.walk(a) if isinstance(inner, ast.BinOp)
            ]


def test_the_valuation_touches_no_outer_layer_or_realized_pnl() -> None:
    modules: set[str] = set()
    names: set[str] = set()
    for node in ast.walk(_tree()):
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)

    for module in modules:
        assert module.split(".")[0] not in {
            "northstar_infrastructure",
            "sqlite3",
            "databento",
            "exchange_calendars",
            "requests",
            "urllib",
            "time",
            "datetime",
        }
        assert "realized" not in module and "acquire" not in module
    assert not names & {
        "FuturesHistoricalMarketDataSource",
        "FuturesTradingSessionResolver",
        "CalculateFuturesRealizedPnlUseCase",
        "Price",
    }


def test_the_public_surface_is_exported_without_helpers() -> None:
    import northstar_application.application_services as services
    import northstar_application.ports as ports

    assert "FuturesProductEconomicsRepository" in ports.__all__
    for name in (
        "FuturesContractUnrealizedPnl",
        "ValueFuturesPaperPortfolioUseCase",
        "FuturesProductEconomicsNotFoundError",
        "FuturesProductEconomicsContractViolationError",
    ):
        assert name in services.__all__
    for private in ("_PNL_CONTEXT", "_last_close", "_DAILY"):
        assert not hasattr(services, private)
