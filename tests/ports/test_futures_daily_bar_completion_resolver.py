"""Contract tests for the FuturesDailyBarCompletionResolver Application port.

The port cannot enforce its own semantics, so the obligations are pinned here
against a conforming in-memory reference implementation. An adapter that fails
these behaviours is not a FuturesDailyBarCompletionResolver, whatever calendar
it is built on.

No calendar library is used or imported here. The reference implementation is a
lookup table, because this port's contract is about what the answers *mean*,
not about how a venue's schedule is computed.
"""

from __future__ import annotations

import inspect
from datetime import date

import pytest
from northstar_core.foundation.value_objects import ExchangeCode, PointInTime, Symbol
from northstar_core.futures import FuturesContract, FuturesProductReference

from northstar_application.ports import (
    FuturesDailyBarCompletionResolver,
    FuturesSessionResolutionError,
    TradingSessionResolutionError,
    TradingSessionResolver,
)

_CME = ExchangeCode("CME")
_NYMEX = ExchangeCode("NYMEX")
_NASDAQ = ExchangeCode("NASDAQ")

_ES = FuturesProductReference(Symbol("ES"), _CME)
_MES = FuturesProductReference(Symbol("MES"), _CME)
_CL = FuturesProductReference(Symbol("CL"), _NYMEX)
_EQUITY_VENUE_PRODUCT = FuturesProductReference(Symbol("ES"), _NASDAQ)

# A normal session, an early close, a weekend and a holiday, as a venue would
# actually report them. The completion instants match the real CME shape: a
# session labelled D completes at 22:00Z, and an early close ends sooner.
_SESSIONS: dict[date, str] = {
    date(2026, 9, 15): "2026-09-15T22:00:00Z",
    date(2026, 9, 16): "2026-09-16T22:00:00Z",
    date(2026, 3, 6): "2026-03-06T23:00:00Z",
    date(2026, 7, 3): "2026-07-03T17:00:00Z",
    date(2026, 11, 27): "2026-11-27T18:00:00Z",
}
_NON_SESSIONS = (date(2026, 9, 19), date(2026, 9, 20), date(2026, 12, 25))

_SUPPORTED_VENUES = frozenset({"CME", "NYMEX"})


class ReferenceFuturesDailyBarCompletionResolver(FuturesDailyBarCompletionResolver):
    """A minimal resolver honouring every documented obligation."""

    def __init__(self) -> None:
        self.calls: list[tuple[FuturesProductReference, date]] = []

    def resolve_completion(
        self, product: FuturesProductReference, trading_date: date
    ) -> PointInTime | None:
        if not isinstance(product, FuturesProductReference):
            raise TypeError("product must be a FuturesProductReference.")
        if not isinstance(trading_date, date):
            raise TypeError("trading_date must be a datetime.date.")

        self.calls.append((product, trading_date))

        if product.exchange_code.value not in _SUPPORTED_VENUES:
            raise FuturesSessionResolutionError(
                f"Unsupported futures venue: {product.exchange_code.value}."
            )

        completion = _SESSIONS.get(trading_date)
        return PointInTime(completion) if completion is not None else None


@pytest.fixture
def resolver() -> ReferenceFuturesDailyBarCompletionResolver:
    return ReferenceFuturesDailyBarCompletionResolver()


# ---------------------------------------------------------------------------
# A session resolves to its completion instant
# ---------------------------------------------------------------------------


def test_a_trading_session_resolves_to_a_point_in_time(
    resolver: ReferenceFuturesDailyBarCompletionResolver,
) -> None:
    completion = resolver.resolve_completion(_ES, date(2026, 9, 15))

    assert isinstance(completion, PointInTime)
    assert completion == PointInTime("2026-09-15T22:00:00Z")


@pytest.mark.parametrize(("trading_date", "expected"), sorted(_SESSIONS.items()))
def test_every_session_resolves_to_its_own_completion_instant(
    resolver: ReferenceFuturesDailyBarCompletionResolver, trading_date: date, expected: str
) -> None:
    assert resolver.resolve_completion(_ES, trading_date) == PointInTime(expected)


def test_the_completion_instant_falls_on_the_session_label_date(
    resolver: ReferenceFuturesDailyBarCompletionResolver,
) -> None:
    """The label names the session; a CME session opens the previous civil day."""
    completion = resolver.resolve_completion(_ES, date(2026, 9, 15))

    assert completion is not None
    assert completion.value.startswith("2026-09-15T")


def test_sessions_are_not_all_the_same_length(
    resolver: ReferenceFuturesDailyBarCompletionResolver,
) -> None:
    """An early close ends sooner, so fixed-duration arithmetic cannot be used."""
    full = resolver.resolve_completion(_ES, date(2026, 9, 15))
    early = resolver.resolve_completion(_ES, date(2026, 7, 3))

    assert full is not None and early is not None
    assert full.value.endswith("T22:00:00Z")
    assert early.value.endswith("T17:00:00Z")


def test_the_resolver_returns_a_canonical_point_in_time(
    resolver: ReferenceFuturesDailyBarCompletionResolver,
) -> None:
    completion = resolver.resolve_completion(_ES, date(2026, 3, 6))

    assert completion == PointInTime("2026-03-06T23:00:00Z")
    assert completion == PointInTime("2026-03-07T04:30:00+05:30")


# ---------------------------------------------------------------------------
# A non-session returns None
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("trading_date", _NON_SESSIONS)
def test_a_non_session_returns_none(
    resolver: ReferenceFuturesDailyBarCompletionResolver, trading_date: date
) -> None:
    assert resolver.resolve_completion(_ES, trading_date) is None


def test_none_and_an_error_are_different_facts(
    resolver: ReferenceFuturesDailyBarCompletionResolver,
) -> None:
    """ "Not a session" must never be confused with "cannot answer"."""
    assert resolver.resolve_completion(_ES, date(2026, 12, 25)) is None

    with pytest.raises(FuturesSessionResolutionError):
        resolver.resolve_completion(_EQUITY_VENUE_PRODUCT, date(2026, 12, 25))


# ---------------------------------------------------------------------------
# Error contract
# ---------------------------------------------------------------------------


def test_an_unsupported_venue_raises(
    resolver: ReferenceFuturesDailyBarCompletionResolver,
) -> None:
    with pytest.raises(FuturesSessionResolutionError, match="Unsupported futures venue"):
        resolver.resolve_completion(_EQUITY_VENUE_PRODUCT, date(2026, 9, 15))


def test_the_error_is_a_runtime_error() -> None:
    assert issubclass(FuturesSessionResolutionError, RuntimeError)


def test_the_futures_error_is_not_the_equity_error() -> None:
    """Separate hierarchies, so catching one never swallows the other."""
    assert not issubclass(FuturesSessionResolutionError, TradingSessionResolutionError)
    assert not issubclass(TradingSessionResolutionError, FuturesSessionResolutionError)

    with pytest.raises(FuturesSessionResolutionError):
        raise FuturesSessionResolutionError("futures")


# ---------------------------------------------------------------------------
# Signature
# ---------------------------------------------------------------------------


def test_the_port_takes_a_product_reference_and_a_date() -> None:
    signature = inspect.signature(FuturesDailyBarCompletionResolver.resolve_completion)

    assert list(signature.parameters) == ["self", "product", "trading_date"]
    assert signature.parameters["product"].annotation == "FuturesProductReference"
    assert signature.parameters["trading_date"].annotation == "date"
    assert signature.return_annotation == "PointInTime | None"


def test_the_port_does_not_take_a_futures_contract(
    resolver: ReferenceFuturesDailyBarCompletionResolver,
) -> None:
    """An expiry does not determine the daily session schedule."""
    from northstar_core.derivatives import ExpirationDate

    contract = FuturesContract(_ES, ExpirationDate("2026-12-18"))

    with pytest.raises(TypeError):
        resolver.resolve_completion(contract, date(2026, 9, 15))  # type: ignore[arg-type]


def test_the_port_does_not_take_a_bare_exchange_code(
    resolver: ReferenceFuturesDailyBarCompletionResolver,
) -> None:
    with pytest.raises(TypeError):
        resolver.resolve_completion(_CME, date(2026, 9, 15))  # type: ignore[arg-type]


def test_the_trading_date_is_a_plain_date(
    resolver: ReferenceFuturesDailyBarCompletionResolver,
) -> None:
    """A civil date carries no clock and no timezone, which is what a label needs."""
    with pytest.raises(TypeError):
        resolver.resolve_completion(_ES, "2026-09-15")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        resolver.resolve_completion(_ES, PointInTime("2026-09-15T22:00:00Z"))  # type: ignore[arg-type]


def test_products_sharing_a_venue_resolve_alike(
    resolver: ReferenceFuturesDailyBarCompletionResolver,
) -> None:
    """Schedules resolve at venue level today; the product is the stable input."""
    assert resolver.resolve_completion(_ES, date(2026, 9, 15)) == resolver.resolve_completion(
        _MES, date(2026, 9, 15)
    )


def test_the_product_reaches_the_implementation_intact(
    resolver: ReferenceFuturesDailyBarCompletionResolver,
) -> None:
    """A per-product schedule must be possible without changing this signature."""
    resolver.resolve_completion(_CL, date(2026, 9, 15))

    assert resolver.calls == [(_CL, date(2026, 9, 15))]
    assert resolver.calls[0][0].product_code == Symbol("CL")


# ---------------------------------------------------------------------------
# Port shape and independence from the equity port
# ---------------------------------------------------------------------------


def test_the_port_is_abstract() -> None:
    with pytest.raises(TypeError):
        FuturesDailyBarCompletionResolver()  # type: ignore[abstract]


def test_the_port_exposes_only_resolve_completion() -> None:
    assert FuturesDailyBarCompletionResolver.__abstractmethods__ == frozenset(
        {"resolve_completion"}
    )


def test_the_futures_port_is_not_an_equity_resolver() -> None:
    assert not issubclass(FuturesDailyBarCompletionResolver, TradingSessionResolver)
    assert not issubclass(TradingSessionResolver, FuturesDailyBarCompletionResolver)
    assert not hasattr(FuturesDailyBarCompletionResolver, "resolve_session_close")


def test_the_port_module_does_not_import_the_equity_port() -> None:
    """Checked through the AST so prose in a docstring cannot trip it."""
    import ast
    from pathlib import Path

    import northstar_application.ports.futures_daily_bar_completion_resolver as module

    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }

    assert "TradingSessionResolver" not in imported
    assert "TradingSessionResolutionError" not in imported


def test_the_port_module_imports_no_calendar_or_provider_library() -> None:
    import ast
    from pathlib import Path

    import northstar_application.ports.futures_daily_bar_completion_resolver as module

    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    } | {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }

    for forbidden in ("exchange_calendars", "databento", "pandas", "zoneinfo", "pytz"):
        assert not any(name.split(".")[0] == forbidden for name in modules), (
            f"port imports {forbidden}"
        )


# ---------------------------------------------------------------------------
# The equity port is untouched
# ---------------------------------------------------------------------------


def test_the_equity_port_keeps_its_own_contract() -> None:
    signature = inspect.signature(TradingSessionResolver.resolve_session_close)

    assert list(signature.parameters) == ["self", "exchange_code", "trading_date"]
    assert TradingSessionResolver.__abstractmethods__ == frozenset({"resolve_session_close"})
    assert issubclass(TradingSessionResolutionError, RuntimeError)


def test_the_equity_port_gained_no_futures_concept() -> None:
    for absent in ("resolve_completion", "product", "futures"):
        assert not hasattr(TradingSessionResolver, absent)


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


def test_the_new_names_are_exported() -> None:
    import northstar_application.ports as ports

    assert "FuturesDailyBarCompletionResolver" in ports.__all__
    assert "FuturesSessionResolutionError" in ports.__all__
    assert ports.FuturesDailyBarCompletionResolver is FuturesDailyBarCompletionResolver
    assert ports.FuturesSessionResolutionError is FuturesSessionResolutionError


def test_no_deferred_acquisition_concept_is_exported_yet() -> None:
    import northstar_application.ports as ports

    for deferred in (
        "FuturesHistoricalMarketDataSource",
        "FuturesContractResolver",
        "FuturesSettlementScheduleResolver",
        "ProviderSymbolResolver",
    ):
        assert deferred not in ports.__all__
