"""Contract tests for the option expiration resolver port and its resolved value."""

from __future__ import annotations

import ast
import inspect
from dataclasses import FrozenInstanceError
from datetime import date, datetime
from pathlib import Path
from typing import get_type_hints

import pytest
from northstar_core.derivatives import ExpirationDate
from northstar_core.foundation.value_objects import ExchangeCode, Symbol
from northstar_core.futures import FuturesProductReference
from northstar_core.options import OptionProductReference

import northstar_application.ports as ports
import northstar_application.ports.option_expiration_resolver as module
from northstar_application.ports import (
    FuturesSessionResolutionError,
    InvalidResolvedOptionExpirationError,
    OptionExpirationResolutionError,
    OptionExpirationResolver,
    ResolvedOptionExpiration,
)

_NIFTY = OptionProductReference(Symbol("NIFTY"), ExchangeCode("NSE"))
_SOURCE = "NSE/FAOP/68747 (2025-06-25)"


def _resolved(
    product: object = _NIFTY,
    nominal: object = date(2026, 11, 24),
    expiration: object = ExpirationDate("2026-11-23"),
    source: object = _SOURCE,
) -> ResolvedOptionExpiration:
    return ResolvedOptionExpiration(product, nominal, expiration, source)


# ---------------------------------------------------------------------------
# ResolvedOptionExpiration
# ---------------------------------------------------------------------------


def test_the_members_are_preserved() -> None:
    resolved = _resolved()

    assert resolved.product == _NIFTY
    assert resolved.nominal_date == date(2026, 11, 24)
    assert resolved.expiration_date == ExpirationDate("2026-11-23")
    assert resolved.rule_source == _SOURCE


def test_the_field_shape_is_exact() -> None:
    assert ResolvedOptionExpiration.__slots__ == (
        "product",
        "nominal_date",
        "expiration_date",
        "rule_source",
    )


def test_an_adjusted_expiration_reports_it() -> None:
    assert _resolved().is_adjusted is True


def test_an_unadjusted_expiration_reports_it() -> None:
    resolved = _resolved(nominal=date(2026, 10, 27), expiration=ExpirationDate("2026-10-27"))

    assert resolved.is_adjusted is False


def test_is_adjusted_is_derived_not_stored() -> None:
    assert "is_adjusted" not in ResolvedOptionExpiration.__slots__
    with pytest.raises(AttributeError):
        _resolved().is_adjusted = False  # type: ignore[misc]


def test_the_expiration_can_never_follow_its_nominal_date() -> None:
    """The adjustment only ever walks backward."""
    with pytest.raises(InvalidResolvedOptionExpirationError, match="cannot be after"):
        _resolved(nominal=date(2026, 11, 23), expiration=ExpirationDate("2026-11-24"))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("product", None, "product cannot be None"),
        ("product", FuturesProductReference(Symbol("NIFTY"), ExchangeCode("NSE")), "product must"),
        ("product", "NIFTY@NSE", "product must"),
        ("nominal", None, "nominal date cannot be None"),
        ("nominal", datetime(2026, 11, 24), "nominal date must be a plain date"),
        ("nominal", "2026-11-24", "nominal date must be a plain date"),
        ("expiration", None, "expiration date cannot be None"),
        ("expiration", date(2026, 11, 23), "must be an ExpirationDate"),
        ("expiration", "2026-11-23", "must be an ExpirationDate"),
        ("source", None, "rule source cannot be None"),
        ("source", "", "rule source cannot be empty"),
        ("source", "   ", "rule source cannot be empty"),
        ("source", 68747, "rule source must be a string"),
    ],
)
def test_invalid_members_are_rejected(field: str, value: object, message: str) -> None:
    with pytest.raises(InvalidResolvedOptionExpirationError, match=message):
        _resolved(**{field: value})


def test_the_value_error_is_distinct_from_the_resolution_error() -> None:
    assert issubclass(InvalidResolvedOptionExpirationError, ValueError)
    assert issubclass(OptionExpirationResolutionError, RuntimeError)
    assert not issubclass(OptionExpirationResolutionError, FuturesSessionResolutionError)
    assert not issubclass(FuturesSessionResolutionError, OptionExpirationResolutionError)


def test_equal_values_compare_and_hash_equal() -> None:
    assert _resolved() == _resolved()
    assert hash(_resolved()) == hash(_resolved())
    assert _resolved() != _resolved(source="another source")


def test_the_value_is_immutable() -> None:
    resolved = _resolved()

    with pytest.raises(FrozenInstanceError):
        resolved.nominal_date = date(2026, 11, 23)  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        resolved.expiration_date = ExpirationDate("2026-11-24")  # type: ignore[misc]


def test_string_and_repr_forms() -> None:
    resolved = _resolved()

    assert str(resolved) == "NIFTY@NSE 2026-11-23 (nominal 2026-11-24)"
    assert repr(resolved) == (
        "ResolvedOptionExpiration("
        "product=OptionProductReference("
        "product_code=Symbol(value='NIFTY'), exchange_code=ExchangeCode(value='NSE')), "
        "nominal_date=datetime.date(2026, 11, 24), "
        "expiration_date=ExpirationDate(value='2026-11-23'), "
        "rule_source='NSE/FAOP/68747 (2025-06-25)')"
    )


def test_the_value_carries_no_listing_or_series_concept() -> None:
    resolved = _resolved()

    for absent in (
        "series",
        "expiry_series",
        "is_weekly",
        "is_monthly",
        "strike",
        "strikes",
        "right",
        "lot_size",
        "instrument_key",
        "contract",
        "listed",
    ):
        assert not hasattr(resolved, absent)


# ---------------------------------------------------------------------------
# Port shape
# ---------------------------------------------------------------------------


def test_the_port_is_abstract_with_exactly_two_operations() -> None:
    assert inspect.isabstract(OptionExpirationResolver)
    assert OptionExpirationResolver.__abstractmethods__ == frozenset(
        {"weekly_expiration", "monthly_expiration"}
    )
    with pytest.raises(TypeError):
        OptionExpirationResolver()  # type: ignore[abstract]


def test_the_operations_are_option_typed() -> None:
    weekly = get_type_hints(OptionExpirationResolver.weekly_expiration)
    monthly = get_type_hints(OptionExpirationResolver.monthly_expiration)

    assert weekly == {
        "product": OptionProductReference,
        "iso_year": int,
        "iso_week": int,
        "return": ResolvedOptionExpiration,
    }
    assert monthly == {
        "product": OptionProductReference,
        "year": int,
        "month": int,
        "return": ResolvedOptionExpiration,
    }


def test_the_operations_take_no_default_period() -> None:
    for operation in (
        OptionExpirationResolver.weekly_expiration,
        OptionExpirationResolver.monthly_expiration,
    ):
        for parameter in inspect.signature(operation).parameters.values():
            assert parameter.default is inspect.Parameter.empty


def test_the_port_surface_is_exported() -> None:
    for name in (
        "InvalidResolvedOptionExpirationError",
        "OptionExpirationResolutionError",
        "OptionExpirationResolver",
        "ResolvedOptionExpiration",
    ):
        assert name in ports.__all__
        assert getattr(ports, name) is getattr(module, name)


def test_no_expiry_series_type_is_introduced() -> None:
    for name in ("OptionExpirySeries", "OptionExpiryKind", "OptionExpirationSeries"):
        assert not hasattr(ports, name)
        assert not hasattr(module, name)


def test_the_module_depends_on_no_futures_clock_provider_or_network() -> None:
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    imported = {
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module
    } | {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }

    for name in imported:
        assert not name.startswith(("northstar_core.futures", "northstar_infrastructure"))
        assert name.split(".")[0] not in {
            "time",
            "zoneinfo",
            "requests",
            "httpx",
            "urllib",
            "socket",
            "sqlite3",
            "exchange_calendars",
        }
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert node.func.attr not in {"now", "utcnow", "today", "time", "monotonic"}
