"""Tests for the option chain read ports and their queries."""

from __future__ import annotations

import inspect
from dataclasses import FrozenInstanceError, fields
from decimal import Decimal

import pytest
from northstar_core.derivatives import ExpirationDate
from northstar_core.foundation.value_objects import ExchangeCode, PointInTime, Symbol
from northstar_core.options import OptionProductReference

import northstar_application.ports as ports
from northstar_application.ports import (
    InvalidOptionChainDailyBarQueryError,
    InvalidOptionListedContractQueryError,
    OptionChainDailyBarQuery,
    OptionChainDailyBarRepository,
    OptionListedContractQuery,
    OptionListedContractRepository,
)

_NIFTY = OptionProductReference(Symbol("NIFTY"), ExchangeCode("NSE"))
_EXPIRY = ExpirationDate("2026-10-27")
_AS_OF = PointInTime("2026-10-08T10:10:00Z")


def test_the_listed_contract_query_has_exactly_its_three_coordinates() -> None:
    query = OptionListedContractQuery(_NIFTY, _EXPIRY, _AS_OF)

    assert [field.name for field in fields(query)] == ["product", "expiration_date", "known_by"]
    with pytest.raises(FrozenInstanceError):
        query.known_by = _AS_OF  # type: ignore[misc]


def test_the_chain_bar_query_has_exactly_its_three_coordinates() -> None:
    query = OptionChainDailyBarQuery(_NIFTY, _EXPIRY, _AS_OF)

    assert [field.name for field in fields(query)] == ["product", "expiration_date", "as_of"]
    with pytest.raises(FrozenInstanceError):
        query.as_of = _AS_OF  # type: ignore[misc]


@pytest.mark.parametrize(
    ("query_type", "error"),
    [
        (OptionListedContractQuery, InvalidOptionListedContractQueryError),
        (OptionChainDailyBarQuery, InvalidOptionChainDailyBarQueryError),
    ],
)
@pytest.mark.parametrize(
    "arguments",
    [
        ("NIFTY@NSE", _EXPIRY, _AS_OF),
        (_NIFTY, "2026-10-27", _AS_OF),
        (_NIFTY, _EXPIRY, "2026-10-08T10:10:00Z"),
        (_NIFTY, _EXPIRY, Decimal("1")),
    ],
    ids=["product", "expiration", "instant-text", "instant-number"],
)
def test_the_queries_refuse_foreign_values(query_type, error, arguments) -> None:
    with pytest.raises(error):
        query_type(*arguments)


@pytest.mark.parametrize(
    ("port", "method"),
    [
        (OptionListedContractRepository, "listed_contracts"),
        (OptionChainDailyBarRepository, "daily_bars_at"),
    ],
)
def test_each_port_has_exactly_one_abstract_read(port, method) -> None:
    assert port.__abstractmethods__ == frozenset({method})
    assert list(inspect.signature(getattr(port, method)).parameters) == ["self", "query"]
    with pytest.raises(TypeError):
        port()


def test_the_ports_expose_no_provider_listing_detail() -> None:
    for query_type in (OptionListedContractQuery, OptionChainDailyBarQuery):
        names = {field.name for field in fields(query_type)}
        assert not names & {"provider", "instrument_key", "lot_size", "established_at", "timeframe"}


def test_the_ports_and_queries_are_exported() -> None:
    for name in (
        "InvalidOptionChainDailyBarQueryError",
        "InvalidOptionListedContractQueryError",
        "OptionChainDailyBarQuery",
        "OptionChainDailyBarRepository",
        "OptionListedContractQuery",
        "OptionListedContractRepository",
    ):
        assert name in ports.__all__
