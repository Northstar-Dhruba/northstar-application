"""Contract tests for the futures trade recommendation and contract cutoff ports."""

from __future__ import annotations

import dataclasses
from datetime import date

import pytest
from northstar_core.derivatives import ExpirationDate
from northstar_core.foundation.value_objects import ExchangeCode, PointInTime, Symbol
from northstar_core.futures import FuturesContract, FuturesProductReference

import northstar_application.ports as ports
from northstar_application.ports import (
    FuturesContractCutoff,
    FuturesContractCutoffSource,
    FuturesTradeRecommendationConflictError,
    FuturesTradeRecommendationExclusivityError,
    FuturesTradeRecommendationQuery,
    FuturesTradeRecommendationRepository,
    FuturesTradeRecommendationStore,
    FuturesTradingSession,
    InvalidFuturesContractCutoffError,
    InvalidFuturesTradeRecommendationQueryError,
)

_NIFTY = FuturesProductReference(Symbol("NIFTY"), ExchangeCode("NSE"))
_OCT = FuturesContract(_NIFTY, ExpirationDate("2026-10-27"))
_LAST = FuturesTradingSession(
    date(2026, 10, 27), PointInTime("2026-10-27T03:45:00Z"), PointInTime("2026-10-27T10:00:00Z")
)


def test_the_query_is_by_product_only() -> None:
    query = FuturesTradeRecommendationQuery(_NIFTY)

    assert query.product == _NIFTY
    assert {field.name for field in dataclasses.fields(query)} == {"product"}


@pytest.mark.parametrize("value", [None, "NIFTY@NSE", _OCT])
def test_the_query_refuses_anything_but_a_product(value: object) -> None:
    with pytest.raises(InvalidFuturesTradeRecommendationQueryError):
        FuturesTradeRecommendationQuery(value)  # type: ignore[arg-type]


def test_a_cutoff_names_its_contract_session_and_authority() -> None:
    cutoff = FuturesContractCutoff(_OCT, _LAST, "  synthetic verified reference  ")

    assert (cutoff.contract, cutoff.last_session, cutoff.authority) == (
        _OCT,
        _LAST,
        "synthetic verified reference",
    )


@pytest.mark.parametrize(
    "contract, session, authority",
    [
        (None, _LAST, "a"),
        (_OCT, None, "a"),
        (_OCT, PointInTime("2026-10-27T10:00:00Z"), "a"),
        (_OCT, _LAST, ""),
        (_OCT, _LAST, "   "),
        (_OCT, _LAST, None),
    ],
)
def test_an_invalid_cutoff_is_refused(contract: object, session: object, authority: object) -> None:
    with pytest.raises(InvalidFuturesContractCutoffError):
        FuturesContractCutoff(contract, session, authority)  # type: ignore[arg-type]


def test_the_cutoff_cannot_be_built_from_an_expiration_date_alone() -> None:
    with pytest.raises(InvalidFuturesContractCutoffError):
        FuturesContractCutoff(_OCT, _OCT.expiration_date, "expiration")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "port",
    [
        FuturesContractCutoffSource,
        FuturesTradeRecommendationRepository,
        FuturesTradeRecommendationStore,
    ],
)
def test_the_ports_are_abstract(port: type) -> None:
    with pytest.raises(TypeError):
        port()


def test_the_store_errors_are_distinct_value_errors() -> None:
    assert issubclass(FuturesTradeRecommendationConflictError, ValueError)
    assert issubclass(FuturesTradeRecommendationExclusivityError, ValueError)
    assert not issubclass(
        FuturesTradeRecommendationConflictError, FuturesTradeRecommendationExclusivityError
    )


def test_no_publication_or_delivery_state_is_modelled() -> None:
    names = {name.lower() for name in ports.__all__}
    for fragment in ("publish", "deliver", "notif", "subscriber", "executed"):
        assert not any(fragment in name for name in names), fragment


def test_public_exports() -> None:
    for name in (
        "FuturesContractCutoff",
        "FuturesContractCutoffSource",
        "InvalidFuturesContractCutoffError",
        "FuturesTradeRecommendationQuery",
        "FuturesTradeRecommendationRepository",
        "InvalidFuturesTradeRecommendationQueryError",
        "FuturesTradeRecommendationConflictError",
        "FuturesTradeRecommendationExclusivityError",
        "FuturesTradeRecommendationStore",
    ):
        assert name in ports.__all__
