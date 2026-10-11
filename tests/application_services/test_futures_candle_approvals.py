"""Tests for the reviewed-candle approval manifest and its fail-closed lookups.

Every contract, session, price and fingerprint is synthetic. Parser tests use
artificial digests (one repeated hex digit); verification tests derive their
fingerprints from the already-tested fdc1 implementation over synthetic bars.
"""

from __future__ import annotations

import ast
import dataclasses
import re
from datetime import date, datetime
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

import northstar_application.application_services.futures_candle_approvals as module
from northstar_application.application_services import (
    FUTURES_DAILY_CANDLE_FINGERPRINT_PREFIX,
    FuturesCandleApproval,
    FuturesCandleApprovalError,
    FuturesCandleApprovalMismatchError,
    FuturesCandleApprovalMissingError,
    FuturesCandleApprovalSet,
    InvalidFuturesCandleApprovalSetError,
    InvalidFuturesDailyCandleIdentityError,
    futures_daily_candle_fingerprint,
    parse_futures_candle_approval_manifest,
)

_NIFTY = FuturesProductReference(Symbol("NIFTY"), ExchangeCode("NSE"))
_OCT = FuturesContract(_NIFTY, ExpirationDate("2026-10-27"))
_NOV = FuturesContract(_NIFTY, ExpirationDate("2026-11-24"))
_D1, _D2, _D3 = date(2026, 10, 1), date(2026, 10, 5), date(2026, 10, 6)


def _fp(digit: str) -> str:
    """An artificial fingerprint: the prefix and one hex digit repeated."""
    return FUTURES_DAILY_CANDLE_FINGERPRINT_PREFIX + digit * 64


_A, _B, _C = _fp("a"), _fp("b"), _fp("c")


def _entry(expiration: str, session: str, fingerprint: str) -> str:
    return f"{expiration}/{session}={fingerprint}"


def _parse(text: str, product: FuturesProductReference = _NIFTY) -> FuturesCandleApprovalSet:
    return parse_futures_candle_approval_manifest(text, product)


def _refused(text: str, message: str) -> None:
    with pytest.raises(InvalidFuturesCandleApprovalSetError) as raised:
        _parse(text)
    assert re.search(message, str(raised.value)), str(raised.value)
    # A refusal never repeats a digest or any hex run from the manifest.
    assert not re.search(r"[0-9a-fA-F]{16}", str(raised.value))


def _bar(contract: FuturesContract = _OCT, close: str = "100") -> FuturesOHLCVBar:
    return FuturesOHLCVBar(
        contract=contract,
        point_in_time=PointInTime("2026-10-01T10:10:00Z"),
        timeframe=Timeframe("1d"),
        open=QuoteValue(Decimal("101.25")),
        high=QuoteValue(Decimal("102.5")),
        low=QuoteValue(Decimal("99.75")),
        close=QuoteValue(Decimal(close)),
        volume=Quantity(Decimal("42")),
    )


_TWO = ",".join([_entry("2026-10-27", "2026-10-01", _A), _entry("2026-10-27", "2026-10-05", _B)])


# ---------------------------------------------------------------------------
# Valid manifests and exact lookups
# ---------------------------------------------------------------------------


def test_a_single_valid_approval() -> None:
    approvals = _parse(_entry("2026-10-27", "2026-10-01", _A))

    assert approvals.product == _NIFTY
    assert approvals.approvals == (FuturesCandleApproval(_OCT, _D1, _A),)
    assert approvals.approved_fingerprint(_OCT, _D1) == _A


def test_multiple_valid_approvals_across_contracts() -> None:
    text = ",".join(
        [
            _entry("2026-10-27", "2026-10-01", _A),
            _entry("2026-10-27", "2026-10-05", _B),
            _entry("2026-11-24", "2026-10-01", _C),
        ]
    )
    approvals = _parse(text)

    assert approvals.approvals == (
        FuturesCandleApproval(_OCT, _D1, _A),
        FuturesCandleApproval(_OCT, _D2, _B),
        FuturesCandleApproval(_NOV, _D1, _C),
    )
    assert approvals.approved_fingerprint(_OCT, _D1) == _A
    assert approvals.approved_fingerprint(_OCT, _D2) == _B
    assert approvals.approved_fingerprint(_NOV, _D1) == _C


def test_reordered_entries_give_the_same_set() -> None:
    reordered = ",".join(reversed(_TWO.split(",")))

    assert _parse(reordered) == _parse(_TWO)
    assert _parse(reordered).approvals == _parse(_TWO).approvals


def test_parsing_is_deterministic() -> None:
    results = {_parse(_TWO) for _ in range(20)}

    assert len(results) == 1
    assert repr(_parse(_TWO)) == repr(_parse(_TWO))


def test_an_empty_manifest_is_valid_and_approves_nothing() -> None:
    approvals = _parse("")

    assert approvals.approvals == ()
    with pytest.raises(FuturesCandleApprovalMissingError) as raised:
        approvals.approved_fingerprint(_OCT, _D1)
    assert raised.value.trading_dates == (_D1,)
    # Nothing makes an empty set falsy, so "if approvals:" can never skip a check.
    assert bool(approvals) is True
    assert not hasattr(approvals, "__len__")


# ---------------------------------------------------------------------------
# Missing approvals and no fallback
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("contract", "session"),
    [
        (FuturesContract(_NIFTY, ExpirationDate("2026-11-24")), _D1),  # wrong expiry
        (
            FuturesContract(
                FuturesProductReference(Symbol("BANKNIFTY"), ExchangeCode("NSE")),
                ExpirationDate("2026-10-27"),
            ),
            _D1,
        ),  # wrong product
        (
            FuturesContract(
                FuturesProductReference(Symbol("NIFTY"), ExchangeCode("BSE")),
                ExpirationDate("2026-10-27"),
            ),
            _D1,
        ),  # wrong exchange
        (_OCT, date(2026, 10, 2)),  # wrong trading date, between two approved ones
        (_OCT, date(2026, 9, 30)),  # the day before an approved session
    ],
    ids=["wrong-expiry", "wrong-product", "wrong-exchange", "wrong-date", "previous-day"],
)
def test_no_lookup_falls_back_to_another_approval(contract: FuturesContract, session: date) -> None:
    approvals = _parse(_TWO)

    with pytest.raises(FuturesCandleApprovalMissingError) as raised:
        approvals.approved_fingerprint(contract, session)
    assert raised.value.contract == contract
    assert raised.value.trading_dates == (session,)
    assert str(raised.value) == (
        f"No approved candle version for {contract} sessions: {session.isoformat()}."
    )


def test_one_missing_session_in_a_larger_set_is_named() -> None:
    approvals = _parse(_TWO)

    approvals.require_approvals(_OCT, [_D2, _D1])  # all approved: no error
    with pytest.raises(FuturesCandleApprovalMissingError) as raised:
        approvals.require_approvals(_OCT, [_D3, _D1, date(2026, 10, 7), _D2])
    assert raised.value.trading_dates == (_D3, date(2026, 10, 7))
    assert "2026-10-06, 2026-10-07" in str(raised.value)


def test_a_removed_entry_is_a_missing_approval() -> None:
    before = _parse(_TWO)
    revoked = _parse(_entry("2026-10-27", "2026-10-01", _A))

    assert before.approved_fingerprint(_OCT, _D2) == _B
    with pytest.raises(FuturesCandleApprovalMissingError) as raised:
        revoked.approved_fingerprint(_OCT, _D2)
    assert raised.value.trading_dates == (_D2,)
    assert revoked.approved_fingerprint(_OCT, _D1) == _A


def test_missing_is_never_an_approval_and_never_a_configuration_error() -> None:
    assert issubclass(FuturesCandleApprovalMissingError, FuturesCandleApprovalError)
    assert issubclass(FuturesCandleApprovalMismatchError, FuturesCandleApprovalError)
    assert not issubclass(FuturesCandleApprovalMissingError, InvalidFuturesCandleApprovalSetError)
    assert not issubclass(InvalidFuturesCandleApprovalSetError, FuturesCandleApprovalError)


# ---------------------------------------------------------------------------
# Fingerprint verification, using real fdc1 fingerprints of synthetic bars
# ---------------------------------------------------------------------------


def test_a_matching_bar_is_accepted() -> None:
    fingerprint = futures_daily_candle_fingerprint(_OCT, _D1, _bar())
    approvals = _parse(_entry("2026-10-27", "2026-10-01", fingerprint))

    assert approvals.verify_bar(_OCT, _D1, _bar()) is None
    assert approvals.verify_fingerprints(_OCT, {_D1: fingerprint}) is None


def test_a_different_bar_is_a_mismatch() -> None:
    approved = futures_daily_candle_fingerprint(_OCT, _D1, _bar(close="100"))
    approvals = _parse(_entry("2026-10-27", "2026-10-01", approved))

    with pytest.raises(FuturesCandleApprovalMismatchError) as raised:
        approvals.verify_bar(_OCT, _D1, _bar(close="100.05"))
    assert raised.value.contract == _OCT
    assert raised.value.trading_dates == (_D1,)
    assert str(raised.value) == (
        f"Candles for {_OCT} differ from their approved versions for sessions: 2026-10-01."
    )
    assert approved not in str(raised.value)


def test_fingerprints_of_different_sessions_are_independent() -> None:
    first = futures_daily_candle_fingerprint(_OCT, _D1, _bar())
    second = futures_daily_candle_fingerprint(_OCT, _D2, _bar())
    approvals = _parse(
        ",".join(
            [_entry("2026-10-27", "2026-10-01", first), _entry("2026-10-27", "2026-10-05", second)]
        )
    )

    assert first != second
    approvals.verify_fingerprints(_OCT, {_D1: first, _D2: second})
    with pytest.raises(FuturesCandleApprovalMismatchError) as raised:
        approvals.verify_fingerprints(_OCT, {_D1: second, _D2: first})  # swapped
    assert raised.value.trading_dates == (_D1, _D2)


def test_a_missing_approval_is_reported_before_any_comparison() -> None:
    approvals = _parse(_entry("2026-10-27", "2026-10-01", _A))

    with pytest.raises(FuturesCandleApprovalMissingError) as raised:
        approvals.verify_fingerprints(_OCT, {_D1: _B, _D2: _B})
    assert raised.value.trading_dates == (_D2,)


def test_a_bar_for_another_contract_is_never_approved_by_this_contracts_entry() -> None:
    fingerprint = futures_daily_candle_fingerprint(_OCT, _D1, _bar())
    approvals = _parse(_entry("2026-10-27", "2026-10-01", fingerprint))

    with pytest.raises(FuturesCandleApprovalMissingError):
        approvals.verify_bar(_NOV, _D1, _bar(contract=_NOV))
    with pytest.raises(InvalidFuturesDailyCandleIdentityError, match="not the requested"):
        approvals.verify_bar(_OCT, _D1, _bar(contract=_NOV))


@pytest.mark.parametrize("calculated", ["", _A.upper(), _A[:-1], "sha256:" + "a" * 64, None])
def test_a_malformed_calculated_fingerprint_is_refused(calculated: object) -> None:
    approvals = _parse(_entry("2026-10-27", "2026-10-01", _A))

    with pytest.raises(InvalidFuturesDailyCandleIdentityError, match="not an fdc1 fingerprint"):
        approvals.verify_fingerprints(_OCT, {_D1: calculated})  # type: ignore[dict-item]


def test_fingerprints_from_the_fdc1_implementation_round_trip_through_the_manifest() -> None:
    fingerprint = futures_daily_candle_fingerprint(_OCT, _D1, _bar())

    assert fingerprint.startswith(FUTURES_DAILY_CANDLE_FINGERPRINT_PREFIX)
    assert (
        _parse(_entry("2026-10-27", "2026-10-01", fingerprint)).approved_fingerprint(_OCT, _D1)
        == fingerprint
    )


# ---------------------------------------------------------------------------
# Strict parsing: every defect refuses the whole manifest
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("second", [_A, _B], ids=["identical", "conflicting"])
def test_a_repeated_key_is_refused_whatever_its_fingerprint(second: str) -> None:
    text = ",".join(
        [_entry("2026-10-27", "2026-10-01", _A), _entry("2026-10-27", "2026-10-01", second)]
    )

    _refused(text, r"entries 1 and 2 both approve expiration 2026-10-27 session 2026-10-01\.")


@pytest.mark.parametrize(
    ("fingerprint", "label"),
    [
        ("fdc2-sha256:" + "a" * 64, "fdc2-sha256"),
        ("fdc1-sha512:" + "a" * 128, "fdc1-sha512"),
        ("fdc1-md5:" + "a" * 32, "fdc1-md5"),
        ("fdc0-sha256:" + "a" * 64, "fdc0-sha256"),
    ],
)
def test_an_unsupported_version_or_algorithm_is_refused(fingerprint: str, label: str) -> None:
    _refused(
        _entry("2026-10-27", "2026-10-01", fingerprint),
        rf"entry 1 uses unsupported fingerprint format {label}; only fdc1-sha256: is accepted\.",
    )


@pytest.mark.parametrize(
    "fingerprint",
    [
        "fdc1-sha256:" + "a" * 63,  # truncated
        "fdc1-sha256:" + "a" * 65,  # too long
        "fdc1-sha256:",  # empty digest
        "fdc1-sha256:" + "g" * 64,  # not hex
        "fdc1-sha256:" + "A" * 64,  # uppercase
        "fdc1-sha256:" + "a" * 63 + "F",  # one uppercase digit
        "FDC1-SHA256:" + "a" * 64,  # uppercase prefix
        "fdc1-sha256" + "a" * 64,  # missing colon
        "a" * 64,  # bare digest
        "sha256:" + "a" * 64,  # no format version
    ],
    ids=[
        "truncated",
        "too-long",
        "empty-digest",
        "non-hex",
        "uppercase",
        "one-uppercase",
        "uppercase-prefix",
        "no-colon",
        "bare",
        "no-version",
    ],
)
def test_a_malformed_fingerprint_is_refused(fingerprint: str) -> None:
    _refused(_entry("2026-10-27", "2026-10-01", fingerprint), r"entry 1 is malformed; expected")


@pytest.mark.parametrize(
    "element",
    [
        f"2026-10-27={_A}",  # no trading date
        "2026-10-27/2026-10-01=",  # no fingerprint
        f"2026-10-27/2026-10-01{_A}",  # no "="
        f"/2026-10-01={_A}",  # empty expiration
        f"2026-10-27/={_A}",  # empty trading date
    ],
    ids=["no-session", "no-fingerprint", "no-equals", "no-expiration", "empty-session"],
)
def test_a_missing_field_is_refused(element: str) -> None:
    _refused(element, r"entry 1 is malformed")


@pytest.mark.parametrize(
    "element",
    [
        f"2026-10-27/2026-10-01/NIFTY={_A}",
        f"NIFTY/2026-10-27/2026-10-01={_A}",
        f"2026-10-27/2026-10-01={_A};note=x",
        f"2026-10-27/2026-10-01={_A}=extra",
        "product=NIFTY",
    ],
    ids=["extra-key-part", "leading-field", "trailing-field", "extra-value", "named-field"],
)
def test_an_unknown_field_is_refused(element: str) -> None:
    _refused(element, r"entry 1 is malformed")


@pytest.mark.parametrize(
    ("text", "message"),
    [
        (
            f"{_entry('2026-10-27', '2026-10-01', _A)};{_entry('2026-10-27', '2026-10-05', _B)}",
            r"entry 1 is malformed",
        ),
        (
            f"{_entry('2026-10-27', '2026-10-01', _A)}|{_entry('2026-10-27', '2026-10-05', _B)}",
            r"entry 1 is malformed",
        ),
        (f"2026-10-27:2026-10-01={_A}", r"entry 1 is malformed"),
        (f"2026-10-27/2026-10-01:{_A}", r"entry 1 is malformed"),
        (f"{_entry('2026-10-27', '2026-10-01', _A)},", r"entry 2 is empty"),
        (f",{_entry('2026-10-27', '2026-10-01', _A)}", r"entry 1 is empty"),
        (
            f"{_entry('2026-10-27', '2026-10-01', _A)},,{_entry('2026-10-27', '2026-10-05', _B)}",
            r"entry 2 is empty",
        ),
        (",", r"entry 1 is empty"),
    ],
    ids=[
        "semicolon",
        "pipe",
        "colon-key",
        "colon-value",
        "trailing-comma",
        "leading-comma",
        "double-comma",
        "lone-comma",
    ],
)
def test_an_unexpected_delimiter_is_refused(text: str, message: str) -> None:
    _refused(text, message)


@pytest.mark.parametrize(
    "text",
    [
        f"{_entry('2026-10-27', '2026-10-01', _A)}, {_entry('2026-10-27', '2026-10-05', _B)}",
        f" {_entry('2026-10-27', '2026-10-01', _A)}",
        f"{_entry('2026-10-27', '2026-10-01', _A)} ",
        f"{_entry('2026-10-27', '2026-10-01', _A)}\n",
        f"2026-10-27/2026-10-01 ={_A}",
        f"2026-10-27/2026-10-01=\t{_A}",
        f"{_entry('2026-10-27', '2026-10-01', _A)} ",
        " ",
    ],
    ids=[
        "after-comma",
        "leading",
        "trailing",
        "newline",
        "before-equals",
        "tab",
        "nbsp",
        "only-space",
    ],
)
def test_whitespace_anywhere_is_refused(text: str) -> None:
    _refused(text, r"must contain no whitespace\.")


@pytest.mark.parametrize(
    ("element", "message"),
    [
        (f"2026-13-01/2026-10-01={_A}", r"entry 1 has an invalid expiration date\."),
        (f"2026-02-30/2026-02-27={_A}", r"entry 1 has an invalid expiration date\."),
        (f"2026-10-27/2026-02-30={_A}", r"entry 1 has an invalid trading date\."),
        (f"2026-10-27/2026-00-10={_A}", r"entry 1 has an invalid trading date\."),
        (f"26-10-27/2026-10-01={_A}", r"entry 1 is malformed"),
        (f"2026/10/27/2026-10-01={_A}", r"entry 1 is malformed"),
        (f"20261027/2026-10-01={_A}", r"entry 1 is malformed"),
        (f"2026-10-27/2026-10-1={_A}", r"entry 1 is malformed"),
        (f"2026-10-27/01-10-2026={_A}", r"entry 1 is malformed"),
        (f"2026-10-27/2026-W40-4={_A}", r"entry 1 is malformed"),
        (f"2026-10-27/2026-10-28={_A}", r"entry 1: .*after the expiration"),
    ],
)
def test_a_malformed_or_impossible_date_is_refused(element: str, message: str) -> None:
    _refused(element, message)


def test_non_ascii_digits_and_characters_are_refused() -> None:
    _refused(f"２０２６-10-27/2026-10-01={_A}", r"must be ASCII text\.")
    _refused(f"2026-10-27/2026-10-01={_A}é", r"must be ASCII text\.")


def test_one_bad_entry_refuses_the_whole_manifest() -> None:
    text = ",".join(
        [
            _entry("2026-10-27", "2026-10-01", _A),
            _entry("2026-10-27", "2026-10-05", _B),
            _entry("2026-10-27", "2026-10-06", "fdc1-sha256:" + "c" * 63),
        ]
    )

    _refused(text, r"entry 3 is malformed")


def test_the_parser_refuses_wrong_input_types() -> None:
    with pytest.raises(TypeError, match="text must be a string"):
        parse_futures_candle_approval_manifest(None, _NIFTY)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="text must be a string"):
        parse_futures_candle_approval_manifest(_TWO.encode(), _NIFTY)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="product must be a FuturesProductReference"):
        parse_futures_candle_approval_manifest(_TWO, _OCT)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# The parsed representation
# ---------------------------------------------------------------------------


def test_the_parsed_representation_is_immutable() -> None:
    approvals = _parse(_TWO)
    approval = approvals.approvals[0]

    with pytest.raises(dataclasses.FrozenInstanceError):
        approvals.approvals = ()  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        approval.fingerprint = _C  # type: ignore[misc]
    with pytest.raises(TypeError):
        approvals._index[(_OCT, _D1)] = _C  # type: ignore[index]
    assert isinstance(approvals.approvals, tuple)
    assert approvals.approved_fingerprint(_OCT, _D1) == _A


def test_direct_construction_is_as_strict_as_parsing() -> None:
    with pytest.raises(InvalidFuturesCandleApprovalSetError, match="more than once"):
        FuturesCandleApprovalSet(
            _NIFTY, (FuturesCandleApproval(_OCT, _D1, _A), FuturesCandleApproval(_OCT, _D1, _A))
        )
    other = FuturesContract(
        FuturesProductReference(Symbol("BANKNIFTY"), ExchangeCode("NSE")), _OCT.expiration_date
    )
    with pytest.raises(InvalidFuturesCandleApprovalSetError, match="is for BANKNIFTY@NSE"):
        FuturesCandleApprovalSet(_NIFTY, (FuturesCandleApproval(other, _D1, _A),))
    with pytest.raises(InvalidFuturesCandleApprovalSetError, match="must be a tuple"):
        FuturesCandleApprovalSet(_NIFTY, [FuturesCandleApproval(_OCT, _D1, _A)])  # type: ignore[arg-type]
    with pytest.raises(InvalidFuturesCandleApprovalSetError, match="64 lowercase hex"):
        FuturesCandleApproval(_OCT, _D1, _A.upper())
    with pytest.raises(InvalidFuturesCandleApprovalSetError, match="plain date"):
        FuturesCandleApproval(_OCT, datetime(2026, 10, 1), _A)  # type: ignore[arg-type]


def test_nothing_derived_from_a_fingerprint_is_printed() -> None:
    approvals = _parse(_TWO)

    assert "a" * 16 not in repr(approvals)
    assert repr(approvals.approvals[0]) == (
        "FuturesCandleApproval(contract=NIFTY@NSE 2026-10-27, trading_date=2026-10-01)"
    )


def test_lookups_refuse_wrong_argument_types() -> None:
    approvals = _parse(_TWO)

    with pytest.raises(TypeError, match="plain date"):
        approvals.approved_fingerprint(_OCT, datetime(2026, 10, 1))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="FuturesContract"):
        approvals.approved_fingerprint(_NIFTY, _D1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="At least one trading date"):
        approvals.require_approvals(_OCT, [])
    with pytest.raises(ValueError, match="only once"):
        approvals.require_approvals(_OCT, [_D1, _D1])


# ---------------------------------------------------------------------------
# Boundaries and public surface
# ---------------------------------------------------------------------------


def test_the_module_is_pure_and_reads_no_configuration() -> None:
    source = Path(module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {
        (node.module or "").split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    } | {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }

    assert imported <= {
        "__future__",
        "collections",
        "dataclasses",
        "datetime",
        "re",
        "types",
        "northstar_core",
        "northstar_application",
    }
    assert not re.search(
        r"\b(upstox|databento|sqlite|environ|getenv|dotenv|open\(|logging|print)\b", source, re.I
    )
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            assert node.attr not in {"now", "utcnow", "today", "read_text", "environ"}
        assert not isinstance(node, ast.Global | ast.Nonlocal)


def test_the_public_surface_is_exported() -> None:
    import northstar_application.application_services as services

    for name in (
        "FuturesCandleApproval",
        "FuturesCandleApprovalError",
        "FuturesCandleApprovalMismatchError",
        "FuturesCandleApprovalMissingError",
        "FuturesCandleApprovalSet",
        "InvalidFuturesCandleApprovalSetError",
        "parse_futures_candle_approval_manifest",
    ):
        assert name in services.__all__
    assert not hasattr(services, "_malformed")
