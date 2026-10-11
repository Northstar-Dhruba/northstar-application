"""Tests for the fdc1 canonical identity and SHA-256 fingerprint of a daily candle.

Every value is synthetic. Expected identities are written out literally from
the documented byte format, and the known-answer digests were computed outside
Python (``sha256sum`` and PowerShell ``Get-FileHash`` over the same bytes), so
no expectation is derived from the implementation under test.
"""

from __future__ import annotations

import ast
import hashlib
import re
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

import pytest
from northstar_core.derivatives import ExpirationDate, QuoteValue
from northstar_core.foundation.exceptions.validation import ValidationError
from northstar_core.foundation.value_objects import (
    ExchangeCode,
    PointInTime,
    Quantity,
    Symbol,
    Timeframe,
)
from northstar_core.futures import FuturesContract, FuturesOHLCVBar, FuturesProductReference

import northstar_application.application_services.futures_daily_candle_identity as module
from northstar_application.application_services import (
    FUTURES_DAILY_CANDLE_FINGERPRINT_PREFIX,
    FUTURES_DAILY_CANDLE_IDENTITY_FORMAT,
    InvalidFuturesDailyCandleIdentityError,
    canonical_futures_daily_candle_identity,
    futures_daily_candle_fingerprint,
)

_CONTRACT = FuturesContract(
    FuturesProductReference(Symbol("NIFTY"), ExchangeCode("NSE")), ExpirationDate("2026-10-27")
)
_SESSION = date(2026, 10, 1)
_CLOSE = PointInTime("2026-10-01T10:10:00Z")

# Known answer 1: the documented layout, typed out byte for byte.
_KAT1_TEXT = (
    "northstar.futures-daily-candle-identity/1\n"
    "product=NIFTY\n"
    "exchange=NSE\n"
    "expiration=2026-10-27\n"
    "trading_date=2026-10-01\n"
    "timeframe=1d\n"
    "open=101.25\n"
    "high=102.5\n"
    "low=99.75\n"
    "close=100\n"
    "volume_contracts=42\n"
)
_KAT1_DIGEST = "fdc1-sha256:a165a7760c27af793ac47c39af0db1c34fcef7ef0bcf6413669855eecd94e919"

# Known answer 2: negative values, zero and a 33-digit fraction.
_LONG_FRACTION = "-1.123456789012345678901234567890123"
_KAT2_TEXT = (
    "northstar.futures-daily-candle-identity/1\n"
    "product=NIFTY\n"
    "exchange=NSE\n"
    "expiration=2026-10-27\n"
    "trading_date=2026-10-01\n"
    "timeframe=1d\n"
    "open=-0.05\n"
    "high=0\n"
    f"low={_LONG_FRACTION}\n"
    "close=0\n"
    "volume_contracts=0\n"
)
_KAT2_DIGEST = "fdc1-sha256:66b751fb58f0def37811792eec15f6c5c3f07bea644b9dd9ec861c5e2f46a349"


def _bar(
    *,
    contract: FuturesContract = _CONTRACT,
    timeframe: str = "1d",
    open: str = "101.25",
    high: str = "102.5",
    low: str = "99.75",
    close: str = "100",
    volume: str = "42",
) -> FuturesOHLCVBar:
    return FuturesOHLCVBar(
        contract=contract,
        point_in_time=_CLOSE,
        timeframe=Timeframe(timeframe),
        open=QuoteValue(Decimal(open)),
        high=QuoteValue(Decimal(high)),
        low=QuoteValue(Decimal(low)),
        close=QuoteValue(Decimal(close)),
        volume=Quantity(Decimal(volume)),
    )


def _contract(product: str = "NIFTY", exchange: str = "NSE", expiry: str = "2026-10-27"):
    return FuturesContract(
        FuturesProductReference(Symbol(product), ExchangeCode(exchange)), ExpirationDate(expiry)
    )


def _fingerprint(bar: FuturesOHLCVBar, trading_date: date = _SESSION) -> str:
    return futures_daily_candle_fingerprint(bar.contract, trading_date, bar)


def _tamper(value: object, attribute: str, replacement: object) -> None:
    """Bypass a frozen value object's own validation, as a defect elsewhere might."""
    object.__setattr__(value, attribute, replacement)


# ---------------------------------------------------------------------------
# Known answers and the exact layout
# ---------------------------------------------------------------------------


def test_the_identity_is_exactly_the_documented_bytes() -> None:
    text = canonical_futures_daily_candle_identity(_CONTRACT, _SESSION, _bar())

    assert text == _KAT1_TEXT
    assert text.encode("ascii") == _KAT1_TEXT.encode("ascii")
    assert len(text.encode("ascii")) == 191


def test_known_answer_vector_one() -> None:
    assert _fingerprint(_bar()) == _KAT1_DIGEST
    # The literal digest is also what the standard library gives for the literal bytes.
    assert _KAT1_DIGEST == "fdc1-sha256:" + hashlib.sha256(_KAT1_TEXT.encode("ascii")).hexdigest()


def test_known_answer_vector_two_with_negatives_zero_and_long_fraction() -> None:
    bar = _bar(open="-0.05", high="0", low=_LONG_FRACTION, close="-0", volume="0")

    assert canonical_futures_daily_candle_identity(_CONTRACT, _SESSION, bar) == _KAT2_TEXT
    assert _fingerprint(bar) == _KAT2_DIGEST


def test_lines_are_in_fixed_order_lf_terminated_without_spaces() -> None:
    text = canonical_futures_daily_candle_identity(_CONTRACT, _SESSION, _bar())
    lines = text.split("\n")

    assert lines[-1] == ""  # LF after the last line as well
    assert [line.split("=", 1)[0] for line in lines[1:-1]] == [
        "product",
        "exchange",
        "expiration",
        "trading_date",
        "timeframe",
        "open",
        "high",
        "low",
        "close",
        "volume_contracts",
    ]
    assert " " not in text and "\r" not in text and "\t" not in text


def test_the_fingerprint_shape_is_prefix_and_64_lowercase_hex() -> None:
    assert re.fullmatch(r"fdc1-sha256:[0-9a-f]{64}", _fingerprint(_bar()))
    assert FUTURES_DAILY_CANDLE_FINGERPRINT_PREFIX == "fdc1-sha256:"


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_identical_candles_built_separately_share_one_fingerprint() -> None:
    assert _fingerprint(_bar()) == _fingerprint(_bar()) == _KAT1_DIGEST


def test_repeated_calls_return_identical_output() -> None:
    bar = _bar()

    assert {_fingerprint(bar) for _ in range(50)} == {_KAT1_DIGEST}
    assert {
        canonical_futures_daily_candle_identity(_CONTRACT, _SESSION, bar) for _ in range(50)
    } == {_KAT1_TEXT}


@pytest.mark.parametrize(
    "spelling",
    [
        {"open": "101.250", "high": "102.50000", "low": "99.750", "close": "100.00"},
        {"open": "1.0125E+2", "high": "1.025E2", "low": "9975E-2", "close": "1E+2"},
        {"close": "100.0", "volume": "42.000"},
        {"volume": "4.2E+1"},
    ],
)
def test_equivalent_decimal_spellings_share_the_identity(spelling: dict[str, str]) -> None:
    bar = _bar(**spelling)

    assert canonical_futures_daily_candle_identity(_CONTRACT, _SESSION, bar) == _KAT1_TEXT
    assert _fingerprint(bar) == _KAT1_DIGEST


def test_negative_zero_is_zero() -> None:
    bar = _bar(open="-0.00", high="0", low="-0.00", close="-0E+3", volume="-0")
    text = canonical_futures_daily_candle_identity(_CONTRACT, _SESSION, bar)

    assert "open=0\nhigh=0\nlow=0\nclose=0\nvolume_contracts=0\n" in text
    assert not re.search(r"=-0(\.0*)?\n", text)


def test_the_identity_ignores_decimal_context_and_locale(monkeypatch) -> None:
    import decimal
    import locale

    with decimal.localcontext() as context:
        context.prec = 3
        context.rounding = decimal.ROUND_DOWN
        bar = _bar(low=_LONG_FRACTION.lstrip("-"))
        text = canonical_futures_daily_candle_identity(_CONTRACT, _SESSION, bar)
    monkeypatch.setattr(locale, "localeconv", lambda: {"decimal_point": ",", "thousands_sep": "."})

    assert f"low={_LONG_FRACTION.lstrip('-')}\n" in text
    assert canonical_futures_daily_candle_identity(_CONTRACT, _SESSION, bar) == text


# ---------------------------------------------------------------------------
# Every identifying and economic field changes the fingerprint
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("change", "line"),
    [
        ({"open": "101.3"}, "open=101.3"),
        ({"high": "102.55"}, "high=102.55"),
        ({"low": "99.7"}, "low=99.7"),
        ({"close": "100.05"}, "close=100.05"),
        ({"volume": "43"}, "volume_contracts=43"),
    ],
)
def test_each_economic_value_changes_the_fingerprint(change: dict[str, str], line: str) -> None:
    bar = _bar(**change)

    assert f"{line}\n" in canonical_futures_daily_candle_identity(_CONTRACT, _SESSION, bar)
    assert _fingerprint(bar) != _KAT1_DIGEST


def test_a_one_digit_change_deep_in_a_fraction_changes_the_fingerprint() -> None:
    first = _bar(low="99.123456789012345678901234567890")
    second = _bar(low="99.123456789012345678901234567891")

    assert _fingerprint(first) != _fingerprint(second)


@pytest.mark.parametrize(
    ("contract", "line"),
    [
        (_contract(product="BANKNIFTY"), "product=BANKNIFTY"),
        (_contract(exchange="BSE"), "exchange=BSE"),
        (_contract(expiry="2026-11-24"), "expiration=2026-11-24"),
    ],
)
def test_each_contract_field_changes_the_fingerprint(contract: FuturesContract, line: str) -> None:
    bar = _bar(contract=contract)

    assert f"{line}\n" in canonical_futures_daily_candle_identity(contract, _SESSION, bar)
    assert _fingerprint(bar) != _KAT1_DIGEST


def test_the_session_date_changes_the_fingerprint() -> None:
    other = date(2026, 10, 5)
    text = canonical_futures_daily_candle_identity(_CONTRACT, other, _bar())

    assert "trading_date=2026-10-05\n" in text
    assert _fingerprint(_bar(), other) != _KAT1_DIGEST


def test_a_non_daily_timeframe_has_no_identity() -> None:
    # The format names 1d literally; another timeframe is refused, never hashed.
    for timeframe in ("1h", "1w", "1m"):
        with pytest.raises(InvalidFuturesDailyCandleIdentityError, match="daily bars only"):
            _fingerprint(_bar(timeframe=timeframe))


def test_the_version_line_is_part_of_the_hashed_bytes() -> None:
    text = canonical_futures_daily_candle_identity(_CONTRACT, _SESSION, _bar())
    unversioned = text.split("\n", 1)[1]
    other_version = text.replace("identity/1\n", "identity/2\n", 1)

    assert text.startswith(f"{FUTURES_DAILY_CANDLE_IDENTITY_FORMAT}\n")
    assert FUTURES_DAILY_CANDLE_IDENTITY_FORMAT == "northstar.futures-daily-candle-identity/1"
    for variant in (unversioned, other_version):
        digest = "fdc1-sha256:" + hashlib.sha256(variant.encode("ascii")).hexdigest()
        assert digest != _KAT1_DIGEST


# ---------------------------------------------------------------------------
# Strict input validation
# ---------------------------------------------------------------------------


def test_missing_inputs_are_refused() -> None:
    with pytest.raises(TypeError, match="contract"):
        canonical_futures_daily_candle_identity(None, _SESSION, _bar())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="bar"):
        canonical_futures_daily_candle_identity(_CONTRACT, _SESSION, None)  # type: ignore[arg-type]
    with pytest.raises(InvalidFuturesDailyCandleIdentityError, match="cannot be None"):
        canonical_futures_daily_candle_identity(_CONTRACT, None, _bar())  # type: ignore[arg-type]


def test_a_bar_missing_a_required_value_cannot_exist() -> None:
    with pytest.raises(ValidationError):
        FuturesOHLCVBar(
            contract=_CONTRACT,
            point_in_time=_CLOSE,
            timeframe=Timeframe("1d"),
            open=None,  # type: ignore[arg-type]
            high=QuoteValue(Decimal("102.5")),
            low=QuoteValue(Decimal("99.75")),
            close=QuoteValue(Decimal("100")),
            volume=Quantity(Decimal("42")),
        )


@pytest.mark.parametrize("field_name", ["open", "high", "low", "close", "volume"])
def test_a_missing_value_that_bypassed_validation_is_refused(field_name: str) -> None:
    bar = _bar()
    _tamper(bar, field_name, None)

    with pytest.raises(InvalidFuturesDailyCandleIdentityError):
        _fingerprint(bar)


def test_a_datetime_session_label_is_refused() -> None:
    with pytest.raises(InvalidFuturesDailyCandleIdentityError, match="plain date"):
        canonical_futures_daily_candle_identity(_CONTRACT, datetime(2026, 10, 1), _bar())
    with pytest.raises(InvalidFuturesDailyCandleIdentityError, match="plain date"):
        canonical_futures_daily_candle_identity(_CONTRACT, "2026-10-01", _bar())  # type: ignore[arg-type]


def test_a_bar_for_another_contract_is_refused() -> None:
    with pytest.raises(InvalidFuturesDailyCandleIdentityError, match="not the requested"):
        canonical_futures_daily_candle_identity(_contract(expiry="2026-11-24"), _SESSION, _bar())


@pytest.mark.parametrize("raw", ["abc", "", "1,5", "NaN", "sNaN", "Infinity", "-Infinity"])
def test_invalid_and_non_finite_numbers_never_become_quote_values(raw: str) -> None:
    with pytest.raises((ValidationError, ArithmeticError)):
        QuoteValue(Decimal(raw) if raw not in {"abc", "", "1,5"} else raw)


@pytest.mark.parametrize(
    "bad", [Decimal("NaN"), Decimal("sNaN"), Decimal("Infinity"), Decimal("-Infinity"), "1.5", 1.5]
)
@pytest.mark.parametrize("field_name", ["open", "close"])
def test_a_non_finite_or_non_decimal_price_that_bypassed_validation_is_refused(
    bad: object, field_name: str
) -> None:
    bar = _bar()
    _tamper(getattr(bar, field_name), "value", bad)

    with pytest.raises(InvalidFuturesDailyCandleIdentityError, match="finite Decimal"):
        _fingerprint(bar)


@pytest.mark.parametrize("bad", [Decimal("NaN"), Decimal("Infinity"), 42.0])
def test_a_non_finite_or_float_volume_that_bypassed_validation_is_refused(bad: object) -> None:
    bar = _bar()
    _tamper(bar.volume, "value", bad)

    with pytest.raises(InvalidFuturesDailyCandleIdentityError, match="finite Decimal"):
        _fingerprint(bar)


@pytest.mark.parametrize("non_canonical", ["100.50", "1E+2", "-0", "1.0"])
def test_a_non_canonical_spelling_that_bypassed_validation_is_refused(non_canonical: str) -> None:
    bar = _bar()
    _tamper(bar.close, "value", Decimal(non_canonical))

    with pytest.raises(InvalidFuturesDailyCandleIdentityError, match="canonical"):
        _fingerprint(bar)


def test_a_fractional_contract_volume_is_refused() -> None:
    with pytest.raises(InvalidFuturesDailyCandleIdentityError, match="whole number"):
        _fingerprint(_bar(volume="42.5"))


def test_a_magnitude_without_a_plain_str_spelling_is_refused_not_guessed() -> None:
    # str() gives 1E-7 while the format requires plain notation 0.0000001.
    assert str(QuoteValue(Decimal("0.0000001"))) == "1E-7"
    with pytest.raises(InvalidFuturesDailyCandleIdentityError, match="plain canonical"):
        _fingerprint(_bar(open="0.0000001", low="0.0000001", high="1", close="1"))
    # The smallest magnitude whose str() is already plain is accepted unchanged.
    text = canonical_futures_daily_candle_identity(
        _CONTRACT, _SESSION, _bar(open="0.000001", low="0.000001", high="1", close="1")
    )
    assert "open=0.000001\n" in text


def test_no_binary_float_reaches_or_leaves_the_identity() -> None:
    with pytest.raises(ValidationError):
        QuoteValue(0.1)  # type: ignore[arg-type]
    text = canonical_futures_daily_candle_identity(
        _CONTRACT, _SESSION, _bar(open="0.1", low="0.1", high="0.3", close="0.3")
    )
    # 0.1 + 0.2 style artefacts never appear: the exact decimal text is kept.
    assert "open=0.1\nhigh=0.3\nlow=0.1\nclose=0.3\n" in text


# ---------------------------------------------------------------------------
# Encoding boundaries
# ---------------------------------------------------------------------------


def test_identity_text_is_normalised_by_core_before_it_gets_here() -> None:
    lower = _contract(product="nifty", exchange="nse")

    assert (
        canonical_futures_daily_candle_identity(lower, _SESSION, _bar(contract=lower)) == _KAT1_TEXT
    )


@pytest.mark.parametrize("injected", ["NİFTY", "NIF TY", "NIFTY=1", "NIFTY\nopen=1", "NIFTY "])
def test_a_contract_text_that_bypassed_validation_is_refused(injected: str) -> None:
    symbol = Symbol("NIFTY")
    _tamper(symbol, "value", injected)
    contract = FuturesContract(
        FuturesProductReference(symbol, ExchangeCode("NSE")), _CONTRACT.expiration_date
    )

    with pytest.raises(InvalidFuturesDailyCandleIdentityError, match="printable ASCII"):
        canonical_futures_daily_candle_identity(contract, _SESSION, _bar(contract=contract))


def test_core_already_refuses_non_ascii_contract_text() -> None:
    with pytest.raises(ValidationError):
        Symbol("NİFTY")
    with pytest.raises(ValidationError):
        ExchangeCode("NSEé")


# ---------------------------------------------------------------------------
# Boundaries and public surface
# ---------------------------------------------------------------------------


def test_the_module_is_pure_and_provider_free() -> None:
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

    assert imported <= {"__future__", "hashlib", "re", "datetime", "decimal", "northstar_core"}
    assert not re.search(
        r"\b(upstox|databento|sqlite|environ|getenv|logging|print)\b", source, re.I
    )
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            assert node.attr not in {"now", "utcnow", "today", "localeconv", "getcontext"}
        assert not isinstance(node, ast.Global | ast.Nonlocal)


def test_the_public_surface_is_exported() -> None:
    import northstar_application.application_services as services

    for name in (
        "FUTURES_DAILY_CANDLE_FINGERPRINT_PREFIX",
        "FUTURES_DAILY_CANDLE_IDENTITY_FORMAT",
        "InvalidFuturesDailyCandleIdentityError",
        "futures_daily_candle_fingerprint",
        "canonical_futures_daily_candle_identity",
    ):
        assert name in services.__all__
    assert not hasattr(services, "_canonical")
    assert issubclass(InvalidFuturesDailyCandleIdentityError, ValueError)
