"""Canonical identity and SHA-256 fingerprint of one futures daily candle.

An operator reviews one version of a daily candle and approves it. This module
turns the canonical bar Northstar would persist into one exact byte string,
and that byte string into a fingerprint, so that an approved version can later
be compared with the bar about to be stored. It decides nothing itself: it
does not approve, store or compare anything.

Format ``fdc1``
---------------
The identity is ASCII text, one ``name=value`` line per field, LF after every
line including the last, in exactly this order and with no spaces::

    northstar.futures-daily-candle-identity/1
    product=<product code>
    exchange=<exchange code>
    expiration=<YYYY-MM-DD>
    trading_date=<YYYY-MM-DD>
    timeframe=1d
    open=<canonical decimal>
    high=<canonical decimal>
    low=<canonical decimal>
    close=<canonical decimal>
    volume_contracts=<canonical integer>

The fingerprint is ``fdc1-sha256:`` followed by the 64 lowercase hex digits of
SHA-256 over those bytes. It binds one version of the values; it says nothing
about whether that version is trustworthy.

Only what is stored and consumed is identified
----------------------------------------------
The fields are exactly the persisted bar's contract, session label, timeframe,
prices and contract volume. Provider timestamps, request times, endpoints,
raw provider volume, lot size and open interest are provenance, not part of
the stored bar, and are deliberately absent. The format names ``1d``
literally, so a bar of any other timeframe has no ``fdc1`` identity and is
refused rather than given one.

Numbers are never re-normalised here
------------------------------------
Each number is the text of the existing QuoteValue or Quantity, which Core
already canonicalises without a decimal context: plain notation, trailing
fractional zeros removed, negative zero folded to zero, every digit kept. So
``100.50`` and ``100.5`` share one identity, and different values never do.
This module only checks that the text really has that canonical shape, so a
value that bypassed its own validation cannot produce an identity.

One magnitude is refused, not guessed. ``str()`` of a Decimal whose magnitude
is below 0.000001 (and not zero) uses an exponent -- ``1E-7`` -- while the
format requires plain notation -- ``0.0000001``. The format does not say which
spelling wins, so such a value has no ``fdc1`` identity. Every accepted value
reads the same either way.
"""

from __future__ import annotations

import hashlib
import re
from datetime import date, datetime
from decimal import Decimal

from northstar_core.derivatives import QuoteValue
from northstar_core.foundation.value_objects import Quantity, Timeframe
from northstar_core.futures import FuturesContract, FuturesOHLCVBar

FUTURES_DAILY_CANDLE_IDENTITY_FORMAT = "northstar.futures-daily-candle-identity/1"
FUTURES_DAILY_CANDLE_FINGERPRINT_PREFIX = "fdc1-sha256:"

_SUBJECT = "Futures daily candle identity"
_DAILY = Timeframe("1d")

# Printable ASCII without space or "=": a field value can never end a line,
# start another field or be mistaken for its name.
_FIELD_TEXT = re.compile(r"[\x21-\x3c\x3e-\x7e]+")
# The shapes canonical_decimal produces: no exponent, no leading zeros, no
# trailing fractional zeros, and never "-0".
_CANONICAL_DECIMAL = re.compile(r"(?!-0\Z)-?(?:0|[1-9][0-9]*)(?:\.[0-9]*[1-9])?")
_CANONICAL_INTEGER = re.compile(r"0|[1-9][0-9]*")


class InvalidFuturesDailyCandleIdentityError(ValueError):
    """Raised when a futures daily candle cannot be given a canonical identity."""


def canonical_futures_daily_candle_identity(
    contract: FuturesContract, trading_date: date, bar: FuturesOHLCVBar
) -> str:
    """Return the ``fdc1`` canonical identity text of one daily bar.

    ``contract`` is the contract the caller is operating and must equal the
    bar's own. ``trading_date`` is the session label the bar was matched to;
    it is a plain date, never an instant.
    """
    if not isinstance(contract, FuturesContract):
        raise TypeError(f"{_SUBJECT} contract must be a FuturesContract.")
    if not isinstance(bar, FuturesOHLCVBar):
        raise TypeError(f"{_SUBJECT} bar must be a FuturesOHLCVBar.")
    if trading_date is None:
        raise InvalidFuturesDailyCandleIdentityError(f"{_SUBJECT} trading date cannot be None.")
    # A datetime carries a clock; a session label has none.
    if isinstance(trading_date, datetime) or not isinstance(trading_date, date):
        raise InvalidFuturesDailyCandleIdentityError(
            f"{_SUBJECT} trading date must be a plain date."
        )
    if bar.contract != contract:
        raise InvalidFuturesDailyCandleIdentityError(
            f"{_SUBJECT} bar is for {bar.contract}, not the requested contract {contract}."
        )
    if bar.timeframe != _DAILY:
        raise InvalidFuturesDailyCandleIdentityError(
            f"{_SUBJECT} is defined for daily bars only; received timeframe {bar.timeframe}."
        )

    product = contract.product
    lines = (
        FUTURES_DAILY_CANDLE_IDENTITY_FORMAT,
        _line("product", _text(product.product_code.value, "product")),
        _line("exchange", _text(product.exchange_code.value, "exchange")),
        _line("expiration", _text(contract.expiration_date.value, "expiration")),
        _line("trading_date", trading_date.isoformat()),
        _line("timeframe", _DAILY.value),
        _line("open", _quote(bar.open, "open")),
        _line("high", _quote(bar.high, "high")),
        _line("low", _quote(bar.low, "low")),
        _line("close", _quote(bar.close, "close")),
        _line("volume_contracts", _contracts(bar.volume)),
    )
    return "".join(f"{line}\n" for line in lines)


def futures_daily_candle_fingerprint(
    contract: FuturesContract, trading_date: date, bar: FuturesOHLCVBar
) -> str:
    """Return ``fdc1-sha256:`` and the SHA-256 of the bar's canonical identity."""
    text = canonical_futures_daily_candle_identity(contract, trading_date, bar)
    try:
        encoded = text.encode("ascii")
    except UnicodeEncodeError as exc:
        raise InvalidFuturesDailyCandleIdentityError(f"{_SUBJECT} must be ASCII text.") from exc
    return FUTURES_DAILY_CANDLE_FINGERPRINT_PREFIX + hashlib.sha256(encoded).hexdigest()


def _line(name: str, value: str) -> str:
    return f"{name}={value}"


def _text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or _FIELD_TEXT.fullmatch(value) is None:
        raise InvalidFuturesDailyCandleIdentityError(
            f"{_SUBJECT} {field_name} must be printable ASCII without spaces or '='."
        )
    return value


def _quote(value: object, field_name: str) -> str:
    if not isinstance(value, QuoteValue):
        raise InvalidFuturesDailyCandleIdentityError(
            f"{_SUBJECT} {field_name} must be a QuoteValue."
        )
    return _canonical(value.value, field_name, _CANONICAL_DECIMAL)


def _contracts(value: object) -> str:
    if not isinstance(value, Quantity):
        raise InvalidFuturesDailyCandleIdentityError(
            f"{_SUBJECT} volume_contracts must be a Quantity."
        )
    number = value.value
    if isinstance(number, Decimal) and number.is_finite() and number != number.to_integral_value():
        raise InvalidFuturesDailyCandleIdentityError(
            f"{_SUBJECT} volume_contracts must be a whole number of contracts."
        )
    return _canonical(number, "volume_contracts", _CANONICAL_INTEGER)


def _canonical(number: object, field_name: str, shape: re.Pattern[str]) -> str:
    """Return a Core-canonical number's text, refusing anything not in that shape."""
    if not isinstance(number, Decimal) or not number.is_finite():
        raise InvalidFuturesDailyCandleIdentityError(
            f"{_SUBJECT} {field_name} must be a finite Decimal."
        )
    text = str(number)
    if shape.fullmatch(text) is None:
        raise InvalidFuturesDailyCandleIdentityError(
            f"{_SUBJECT} {field_name} has no plain canonical decimal form."
        )
    return text
