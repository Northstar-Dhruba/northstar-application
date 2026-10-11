"""Approved ``fdc1`` fingerprints of reviewed futures daily candles.

An operator who has reviewed one version of a daily candle approves exactly
that version by publishing its ``fdc1`` fingerprint. This module parses those
published approvals and answers two questions, and nothing else: which
fingerprint is approved for one contract and session, and whether a calculated
fingerprint is that one. It reads no file, environment variable, network or
clock; where the text comes from is a composition concern.

Manifest text
-------------
The manifest is comma-separated entries, with no whitespace anywhere::

    <expiration YYYY-MM-DD>/<trading date YYYY-MM-DD>=fdc1-sha256:<64 lowercase hex>

The empty text is a valid manifest with no approvals. Nothing else is
lenient: every entry must match exactly, both dates must be real calendar
dates, and one malformed entry refuses the whole manifest. There is never a
partial result, and no entry is ever skipped.

The version is the fingerprint prefix. ``fdc1-sha256:`` names both the
identity format and the hash, so any other prefix -- another format version
or another algorithm -- is refused rather than guessed at.

Exact keys, no fallback
-----------------------
The key in the text binds a contract's expiration and one session. The
product and exchange are not in the text; they are the product the manifest
is parsed for, which the caller supplies. So every approval belongs to exactly
one FuturesContract and one trading date, and a lookup for any other product,
exchange, expiration or date finds nothing. Nothing ever falls back to a
neighbouring session, another expiry or a looser approval rule.

A key may appear once. A repeated key is refused even when both fingerprints
are equal, because a manifest that says one thing twice has a defect the
operator should see.

Missing is not permission
-------------------------
A session with no approval -- never approved, or revoked by removing its
entry -- raises FuturesCandleApprovalMissingError. A calculated fingerprint
that differs from the approved one raises FuturesCandleApprovalMismatchError.
Both are failures; neither is ever read as consent to ingest. A manifest that
cannot be parsed raises InvalidFuturesCandleApprovalSetError, which is a
configuration problem and deliberately not an approval failure.

What a manifest is not
----------------------
An entry proves only that bytes match. It is not evidence of who approved
them: anyone able to write the configuration could add one. Fingerprints are
derived from provider data, so no message here ever contains one.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from types import MappingProxyType

from northstar_core.derivatives import ExpirationDate
from northstar_core.foundation.exceptions.validation import ValidationError
from northstar_core.futures import FuturesContract, FuturesOHLCVBar, FuturesProductReference

from northstar_application.application_services.futures_daily_candle_identity import (
    FUTURES_DAILY_CANDLE_FINGERPRINT_PREFIX,
    InvalidFuturesDailyCandleIdentityError,
    futures_daily_candle_fingerprint,
)

_SUBJECT = "Futures candle approval manifest"

# ASCII digits only: \d would also accept other scripts' digits.
_DATE = r"[0-9]{4}-[0-9]{2}-[0-9]{2}"
_FINGERPRINT = re.compile(re.escape(FUTURES_DAILY_CANDLE_FINGERPRINT_PREFIX) + r"[0-9a-f]{64}")
_ENTRY = re.compile(
    rf"(?P<expiration>{_DATE})/(?P<trading_date>{_DATE})="
    rf"(?P<fingerprint>{re.escape(FUTURES_DAILY_CANDLE_FINGERPRINT_PREFIX)}[0-9a-f]{{64}})"
)
# An entry whose key is well formed but whose fingerprint has some other
# prefix: reported as an unsupported format rather than as generic noise.
_OTHER_FORMAT = re.compile(rf"{_DATE}/{_DATE}=(?P<label>[a-z0-9]{{1,16}}-[a-z0-9]{{1,16}}):.*")


class InvalidFuturesCandleApprovalSetError(ValueError):
    """Raised when approval manifest text or an approval set is invalid."""


class FuturesCandleApprovalError(ValueError):
    """Base of the approval failures for a contract's sessions.

    ``contract`` is the contract the caller asked about. ``trading_dates``
    are the affected sessions, ascending and never empty.
    """

    def __init__(self, message: str, contract: FuturesContract, trading_dates: tuple[date, ...]):
        super().__init__(message)
        self.contract = contract
        self.trading_dates = trading_dates


class FuturesCandleApprovalMissingError(FuturesCandleApprovalError):
    """Raised when a session has no approved fingerprint, including a revoked one."""


class FuturesCandleApprovalMismatchError(FuturesCandleApprovalError):
    """Raised when a calculated fingerprint is not the one approved for its session."""


@dataclass(frozen=True, slots=True)
class FuturesCandleApproval:
    """One approved fingerprint for exactly one contract and trading session."""

    contract: FuturesContract
    trading_date: date
    fingerprint: str

    def __post_init__(self) -> None:
        if not isinstance(self.contract, FuturesContract):
            raise InvalidFuturesCandleApprovalSetError(
                "FuturesCandleApproval contract must be a FuturesContract."
            )
        if isinstance(self.trading_date, datetime) or not isinstance(self.trading_date, date):
            raise InvalidFuturesCandleApprovalSetError(
                "FuturesCandleApproval trading date must be a plain date."
            )
        if self.trading_date > date.fromisoformat(self.contract.expiration_date.value):
            raise InvalidFuturesCandleApprovalSetError(
                f"FuturesCandleApproval trading date {self.trading_date.isoformat()} is after "
                f"the expiration of {self.contract}."
            )
        if (
            not isinstance(self.fingerprint, str)
            or _FINGERPRINT.fullmatch(self.fingerprint) is None
        ):
            raise InvalidFuturesCandleApprovalSetError(
                "FuturesCandleApproval fingerprint must be fdc1-sha256: and 64 lowercase hex."
            )

    def __repr__(self) -> str:
        """Deliberately omits the fingerprint, which is derived from provider data."""
        return (
            f"FuturesCandleApproval(contract={self.contract}, "
            f"trading_date={self.trading_date.isoformat()})"
        )


@dataclass(frozen=True, slots=True)
class FuturesCandleApprovalSet:
    """Every approved fingerprint for one product, keyed by contract and session.

    ``approvals`` is held in one canonical order -- expiration, then trading
    date -- so two sets with the same approvals are equal whatever order they
    were supplied in.
    """

    product: FuturesProductReference
    approvals: tuple[FuturesCandleApproval, ...]
    _index: Mapping[tuple[FuturesContract, date], str] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if not isinstance(self.product, FuturesProductReference):
            raise InvalidFuturesCandleApprovalSetError(
                "FuturesCandleApprovalSet product must be a FuturesProductReference."
            )
        if not isinstance(self.approvals, tuple):
            raise InvalidFuturesCandleApprovalSetError(
                "FuturesCandleApprovalSet approvals must be a tuple of FuturesCandleApproval."
            )
        index: dict[tuple[FuturesContract, date], str] = {}
        for position, approval in enumerate(self.approvals, start=1):
            if not isinstance(approval, FuturesCandleApproval):
                raise InvalidFuturesCandleApprovalSetError(
                    f"FuturesCandleApprovalSet approval {position} must be a FuturesCandleApproval."
                )
            if approval.contract.product != self.product:
                raise InvalidFuturesCandleApprovalSetError(
                    f"FuturesCandleApprovalSet approval {position} is for "
                    f"{approval.contract.product}, not {self.product}."
                )
            key = (approval.contract, approval.trading_date)
            if key in index:
                raise InvalidFuturesCandleApprovalSetError(
                    f"FuturesCandleApprovalSet approves {approval.contract} session "
                    f"{approval.trading_date.isoformat()} more than once."
                )
            index[key] = approval.fingerprint
        ordered = tuple(
            sorted(
                self.approvals,
                key=lambda a: (a.contract.expiration_date.value, a.trading_date),
            )
        )
        object.__setattr__(self, "approvals", ordered)
        object.__setattr__(self, "_index", MappingProxyType(index))

    def approved_fingerprint(self, contract: FuturesContract, trading_date: date) -> str:
        """Return the fingerprint approved for exactly this contract and session."""
        _require_contract(contract)
        _require_session(trading_date, "Approval lookup trading date")
        fingerprint = self._index.get((contract, trading_date))
        if fingerprint is None:
            raise _missing(contract, (trading_date,))
        return fingerprint

    def require_approvals(self, contract: FuturesContract, trading_dates: Iterable[date]) -> None:
        """Require an approval for every session, naming every one that has none."""
        _require_contract(contract)
        dates = _sessions(trading_dates)
        missing = tuple(d for d in dates if (contract, d) not in self._index)
        if missing:
            raise _missing(contract, missing)

    def verify_fingerprints(
        self, contract: FuturesContract, fingerprints: Mapping[date, str]
    ) -> None:
        """Require each calculated fingerprint to be exactly the approved one.

        Every session must be approved first; any missing approval is reported
        before any comparison. Then every mismatch is reported together.
        """
        _require_contract(contract)
        if not isinstance(fingerprints, Mapping):
            raise TypeError("Calculated fingerprints must be a mapping of date to fingerprint.")
        dates = _sessions(fingerprints)
        for trading_date in dates:
            calculated = fingerprints[trading_date]
            if not isinstance(calculated, str) or _FINGERPRINT.fullmatch(calculated) is None:
                raise InvalidFuturesDailyCandleIdentityError(
                    f"The calculated fingerprint for {trading_date.isoformat()} is not an fdc1 "
                    "fingerprint."
                )
        self.require_approvals(contract, dates)
        mismatched = tuple(d for d in dates if fingerprints[d] != self._index[(contract, d)])
        if mismatched:
            raise FuturesCandleApprovalMismatchError(
                f"Candles for {contract} differ from their approved versions for sessions: "
                + ", ".join(d.isoformat() for d in mismatched)
                + ".",
                contract,
                mismatched,
            )

    def verify_bar(
        self, contract: FuturesContract, trading_date: date, bar: FuturesOHLCVBar
    ) -> None:
        """Require one bar to be exactly the approved version of its session."""
        _require_session(trading_date, "Approval lookup trading date")
        calculated = futures_daily_candle_fingerprint(contract, trading_date, bar)
        self.verify_fingerprints(contract, {trading_date: calculated})


def parse_futures_candle_approval_manifest(
    text: str, product: FuturesProductReference
) -> FuturesCandleApprovalSet:
    """Parse manifest text for one product; any defect refuses the whole manifest."""
    if not isinstance(text, str):
        raise TypeError(f"{_SUBJECT} text must be a string.")
    if not isinstance(product, FuturesProductReference):
        raise TypeError(f"{_SUBJECT} product must be a FuturesProductReference.")
    if text == "":
        return FuturesCandleApprovalSet(product, ())
    if any(character.isspace() for character in text):
        raise InvalidFuturesCandleApprovalSetError(f"{_SUBJECT} must contain no whitespace.")
    if not text.isascii():
        raise InvalidFuturesCandleApprovalSetError(f"{_SUBJECT} must be ASCII text.")

    approvals: list[FuturesCandleApproval] = []
    seen: dict[tuple[str, str], int] = {}
    for position, element in enumerate(text.split(","), start=1):
        match = _ENTRY.fullmatch(element)
        if match is None:
            raise InvalidFuturesCandleApprovalSetError(_malformed(element, position))
        key = (match["expiration"], match["trading_date"])
        if key in seen:
            raise InvalidFuturesCandleApprovalSetError(
                f"{_SUBJECT} entries {seen[key]} and {position} both approve expiration "
                f"{key[0]} session {key[1]}."
            )
        seen[key] = position
        approvals.append(_approval(match, product, position))
    return FuturesCandleApprovalSet(product, tuple(approvals))


def _approval(
    match: re.Match[str], product: FuturesProductReference, position: int
) -> FuturesCandleApproval:
    try:
        expiration = ExpirationDate(match["expiration"])
    except ValidationError as exc:
        raise InvalidFuturesCandleApprovalSetError(
            f"{_SUBJECT} entry {position} has an invalid expiration date."
        ) from exc
    try:
        trading_date = date.fromisoformat(match["trading_date"])
    except ValueError as exc:
        raise InvalidFuturesCandleApprovalSetError(
            f"{_SUBJECT} entry {position} has an invalid trading date."
        ) from exc
    try:
        return FuturesCandleApproval(
            FuturesContract(product, expiration), trading_date, match["fingerprint"]
        )
    except InvalidFuturesCandleApprovalSetError as exc:
        raise InvalidFuturesCandleApprovalSetError(f"{_SUBJECT} entry {position}: {exc}") from exc


def _malformed(element: str, position: int) -> str:
    """Describe a malformed entry without repeating any of its content."""
    if element == "":
        return f"{_SUBJECT} entry {position} is empty."
    other = _OTHER_FORMAT.fullmatch(element)
    if other is not None and f"{other['label']}:" != FUTURES_DAILY_CANDLE_FINGERPRINT_PREFIX:
        return (
            f"{_SUBJECT} entry {position} uses unsupported fingerprint format "
            f"{other['label']}; only {FUTURES_DAILY_CANDLE_FINGERPRINT_PREFIX} is accepted."
        )
    return (
        f"{_SUBJECT} entry {position} is malformed; expected "
        f"<YYYY-MM-DD>/<YYYY-MM-DD>={FUTURES_DAILY_CANDLE_FINGERPRINT_PREFIX}"
        "<64 lowercase hex digits>."
    )


def _missing(
    contract: FuturesContract, dates: tuple[date, ...]
) -> FuturesCandleApprovalMissingError:
    return FuturesCandleApprovalMissingError(
        f"No approved candle version for {contract} sessions: "
        + ", ".join(d.isoformat() for d in dates)
        + ".",
        contract,
        dates,
    )


def _require_contract(contract: object) -> None:
    if not isinstance(contract, FuturesContract):
        raise TypeError("Approval lookup contract must be a FuturesContract.")


def _require_session(value: object, subject: str) -> None:
    # A datetime carries a clock; a session label has none.
    if isinstance(value, datetime) or not isinstance(value, date):
        raise TypeError(f"{subject} must be a plain date.")


def _sessions(trading_dates: Iterable[date]) -> tuple[date, ...]:
    dates = tuple(trading_dates)
    if not dates:
        raise ValueError("At least one trading date is required.")
    for value in dates:
        _require_session(value, "Approval lookup trading date")
    if len(set(dates)) != len(dates):
        raise ValueError("Each trading date may be given only once.")
    return tuple(sorted(dates))
