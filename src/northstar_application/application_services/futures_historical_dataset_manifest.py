"""Immutable manifest and fingerprint of one historical futures dataset (R-1A).

A historical dataset is a fixed set of stored daily bars for one product's
individual contracts. This module identifies such a dataset and lets its
content be checked later; it stores, imports, fetches and approves nothing.
It is a pure contract for the later importer (R-1C) and acceptance (R-1D).

Four things are kept apart
--------------------------
- Per-candle fingerprint -- ``fdc1-sha256:``, reused unchanged from
  futures_daily_candle_identity. It binds one bar's contract, trading date,
  prices and contract volume.
- Whole-dataset fingerprint -- ``fhd1-sha256:``, SHA-256 over the canonical
  encoding below. It binds the ordered candle inventory and every material
  source, normalisation and coverage field. Identical content always has the
  same fingerprint, whatever it is called.
- Dataset ID -- the caller's immutable name for one version. It is not hashed:
  one ID may name only one fingerprint (verify_futures_historical_dataset_catalog),
  and a revision is a new ID that ``supersedes`` the old.
- Authorization evidence -- ``rights_reference`` is recorded as the caller
  states it and is never verified here. No trusted rights-verification
  mechanism exists, so ``rights_verified`` is always False and no manifest
  makes data usable as market evidence. The R-0 evaluator stays SYNTHETIC-only.

A hash proves only that encoded content equals content fingerprinted earlier.
It does not prove the source was legitimate, the prices genuine or the use
licensed, and nothing here is signed.

Dataset fingerprint ``fhd1``
----------------------------
SHA-256 over an injective encoding: each part is UTF-8, prefixed by its byte
length and ``:``, concatenated, in exactly this order::

    northstar.futures-historical-dataset/1
    <manifest schema>
    <source category> <provider> <release: "absent" | "present", value>
    <normalization version> <timeframe> <source volume unit>
    <product code> <exchange code>
    "contracts" <count> <expiration>...                (ascending)
    "coverage" <first trading date> <last trading date>
    <calendar reference: absent | present, value>
    <contract reference: absent | present, value>
    "expected" absent | present <count> (<expiration> <trading date>)...
    "bars" <count> (<expiration> <trading date> <fdc1 fingerprint>)...

Sequences are sorted by expiration then trading date, so input order never
matters. Dates are ISO ``YYYY-MM-DD``. The dataset ID, ``supersedes``, the
acquisition instant and the rights reference are deliberately not hashed:
they name, link and attest a version, they are not its content.

Coverage and completeness
-------------------------
Without expected-session evidence completeness is UNVERIFIED, and the bounds
must be exactly the first and last stored trading dates: the manifest claims
nothing it does not hold. Expected sessions may be supplied only with both a
named calendar reference and a named contract reference; then every bar must
be an expected session, the bounds are the expected extremes, and the manifest
derives COMPLETE or INCOMPLETE with every missing session listed. Nothing is
interpolated. Completeness is relative to the named references, whose own
verification belongs to R-1B.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum
from typing import Any

from northstar_core.derivatives import ExpirationDate
from northstar_core.foundation.value_objects import ExchangeCode, PointInTime, Symbol, Timeframe
from northstar_core.futures import FuturesContract, FuturesOHLCVBar, FuturesProductReference

from northstar_application.application_services.evaluate_futures_recommendation_policy import (
    FuturesEvaluationEvidence,
    FuturesEvaluationEvidenceClassification,
)
from northstar_application.application_services.futures_daily_candle_identity import (
    FUTURES_DAILY_CANDLE_FINGERPRINT_PREFIX,
    futures_daily_candle_fingerprint,
)

FUTURES_HISTORICAL_DATASET_MANIFEST_SCHEMA = "northstar.futures-historical-dataset-manifest/1"
FUTURES_HISTORICAL_DATASET_FINGERPRINT_NAMESPACE = "northstar.futures-historical-dataset/1"
FUTURES_HISTORICAL_DATASET_FINGERPRINT_PREFIX = "fhd1-sha256:"

_DAILY = Timeframe("1d")
_CANDLE_FINGERPRINT = re.compile(
    re.escape(FUTURES_DAILY_CANDLE_FINGERPRINT_PREFIX) + "[0-9a-f]{64}"
)
_DATASET_FINGERPRINT = re.compile(
    re.escape(FUTURES_HISTORICAL_DATASET_FINGERPRINT_PREFIX) + "[0-9a-f]{64}"
)
# A dataset ID is a stable name: no spaces, no path or URL syntax.
_DATASET_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
# Free reference text: printable ASCII, no leading or trailing space.
_REFERENCE = re.compile(r"[\x21-\x7e](?:[\x20-\x7e]{0,254}[\x21-\x7e])?")
_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


class InvalidFuturesHistoricalDatasetManifestError(ValueError):
    """Raised when a historical dataset manifest or its encoding is invalid."""


class FuturesHistoricalDatasetVersionError(ValueError):
    """Raised when dataset IDs, fingerprints and revisions contradict each other."""


class FuturesHistoricalDatasetSourceCategory(StrEnum):
    """What kind of source a dataset came from; a category, never a permission."""

    SYNTHETIC = "SYNTHETIC"
    PROVIDER_API = "PROVIDER_API"
    EXCHANGE_PRODUCT = "EXCHANGE_PRODUCT"
    LICENSED_VENDOR = "LICENSED_VENDOR"
    OWNER_SUPPLIED = "OWNER_SUPPLIED"


class FuturesHistoricalDatasetVolumeUnit(StrEnum):
    """The unit the source reported volume in; stored bars are always in contracts."""

    CONTRACTS = "CONTRACTS"
    UNDERLYING_UNITS = "UNDERLYING_UNITS"


class FuturesHistoricalDatasetCompleteness(StrEnum):
    """Whether every expected session has its bar, relative to the named references."""

    UNVERIFIED = "UNVERIFIED"
    COMPLETE = "COMPLETE"
    INCOMPLETE = "INCOMPLETE"


def _error(message: str) -> InvalidFuturesHistoricalDatasetManifestError:
    return InvalidFuturesHistoricalDatasetManifestError(message)


def _require(value: object, expected: type, name: str) -> None:
    if not isinstance(value, expected):
        raise _error(f"Historical dataset {name} must be a {expected.__name__}.")


def _require_date(value: object, name: str) -> None:
    # A datetime carries a clock; a trading date has none.
    if isinstance(value, datetime) or not isinstance(value, date):
        raise _error(f"Historical dataset {name} must be a plain date.")


def _require_text(value: object, pattern: re.Pattern[str], name: str) -> None:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise _error(f"Historical dataset {name} is missing or not valid reference text.")


def _require_optional_text(value: object, name: str) -> None:
    if value is not None:
        _require_text(value, _REFERENCE, name)


def _require_tuple_of(value: object, expected: type, name: str) -> None:
    if not isinstance(value, tuple) or not all(isinstance(item, expected) for item in value):
        raise _error(f"Historical dataset {name} must be a tuple of {expected.__name__} values.")


def _session_key(contract: FuturesContract, trading_date: date) -> tuple[str, str]:
    return (contract.expiration_date.value, trading_date.isoformat())


@dataclass(frozen=True, slots=True)
class FuturesHistoricalDatasetSession:
    """One contract's trading session, named by its local exchange trading date."""

    contract: FuturesContract
    trading_date: date

    def __post_init__(self) -> None:
        _require(self.contract, FuturesContract, "session contract")
        _require_date(self.trading_date, "session trading date")

    @property
    def sort_key(self) -> tuple[str, str]:
        """Return the canonical order: expiration, then trading date."""
        return _session_key(self.contract, self.trading_date)


@dataclass(frozen=True, slots=True)
class FuturesHistoricalDatasetBarReference:
    """One stored daily bar, identified by contract, trading date and fdc1 fingerprint.

    The reference never holds prices: the fingerprint binds them.
    ``trading_date`` is the session label the bar was matched to; checking it
    against a verified session close is the importer's job (R-1C).
    """

    contract: FuturesContract
    trading_date: date
    candle_fingerprint: str

    def __post_init__(self) -> None:
        _require(self.contract, FuturesContract, "bar contract")
        _require_date(self.trading_date, "bar trading date")
        if (
            not isinstance(self.candle_fingerprint, str)
            or _CANDLE_FINGERPRINT.fullmatch(self.candle_fingerprint) is None
        ):
            raise _error(
                "Historical dataset bar fingerprint must be fdc1-sha256: and 64 lowercase hex."
            )

    @classmethod
    def from_bar(
        cls, contract: FuturesContract, trading_date: date, bar: FuturesOHLCVBar
    ) -> FuturesHistoricalDatasetBarReference:
        """Return the reference of ``bar``, fingerprinted with the existing fdc1 identity."""
        return cls(
            contract, trading_date, futures_daily_candle_fingerprint(contract, trading_date, bar)
        )

    @property
    def session(self) -> FuturesHistoricalDatasetSession:
        """Return the contract session this bar belongs to."""
        return FuturesHistoricalDatasetSession(self.contract, self.trading_date)

    @property
    def sort_key(self) -> tuple[str, str]:
        """Return the canonical order: expiration, then trading date."""
        return _session_key(self.contract, self.trading_date)


@dataclass(frozen=True, slots=True)
class FuturesHistoricalDatasetSource:
    """Where a dataset came from, as the caller states it.

    ``release_reference`` is the source's own file or release version, None
    when the source supplies none. ``rights_reference`` names the written
    permission the data is held under. It is required for every real source
    category and forbidden for SYNTHETIC, and it is a statement, not proof:
    ``rights_verified`` is always False.
    """

    category: FuturesHistoricalDatasetSourceCategory
    provider: str
    release_reference: str | None
    acquired_at: PointInTime
    rights_reference: str | None

    def __post_init__(self) -> None:
        _require(self.category, FuturesHistoricalDatasetSourceCategory, "source category")
        _require_text(self.provider, _REFERENCE, "source provider")
        _require_optional_text(self.release_reference, "source release reference")
        _require(self.acquired_at, PointInTime, "acquisition instant")
        _require_optional_text(self.rights_reference, "rights reference")
        synthetic = self.category is FuturesHistoricalDatasetSourceCategory.SYNTHETIC
        if synthetic and self.rights_reference is not None:
            raise _error("A SYNTHETIC dataset needs no data-use rights; name none.")
        if not synthetic and self.rights_reference is None:
            raise _error(f"A {self.category} dataset must name its written data-use permission.")

    @property
    def rights_verified(self) -> bool:
        """Return whether the data-use rights were verified; never, in R-1A."""
        return False


@dataclass(frozen=True, slots=True)
class FuturesHistoricalDatasetCoverage:
    """The covered trading dates and the evidence completeness is measured against.

    ``expected_sessions`` is None when no expected-session evidence exists;
    completeness is then UNVERIFIED. It may be supplied only together with a
    named calendar reference and a named contract reference.
    """

    first_trading_date: date
    last_trading_date: date
    calendar_reference: str | None
    contract_reference: str | None
    expected_sessions: tuple[FuturesHistoricalDatasetSession, ...] | None

    def __post_init__(self) -> None:
        _require_date(self.first_trading_date, "first trading date")
        _require_date(self.last_trading_date, "last trading date")
        if self.first_trading_date > self.last_trading_date:
            raise _error("Historical dataset coverage cannot start after it ends.")
        _require_optional_text(self.calendar_reference, "calendar reference")
        _require_optional_text(self.contract_reference, "contract reference")
        if self.expected_sessions is None:
            return
        _require_tuple_of(
            self.expected_sessions, FuturesHistoricalDatasetSession, "expected sessions"
        )
        if self.calendar_reference is None or self.contract_reference is None:
            raise _error(
                "Expected sessions need both a calendar reference and a contract reference; "
                "without them completeness stays UNVERIFIED."
            )
        if not self.expected_sessions:
            raise _error("Expected-session evidence cannot be empty.")
        if len(set(self.expected_sessions)) != len(self.expected_sessions):
            raise _error("Expected-session evidence names one session more than once.")
        ordered = tuple(sorted(self.expected_sessions, key=lambda s: s.sort_key))
        object.__setattr__(self, "expected_sessions", ordered)
        dates = [session.trading_date for session in ordered]
        if min(dates) != self.first_trading_date or max(dates) != self.last_trading_date:
            raise _error("Coverage bounds must be the first and last expected sessions.")


@dataclass(frozen=True, slots=True)
class FuturesHistoricalDatasetManifest:
    """Immutable identity, provenance, coverage and candle inventory of one dataset version.

    Contracts, bars and expected sessions are kept in canonical order, so the
    same content always yields an equal manifest. ``fingerprint`` is derived,
    never supplied.
    """

    dataset_id: str
    source: FuturesHistoricalDatasetSource
    product: FuturesProductReference
    contracts: tuple[FuturesContract, ...]
    timeframe: Timeframe
    source_volume_unit: FuturesHistoricalDatasetVolumeUnit
    normalization_version: str
    coverage: FuturesHistoricalDatasetCoverage
    bars: tuple[FuturesHistoricalDatasetBarReference, ...]
    supersedes: str | None = None
    schema_version: str = FUTURES_HISTORICAL_DATASET_MANIFEST_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != FUTURES_HISTORICAL_DATASET_MANIFEST_SCHEMA:
            raise _error(f"Unsupported historical dataset manifest schema {self.schema_version!r}.")
        _require_text(self.dataset_id, _DATASET_ID, "dataset ID")
        _require(self.source, FuturesHistoricalDatasetSource, "source")
        _require(self.product, FuturesProductReference, "product")
        _require(self.timeframe, Timeframe, "timeframe")
        if self.timeframe != _DAILY:
            raise _error("Historical datasets are defined for native daily (1d) bars only.")
        _require(self.source_volume_unit, FuturesHistoricalDatasetVolumeUnit, "volume unit")
        _require_text(self.normalization_version, _REFERENCE, "normalization version")
        _require(self.coverage, FuturesHistoricalDatasetCoverage, "coverage")
        if self.supersedes is not None:
            _require_text(self.supersedes, _DATASET_ID, "superseded dataset ID")
            if self.supersedes == self.dataset_id:
                raise _error("A dataset cannot supersede itself.")
        self._validate_contracts()
        self._validate_bars()
        self._validate_coverage()

    def _validate_contracts(self) -> None:
        _require_tuple_of(self.contracts, FuturesContract, "contracts")
        if not self.contracts:
            raise _error("A historical dataset must name its contracts.")
        if len(set(self.contracts)) != len(self.contracts):
            raise _error("A historical dataset names one contract more than once.")
        if any(contract.product != self.product for contract in self.contracts):
            raise _error(f"Every dataset contract must be of {self.product}.")
        ordered = tuple(sorted(self.contracts, key=lambda c: c.expiration_date.value))
        object.__setattr__(self, "contracts", ordered)

    def _validate_bars(self) -> None:
        _require_tuple_of(self.bars, FuturesHistoricalDatasetBarReference, "bars")
        if not self.bars:
            raise _error("A historical dataset must reference at least one bar.")
        universe = set(self.contracts)
        for bar in self.bars:
            if bar.contract not in universe:
                raise _error(f"Bar for {bar.contract} is outside the dataset's contracts.")
        if len({bar.sort_key for bar in self.bars}) != len(self.bars):
            raise _error("A historical dataset references one contract session more than once.")
        if len({bar.candle_fingerprint for bar in self.bars}) != len(self.bars):
            raise _error("One candle fingerprint is attached to more than one contract session.")
        ordered = tuple(sorted(self.bars, key=lambda b: b.sort_key))
        object.__setattr__(self, "bars", ordered)

    def _validate_coverage(self) -> None:
        coverage = self.coverage
        if (
            self.source_volume_unit is FuturesHistoricalDatasetVolumeUnit.UNDERLYING_UNITS
            and coverage.contract_reference is None
        ):
            raise _error(
                "Volume reported in underlying units needs a named contract reference "
                "(lot size) to be converted; it is never converted silently."
            )
        dates = [bar.trading_date for bar in self.bars]
        if coverage.expected_sessions is None:
            if coverage.first_trading_date != min(dates) or coverage.last_trading_date != max(
                dates
            ):
                raise _error(
                    "Unverified coverage bounds must be exactly the first and last stored "
                    "trading dates."
                )
            return
        universe = set(self.contracts)
        expected = set(coverage.expected_sessions)
        for session in expected:
            if session.contract not in universe:
                raise _error(f"Expected session for {session.contract} is outside the contracts.")
        for bar in self.bars:
            if bar.session not in expected:
                raise _error(
                    f"Bar {bar.contract} {bar.trading_date} is not an expected session "
                    "under the named references."
                )

    # -- derived views ------------------------------------------------------

    @property
    def completeness(self) -> FuturesHistoricalDatasetCompleteness:
        """Return completeness relative to the named references; never assumed."""
        if self.coverage.expected_sessions is None:
            return FuturesHistoricalDatasetCompleteness.UNVERIFIED
        if self.missing_sessions:
            return FuturesHistoricalDatasetCompleteness.INCOMPLETE
        return FuturesHistoricalDatasetCompleteness.COMPLETE

    @property
    def missing_sessions(self) -> tuple[FuturesHistoricalDatasetSession, ...]:
        """Return every expected session without a bar, in canonical order."""
        if self.coverage.expected_sessions is None:
            return ()
        present = {bar.session for bar in self.bars}
        return tuple(s for s in self.coverage.expected_sessions if s not in present)

    @property
    def bar_counts(self) -> tuple[tuple[FuturesContract, int], ...]:
        """Return each contract's bar count in expiration order, zero counts included."""
        counts = Counter(bar.contract for bar in self.bars)
        return tuple((contract, counts[contract]) for contract in self.contracts)

    @property
    def fingerprint(self) -> str:
        """Return the ``fhd1-sha256:`` whole-dataset fingerprint."""
        digest = hashlib.sha256(_canonical(self._fingerprint_parts())).hexdigest()
        return FUTURES_HISTORICAL_DATASET_FINGERPRINT_PREFIX + digest

    @property
    def is_market_evidence(self) -> bool:
        """Return whether this dataset may support real-market conclusions; never in R-1A."""
        return False

    def synthetic_evaluation_evidence(self) -> FuturesEvaluationEvidence:
        """Return R-0 SYNTHETIC evidence naming this dataset; refused for any real source.

        Real-data evaluation needs a trusted rights-verification and dataset
        validation path (R-1D) that does not exist; no manifest bypasses that.
        """
        if self.source.category is not FuturesHistoricalDatasetSourceCategory.SYNTHETIC:
            raise FuturesHistoricalDatasetVersionError(
                f"{self.source.category} datasets cannot be evaluated: R-0 accepts SYNTHETIC "
                "evidence only, and authorized-dataset evaluation is not available."
            )
        return FuturesEvaluationEvidence(
            FuturesEvaluationEvidenceClassification.SYNTHETIC,
            f"{self.dataset_id}@{self.fingerprint}",
        )

    def _fingerprint_parts(self) -> tuple[str, ...]:
        source = self.source
        coverage = self.coverage
        parts: list[str] = [
            FUTURES_HISTORICAL_DATASET_FINGERPRINT_NAMESPACE,
            self.schema_version,
            source.category.value,
            source.provider,
            *_optional(source.release_reference),
            self.normalization_version,
            self.timeframe.value,
            self.source_volume_unit.value,
            self.product.product_code.value,
            self.product.exchange_code.value,
            "contracts",
            str(len(self.contracts)),
            *(contract.expiration_date.value for contract in self.contracts),
            "coverage",
            coverage.first_trading_date.isoformat(),
            coverage.last_trading_date.isoformat(),
            *_optional(coverage.calendar_reference),
            *_optional(coverage.contract_reference),
            "expected",
        ]
        if coverage.expected_sessions is None:
            parts.append("absent")
        else:
            parts += ["present", str(len(coverage.expected_sessions))]
            for session in coverage.expected_sessions:
                parts += list(session.sort_key)
        parts += ["bars", str(len(self.bars))]
        for bar in self.bars:
            parts += [*bar.sort_key, bar.candle_fingerprint]
        return tuple(parts)

    # -- canonical JSON -----------------------------------------------------

    def canonical_json(self) -> str:
        """Return the manifest's canonical JSON: sorted keys, no spaces, ASCII, no numbers."""
        return json.dumps(self._record(), sort_keys=True, separators=(",", ":"), ensure_ascii=True)

    def _record(self) -> dict[str, Any]:
        source = self.source
        coverage = self.coverage
        expected = (
            None
            if coverage.expected_sessions is None
            else [_session_record(s.contract, s.trading_date) for s in coverage.expected_sessions]
        )
        return {
            "schema": self.schema_version,
            "dataset_id": self.dataset_id,
            "fingerprint": self.fingerprint,
            "supersedes": self.supersedes,
            "source": {
                "category": source.category.value,
                "provider": source.provider,
                "release_reference": source.release_reference,
                "acquired_at": source.acquired_at.value,
                "rights_reference": source.rights_reference,
            },
            "product": {
                "product_code": self.product.product_code.value,
                "exchange_code": self.product.exchange_code.value,
            },
            "contracts": [contract.expiration_date.value for contract in self.contracts],
            "timeframe": self.timeframe.value,
            "source_volume_unit": self.source_volume_unit.value,
            "normalization_version": self.normalization_version,
            "coverage": {
                "first_trading_date": coverage.first_trading_date.isoformat(),
                "last_trading_date": coverage.last_trading_date.isoformat(),
                "calendar_reference": coverage.calendar_reference,
                "contract_reference": coverage.contract_reference,
                "expected_sessions": expected,
            },
            "bars": [
                {
                    **_session_record(bar.contract, bar.trading_date),
                    "fingerprint": bar.candle_fingerprint,
                }
                for bar in self.bars
            ],
        }


def _optional(value: str | None) -> tuple[str, ...]:
    return ("absent",) if value is None else ("present", value)


def _canonical(parts: tuple[str, ...]) -> bytes:
    """Return an injective byte encoding of an ordered string tuple."""
    encoded = bytearray()
    for part in parts:
        raw = part.encode("utf-8")
        encoded += str(len(raw)).encode("ascii")
        encoded += b":"
        encoded += raw
    return bytes(encoded)


def _session_record(contract: FuturesContract, trading_date: date) -> dict[str, str]:
    return {"expiration": contract.expiration_date.value, "trading_date": trading_date.isoformat()}


# ---------------------------------------------------------------------------
# Strict decoding
# ---------------------------------------------------------------------------

_TOP_KEYS = frozenset(
    {
        "schema",
        "dataset_id",
        "fingerprint",
        "supersedes",
        "source",
        "product",
        "contracts",
        "timeframe",
        "source_volume_unit",
        "normalization_version",
        "coverage",
        "bars",
    }
)
_SOURCE_KEYS = frozenset(
    {"category", "provider", "release_reference", "acquired_at", "rights_reference"}
)
_PRODUCT_KEYS = frozenset({"product_code", "exchange_code"})
_COVERAGE_KEYS = frozenset(
    {
        "first_trading_date",
        "last_trading_date",
        "calendar_reference",
        "contract_reference",
        "expected_sessions",
    }
)
_SESSION_KEYS = frozenset({"expiration", "trading_date"})
_BAR_KEYS = frozenset({"expiration", "trading_date", "fingerprint"})


def parse_futures_historical_dataset_manifest(text: str) -> FuturesHistoricalDatasetManifest:
    """Decode canonical manifest JSON strictly.

    Refused: anything but the exact canonical bytes, unknown or missing
    fields, duplicate keys, any JSON number, an unsupported schema, and a
    stated fingerprint that differs from the recomputed one.
    """
    if not isinstance(text, str):
        raise _error("A historical dataset manifest must be text.")
    try:
        record = json.loads(
            text,
            object_pairs_hook=_no_duplicate_keys,
            parse_float=_no_number,
            parse_int=_no_number,
            parse_constant=_no_number,
        )
    except InvalidFuturesHistoricalDatasetManifestError:
        raise
    except ValueError as exc:
        raise _error("A historical dataset manifest must be valid JSON.") from exc
    _keys(record, _TOP_KEYS, "manifest")
    if record["schema"] != FUTURES_HISTORICAL_DATASET_MANIFEST_SCHEMA:
        raise _error(f"Unsupported historical dataset manifest schema {record['schema']!r}.")
    source = _keys(record["source"], _SOURCE_KEYS, "source")
    product_record = _keys(record["product"], _PRODUCT_KEYS, "product")
    coverage = _keys(record["coverage"], _COVERAGE_KEYS, "coverage")
    try:
        product = FuturesProductReference(
            Symbol(_string(product_record["product_code"])),
            ExchangeCode(_string(product_record["exchange_code"])),
        )

        def contract(expiration: object) -> FuturesContract:
            return FuturesContract(product, ExpirationDate(_string(expiration)))

        expected_records = coverage["expected_sessions"]
        expected = (
            None
            if expected_records is None
            else tuple(
                FuturesHistoricalDatasetSession(
                    contract(item["expiration"]), _date(item["trading_date"])
                )
                for item in (
                    _keys(i, _SESSION_KEYS, "expected session") for i in _list(expected_records)
                )
            )
        )
        manifest = FuturesHistoricalDatasetManifest(
            dataset_id=_string(record["dataset_id"]),
            source=FuturesHistoricalDatasetSource(
                category=FuturesHistoricalDatasetSourceCategory(_string(source["category"])),
                provider=_string(source["provider"]),
                release_reference=_optional_string(source["release_reference"]),
                acquired_at=PointInTime(_string(source["acquired_at"])),
                rights_reference=_optional_string(source["rights_reference"]),
            ),
            product=product,
            contracts=tuple(contract(item) for item in _list(record["contracts"])),
            timeframe=Timeframe(_string(record["timeframe"])),
            source_volume_unit=FuturesHistoricalDatasetVolumeUnit(
                _string(record["source_volume_unit"])
            ),
            normalization_version=_string(record["normalization_version"]),
            coverage=FuturesHistoricalDatasetCoverage(
                first_trading_date=_date(coverage["first_trading_date"]),
                last_trading_date=_date(coverage["last_trading_date"]),
                calendar_reference=_optional_string(coverage["calendar_reference"]),
                contract_reference=_optional_string(coverage["contract_reference"]),
                expected_sessions=expected,
            ),
            bars=tuple(
                FuturesHistoricalDatasetBarReference(
                    contract(item["expiration"]),
                    _date(item["trading_date"]),
                    _string(item["fingerprint"]),
                )
                for item in (_keys(i, _BAR_KEYS, "bar") for i in _list(record["bars"]))
            ),
            supersedes=_optional_string(record["supersedes"]),
        )
    except InvalidFuturesHistoricalDatasetManifestError:
        raise
    except ValueError as exc:
        raise _error(f"A historical dataset manifest field is invalid: {exc}") from exc
    stated = record["fingerprint"]
    if not isinstance(stated, str) or _DATASET_FINGERPRINT.fullmatch(stated) is None:
        raise _error("A historical dataset manifest must state an fhd1-sha256: fingerprint.")
    if stated != manifest.fingerprint:
        raise _error("The stated dataset fingerprint does not match the manifest's content.")
    if manifest.canonical_json() != text:
        raise _error("A historical dataset manifest must be in canonical form.")
    return manifest


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    record: dict[str, Any] = {}
    for key, value in pairs:
        if key in record:
            raise _error(f"A historical dataset manifest repeats the field {key!r}.")
        record[key] = value
    return record


def _no_number(text: str) -> Any:
    raise _error(f"A historical dataset manifest contains a JSON number {text!r}; none is allowed.")


def _keys(value: object, keys: frozenset[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise _error(
            f"A historical dataset manifest {name} must have exactly the fields {sorted(keys)}."
        )
    return value


def _list(value: object) -> list[Any]:
    if not isinstance(value, list):
        raise _error("A historical dataset manifest sequence must be a JSON array.")
    return value


def _string(value: object) -> str:
    if not isinstance(value, str):
        raise _error("A historical dataset manifest field must be a JSON string.")
    return value


def _optional_string(value: object) -> str | None:
    return None if value is None else _string(value)


def _date(value: object) -> date:
    text = _string(value)
    if _ISO_DATE.fullmatch(text) is None:
        raise _error(f"A historical dataset manifest date must be YYYY-MM-DD, not {text!r}.")
    return date.fromisoformat(text)


# ---------------------------------------------------------------------------
# Version consistency
# ---------------------------------------------------------------------------


def verify_futures_historical_dataset_revision(
    revision: FuturesHistoricalDatasetManifest, original: FuturesHistoricalDatasetManifest
) -> None:
    """Refuse ``revision`` unless it is a genuine new version superseding ``original``.

    Both manifests must be supplied; a superseded dataset is never assumed to
    exist. A revision names the original, has its own ID, covers the same
    product and differs in content.
    """
    _require_manifest(revision, "revision")
    _require_manifest(original, "original")
    if revision.supersedes != original.dataset_id:
        raise FuturesHistoricalDatasetVersionError(
            f"Dataset {revision.dataset_id} does not supersede {original.dataset_id}."
        )
    if revision.product != original.product:
        raise FuturesHistoricalDatasetVersionError(
            "A dataset revision must cover the same product as the dataset it supersedes."
        )
    if revision.fingerprint == original.fingerprint:
        raise FuturesHistoricalDatasetVersionError(
            "A dataset revision has the same content as the dataset it supersedes."
        )


def verify_futures_historical_dataset_catalog(
    manifests: tuple[FuturesHistoricalDatasetManifest, ...],
) -> None:
    """Refuse a collection where one dataset ID names two contents or revisions form a cycle.

    Only the supplied manifests are checked; a ``supersedes`` naming a dataset
    not supplied is left unresolved, never assumed.
    """
    if not isinstance(manifests, tuple):
        raise FuturesHistoricalDatasetVersionError("Dataset manifests must be a tuple.")
    by_id: dict[str, FuturesHistoricalDatasetManifest] = {}
    for manifest in manifests:
        _require_manifest(manifest, "manifest")
        known = by_id.setdefault(manifest.dataset_id, manifest)
        if known.fingerprint != manifest.fingerprint:
            raise FuturesHistoricalDatasetVersionError(
                f"Dataset ID {manifest.dataset_id} names two different contents."
            )
    for start in by_id:
        seen = {start}
        current = by_id[start].supersedes
        while current is not None and current in by_id:
            if current in seen:
                raise FuturesHistoricalDatasetVersionError(
                    f"Dataset revisions form a cycle through {current}."
                )
            seen.add(current)
            current = by_id[current].supersedes


def _require_manifest(value: object, name: str) -> None:
    if not isinstance(value, FuturesHistoricalDatasetManifest):
        raise FuturesHistoricalDatasetVersionError(
            f"Dataset {name} must be a FuturesHistoricalDatasetManifest."
        )
