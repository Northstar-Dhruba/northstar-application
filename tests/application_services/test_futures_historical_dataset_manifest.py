"""R-1A: historical futures dataset manifest, fingerprint and evidence boundary.

Every value here is synthetic. Expiration dates, sessions and prices are test
labels, not statements about real NIFTY contracts, calendars or markets.
"""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import json
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

import northstar_application.application_services as services
from northstar_application.application_services import (
    FUTURES_HISTORICAL_DATASET_FINGERPRINT_NAMESPACE,
    FUTURES_HISTORICAL_DATASET_FINGERPRINT_PREFIX,
    FUTURES_HISTORICAL_DATASET_MANIFEST_SCHEMA,
    FuturesEvaluationEvidence,
    FuturesEvaluationEvidenceClassification,
    FuturesHistoricalDatasetBarReference,
    FuturesHistoricalDatasetCompleteness,
    FuturesHistoricalDatasetCoverage,
    FuturesHistoricalDatasetManifest,
    FuturesHistoricalDatasetSession,
    FuturesHistoricalDatasetSource,
    FuturesHistoricalDatasetSourceCategory,
    FuturesHistoricalDatasetVersionError,
    FuturesHistoricalDatasetVolumeUnit,
    InvalidFuturesHistoricalDatasetManifestError,
    InvalidFuturesRecommendationEvaluationError,
    futures_daily_candle_fingerprint,
    parse_futures_historical_dataset_manifest,
    verify_futures_historical_dataset_catalog,
    verify_futures_historical_dataset_revision,
)

_NIFTY = FuturesProductReference(Symbol("NIFTY"), ExchangeCode("NSE"))
_BANKNIFTY = FuturesProductReference(Symbol("BANKNIFTY"), ExchangeCode("NSE"))
_OCT = FuturesContract(_NIFTY, ExpirationDate("2026-10-27"))
_NOV = FuturesContract(_NIFTY, ExpirationDate("2026-11-24"))
_DAILY = Timeframe("1d")
_DAYS = (date(2026, 10, 5), date(2026, 10, 6), date(2026, 10, 7))
_SYNTHETIC = FuturesHistoricalDatasetSourceCategory.SYNTHETIC
_CONTRACTS = FuturesHistoricalDatasetVolumeUnit.CONTRACTS
Manifest = FuturesHistoricalDatasetManifest
Ref = FuturesHistoricalDatasetBarReference
Session = FuturesHistoricalDatasetSession


def _bar(contract: FuturesContract, day: date, close: str = "24400.5") -> FuturesOHLCVBar:
    value = Decimal(close)
    return FuturesOHLCVBar(
        contract=contract,
        point_in_time=PointInTime(f"{day.isoformat()}T10:00:00Z"),
        timeframe=_DAILY,
        open=QuoteValue(value - 10),
        high=QuoteValue(value + 20),
        low=QuoteValue(value - 30),
        close=QuoteValue(value),
        volume=Quantity(Decimal("1200")),
    )


def _ref(contract: FuturesContract, day: date, close: str = "24400.5") -> Ref:
    return Ref.from_bar(contract, day, _bar(contract, day, close))


def _source(**changes: object) -> FuturesHistoricalDatasetSource:
    fields: dict[str, object] = {
        "category": _SYNTHETIC,
        "provider": "northstar-test-fixture",
        "release_reference": "fixture-r1",
        "acquired_at": PointInTime("2026-10-16T12:00:00Z"),
        "rights_reference": None,
    }
    fields.update(changes)
    return FuturesHistoricalDatasetSource(**fields)  # type: ignore[arg-type]


def _unverified(first: date = _DAYS[0], last: date = _DAYS[-1]) -> FuturesHistoricalDatasetCoverage:
    return FuturesHistoricalDatasetCoverage(first, last, None, None, None)


def _bars() -> tuple[Ref, ...]:
    return tuple(_ref(c, d) for c in (_OCT, _NOV) for d in _DAYS)


def _manifest(**changes: object) -> Manifest:
    fields: dict[str, object] = {
        "dataset_id": "nifty-synthetic-v1",
        "source": _source(),
        "product": _NIFTY,
        "contracts": (_OCT, _NOV),
        "timeframe": _DAILY,
        "source_volume_unit": _CONTRACTS,
        "normalization_version": "northstar.fhd-normalization/1",
        "coverage": _unverified(),
        "bars": _bars(),
    }
    fields.update(changes)
    return Manifest(**fields)  # type: ignore[arg-type]


def _verified(bars: tuple[Ref, ...], expected: tuple[Session, ...], **coverage: object) -> Manifest:
    dates = [s.trading_date for s in expected]
    fields: dict[str, object] = {
        "first_trading_date": min(dates),
        "last_trading_date": max(dates),
        "calendar_reference": "synthetic-calendar/1",
        "contract_reference": "synthetic-contracts/1",
        "expected_sessions": expected,
    }
    fields.update(coverage)
    return _manifest(coverage=FuturesHistoricalDatasetCoverage(**fields), bars=bars)  # type: ignore[arg-type]


def _all_expected() -> tuple[Session, ...]:
    return tuple(Session(c, d) for c in (_OCT, _NOV) for d in _DAYS)


# ---------------------------------------------------------------------------
# Identity and fingerprint
# ---------------------------------------------------------------------------


def test_bar_reference_reuses_the_fdc1_candle_fingerprint() -> None:
    bar = _bar(_OCT, _DAYS[0])
    reference = Ref.from_bar(_OCT, _DAYS[0], bar)
    assert reference.candle_fingerprint == futures_daily_candle_fingerprint(_OCT, _DAYS[0], bar)
    assert reference.candle_fingerprint.startswith("fdc1-sha256:")


def test_fingerprint_is_stable_and_namespaced() -> None:
    first, second = _manifest(), _manifest()
    assert first == second
    assert first.fingerprint == second.fingerprint
    assert first.fingerprint.startswith(FUTURES_HISTORICAL_DATASET_FINGERPRINT_PREFIX)
    assert len(first.fingerprint) == len(FUTURES_HISTORICAL_DATASET_FINGERPRINT_PREFIX) + 64


def test_fingerprint_is_sha256_of_the_documented_length_prefixed_encoding() -> None:
    manifest = _manifest(
        contracts=(_OCT,),
        bars=(_ref(_OCT, _DAYS[0]),),
        coverage=_unverified(_DAYS[0], _DAYS[0]),
    )
    parts = (
        FUTURES_HISTORICAL_DATASET_FINGERPRINT_NAMESPACE,
        FUTURES_HISTORICAL_DATASET_MANIFEST_SCHEMA,
        "SYNTHETIC", "northstar-test-fixture", "present", "fixture-r1",
        "northstar.fhd-normalization/1", "1d", "CONTRACTS", "NIFTY", "NSE",
        "contracts", "1", "2026-10-27",
        "coverage", "2026-10-05", "2026-10-05", "absent", "absent",
        "expected", "absent",
        "bars", "1", "2026-10-27", "2026-10-05", manifest.bars[0].candle_fingerprint,
    )  # fmt: skip
    encoded = b"".join(f"{len(p.encode())}:".encode() + p.encode() for p in parts)
    assert manifest.fingerprint == "fhd1-sha256:" + hashlib.sha256(encoded).hexdigest()


def test_reordered_inputs_give_the_same_manifest_and_digest() -> None:
    reordered = _manifest(contracts=(_NOV, _OCT), bars=tuple(reversed(_bars())))
    assert reordered == _manifest()
    assert reordered.fingerprint == _manifest().fingerprint
    assert reordered.canonical_json() == _manifest().canonical_json()


def test_a_modified_candle_changes_the_digest() -> None:
    bars = list(_bars())
    bars[2] = _ref(bars[2].contract, bars[2].trading_date, close="24400.55")
    assert _manifest(bars=tuple(bars)).fingerprint != _manifest().fingerprint


def test_a_modified_contract_identity_changes_the_digest() -> None:
    dec = FuturesContract(_NIFTY, ExpirationDate("2026-12-29"))
    bars = tuple(_ref(c, d) for c in (_OCT, dec) for d in _DAYS)
    changed = _manifest(contracts=(_OCT, dec), bars=bars)
    assert changed.fingerprint != _manifest().fingerprint


def test_the_same_fingerprint_under_another_session_is_refused_by_identity() -> None:
    original = _ref(_OCT, _DAYS[0])
    with pytest.raises(InvalidFuturesHistoricalDatasetManifestError, match="more than one"):
        _manifest(
            bars=(original, Ref(_OCT, _DAYS[1], original.candle_fingerprint)),
            contracts=(_OCT,),
            coverage=_unverified(_DAYS[0], _DAYS[1]),
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"normalization_version": "northstar.fhd-normalization/2"},
        {"source": _source(release_reference="fixture-r2")},
        {"source": _source(release_reference=None)},
        {"source": _source(provider="another-fixture")},
        {"source_volume_unit": FuturesHistoricalDatasetVolumeUnit.UNDERLYING_UNITS,
         "coverage": FuturesHistoricalDatasetCoverage(
             _DAYS[0], _DAYS[-1], None, "synthetic-contracts/1", None)},
        {"coverage": FuturesHistoricalDatasetCoverage(
            _DAYS[0], _DAYS[-1], "synthetic-calendar/1", None, None)},
    ],
)  # fmt: skip
def test_a_material_version_field_changes_the_digest(changes: dict[str, object]) -> None:
    assert _manifest(**changes).fingerprint != _manifest().fingerprint


@pytest.mark.parametrize(
    "changes",
    [
        {"dataset_id": "another-name"},
        {"supersedes": "nifty-synthetic-v0"},
        {"source": _source(acquired_at=PointInTime("2026-10-17T00:00:00Z"))},
    ],
)
def test_names_links_and_acquisition_instant_are_not_content(changes: dict[str, object]) -> None:
    assert _manifest(**changes).fingerprint == _manifest().fingerprint


def test_length_prefixing_prevents_field_boundary_collisions() -> None:
    joined = _manifest(source=_source(provider="ab", release_reference="c"))
    split = _manifest(source=_source(provider="a", release_reference="bc"))
    assert joined.fingerprint != split.fingerprint
    absent = _manifest(source=_source(release_reference=None))
    literal = _manifest(source=_source(release_reference="absent"))
    assert absent.fingerprint != literal.fingerprint


def test_repeated_serialization_is_byte_identical() -> None:
    manifest = _manifest()
    assert {manifest.canonical_json() for _ in range(5)} == {manifest.canonical_json()}
    assert manifest.canonical_json() == _manifest().canonical_json()


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------


def test_unverified_coverage_reports_bounds_counts_and_no_gaps() -> None:
    manifest = _manifest()
    assert manifest.completeness is FuturesHistoricalDatasetCompleteness.UNVERIFIED
    assert manifest.missing_sessions == ()
    assert manifest.bar_counts == ((_OCT, 3), (_NOV, 3))
    assert manifest.coverage.first_trading_date == _DAYS[0]


def test_a_contract_without_bars_stays_visible_in_the_counts() -> None:
    dec = FuturesContract(_NIFTY, ExpirationDate("2026-12-29"))
    assert _manifest(contracts=(_OCT, _NOV, dec)).bar_counts[-1] == (dec, 0)


def test_inverted_bounds_are_refused() -> None:
    with pytest.raises(InvalidFuturesHistoricalDatasetManifestError, match="start after"):
        _unverified(_DAYS[1], _DAYS[0])


@pytest.mark.parametrize("bounds", [(_DAYS[0], date(2026, 10, 8)), (date(2026, 10, 2), _DAYS[-1])])
def test_unverified_bounds_must_equal_the_stored_dates(bounds: tuple[date, date]) -> None:
    with pytest.raises(InvalidFuturesHistoricalDatasetManifestError, match="exactly"):
        _manifest(coverage=_unverified(*bounds))


def test_duplicate_candle_reference_is_refused() -> None:
    with pytest.raises(InvalidFuturesHistoricalDatasetManifestError, match="more than once"):
        _manifest(bars=(*_bars(), _ref(_OCT, _DAYS[0], close="1")))


def test_bar_outside_the_contract_universe_is_refused() -> None:
    with pytest.raises(InvalidFuturesHistoricalDatasetManifestError, match="outside"):
        _manifest(contracts=(_OCT,))


@pytest.mark.parametrize(
    "changes",
    [
        {"dataset_id": ""},
        {"dataset_id": "has space"},
        {"dataset_id": "../escape"},
        {"normalization_version": " "},
        {"contracts": ()},
        {"bars": ()},
        {"product": _BANKNIFTY},
        {"timeframe": Timeframe("1h")},
        {"source_volume_unit": "CONTRACTS"},
        {"contracts": [_OCT, _NOV]},
    ],
)
def test_missing_or_invalid_metadata_is_refused(changes: dict[str, object]) -> None:
    with pytest.raises(InvalidFuturesHistoricalDatasetManifestError):
        _manifest(**changes)


@pytest.mark.parametrize(
    "changes",
    [
        {"provider": ""},
        {"provider": " padded"},
        {"release_reference": ""},
        {"acquired_at": "2026-10-16T12:00:00Z"},
        {"category": "SYNTHETIC"},
    ],
)
def test_invalid_source_metadata_is_refused(changes: dict[str, object]) -> None:
    with pytest.raises(InvalidFuturesHistoricalDatasetManifestError):
        _source(**changes)


def test_trading_dates_must_be_plain_dates() -> None:
    with pytest.raises(InvalidFuturesHistoricalDatasetManifestError):
        Ref(_OCT, datetime(2026, 10, 5), _ref(_OCT, _DAYS[0]).candle_fingerprint)  # type: ignore[arg-type]
    with pytest.raises(InvalidFuturesHistoricalDatasetManifestError):
        Ref(_OCT, _DAYS[0], "sha256:" + "0" * 64)


def test_complete_against_named_references() -> None:
    manifest = _verified(_bars(), _all_expected())
    assert manifest.completeness is FuturesHistoricalDatasetCompleteness.COMPLETE
    assert manifest.missing_sessions == ()


def test_known_gaps_stay_visible_and_are_never_filled() -> None:
    bars = tuple(b for b in _bars() if (b.contract, b.trading_date) != (_NOV, _DAYS[1]))
    manifest = _verified(bars, _all_expected())
    assert manifest.completeness is FuturesHistoricalDatasetCompleteness.INCOMPLETE
    assert manifest.missing_sessions == (Session(_NOV, _DAYS[1]),)
    assert len(manifest.bars) == 5
    assert manifest.bar_counts == ((_OCT, 3), (_NOV, 2))


def test_a_missing_edge_session_is_reported_not_trimmed() -> None:
    bars = tuple(b for b in _bars() if b.trading_date != _DAYS[0])
    manifest = _verified(bars, _all_expected())
    assert manifest.coverage.first_trading_date == _DAYS[0]
    assert manifest.missing_sessions == (Session(_OCT, _DAYS[0]), Session(_NOV, _DAYS[0]))


@pytest.mark.parametrize("missing", ["calendar_reference", "contract_reference"])
def test_expected_sessions_need_both_references(missing: str) -> None:
    with pytest.raises(InvalidFuturesHistoricalDatasetManifestError, match="UNVERIFIED"):
        _verified(_bars(), _all_expected(), **{missing: None})


def test_unknown_references_leave_completeness_unverified() -> None:
    calendar_only = FuturesHistoricalDatasetCoverage(
        _DAYS[0], _DAYS[-1], "synthetic-calendar/1", None, None
    )
    manifest = _manifest(coverage=calendar_only)
    assert manifest.completeness is FuturesHistoricalDatasetCompleteness.UNVERIFIED


def test_a_bar_on_an_unexpected_session_is_refused() -> None:
    expected = tuple(s for s in _all_expected() if s != Session(_OCT, _DAYS[1]))
    with pytest.raises(InvalidFuturesHistoricalDatasetManifestError, match="not an expected"):
        _verified(_bars(), expected)


def test_verified_bounds_must_be_the_expected_extremes() -> None:
    with pytest.raises(InvalidFuturesHistoricalDatasetManifestError, match="expected sessions"):
        _verified(_bars(), _all_expected(), last_trading_date=date(2026, 10, 8))


def test_empty_or_duplicate_expected_sessions_are_refused() -> None:
    with pytest.raises(InvalidFuturesHistoricalDatasetManifestError, match="empty"):
        FuturesHistoricalDatasetCoverage(_DAYS[0], _DAYS[0], "c", "k", ())
    with pytest.raises(InvalidFuturesHistoricalDatasetManifestError, match="more than once"):
        _verified(_bars(), (*_all_expected(), Session(_OCT, _DAYS[0])))


def test_expected_session_for_an_unlisted_contract_is_refused() -> None:
    dec = FuturesContract(_NIFTY, ExpirationDate("2026-12-29"))
    with pytest.raises(InvalidFuturesHistoricalDatasetManifestError, match="outside"):
        _verified(_bars(), (*_all_expected(), Session(dec, _DAYS[0])))


def test_underlying_unit_volume_needs_a_contract_reference() -> None:
    with pytest.raises(InvalidFuturesHistoricalDatasetManifestError, match="never converted"):
        _manifest(source_volume_unit=FuturesHistoricalDatasetVolumeUnit.UNDERLYING_UNITS)


# ---------------------------------------------------------------------------
# Versioning
# ---------------------------------------------------------------------------


def test_supported_and_unknown_schema_versions() -> None:
    assert _manifest().schema_version == "northstar.futures-historical-dataset-manifest/1"
    with pytest.raises(InvalidFuturesHistoricalDatasetManifestError, match="schema"):
        _manifest(schema_version="northstar.futures-historical-dataset-manifest/2")


def test_a_dataset_cannot_supersede_itself() -> None:
    with pytest.raises(InvalidFuturesHistoricalDatasetManifestError, match="itself"):
        _manifest(supersedes="nifty-synthetic-v1")


def _revision(**changes: object) -> Manifest:
    bars = list(_bars())
    bars[0] = _ref(bars[0].contract, bars[0].trading_date, close="24401")
    fields: dict[str, object] = {
        "dataset_id": "nifty-synthetic-v2",
        "supersedes": "nifty-synthetic-v1",
        "bars": tuple(bars),
        "source": _source(release_reference="fixture-r2"),
    }
    fields.update(changes)
    return _manifest(**fields)


def test_a_revision_is_a_new_version_with_a_new_fingerprint() -> None:
    original, revision = _manifest(), _revision()
    verify_futures_historical_dataset_revision(revision, original)
    assert revision.fingerprint != original.fingerprint
    assert original.bars == _bars()  # the original is untouched


def test_revision_must_name_and_differ_from_the_original() -> None:
    original = _manifest()
    with pytest.raises(FuturesHistoricalDatasetVersionError, match="does not supersede"):
        verify_futures_historical_dataset_revision(_revision(supersedes="other"), original)
    with pytest.raises(FuturesHistoricalDatasetVersionError, match="same content"):
        verify_futures_historical_dataset_revision(
            _manifest(dataset_id="nifty-synthetic-v2", supersedes="nifty-synthetic-v1"), original
        )
    with pytest.raises(FuturesHistoricalDatasetVersionError):
        verify_futures_historical_dataset_revision(_revision(), "nifty-synthetic-v1")  # type: ignore[arg-type]


def test_revision_must_cover_the_same_product() -> None:
    contract = FuturesContract(_BANKNIFTY, ExpirationDate("2026-10-27"))
    other = _manifest(
        dataset_id="bank-v2",
        supersedes="nifty-synthetic-v1",
        product=_BANKNIFTY,
        contracts=(contract,),
        bars=tuple(_ref(contract, d) for d in _DAYS),
    )
    with pytest.raises(FuturesHistoricalDatasetVersionError, match="same product"):
        verify_futures_historical_dataset_revision(other, _manifest())


def test_one_dataset_id_cannot_name_two_contents() -> None:
    verify_futures_historical_dataset_catalog((_manifest(), _manifest(), _revision()))
    reused = _revision(dataset_id="nifty-synthetic-v1", supersedes=None)
    with pytest.raises(FuturesHistoricalDatasetVersionError, match="two different contents"):
        verify_futures_historical_dataset_catalog((_manifest(), reused))


def test_revision_cycles_are_refused_and_unsupplied_ancestors_left_unresolved() -> None:
    a = _manifest(dataset_id="a", supersedes="b")
    b = _revision(dataset_id="b", supersedes="a")
    with pytest.raises(FuturesHistoricalDatasetVersionError, match="cycle"):
        verify_futures_historical_dataset_catalog((a, b))
    verify_futures_historical_dataset_catalog((_revision(),))  # v1 not supplied: not assumed
    with pytest.raises(FuturesHistoricalDatasetVersionError):
        verify_futures_historical_dataset_catalog([_manifest()])  # type: ignore[arg-type]


def test_manifests_are_immutable() -> None:
    manifest = _manifest()
    with pytest.raises(dataclasses.FrozenInstanceError):
        manifest.dataset_id = "changed"  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        manifest.source.rights_reference = "approved"  # type: ignore[misc]
    assert isinstance(manifest.bars, tuple)
    assert isinstance(manifest.contracts, tuple)


# ---------------------------------------------------------------------------
# Evidence and authorization
# ---------------------------------------------------------------------------


def test_synthetic_manifest_yields_only_r0_synthetic_evidence() -> None:
    manifest = _manifest()
    evidence = manifest.synthetic_evaluation_evidence()
    assert evidence.classification is FuturesEvaluationEvidenceClassification.SYNTHETIC
    assert evidence.dataset_reference == f"nifty-synthetic-v1@{manifest.fingerprint}"
    assert evidence.is_market_evidence is False
    assert manifest.is_market_evidence is False


def test_r0_evidence_classification_is_still_synthetic_only() -> None:
    assert [c.value for c in FuturesEvaluationEvidenceClassification] == ["SYNTHETIC"]
    with pytest.raises(ValueError):
        FuturesEvaluationEvidenceClassification("AUTHORIZED_DATASET")
    with pytest.raises(InvalidFuturesRecommendationEvaluationError):
        FuturesEvaluationEvidence("AUTHORIZED_DATASET", "dataset")  # type: ignore[arg-type]
    with pytest.raises(InvalidFuturesRecommendationEvaluationError):
        FuturesEvaluationEvidence(_SYNTHETIC, "dataset")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "category",
    [c for c in FuturesHistoricalDatasetSourceCategory if c is not _SYNTHETIC],
)
def test_a_real_source_with_a_permission_string_is_still_not_evidence(
    category: FuturesHistoricalDatasetSourceCategory,
) -> None:
    manifest = _manifest(source=_source(category=category, rights_reference="approved"))
    assert manifest.source.rights_verified is False
    assert manifest.is_market_evidence is False
    with pytest.raises(FuturesHistoricalDatasetVersionError, match="SYNTHETIC evidence only"):
        manifest.synthetic_evaluation_evidence()


def test_rights_reference_is_required_for_real_and_forbidden_for_synthetic_sources() -> None:
    with pytest.raises(InvalidFuturesHistoricalDatasetManifestError, match="written data-use"):
        _source(category=FuturesHistoricalDatasetSourceCategory.LICENSED_VENDOR)
    with pytest.raises(InvalidFuturesHistoricalDatasetManifestError, match="needs no"):
        _source(rights_reference="approved")
    assert _source().rights_verified is False


def test_there_is_no_approval_toggle() -> None:
    names = {
        f.name
        for cls in (Manifest, FuturesHistoricalDatasetSource)
        for f in dataclasses.fields(cls)
    }
    assert not {n for n in names if "approv" in n or "authoriz" in n or "verified" in n}


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def test_canonical_json_round_trips_to_an_equal_manifest() -> None:
    for manifest in (
        _manifest(),
        _revision(),
        _verified(_bars()[1:], _all_expected()),
        _manifest(source=_source(release_reference=None)),
    ):
        text = manifest.canonical_json()
        decoded = parse_futures_historical_dataset_manifest(text)
        assert decoded == manifest
        assert decoded.canonical_json() == text
        assert decoded.fingerprint == manifest.fingerprint


def test_canonical_json_shape_dates_instants_and_explicit_nulls() -> None:
    manifest = _manifest(source=_source(acquired_at=PointInTime("2026-10-16T17:30:00+05:30")))
    record = json.loads(manifest.canonical_json())
    assert record["source"]["acquired_at"] == "2026-10-16T12:00:00Z"
    assert record["source"]["rights_reference"] is None
    assert record["supersedes"] is None
    assert record["coverage"]["expected_sessions"] is None
    assert record["coverage"]["first_trading_date"] == "2026-10-05"
    assert record["fingerprint"] == manifest.fingerprint
    assert '"supersedes":null' in manifest.canonical_json()
    assert " " not in manifest.canonical_json().replace("northstar-test-fixture", "")


def test_no_numbers_and_no_price_payloads_are_serialized() -> None:
    text = _manifest().canonical_json()
    record = json.loads(text)

    def leaves(value: object) -> list[object]:
        if isinstance(value, dict):
            return [leaf for v in value.values() for leaf in leaves(v)]
        if isinstance(value, list):
            return [leaf for v in value for leaf in leaves(v)]
        return [value]

    assert all(leaf is None or isinstance(leaf, str) for leaf in leaves(record))
    assert set(record["bars"][0]) == {"expiration", "trading_date", "fingerprint"}
    assert "24400" not in text
    for word in ("open", "high", "low", "close", "volume", "price"):
        assert f'"{word}' not in text


def _mutated(mutate: object) -> str:
    record = json.loads(_manifest().canonical_json())
    mutate(record)  # type: ignore[operator]
    return json.dumps(record, sort_keys=True, separators=(",", ":"))


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda r: r.update(extra="x"), "exactly the fields"),
        (lambda r: r.pop("supersedes"), "exactly the fields"),
        (lambda r: r["source"].update(verified="yes"), "exactly the fields"),
        (lambda r: r.update(schema="northstar.futures-historical-dataset-manifest/2"), "schema"),
        (lambda r: r.update(fingerprint="fhd1-sha256:" + "0" * 64), "does not match"),
        (lambda r: r.update(fingerprint="sha256:abc"), "fhd1-sha256"),
        (lambda r: r.update(dataset_id=1), "JSON number"),
        (lambda r: r["coverage"].update(first_trading_date="2026-10-5"), "YYYY-MM-DD"),
        (lambda r: r["coverage"].update(first_trading_date="2026-10-04"), "exactly"),
        (lambda r: r["bars"][0].update(fingerprint="fdc1-sha256:" + "f" * 64), "does not match"),
        (lambda r: r["bars"].reverse(), "canonical form"),
        (lambda r: r["source"].update(category="LICENSED"), "invalid"),
        (lambda r: r.update(contracts="2026-10-27"), "JSON array"),
    ],
)
def test_strict_decoding_refuses_non_canonical_or_unsupported_input(
    mutate: object, message: str
) -> None:
    with pytest.raises(InvalidFuturesHistoricalDatasetManifestError, match=message):
        parse_futures_historical_dataset_manifest(_mutated(mutate))


def test_strict_decoding_refuses_floats_duplicates_whitespace_and_non_json() -> None:
    text = _manifest().canonical_json()
    with pytest.raises(InvalidFuturesHistoricalDatasetManifestError, match="JSON number"):
        parse_futures_historical_dataset_manifest(
            text.replace('"supersedes":null', '"supersedes":1.5')
        )
    with pytest.raises(InvalidFuturesHistoricalDatasetManifestError, match="repeats"):
        parse_futures_historical_dataset_manifest('{"schema":"a","schema":"b"}')
    with pytest.raises(InvalidFuturesHistoricalDatasetManifestError, match="canonical form"):
        parse_futures_historical_dataset_manifest(text + "\n")
    with pytest.raises(InvalidFuturesHistoricalDatasetManifestError, match="valid JSON"):
        parse_futures_historical_dataset_manifest(text[:-1])
    with pytest.raises(InvalidFuturesHistoricalDatasetManifestError):
        parse_futures_historical_dataset_manifest(text.encode())  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------


def test_the_manifest_module_has_no_outward_dependency_clock_or_io() -> None:
    path = Path(services.__file__).parent / "futures_historical_dataset_manifest.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported = {
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module
    } | {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    for name in imported:
        assert not name.startswith(
            ("sqlite3", "northstar_infrastructure", "northstar_api", "upstox", "requests", "httpx",
             "socket", "urllib", "time", "random", "uuid", "os", "pathlib", "shutil", "pickle",
             "northstar_core.recommendations", "northstar_core.paper_trading",
             "northstar_core.broker_execution")
        ), name  # fmt: skip
    called = {
        node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    }
    assert not called & {"now", "utcnow", "today", "open", "write_text", "write_bytes", "repr"}


def test_public_exports() -> None:
    for name in (
        "FuturesHistoricalDatasetManifest",
        "FuturesHistoricalDatasetBarReference",
        "FuturesHistoricalDatasetCoverage",
        "FuturesHistoricalDatasetSource",
        "FuturesHistoricalDatasetSession",
        "FuturesHistoricalDatasetSourceCategory",
        "FuturesHistoricalDatasetVolumeUnit",
        "FuturesHistoricalDatasetCompleteness",
        "FuturesHistoricalDatasetVersionError",
        "InvalidFuturesHistoricalDatasetManifestError",
        "parse_futures_historical_dataset_manifest",
        "verify_futures_historical_dataset_catalog",
        "verify_futures_historical_dataset_revision",
        "FUTURES_HISTORICAL_DATASET_MANIFEST_SCHEMA",
        "FUTURES_HISTORICAL_DATASET_FINGERPRINT_NAMESPACE",
        "FUTURES_HISTORICAL_DATASET_FINGERPRINT_PREFIX",
    ):
        assert name in services.__all__
        assert getattr(services, name) is not None
