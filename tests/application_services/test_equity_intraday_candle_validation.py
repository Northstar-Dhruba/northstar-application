"""EI-2A: validation and finality gating of synthetic 5-minute equity candles.

Everything here is invented. The research window comes from the real EI-1C-3
use case over in-test port doubles on a 2031 label, shaped like the NSE common
boundary (09:15 to 15:15 IST) only to exercise the arithmetic. Every candle,
source identity and finality statement is synthetic test evidence; none is
NSE or provider data, and no real finality source is approved anywhere.

An acceptable candle is only eligible for later accepted persistence (EI-2B).
Nothing is persisted, and nothing implies a trade, fill or profit.
"""

from __future__ import annotations

import ast
import inspect
from datetime import UTC, date, datetime, timedelta, timezone
from decimal import Decimal

import pytest
from northstar_core.domain.value_objects import ListingReference
from northstar_core.foundation.value_objects import (
    Currency,
    ExchangeCode,
    PointInTime,
    Price,
    Quantity,
    Symbol,
    Timeframe,
)
from northstar_core.market_data import HistoricalOHLCVBar

from northstar_application import application_services
from northstar_application.application_services import (
    NO_APPROVED_EQUITY_CANDLE_FINALITY,
    EquityCandleCoverage,
    EquityCandleFinalityEvidence,
    EquityCandleFinalityPolicy,
    EquityCandleFinding,
    EquityCandleIntervalOutcome,
    EquityCandleObservation,
    EvaluateEquityResearchWindowUseCase,
    InvalidEquityCandleEvidenceError,
    ValidateEquityIntradayCandlesUseCase,
)
from northstar_application.application_services import (
    equity_intraday_candle_validation as validation_module,
)
from northstar_application.ports import (
    EquityListingApplicability,
    EquityListingApplicabilityClassification,
    EquityListingApplicabilitySource,
    EquityVenueDayClassification,
    EquityVenueDayReference,
    EquityVenueDayReferenceSource,
)

_NSE = ExchangeCode("NSE")
_INR = Currency("INR")
_DAY = date(2031, 3, 4)
_IST = timezone(timedelta(hours=5, minutes=30))
_LISTING = ListingReference(Symbol("TESTCO"), _NSE)
_M_AND_M = ListingReference(Symbol("M&M"), _NSE)
_FIRST_END = datetime(2031, 3, 4, 3, 50, tzinfo=UTC)
_SOURCE = "synthetic-source"
_SYNTHETIC = EquityCandleFinalityPolicy(
    "synthetic-ei2a-test/1", frozenset({_SOURCE}), permits_synthetic=True
)
_FINDING = EquityCandleFinding
_OUTCOME = EquityCandleIntervalOutcome
_COVERAGE = EquityCandleCoverage


class _Venue(EquityVenueDayReferenceSource):
    def venue_day(self, exchange_code: ExchangeCode, trading_date: date) -> EquityVenueDayReference:
        return EquityVenueDayReference(
            exchange_code=exchange_code,
            trading_date=trading_date,
            classification=EquityVenueDayClassification.NORMAL_SESSION,
            common_opens_at=PointInTime("2031-03-04T03:45:00Z"),
            common_closes_at=PointInTime("2031-03-04T09:45:00Z"),
            utc_offset=timedelta(hours=5, minutes=30),
            reference="test-venue/1",
            reason="test normal session",
        )


class _Applicability(EquityListingApplicabilitySource):
    def __init__(self, eligible: bool = True) -> None:
        self.eligible = eligible

    def applicability(
        self, listing: ListingReference, trading_date: date
    ) -> EquityListingApplicability:
        classification = (
            EquityListingApplicabilityClassification.ELIGIBLE_FOR_NORMAL_RESEARCH_WINDOW
            if self.eligible
            else EquityListingApplicabilityClassification.UNKNOWN
        )
        return EquityListingApplicability(
            listing, trading_date, classification, "test-applicability/1", "test"
        )


def _pit(value: datetime) -> PointInTime:
    return PointInTime(value.isoformat().replace("+00:00", "Z"))


def _end(index: int) -> datetime:
    """Return the UTC end of the ``index``-th window interval, counting from 1."""
    return _FIRST_END + timedelta(minutes=5 * (index - 1))


def _window(as_of: PointInTime, listing: ListingReference = _LISTING, eligible: bool = True):  # type: ignore[no-untyped-def]
    use_case = EvaluateEquityResearchWindowUseCase(_Venue(), _Applicability(eligible))
    return use_case.execute(listing, _DAY, as_of)


def _bar(
    end: datetime,
    *,
    listing: ListingReference = _LISTING,
    prices: tuple[str, str, str, str] = ("100", "101", "99", "100.5"),
    volume: str = "1000",
    timeframe: str = "5m",
    currency: Currency = _INR,
    adjusted_close: str | None = None,
) -> HistoricalOHLCVBar:
    o, h, low, c = (Price(Decimal(p), currency) for p in prices)
    return HistoricalOHLCVBar(
        symbol=listing.symbol,
        exchange_code=listing.exchange_code,
        point_in_time=_pit(end),
        timeframe=Timeframe(timeframe),
        open=o,
        high=h,
        low=low,
        close=c,
        volume=Quantity(Decimal(volume)),
        adjusted_close=None if adjusted_close is None else Price(Decimal(adjusted_close), currency),
    )


def _final(
    end: datetime,
    *,
    source: str = _SOURCE,
    event: str | None = None,
    revision: str = "r1",
    confirmed: datetime | None = None,
    synthetic: bool = True,
) -> EquityCandleFinalityEvidence:
    return EquityCandleFinalityEvidence(
        source_id=source,
        source_event_id=event or f"evt-{end.isoformat()}",
        source_revision=revision,
        confirmed_at=_pit(confirmed or end),
        evidence_reference="synthetic test statement",
        is_synthetic=synthetic,
    )


_DEFAULT = object()


def _obs(
    end: datetime,
    *,
    bar: HistoricalOHLCVBar | None = None,
    source: str = _SOURCE,
    event: str | None = None,
    revision: str = "r1",
    received: datetime | None = None,
    finality: object = _DEFAULT,
    corrects: str | None = None,
    **bar_overrides: object,
) -> EquityCandleObservation:
    event = event or f"evt-{end.isoformat()}"
    return EquityCandleObservation(
        bar=bar or _bar(end, **bar_overrides),  # type: ignore[arg-type]
        source_id=source,
        source_event_id=event,
        source_revision=revision,
        received_at=_pit(received or end),
        finality=_final(end, event=event, revision=revision) if finality is _DEFAULT else finality,  # type: ignore[arg-type]
        corrects_revision=corrects,
    )


def _validate(
    observations: list[EquityCandleObservation],
    as_of: datetime,
    *,
    policy: EquityCandleFinalityPolicy = _SYNTHETIC,
    listing: ListingReference = _LISTING,
    eligible: bool = True,
):  # type: ignore[no-untyped-def]
    instant = _pit(as_of)
    use_case = ValidateEquityIntradayCandlesUseCase(_INR, policy)
    return use_case.execute(_window(instant, listing, eligible), observations, instant)


_LATE = datetime(2031, 3, 4, 12, 0, tzinfo=UTC)


# ----------------------------------------------------------------------
# A. Valid OHLCV and expected intervals
# ----------------------------------------------------------------------


def test_valid_synthetic_inr_candle_is_acceptable_for_its_interval() -> None:
    observation = _obs(_end(1))
    result = _validate([observation], _LATE)
    first = result.intervals[0]
    assert first.outcome is _OUTCOME.ACCEPTABLE
    assert first.acceptable == (observation,)
    assert first.observations[0].finding is _FINDING.ACCEPTABLE
    assert first.interval.closes_at == PointInTime("2031-03-04T03:50:00Z")
    assert _FIRST_END.astimezone(_IST).time().isoformat() == "09:20:00"
    assert result.unmatched == ()


def test_expected_intervals_are_the_69_window_intervals_of_the_ei1a_grid() -> None:
    result = _validate([], _LATE)
    assert len(result.intervals) == 69
    assert result.intervals[0].interval.closes_at == _pit(_end(1))
    assert result.intervals[-1].interval.closes_at == _pit(_end(69))
    assert _end(69).astimezone(_IST).time().isoformat() == "15:00:00"
    window = _window(_pit(_LATE))
    assert tuple(a.interval for a in result.intervals) == window.intervals


def test_ampersand_listing_is_accepted_under_its_canonical_identity() -> None:
    result = _validate([_obs(_end(1), listing=_M_AND_M)], _LATE, listing=_M_AND_M)
    assert result.listing == _M_AND_M
    assert result.intervals[0].outcome is _OUTCOME.ACCEPTABLE


def test_flat_prices_and_zero_volume_are_valid_boundary_values() -> None:
    observation = _obs(_end(1), prices=("100", "100", "100", "100"), volume="0")
    assert _validate([observation], _LATE).intervals[0].outcome is _OUTCOME.ACCEPTABLE


# ----------------------------------------------------------------------
# B. Invalid OHLCV: structural rules are the Core bar's, reused
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "prices",
    [("100", "99", "98", "100"), ("100", "101", "100.5", "100"), ("100", "101", "102", "101")],
)
def test_invalid_high_low_relationships_are_refused_by_the_core_bar(
    prices: tuple[str, str, str, str],
) -> None:
    with pytest.raises(ValueError, match="high|low"):
        _bar(_end(1), prices=prices)


@pytest.mark.parametrize(
    "overrides",
    [
        {"prices": ("-1", "101", "99", "100")},
        {"prices": ("NaN", "101", "99", "100")},
        {"prices": ("100", "Infinity", "99", "100")},
        {"volume": "-1"},
        {"volume": "NaN"},
    ],
)
def test_negative_or_nonfinite_values_are_refused_by_core_values(
    overrides: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        _bar(_end(1), **overrides)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("overrides", "finding"),
    [
        ({"timeframe": "1m"}, _FINDING.WRONG_TIMEFRAME),
        ({"listing": ListingReference(Symbol("OTHERCO"), _NSE)}, _FINDING.WRONG_LISTING),
        (
            {"listing": ListingReference(Symbol("TESTCO"), ExchangeCode("BSE"))},
            _FINDING.WRONG_LISTING,
        ),
        ({"currency": Currency("USD")}, _FINDING.WRONG_CURRENCY),
        ({"adjusted_close": "100"}, _FINDING.ADJUSTED_VALUE_NOT_ACCEPTED),
    ],
)
def test_candles_for_another_identity_unit_or_basis_are_unmatched(
    overrides: dict[str, object], finding: EquityCandleFinding
) -> None:
    result = _validate([_obs(_end(1), **overrides)], _LATE)
    assert [a.finding for a in result.unmatched] == [finding]
    assert result.intervals[0].outcome is _OUTCOME.MISSING


@pytest.mark.parametrize(
    ("end", "finding"),
    [
        (_end(1) + timedelta(days=1), _FINDING.OUTSIDE_RESEARCH_WINDOW),  # another session date
        (_end(1) - timedelta(minutes=10), _FINDING.OUTSIDE_RESEARCH_WINDOW),  # 09:10, pre-open
        (_end(1) - timedelta(minutes=5), _FINDING.OUTSIDE_RESEARCH_WINDOW),  # 09:15, window open
        (_end(69) + timedelta(minutes=5), _FINDING.OUTSIDE_RESEARCH_WINDOW),  # 15:05, outside
        (_end(69) + timedelta(minutes=25), _FINDING.OUTSIDE_RESEARCH_WINDOW),  # 15:25, auction
        (_end(1) + timedelta(minutes=2), _FINDING.MISALIGNED_INTERVAL_END),
        (_end(1) + timedelta(microseconds=1), _FINDING.MISALIGNED_INTERVAL_END),
    ],
)
def test_candles_not_ending_on_an_expected_interval_are_unmatched(
    end: datetime, finding: EquityCandleFinding
) -> None:
    observation = _obs(end, received=end, finality=_final(end, confirmed=end))
    result = _validate([observation], end + timedelta(days=2) if end.day == 5 else _LATE)
    assert [a.finding for a in result.unmatched] == [finding]
    assert result.acceptable_intervals == ()


# ----------------------------------------------------------------------
# C. Finality
# ----------------------------------------------------------------------


def test_ended_interval_without_finality_evidence_is_unfinalized() -> None:
    result = _validate([_obs(_end(1), finality=None)], _LATE)
    assert result.intervals[0].observations[0].finding is _FINDING.FINALITY_MISSING
    assert result.intervals[0].outcome is _OUTCOME.UNFINALIZED
    assert result.unfinalized_intervals == (result.intervals[0].interval,)


def test_a_still_forming_interval_is_not_ended_even_with_a_candle() -> None:
    as_of = _end(1) - timedelta(microseconds=1)
    result = _validate([_obs(_end(1))], as_of)
    assert result.intervals[0].outcome is _OUTCOME.NOT_ENDED
    assert result.intervals[0].observations[0].finding is _FINDING.INTERVAL_NOT_ENDED


@pytest.mark.parametrize(
    "build",
    [
        lambda: _obs(_end(1), source=""),
        lambda: _obs(_end(1), event=" "),
        lambda: _obs(_end(1), revision=""),
        lambda: _final(_end(1), source=""),
        lambda: _obs(_end(1), corrects="r1"),
    ],
)
def test_missing_source_identity_is_refused(build: object) -> None:
    with pytest.raises(InvalidEquityCandleEvidenceError):
        build()  # type: ignore[operator]


@pytest.mark.parametrize(
    ("finality", "finding"),
    [
        (_final(_end(1), source="other-source"), _FINDING.FINALITY_SOURCE_MISMATCH),
        (_final(_end(1), event="other-event"), _FINDING.FINALITY_EVENT_MISMATCH),
        (_final(_end(1), revision="r2"), _FINDING.FINALITY_REVISION_MISMATCH),
        (
            _final(_end(1), confirmed=_end(1) - timedelta(seconds=1)),
            _FINDING.FINALITY_BEFORE_INTERVAL_END,
        ),
        (
            _final(_end(1), confirmed=_LATE + timedelta(seconds=1)),
            _FINDING.FINALITY_NOT_CONFIRMED_AS_OF,
        ),
    ],
)
def test_invalid_finality_provenance_or_timing_is_unfinalized(
    finality: EquityCandleFinalityEvidence, finding: EquityCandleFinding
) -> None:
    result = _validate([_obs(_end(1), finality=finality)], _LATE)
    assert result.intervals[0].observations[0].finding is finding
    assert result.intervals[0].outcome is _OUTCOME.UNFINALIZED


def test_production_policy_refuses_synthetic_and_unapproved_finality() -> None:
    synthetic = _validate([_obs(_end(1))], _LATE, policy=NO_APPROVED_EQUITY_CANDLE_FINALITY)
    assert (
        synthetic.intervals[0].observations[0].finding is _FINDING.SYNTHETIC_FINALITY_NOT_PERMITTED
    )
    real = _obs(_end(1), finality=_final(_end(1), synthetic=False))
    result = _validate([real], _LATE, policy=NO_APPROVED_EQUITY_CANDLE_FINALITY)
    assert result.intervals[0].observations[0].finding is _FINDING.FINALITY_SOURCE_NOT_APPROVED
    assert result.coverage is _COVERAGE.INCOMPLETE_AS_OF
    assert NO_APPROVED_EQUITY_CANDLE_FINALITY.approved_source_ids == frozenset()


def test_unapproved_source_under_a_synthetic_policy_is_refused() -> None:
    observation = _obs(
        _end(1), source="other-source", finality=_final(_end(1), source="other-source")
    )
    result = _validate([observation], _LATE)
    assert result.intervals[0].observations[0].finding is _FINDING.FINALITY_SOURCE_NOT_APPROVED


def test_a_synthetic_policy_must_say_it_is_synthetic() -> None:
    with pytest.raises(InvalidEquityCandleEvidenceError, match="synthetic"):
        EquityCandleFinalityPolicy("approved/1", frozenset({_SOURCE}), permits_synthetic=True)


def test_receipt_timing_is_point_in_time() -> None:
    late = _obs(_end(1), received=_LATE + timedelta(seconds=1))
    result = _validate([late], _LATE)
    assert result.intervals[0].outcome is _OUTCOME.MISSING
    assert result.intervals[0].observations[0].finding is _FINDING.NOT_RECEIVED_AS_OF
    early = _obs(_end(1), received=_end(1) - timedelta(seconds=1))
    result = _validate([early], _LATE)
    assert result.intervals[0].observations[0].finding is _FINDING.RECEIVED_BEFORE_INTERVAL_END
    assert result.intervals[0].outcome is _OUTCOME.UNFINALIZED


# ----------------------------------------------------------------------
# D. Duplicates and conflicts
# ----------------------------------------------------------------------


def test_identical_repeated_evidence_is_acceptable_without_preference() -> None:
    observation = _obs(_end(1))
    result = _validate([observation, observation], _LATE)
    assert result.intervals[0].outcome is _OUTCOME.ACCEPTABLE
    assert result.intervals[0].acceptable == (observation, observation)


def _outcomes_in_every_order(
    observations: list[EquityCandleObservation], as_of: datetime = _LATE
) -> set[EquityCandleIntervalOutcome]:
    orders = (observations, list(reversed(observations)))
    return {_validate(order, as_of).intervals[0].outcome for order in orders}


def test_identical_repeats_under_the_same_identity_are_acceptable_in_any_order() -> None:
    first = _obs(_end(1))
    repeat = _obs(_end(1), received=_end(1) + timedelta(seconds=30))
    assert first.source_revision == repeat.source_revision
    assert first.source_event_id == repeat.source_event_id
    assert _outcomes_in_every_order([first, repeat]) == {_OUTCOME.ACCEPTABLE}
    result = _validate([first, repeat], _LATE)
    assert result.intervals[0].acceptable == (first, repeat)


def test_identical_values_under_different_revisions_are_unresolved() -> None:
    first, second = _obs(_end(1), revision="r1"), _obs(_end(1), revision="r2")
    assert first.values == second.values
    assert _outcomes_in_every_order([first, second]) == {_OUTCOME.UNRESOLVED_REVISIONS}
    result = _validate([first, second], _LATE)
    assert result.intervals[0].acceptable == ()
    assert [a.observation for a in result.intervals[0].observations] == [first, second]
    assert [a.finding for a in result.intervals[0].observations] == [
        _FINDING.ACCEPTABLE,
        _FINDING.ACCEPTABLE,
    ]


def test_identical_values_under_different_source_events_are_unresolved() -> None:
    first = _obs(_end(1), event="evt-a")
    second = _obs(_end(1), event="evt-b")
    assert _outcomes_in_every_order([first, second]) == {_OUTCOME.UNRESOLVED_REVISIONS}


def test_identical_values_from_different_sources_are_unresolved() -> None:
    approved = EquityCandleFinalityPolicy(
        "synthetic-ei2a-two-sources/1", frozenset({_SOURCE, "synthetic-other"}), True
    )
    first = _obs(_end(1))
    second = _obs(
        _end(1), source="synthetic-other", finality=_final(_end(1), source="synthetic-other")
    )
    for order in ([first, second], [second, first]):
        result = _validate(order, _LATE, policy=approved)
        assert result.intervals[0].outcome is _OUTCOME.UNRESOLVED_REVISIONS


def test_conflicting_values_under_the_same_identity_are_conflicting_in_any_order() -> None:
    first = _obs(_end(1))
    second = _obs(_end(1), prices=("100", "102", "99", "101"))
    assert _outcomes_in_every_order([first, second]) == {_OUTCOME.CONFLICTING_VALUES}


def test_conflicting_values_under_different_revisions_are_unresolved_in_any_order() -> None:
    first = _obs(_end(1), revision="r1")
    second = _obs(_end(1), revision="r2", prices=("100", "102", "99", "101"))
    assert _outcomes_in_every_order([first, second]) == {_OUTCOME.UNRESOLVED_REVISIONS}


def test_an_explicit_correction_is_unresolved_in_any_order() -> None:
    original = _obs(_end(1), revision="r1")
    for correction in (
        _obs(_end(1), revision="r2", corrects="r1"),
        _obs(_end(1), revision="r2", corrects="r1", volume="2000"),
    ):
        assert _outcomes_in_every_order([original, correction]) == {_OUTCOME.UNRESOLVED_REVISIONS}
    alone = _obs(_end(1), revision="r2", corrects="r1")
    assert _outcomes_in_every_order([alone]) == {_OUTCOME.UNRESOLVED_REVISIONS}


@pytest.mark.parametrize("as_of_index", [10, 69])
def test_an_unresolved_revision_blocks_snapshot_completeness(as_of_index: int) -> None:
    candles = [_obs(_end(i)) for i in range(1, as_of_index + 1)]
    candles.append(_obs(_end(5), revision="r2"))
    for order in (candles, list(reversed(candles))):
        result = _validate(order, _end(as_of_index))
        assert result.coverage is _COVERAGE.INCOMPLETE_AS_OF
        assert result.conflicted_intervals == (result.intervals[4].interval,)
        assert len(result.acceptable_intervals) == as_of_index - 1


def test_different_values_under_one_revision_are_conflicting() -> None:
    first = _obs(_end(1))
    second = _obs(_end(1), prices=("100", "102", "99", "101"))
    for order in ([first, second], [second, first]):
        result = _validate(order, _LATE)
        assert result.intervals[0].outcome is _OUTCOME.CONFLICTING_VALUES
        assert result.intervals[0].acceptable == ()
        assert result.conflicted_intervals == (result.intervals[0].interval,)


def test_different_values_across_revisions_are_unresolved_not_ordered() -> None:
    first = _obs(_end(1), revision="r1")
    second = _obs(_end(1), revision="r2", volume="2000", received=_end(1) + timedelta(minutes=1))
    for order in ([first, second], [second, first]):
        result = _validate(order, _LATE)
        assert result.intervals[0].outcome is _OUTCOME.UNRESOLVED_REVISIONS
        assert result.intervals[0].acceptable == ()


def test_correction_evidence_is_recorded_but_never_applied() -> None:
    original = _obs(_end(1), revision="r1")
    correction = _obs(_end(1), revision="r2", corrects="r1", prices=("100", "102", "99", "101"))
    result = _validate([original, correction], _LATE)
    assert result.intervals[0].outcome is _OUTCOME.UNRESOLVED_REVISIONS
    same_values = _obs(_end(1), revision="r2", corrects="r1")
    assert _validate([original, same_values], _LATE).intervals[0].outcome is (
        _OUTCOME.UNRESOLVED_REVISIONS
    )


def test_a_conflict_is_not_hidden_by_an_invalid_observation() -> None:
    good = _obs(_end(1))
    unfinal = _obs(_end(1), prices=("100", "102", "99", "101"), finality=None)
    result = _validate([good, unfinal], _LATE)
    assert result.intervals[0].outcome is _OUTCOME.CONFLICTING_VALUES


def test_missing_intervals_are_reported_and_never_fabricated() -> None:
    observations = [_obs(_end(i)) for i in (1, 2, 4)]
    result = _validate(observations, _end(4))
    assert result.missing_intervals == (result.intervals[2].interval,)
    assert result.intervals[2].observations == ()
    assert result.intervals[2].acceptable == ()


# ----------------------------------------------------------------------
# E. Snapshot completeness and observation time
# ----------------------------------------------------------------------


def _all_candles() -> list[EquityCandleObservation]:
    return [_obs(_end(i)) for i in range(1, 70)]


@pytest.mark.parametrize(
    ("as_of", "ended", "coverage"),
    [
        (datetime(2031, 3, 4, 3, 0, tzinfo=UTC), 0, _COVERAGE.NO_INTERVALS_ENDED),
        (_end(1) - timedelta(microseconds=1), 0, _COVERAGE.NO_INTERVALS_ENDED),
        (_end(1), 1, _COVERAGE.COMPLETE_AS_OF),
        (_end(1) + timedelta(minutes=1), 1, _COVERAGE.COMPLETE_AS_OF),
        (_end(27) + timedelta(seconds=30), 27, _COVERAGE.COMPLETE_AS_OF),
        (_end(69) - timedelta(microseconds=1), 68, _COVERAGE.COMPLETE_AS_OF),
        (_end(69), 69, _COVERAGE.WINDOW_COMPLETE),
        (_end(69) + timedelta(hours=3), 69, _COVERAGE.WINDOW_COMPLETE),
    ],
)
def test_coverage_is_judged_over_ended_intervals_only(
    as_of: datetime, ended: int, coverage: EquityCandleCoverage
) -> None:
    result = _validate(_all_candles(), as_of)
    assert len(result.ended_intervals) == ended
    assert result.coverage is coverage
    assert len(result.acceptable_intervals) == ended


def test_a_missing_candle_within_completed_coverage_is_incomplete() -> None:
    candles = [_obs(_end(i)) for i in range(1, 11) if i != 6]
    result = _validate(candles, _end(10))
    assert result.coverage is _COVERAGE.INCOMPLETE_AS_OF
    assert result.missing_intervals == (result.intervals[5].interval,)


def test_a_full_window_needs_every_interval_through_15_00() -> None:
    result = _validate(_all_candles()[:-1], _end(69))
    assert result.coverage is _COVERAGE.INCOMPLETE_AS_OF
    assert result.missing_intervals == (result.intervals[-1].interval,)


def test_an_ineligible_window_validates_nothing() -> None:
    result = _validate([_obs(_end(1))], _LATE, eligible=False)
    assert result.coverage is _COVERAGE.WINDOW_NOT_ELIGIBLE
    assert result.intervals == ()
    assert [a.finding for a in result.unmatched] == [_FINDING.WINDOW_NOT_ELIGIBLE]


def test_repeated_evaluation_is_deterministic() -> None:
    candles = _all_candles()[:20]
    as_of = _end(25) + timedelta(seconds=7)
    assert _validate(candles, as_of) == _validate(candles, as_of)


def test_one_observation_instant_governs_and_must_be_explicit() -> None:
    use_case = ValidateEquityIntradayCandlesUseCase(_INR, _SYNTHETIC)
    window = _window(_pit(_LATE))
    with pytest.raises(ValueError, match="as_of must equal"):
        use_case.execute(window, [], _pit(_end(3)))
    with pytest.raises(TypeError):
        use_case.execute(window, [], "2031-03-04T12:00:00Z")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        PointInTime("2031-03-04T12:00:00")  # no offset: not timezone-aware


def test_inputs_are_type_checked() -> None:
    use_case = ValidateEquityIntradayCandlesUseCase(_INR, _SYNTHETIC)
    window = _window(_pit(_LATE))
    with pytest.raises(TypeError):
        use_case.execute(window, [_bar(_end(1))], _pit(_LATE))  # type: ignore[list-item]
    with pytest.raises(TypeError):
        ValidateEquityIntradayCandlesUseCase("INR")  # type: ignore[arg-type]


# ----------------------------------------------------------------------
# F. Scope
# ----------------------------------------------------------------------


def test_module_reads_no_clock_and_has_no_persistence_or_provider_coupling() -> None:
    tree = ast.parse(inspect.getsource(validation_module))
    called = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert not called & {"now", "today", "utcnow", "monotonic", "time"}
    imported = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    for module in imported:
        assert module is not None
        assert not module.startswith("northstar_infrastructure")
        for forbidden in ("sqlite", "repository", "store", "ingest", "recommendation", "ports"):
            assert forbidden not in module, module


def test_packages_export_the_validation_contracts() -> None:
    for name in (
        "ValidateEquityIntradayCandlesUseCase",
        "EquityCandleObservation",
        "EquityCandleFinalityEvidence",
        "EquityCandleFinalityPolicy",
        "NO_APPROVED_EQUITY_CANDLE_FINALITY",
    ):
        assert name in application_services.__all__
