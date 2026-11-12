"""EI-5C: synthetic equity recommendation lifecycle evaluation (ADR-024).

These tests run the real stack end to end, in memory:

- the EI-4C generator (``test_generate_equity_trade_recommendation`` fixtures)
  produces a genuinely PROPOSED disposition, which the production verifier
  re-verifies at levels A to C;
- every lifecycle snapshot is a real EI-2B snapshot, issued by
  ``IssueEquityCandleEvidenceSnapshotUseCase`` over the same journal, so every
  interval outcome is EI-2A's own classification;
- the EI-5B Core lifecycle makes every transition.

Nothing here is an issuance. The lifecycle starts ``RECORDED`` only under an
explicit ``synthetic-`` label, and no issued history, ``eiris1`` identity or
persistent event exists. All prices and times are synthetic.

The proposal's synthetic envelope (IST): entry zone [101.5, 102.5], stop 99,
target 109.5; modeled issuance 09:36; ``eligible_from`` 09:40; last entry bar
10:45-10:50; modeled exit 14:00. Scenario identifiers are the EI-5C rows of the
EI-5 design acceptance matrix.
"""

from __future__ import annotations

import ast
import inspect
from dataclasses import dataclass, fields, replace
from datetime import date, datetime, timedelta
from decimal import Decimal

import pytest
import test_generate_equity_trade_recommendation as ei4c
import test_record_equity_recommendation_generation as rec
from northstar_core.foundation.value_objects import PointInTime, Price, Quantity, Timeframe
from northstar_core.market_data import HistoricalOHLCVBar
from northstar_core.recommendations.equity.lifecycle import (
    EquityEvidenceDeficiencyKind,
    EquityIntrabarOrdering,
    EquityLifecycleAdvanceStatus,
    EquityLifecycleEventKind,
    EquityLifecycleIntervalOutcome,
    EquityLifecycleLevel,
    EquityLifecycleState,
    EquityRevisionMateriality,
)

from northstar_application.application_services import (
    NSE_EQUITY_INTRADAY_V1_RESEARCH_WINDOW,
    EquityCandleFinalityEvidence,
    EquityCandleFinalityRecord,
    EquityCandleObservation,
    EquityCandleObservationRecord,
    EquityRecommendationGenerationVerifier,
    EquityRecommendationReplayMismatchError,
    EquityRecommendationSourceUnavailableError,
    EquityRecommendationSourceVerificationError,
    EquitySyntheticLifecycleEvaluation,
    EquitySyntheticLifecycleRefusal,
    EquitySyntheticLifecycleRefusedError,
    EquitySyntheticLifecycleRequest,
    EquitySyntheticLifecycleSnapshotPosition,
    EvaluateEquityResearchWindowUseCase,
    EvaluateSyntheticEquityRecommendationLifecycleUseCase,
    IncompatibleEquityRecommendationRecordError,
    InvalidEquityRecommendationRecordError,
    InvalidEquitySyntheticLifecycleRequestError,
    VerifyStoredEquityRecommendationGenerationUseCase,
    equity_recommendation_disposition,
)
from northstar_application.application_services import (
    equity_candle_evidence_snapshot as snapshot_module,
)
from northstar_application.application_services import (
    evaluate_synthetic_equity_recommendation_lifecycle as lifecycle_module,
)
from northstar_application.application_services.equity_intraday_candle_validation import (
    NO_APPROVED_EQUITY_CANDLE_FINALITY,
)
from northstar_application.ports import EquityCandleEvidenceSnapshotStore

K = EquityLifecycleEventKind
S = EquityLifecycleState
Status = EquityLifecycleAdvanceStatus
Outcome = EquityLifecycleIntervalOutcome
Refusal = EquitySyntheticLifecycleRefusal

_LABEL = "synthetic-ei5c-acceptance"
_QUIET = ("103", "104")  # above the zone, below the target: touches nothing
_ENTRY = ("101.8", "102.2")


# ---------------------------------------------------------------------------
# Fixture: a verified stored proposal and an EI-2B world to issue snapshots in
# ---------------------------------------------------------------------------


def _ist(clock: str) -> datetime:
    """Return the UTC instant of an IST wall-clock time on the fixture date."""
    parts = clock.split(":")
    hour, minute = int(parts[0]), int(parts[1])
    second, micro = 0, 0
    if len(parts) == 3:
        whole, _, fraction = parts[2].partition(".")
        second, micro = int(whole), int(fraction.ljust(6, "0")) if fraction else 0
    return ei4c._ist(hour, minute, second) + timedelta(microseconds=micro)


def _pit(clock: str) -> PointInTime:
    return ei4c._pit(_ist(clock))


def _index(opening: str) -> int:
    hour, minute = (int(part) for part in opening.split(":"))
    return (hour * 60 + minute - (9 * 60 + 15)) // 5 + 1


def _openings(start: str, end: str) -> list[str]:
    current, stop = datetime.strptime(start, "%H:%M"), datetime.strptime(end, "%H:%M")
    found = []
    while current < stop:
        found.append(current.strftime("%H:%M"))
        current += timedelta(minutes=5)
    return found


@dataclass
class _Scenario:
    world: ei4c._World
    ports: ei4c._Ports
    disposition: rec.Disposition
    store: rec._Store
    use_case: EvaluateSyntheticEquityRecommendationLifecycleUseCase

    @property
    def key(self) -> str:
        return self.disposition.generation_key

    def put(
        self,
        opening: str,
        bar: tuple[str, str],
        *,
        received: float = 5,
        confirmed: float | None = 10,
        revision: str = "r1",
        suffix: str = "",
    ) -> None:
        """Journal one synthetic candle, and its finality unless ``confirmed`` is None."""
        index = _index(opening)
        end, event = ei4c._end(index), f"evt-{index}"
        low, high = (Price(Decimal(value), ei4c._INR) for value in bar)
        candle = HistoricalOHLCVBar(
            symbol=ei4c._LISTING.symbol,
            exchange_code=ei4c._NSE,
            point_in_time=ei4c._pit(end),
            timeframe=Timeframe("5m"),
            open=low,
            high=high,
            low=low,
            close=high,
            volume=Quantity(Decimal("1000")),
        )
        observed = ei4c._pit(end + timedelta(seconds=received))
        self.world.journal.append(
            EquityCandleObservationRecord(
                f"obs-{event}-{revision}{suffix}",
                EquityCandleObservation(candle, ei4c._SOURCE, event, revision, observed, None),
                True,
            )
        )
        if confirmed is not None:
            final = ei4c._pit(end + timedelta(seconds=confirmed))
            self.world.journal.append(
                EquityCandleFinalityRecord(
                    f"fin-{event}-{revision}{suffix}",
                    EquityCandleFinalityEvidence(
                        ei4c._SOURCE, event, revision, final, "synthetic", True
                    ),
                    final,
                )
            )

    def fill(self, start: str, end: str, bars: dict[str, tuple[str, str] | None] | None = None):  # type: ignore[no-untyped-def]
        """Journal quiet candles from ``start`` up to ``end``; ``bars`` override or omit (None)."""
        bars = bars or {}
        for opening in _openings(start, end):
            bar = bars.get(opening, _QUIET)
            if bar is not None:
                self.put(opening, bar)

    def snap(self, clock: str) -> str:
        return self.world.issue(_ist(clock)).snapshot_id

    def run(self, *snapshot_ids: str) -> EquitySyntheticLifecycleEvaluation:
        return self.use_case.execute(
            EquitySyntheticLifecycleRequest(_LABEL, self.key, tuple(snapshot_ids))
        )

    def refused(self, request: EquitySyntheticLifecycleRequest) -> Refusal:
        with pytest.raises(EquitySyntheticLifecycleRefusedError) as raised:
            self.use_case.execute(request)
        return raised.value.reason


def _verifier(world: ei4c._World) -> EquityRecommendationGenerationVerifier:
    return EquityRecommendationGenerationVerifier(world.timelines, world.snapshots)


def _scenario() -> _Scenario:
    world = ei4c._World({index: ei4c._MORNING[index] for index in range(1, 6)})
    world.commit(ei4c._ist(9, 31), ei4c._ist(9, 36), ei4c._ist(9, 41))
    ports = ei4c._Ports(world)
    disposition = equity_recommendation_disposition(ports.run())
    assert disposition is not None and disposition.proposal is not None
    store = rec._Store()
    store.dispositions[disposition.generation_key] = disposition
    verify = VerifyStoredEquityRecommendationGenerationUseCase(store, _verifier(world))
    use_case = EvaluateSyntheticEquityRecommendationLifecycleUseCase(verify, world.snapshots)
    return _Scenario(world, ports, disposition, store, use_case)


def _kinds(evaluation: EquitySyntheticLifecycleEvaluation) -> list[EquityLifecycleEventKind]:
    return [event.kind for event in evaluation.new_events]


def _active(scenario: _Scenario) -> EquitySyntheticLifecycleEvaluation:
    """Activate on the bar 09:45-09:50; the cursor is then 09:50."""
    scenario.fill("09:40", "09:50", {"09:45": _ENTRY})
    evaluation = scenario.run(scenario.snap("09:51"))
    assert evaluation.state is S.ACTIVE
    return evaluation


def _then(
    scenario: _Scenario, before: EquitySyntheticLifecycleEvaluation, *snapshot_ids: str
) -> EquitySyntheticLifecycleEvaluation:
    return scenario.use_case.execute(
        EquitySyntheticLifecycleRequest.following(before, tuple(snapshot_ids))
    )


def test_the_fixture_proposal_matches_the_documented_synthetic_envelope() -> None:
    scenario = _scenario()
    window = scenario.run(scenario.snap("09:41")).window
    levels = window.levels
    assert (levels.entry_zone.low.amount, levels.entry_zone.high.amount) == (
        Decimal("101.5"),
        Decimal("102.5"),
    )
    assert (levels.stop.amount, levels.target.amount) == (Decimal("99"), Decimal("109.5"))
    assert window.eligible_from == _pit("09:40")
    assert window.latest_entry_bar_opens_at == _pit("10:45")
    assert window.entry_window_ends_at == _pit("10:50")
    assert window.modeled_exit_at == _pit("14:00")


# ---------------------------------------------------------------------------
# A. Stored-generation verification (EI-4D-A levels A to C, reused unchanged)
# ---------------------------------------------------------------------------


def test_a_verified_proposed_disposition_is_evaluated_with_its_provenance() -> None:
    scenario = _scenario()
    snapshot_id = scenario.snap("09:41")
    evaluation = scenario.run(snapshot_id)
    assert evaluation.generation_key == scenario.key
    assert evaluation.proposal_fingerprint == scenario.disposition.proposal_fingerprint
    assert evaluation.synthetic_label == _LABEL
    assert evaluation.last_evaluated == EquitySyntheticLifecycleSnapshotPosition(
        snapshot_id, _pit("09:41"), scenario.world.journal.watermark()
    )


@pytest.mark.parametrize("kind", ["not_eligible", "unresolved"])
def test_not_eligible_and_unresolved_dispositions_are_refused(kind: str) -> None:
    scenario = _scenario()
    audit = (rec._not_eligible if kind == "not_eligible" else rec._unresolved)(scenario.ports)
    other = equity_recommendation_disposition(audit)
    assert other is not None
    scenario.store.dispositions[other.generation_key] = other
    request = EquitySyntheticLifecycleRequest(
        _LABEL, other.generation_key, (scenario.snap("09:41"),)
    )
    assert scenario.refused(request) is Refusal.DISPOSITION_NOT_PROPOSED


def test_a_missing_disposition_is_refused() -> None:
    scenario = _scenario()
    request = EquitySyntheticLifecycleRequest(_LABEL, "eirk1-sha256:" + "0" * 64, ("x",))
    assert scenario.refused(request) is Refusal.GENERATION_NOT_FOUND


@pytest.mark.parametrize(
    ("failure", "error"),
    [
        ("undecodable", InvalidEquityRecommendationRecordError),
        ("incompatible", IncompatibleEquityRecommendationRecordError),
    ],
    ids=["corrupted-or-fingerprint-mismatch", "evaluator-version-conflict"],
)
def test_an_undecodable_or_incompatible_stored_record_fails_closed(
    failure: str, error: type[Exception]
) -> None:
    scenario = _scenario()
    scenario.store.load_failure = failure
    with pytest.raises(error):
        scenario.run(scenario.snap("09:41"))


def test_a_proposal_whose_documents_no_longer_reconcile_cannot_be_constructed() -> None:
    scenario = _scenario()
    with pytest.raises(InvalidEquityRecommendationRecordError):
        rec._rebuilt(scenario.disposition, proposal_fingerprint="eirp1-sha256:" + "0" * 64)
    with pytest.raises(InvalidEquityRecommendationRecordError):
        rec._rebuilt(scenario.disposition, proposal_document=None, proposal_fingerprint=None)


def test_invalid_candidate_linkage_fails_source_verification() -> None:
    scenario = _scenario()
    relinked = rec._manifest_relinked(scenario.disposition)
    scenario.store.dispositions[scenario.key] = relinked
    with pytest.raises(EquityRecommendationSourceVerificationError):
        scenario.run(scenario.snap("09:41"))


def test_a_stored_record_that_does_not_replay_fails_closed() -> None:
    scenario = _scenario()
    original = equity_recommendation_disposition(rec._not_eligible(scenario.ports))
    assert original is not None

    def misstate(document):  # type: ignore[no-untyped-def]
        document["core_outcome"]["outcome"]["reasons"] = ["INSUFFICIENT_RISK_BUDGET"]

    altered = rec._tampered(original, misstate)
    scenario.store.dispositions[altered.generation_key] = altered
    request = EquitySyntheticLifecycleRequest(_LABEL, altered.generation_key, ("x",))
    with pytest.raises(EquityRecommendationReplayMismatchError):
        scenario.use_case.execute(request)


def test_unavailable_historical_evidence_is_typed_unavailable_never_substituted() -> None:
    scenario = _scenario()
    snapshot_id = scenario.snap("09:41")
    scenario.world.snapshots.unavailable = True
    with pytest.raises(EquityRecommendationSourceUnavailableError):
        scenario.run(snapshot_id)


# ---------------------------------------------------------------------------
# B. Snapshot resolution (LA-01 to LA-06, SE-03)
# ---------------------------------------------------------------------------


def _foreign_snapshot(scenario: _Scenario, listing=ei4c._LISTING, day=ei4c._DAY) -> str:  # type: ignore[no-untyped-def]
    as_of = ei4c._pit(_ist("09:51"))
    window = EvaluateEquityResearchWindowUseCase(
        scenario.world.venue, ei4c._Applicability(), NSE_EQUITY_INTRADAY_V1_RESEARCH_WINDOW
    ).execute(listing, day, as_of)
    return scenario.world.issuer.execute(window, as_of, as_of).snapshot_id


def test_la05_a_snapshot_of_another_listing_is_refused() -> None:
    scenario = _scenario()
    request = EquitySyntheticLifecycleRequest(
        _LABEL, scenario.key, (_foreign_snapshot(scenario, listing=ei4c._OTHER),)
    )
    assert scenario.refused(request) is Refusal.SNAPSHOT_LISTING_MISMATCH


class _NextDayVenue(ei4c.EquityVenueDayReferenceSource):
    """A NORMAL 09:15-15:30 IST venue day on 2026-08-13, under the same reference."""

    def venue_day(self, exchange_code, trading_date):  # type: ignore[no-untyped-def]
        def at(hour: int, minute: int) -> PointInTime:
            local = datetime(2026, 8, 13, hour, minute, tzinfo=ei4c._IST)
            return ei4c._pit(local.astimezone(ei4c.UTC))

        return ei4c.EquityVenueDayReference(
            exchange_code=exchange_code,
            trading_date=trading_date,
            classification=ei4c.EquityVenueDayClassification.NORMAL_SESSION,
            common_opens_at=at(9, 15),
            common_closes_at=at(15, 30),
            utc_offset=timedelta(hours=5, minutes=30),
            reference="test-venue/1",
            reason="test",
        )


def test_se03_a_next_session_snapshot_is_refused() -> None:
    scenario = _scenario()
    next_day = date(2026, 8, 13)
    as_of = ei4c._pit(datetime(2026, 8, 13, 9, 51, tzinfo=ei4c._IST).astimezone(ei4c.UTC))
    window = EvaluateEquityResearchWindowUseCase(
        _NextDayVenue(), ei4c._Applicability(), NSE_EQUITY_INTRADAY_V1_RESEARCH_WINDOW
    ).execute(ei4c._LISTING, next_day, as_of)
    snapshot_id = scenario.world.issuer.execute(window, as_of, as_of).snapshot_id
    request = EquitySyntheticLifecycleRequest(_LABEL, scenario.key, (snapshot_id,))
    assert scenario.refused(request) is Refusal.SNAPSHOT_SESSION_MISMATCH


def test_a_snapshot_resolved_against_another_venue_reference_is_refused() -> None:
    scenario = _scenario()
    scenario.world.venue.reference = "test-venue/2"
    request = EquitySyntheticLifecycleRequest(_LABEL, scenario.key, (scenario.snap("09:51"),))
    assert scenario.refused(request) is Refusal.SNAPSHOT_SESSION_MISMATCH


def test_an_unknown_snapshot_identity_is_refused() -> None:
    scenario = _scenario()
    request = EquitySyntheticLifecycleRequest(_LABEL, scenario.key, ("ecs1-sha256:" + "0" * 64,))
    assert scenario.refused(request) is Refusal.SNAPSHOT_NOT_FOUND


class _Substituting(EquityCandleEvidenceSnapshotStore):
    """A faulty store that answers another snapshot than the one asked for."""

    def __init__(self, inner: EquityCandleEvidenceSnapshotStore, answer: str) -> None:
        self.inner, self.answer = inner, answer

    def issue(self, snapshot):  # type: ignore[no-untyped-def]
        raise AssertionError("EI-5C never issues snapshots")

    def get_snapshot(self, snapshot_id):  # type: ignore[no-untyped-def]
        return self.inner.get_snapshot(self.answer)


def test_a_store_answering_another_snapshot_is_refused() -> None:
    scenario = _scenario()
    asked, other = scenario.snap("09:41"), scenario.snap("09:42")
    use_case = EvaluateSyntheticEquityRecommendationLifecycleUseCase(
        scenario.use_case._generations,  # type: ignore[attr-defined]
        _Substituting(scenario.world.snapshots, other),
    )
    with pytest.raises(EquitySyntheticLifecycleRefusedError) as raised:
        use_case.execute(EquitySyntheticLifecycleRequest(_LABEL, scenario.key, (asked,)))
    assert raised.value.reason is Refusal.SNAPSHOT_IDENTITY_MISMATCH


def test_la04_a_snapshot_before_the_modeled_issuance_is_refused() -> None:
    scenario = _scenario()
    request = EquitySyntheticLifecycleRequest(_LABEL, scenario.key, (scenario.snap("09:35"),))
    assert scenario.refused(request) is Refusal.SNAPSHOT_BEFORE_ISSUANCE


def test_la02_an_older_snapshot_after_a_newer_one_is_refused() -> None:
    scenario = _scenario()
    newer, older = scenario.snap("10:00"), scenario.snap("09:50")
    request = EquitySyntheticLifecycleRequest(_LABEL, scenario.key, (newer, older))
    assert scenario.refused(request) is Refusal.SNAPSHOT_OUT_OF_ORDER


def test_la03_a_different_snapshot_at_the_same_instant_is_refused() -> None:
    scenario = _scenario()
    first = scenario.snap("10:00")
    scenario.put("09:40", _QUIET)  # the journal grows: another manifest at the same as_of
    second = scenario.snap("10:00")
    assert first != second
    request = EquitySyntheticLifecycleRequest(_LABEL, scenario.key, (first, second))
    assert scenario.refused(request) is Refusal.SNAPSHOT_OUT_OF_ORDER


def test_la03_repeating_the_last_snapshot_is_idempotent() -> None:
    scenario = _scenario()
    scenario.fill("09:40", "09:50", {"09:45": _ENTRY})
    snapshot_id = scenario.snap("09:51")
    once = scenario.run(snapshot_id)
    twice = scenario.run(snapshot_id, snapshot_id)
    assert twice.lifecycle == once.lifecycle
    assert twice.steps[1].repeated and twice.steps[1].events == () and twice.steps[1].status is None
    resumed = _then(scenario, once, snapshot_id)
    assert resumed.new_events == () and resumed.lifecycle == once.lifecycle


def test_la06_a_watermark_regression_is_refused() -> None:
    scenario = _scenario()
    low_watermark = scenario.snap("10:40")
    scenario.fill("09:40", "10:00")
    high_watermark = scenario.snap("10:30")
    request = EquitySyntheticLifecycleRequest(_LABEL, scenario.key, (high_watermark, low_watermark))
    assert scenario.refused(request) is Refusal.SNAPSHOT_WATERMARK_REGRESSION


@pytest.mark.parametrize(
    ("setup", "outcome"),
    [
        ("missing", Outcome.MISSING),
        ("unfinalized", Outcome.UNFINALIZED),
        ("conflicting", Outcome.CONFLICTING_VALUES),
        ("revised", Outcome.UNRESOLVED_REVISIONS),
    ],
)
def test_ev01_to_ev04_real_ei2a_deficiencies_block_monitoring_before_the_boundary(
    setup: str, outcome: EquityLifecycleIntervalOutcome
) -> None:
    scenario = _scenario()
    scenario.fill("09:40", "10:20", {"10:00": None})
    if setup == "unfinalized":
        scenario.put("10:00", _QUIET, confirmed=None)
    elif setup == "conflicting":
        scenario.put("10:00", _QUIET)
        scenario.put("10:00", ("103.1", "104"), confirmed=None, suffix="-b")
    elif setup == "revised":
        scenario.put("10:00", _QUIET)
        scenario.put("10:00", _QUIET, revision="r2")
    evaluation = scenario.run(scenario.snap("10:21"))
    (step,) = evaluation.steps
    assert step.status is Status.BLOCKED and _kinds(evaluation) == [K.ENTRY_WINDOW_OPENED]
    assert step.halted_at is not None and step.halted_at.opens_at == _pit("10:00")
    assert step.halted_at.outcome is outcome
    assert evaluation.state is S.WAITING_FOR_ENTRY


def test_ev05_la01_an_interval_not_ended_at_as_of_is_awaited_without_look_ahead() -> None:
    scenario = _scenario()
    scenario.fill("09:40", "09:50", {"09:45": _ENTRY})  # journaled, but it ends at 09:50
    evaluation = scenario.run(scenario.snap("09:49"))
    (step,) = evaluation.steps
    assert step.status is Status.AWAITING_EVIDENCE
    assert step.halted_at is not None and step.halted_at.outcome is Outcome.NOT_ENDED
    assert _kinds(evaluation) == [K.ENTRY_WINDOW_OPENED]


def test_ev06_a_later_acceptable_bar_is_never_used_past_a_gap() -> None:
    scenario = _scenario()
    scenario.fill("09:40", "10:30", {"09:55": None, "10:00": ("97", "98")})
    evaluation = scenario.run(scenario.snap("10:31"))
    assert evaluation.steps[0].status is Status.BLOCKED
    assert evaluation.state is S.WAITING_FOR_ENTRY


# ---------------------------------------------------------------------------
# C. Lifecycle outcomes through Application orchestration (Core decides)
# ---------------------------------------------------------------------------


def test_lt01_the_first_eligible_snapshot_opens_the_entry_window() -> None:
    scenario = _scenario()
    snapshot_id = scenario.snap("09:41")
    evaluation = scenario.run(snapshot_id)
    (event,) = evaluation.new_events
    assert event.kind is K.ENTRY_WINDOW_OPENED and event.to_state is S.WAITING_FOR_ENTRY
    assert event.evidence.reference == snapshot_id
    assert event.evaluated_at == _pit("09:41")


def test_lt02_a_bar_inside_the_zone_activates_and_cites_its_real_candle() -> None:
    scenario = _scenario()
    evaluation = _active(scenario)
    activation = evaluation.new_events[-1]
    assert activation.kind is K.ENTRY_ACTIVATED
    assert activation.evidence.opens_at == _pit("09:45")
    (candle,) = activation.evidence.observations
    sequence, source, event, revision = candle.split(":")
    assert (source, event, revision) == (ei4c._SOURCE, f"evt-{_index('09:45')}", "r1")
    assert int(sequence) >= 1


@pytest.mark.parametrize(
    ("bar", "kind", "state"),
    [
        (("98.8", "99.5"), K.ENTRY_CANCELLED, S.CANCELLED_BEFORE_ENTRY),  # LT-06, PE-01
        (("97", "98"), K.ENTRY_CANCELLED, S.CANCELLED_BEFORE_ENTRY),  # stop gap-through
        (("98.9", "101.8"), K.ENTRY_INDETERMINATE, S.INDETERMINATE_ENTRY),  # entry + stop
        (("102", "110"), K.ENTRY_INDETERMINATE, S.INDETERMINATE_ENTRY),  # entry + target
        (("102.6", "103"), None, S.WAITING_FOR_ENTRY),  # above the zone: no gap entry
    ],
    ids=["stop-only", "stop-gap-through", "entry-and-stop", "entry-and-target", "above-zone"],
)
def test_waiting_bar_outcomes_are_cores(
    bar: tuple[str, str], kind: EquityLifecycleEventKind | None, state: EquityLifecycleState
) -> None:
    scenario = _scenario()
    scenario.fill("09:40", "09:55", {"09:50": bar})
    evaluation = scenario.run(scenario.snap("09:56"))
    assert evaluation.state is state
    assert _kinds(evaluation)[1:] == ([] if kind is None else [kind])
    if kind is K.ENTRY_INDETERMINATE:
        assert evaluation.new_events[-1].ordering is EquityIntrabarOrdering.UNKNOWN


@pytest.mark.parametrize(
    ("bar", "kind", "state"),
    [
        (("97", "98"), K.STOP_REACHED, S.CLOSED_STOP),  # LT-10 gap-through
        (("110", "111"), K.TARGET_REACHED, S.CLOSED_TARGET),  # LT-09 gap-through
        (("98.9", "110"), K.EXIT_INDETERMINATE, S.INDETERMINATE_EXIT),  # LT-11
    ],
    ids=["stop-gap", "target-gap", "stop-and-target"],
)
def test_active_bar_outcomes_are_cores(
    bar: tuple[str, str], kind: EquityLifecycleEventKind, state: EquityLifecycleState
) -> None:
    scenario = _scenario()
    active = _active(scenario)
    scenario.fill("09:50", "11:00", {"10:30": bar})
    evaluation = _then(scenario, active, scenario.snap("11:01"))
    assert _kinds(evaluation) == [kind] and evaluation.state is state
    if kind is K.EXIT_INDETERMINATE:
        event = evaluation.new_events[0]
        assert event.ordering is EquityIntrabarOrdering.UNKNOWN
        assert event.touched == (EquityLifecycleLevel.STOP, EquityLifecycleLevel.TARGET)


def test_lt07_a_complete_untouched_entry_window_expires() -> None:
    scenario = _scenario()
    scenario.fill("09:40", "10:50")
    evaluation = scenario.run(scenario.snap("10:51"))
    assert _kinds(evaluation) == [K.ENTRY_WINDOW_OPENED, K.ENTRY_WINDOW_EXPIRED]
    assert evaluation.new_events[-1].evidence.opens_at == _pit("10:45")


def test_lt12_complete_active_coverage_reaches_the_session_cutoff() -> None:
    scenario = _scenario()
    active = _active(scenario)
    scenario.fill("09:50", "14:00")
    evaluation = _then(scenario, active, scenario.snap("14:01"))
    assert _kinds(evaluation) == [K.SESSION_CUTOFF_REACHED]
    assert evaluation.new_events[0].evidence.closes_at == _pit("14:00")


@pytest.mark.parametrize(
    ("waiting", "clock", "gap"),
    [(True, "10:51", "10:00"), (False, "14:01", "12:00")],
    ids=["lt08-waiting", "lt13-active"],
)
def test_lt08_lt13_a_gap_still_open_after_the_boundary_is_incomplete_evidence(
    waiting: bool, clock: str, gap: str
) -> None:
    scenario = _scenario()
    if waiting:
        scenario.fill("09:40", "10:50", {gap: None})
        evaluation = scenario.run(scenario.snap(clock))
    else:
        active = _active(scenario)
        scenario.fill("09:50", "14:00", {gap: None})
        evaluation = _then(scenario, active, scenario.snap(clock))
    event = evaluation.new_events[-1]
    assert event.kind is K.EVIDENCE_INCOMPLETE and evaluation.state is S.INCOMPLETE_EVIDENCE
    assert [(d.opens_at, d.kind) for d in event.deficiencies] == [
        (_pit(gap), EquityEvidenceDeficiencyKind.MISSING)
    ]


def test_tb01_tb02_a_temporary_block_resumes_once_the_interval_is_acceptable() -> None:
    scenario = _scenario()
    scenario.fill("09:40", "10:15", {"10:00": None})
    blocked = scenario.run(scenario.snap("10:16"))
    assert blocked.steps[0].status is Status.BLOCKED
    scenario.put("10:00", _QUIET)
    scenario.put("10:15", _ENTRY)
    resumed = _then(scenario, blocked, scenario.snap("10:21"))
    assert _kinds(resumed) == [K.ENTRY_ACTIVATED]
    assert resumed.new_events[0].evidence.opens_at == _pit("10:15")


def test_ew01_ew04_bars_outside_the_entry_window_never_activate() -> None:
    scenario = _scenario()
    # EW-01: the real bar 09:35-09:40 (low 102, high 104) intersects the entry zone, but
    # opens before eligible_from (09:40). EW-04: 10:50 opens after the last entry opening.
    low, high = (Decimal(value) for value in ei4c._MORNING[_index("09:35")][1:3][::-1])
    assert low <= Decimal("102.5") and high >= Decimal("101.5")
    scenario.fill("09:40", "11:00", {"10:50": _ENTRY})
    evaluation = scenario.run(scenario.snap("11:01"))
    assert _kinds(evaluation) == [K.ENTRY_WINDOW_OPENED, K.ENTRY_WINDOW_EXPIRED]
    assert all(
        event.evidence.opens_at.compare(_pit("09:40")) >= 0 for event in evaluation.new_events
    )


def test_a_terminal_lifecycle_stays_frozen_across_later_snapshots() -> None:
    scenario = _scenario()
    scenario.fill("09:40", "09:55", {"09:50": ("97", "98")})
    cancelled = scenario.run(scenario.snap("09:56"))
    assert cancelled.state is S.CANCELLED_BEFORE_ENTRY
    scenario.fill("09:55", "10:30", {"10:00": _ENTRY})
    later = _then(scenario, cancelled, scenario.snap("10:31"))
    assert later.state is S.CANCELLED_BEFORE_ENTRY
    assert later.steps[0].status is Status.TERMINAL and later.new_events == ()


# ---------------------------------------------------------------------------
# D. Exact-boundary finality (EI5-D7 clarification) with real EI-2A outcomes
# ---------------------------------------------------------------------------


def test_entry_boundary_a_last_bar_not_yet_received_at_10_50_blocks() -> None:
    scenario = _scenario()
    scenario.fill("09:40", "10:50")  # the last bar is received at 10:50:05
    evaluation = scenario.run(scenario.snap("10:50"))
    (step,) = evaluation.steps
    assert step.status is Status.BLOCKED
    assert step.halted_at is not None and step.halted_at.outcome is Outcome.MISSING
    assert evaluation.state is S.WAITING_FOR_ENTRY


def test_entry_boundary_a_received_unfinalized_last_bar_at_10_50_blocks() -> None:
    scenario = _scenario()
    scenario.fill("09:40", "10:45")
    scenario.put("10:45", _QUIET, received=0, confirmed=None)
    evaluation = scenario.run(scenario.snap("10:50"))
    assert evaluation.steps[0].status is Status.BLOCKED
    assert evaluation.steps[0].halted_at.outcome is Outcome.UNFINALIZED  # type: ignore[union-attr]


def test_entry_boundary_a_last_bar_final_exactly_at_10_50_is_evaluated_normally() -> None:
    scenario = _scenario()
    scenario.fill("09:40", "10:45")
    scenario.put("10:45", _QUIET, received=0, confirmed=0)
    evaluation = scenario.run(scenario.snap("10:50"))
    assert _kinds(evaluation) == [K.ENTRY_WINDOW_OPENED, K.ENTRY_WINDOW_EXPIRED]


def test_entry_boundary_after_10_50_a_still_unfinalized_last_bar_is_incomplete() -> None:
    scenario = _scenario()
    scenario.fill("09:40", "10:45")
    scenario.put("10:45", _QUIET, received=0, confirmed=None)
    evaluation = scenario.run(scenario.snap("10:50:00.000001"))
    assert _kinds(evaluation)[-1] is K.EVIDENCE_INCOMPLETE


def test_entry_boundary_an_earlier_genuine_gap_at_10_50_is_still_terminal() -> None:
    scenario = _scenario()
    scenario.fill("09:40", "10:50", {"10:00": None})
    evaluation = scenario.run(scenario.snap("10:50"))
    assert _kinds(evaluation)[-1] is K.EVIDENCE_INCOMPLETE
    assert evaluation.new_events[-1].deficiencies[0].opens_at == _pit("10:00")


@pytest.mark.parametrize(
    ("bar", "received", "confirmed", "kind"),
    [
        (_QUIET, 5, 10, None),  # MISSING at 14:00: blocked
        (_QUIET, 0, None, None),  # UNFINALIZED at 14:00: blocked
        (_QUIET, 0, 0, K.SESSION_CUTOFF_REACHED),  # final exactly at 14:00
        (("97", "98"), 0, 0, K.STOP_REACHED),
        (("110", "111"), 0, 0, K.TARGET_REACHED),
    ],
    ids=["missing", "unfinalized", "cutoff", "stop", "target"],
)
def test_exit_boundary_at_14_00(
    bar: tuple[str, str],
    received: float,
    confirmed: float | None,
    kind: EquityLifecycleEventKind | None,
) -> None:
    scenario = _scenario()
    active = _active(scenario)
    scenario.fill("09:50", "13:55")
    scenario.put("13:55", bar, received=received, confirmed=confirmed)
    evaluation = _then(scenario, active, scenario.snap("14:00"))
    if kind is None:
        assert evaluation.steps[0].status is Status.BLOCKED and evaluation.state is S.ACTIVE
    else:
        assert _kinds(evaluation) == [kind]


def test_ew07_no_candle_after_the_modeled_exit_is_read() -> None:
    scenario = _scenario()
    active = _active(scenario)
    scenario.fill("09:50", "14:00")
    scenario.put("14:00", ("97", "111"))  # after the exit: it touches everything
    evaluation = _then(scenario, active, scenario.snap("14:10"))
    assert _kinds(evaluation) == [K.SESSION_CUTOFF_REACHED]
    assert all(
        event.evidence.closes_at.compare(_pit("14:00")) <= 0
        for event in evaluation.lifecycle.events
    )


# ---------------------------------------------------------------------------
# E. Snapshot sequences, quiet bars, revisions and replay
# ---------------------------------------------------------------------------


def test_rr02_identical_requests_give_identical_results() -> None:
    scenario = _scenario()
    scenario.fill("09:40", "10:30", {"10:00": _ENTRY, "10:20": ("110", "111")})
    snapshots = (scenario.snap("09:52"), scenario.snap("10:06"), scenario.snap("10:31"))
    first, second = scenario.run(*snapshots), scenario.run(*snapshots)
    assert first == second
    assert _kinds(first) == [K.ENTRY_WINDOW_OPENED, K.ENTRY_ACTIVATED, K.TARGET_REACHED]
    assert [e.sequence for e in first.lifecycle.events] == [1, 2, 3]


def test_continuing_in_several_calls_equals_one_call() -> None:
    scenario = _scenario()
    scenario.fill("09:40", "10:30", {"10:00": _ENTRY, "10:20": ("97", "98")})
    snapshots = (scenario.snap("09:52"), scenario.snap("10:06"), scenario.snap("10:31"))
    whole = scenario.run(*snapshots)
    part = scenario.run(snapshots[0])
    for snapshot_id in snapshots[1:]:
        part = _then(scenario, part, snapshot_id)
    assert part.lifecycle == whole.lifecycle and part.last_evaluated == whole.last_evaluated


def test_quiet_bars_are_re_read_and_a_correction_to_one_blocks_rather_than_rewrites() -> None:
    scenario = _scenario()
    scenario.fill("09:40", "10:00")
    quiet = scenario.run(scenario.snap("10:01"))
    assert _kinds(quiet) == [K.ENTRY_WINDOW_OPENED]
    scenario.put("09:45", _ENTRY, revision="r2")  # a later correction to a quiet bar
    later = _then(scenario, quiet, scenario.snap("10:06"))
    assert later.steps[0].status is Status.BLOCKED
    assert later.steps[0].halted_at.outcome is Outcome.UNRESOLVED_REVISIONS  # type: ignore[union-attr]
    assert later.new_events == () and later.state is S.WAITING_FOR_ENTRY


def test_rv02_a_correction_to_the_activation_bar_is_recorded_without_rewriting() -> None:
    scenario = _scenario()
    active = _active(scenario)
    activation = active.lifecycle.events[-1]
    scenario.put("09:45", ("101.6", "102.3"), revision="r2")  # corrects the cited bar
    scenario.fill("09:50", "10:00")
    revised = _then(scenario, active, scenario.snap("10:01"))
    revision = revised.new_events[-1]
    assert revision.kind is K.SOURCE_REVISION_RECORDED and revised.state is S.ACTIVE
    assert revision.cited_sequence == activation.sequence
    assert revision.superseded_evidence == activation.evidence
    assert revision.materiality is EquityRevisionMateriality.UNASSESSED
    assert revised.lifecycle.events[: len(active.lifecycle.events)] == active.lifecycle.events
    assert len(revision.evidence.observations) == 2


def test_rv03_the_same_correction_seen_again_is_not_recorded_twice() -> None:
    scenario = _scenario()
    active = _active(scenario)
    scenario.put("09:45", ("101.6", "102.3"), revision="r2")
    scenario.fill("09:50", "10:10")
    once = _then(scenario, active, scenario.snap("10:01"))
    twice = _then(scenario, once, scenario.snap("10:11"))
    assert [e.kind for e in twice.new_events] == []
    assert sum(e.kind is K.SOURCE_REVISION_RECORDED for e in twice.lifecycle.events) == 1


def test_an_exact_repeat_of_a_cited_candle_is_not_a_correction() -> None:
    scenario = _scenario()
    active = _active(scenario)
    scenario.put("09:45", _ENTRY, suffix="-repeat")  # same source, event, revision and values
    scenario.fill("09:50", "10:00")
    later = _then(scenario, active, scenario.snap("10:01"))
    assert later.new_events == () and later.state is S.ACTIVE


def test_rv01_a_correction_after_terminality_is_recorded_and_changes_nothing() -> None:
    scenario = _scenario()
    active = _active(scenario)
    scenario.fill("09:50", "10:05", {"10:00": ("110", "111")})
    closed = _then(scenario, active, scenario.snap("10:06"))
    assert closed.state is S.CLOSED_TARGET
    scenario.put("10:00", ("105", "108"), revision="r2")
    revised = _then(scenario, closed, scenario.snap("16:00"))
    assert [e.kind for e in revised.new_events] == [K.SOURCE_REVISION_RECORDED]
    assert revised.state is S.CLOSED_TARGET and revised.steps[0].status is Status.TERMINAL


def test_a_corrupted_prior_event_history_is_rejected_by_core_replay() -> None:
    scenario = _scenario()
    scenario.fill("09:40", "09:50", {"09:45": _ENTRY})
    snapshot_id = scenario.snap("09:51")
    active = scenario.run(snapshot_id)
    reordered = EquitySyntheticLifecycleRequest(
        _LABEL,
        scenario.key,
        (snapshot_id,),
        tuple(reversed(active.lifecycle.events)),
        active.last_evaluated,
    )
    assert scenario.refused(reordered) is Refusal.INVALID_LIFECYCLE_HISTORY
    unpositioned = EquitySyntheticLifecycleRequest(
        _LABEL, scenario.key, (snapshot_id,), active.lifecycle.events
    )
    assert scenario.refused(unpositioned) is Refusal.PRIOR_EVALUATION_INCONSISTENT


def test_a_snapshot_older_than_the_prior_events_is_refused() -> None:
    scenario = _scenario()
    early = scenario.snap("09:50")
    scenario.fill("09:40", "10:00", {"09:55": _ENTRY})
    active = scenario.run(scenario.snap("10:01"))
    request = EquitySyntheticLifecycleRequest.following(active, (early,))
    assert scenario.refused(request) is Refusal.SNAPSHOT_OUT_OF_ORDER


# ---------------------------------------------------------------------------
# Synthetic issuance boundary
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("label", ["issued-ei5c", "synthetic-", "SYNTHETIC-x", " synthetic-x", ""])
def test_only_an_explicitly_synthetic_label_is_accepted(label: str) -> None:
    with pytest.raises(InvalidEquitySyntheticLifecycleRequestError):
        EquitySyntheticLifecycleRequest(label, "eirk1-sha256:x", ("x",))


def test_an_evaluation_establishes_no_issuance_and_is_not_authoritative() -> None:
    scenario = _scenario()
    evaluation = scenario.run(scenario.snap("09:41"))
    assert evaluation.issuance_established is False and evaluation.is_authoritative is False
    names = {field.name for field in fields(evaluation)}
    for forbidden in (
        "issued",
        "issuance_id",
        "recommendation_key",
        "fill",
        "price",
        "order",
        "pnl",
    ):
        assert not any(forbidden in name for name in names), forbidden
    assert scenario.store.writes == 0


# ---------------------------------------------------------------------------
# F. Architecture (AR-02, LA-07)
# ---------------------------------------------------------------------------

_SOURCE = inspect.getsource(lifecycle_module)
_TREE = ast.parse(_SOURCE)


def _imports() -> list[str]:
    found: list[str] = []
    for node in ast.walk(_TREE):
        if isinstance(node, ast.Import):
            found.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            found.append(node.module)
    return found


def test_the_module_imports_only_core_and_application_contracts() -> None:
    for module in _imports():
        assert module in {"__future__", "dataclasses", "enum"} or module.startswith(
            ("northstar_core.", "northstar_application.")
        ), module
        for forbidden in ("infrastructure", "sqlite", "provider", "broker", "order", "paper"):
            assert forbidden not in module, module
    assert "northstar_core.recommendations.futures" not in " ".join(_imports())


def test_the_module_reads_no_clock_and_calls_no_write_or_issuance_operation() -> None:
    called = {
        (node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", ""))
        for node in ast.walk(_TREE)
        if isinstance(node, ast.Call)
    }
    for name in ("now", "utcnow", "today", "time", "monotonic", "perf_counter"):
        assert name not in called, name
    for name in ("issue", "write_generation", "write_attempt", "publish", "commit", "save"):
        assert name not in called, name
    # The two injected dependencies are only ever read: verify, and get a snapshot.
    dependency_calls = {
        node.func.attr
        for node in ast.walk(_TREE)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Attribute)
        and node.func.value.attr in {"_snapshots", "_generations"}
    }
    assert dependency_calls == {"get_snapshot", "execute"}
    assert "eiris" not in _SOURCE.lower().replace("no ``eiris1`` issuance identity", "")


def test_the_use_case_depends_only_on_the_verified_reader_and_the_snapshot_store() -> None:
    parameters = inspect.signature(
        EvaluateSyntheticEquityRecommendationLifecycleUseCase.__init__
    ).parameters
    assert list(parameters) == ["self", "generations", "snapshots"]
    with pytest.raises(TypeError):
        EvaluateSyntheticEquityRecommendationLifecycleUseCase(object(), object())  # type: ignore[arg-type]


def test_la07_every_event_cites_a_snapshot_and_is_decided_at_its_as_of() -> None:
    scenario = _scenario()
    scenario.fill("09:40", "10:30", {"10:00": _ENTRY, "10:20": ("97", "98")})
    snapshots = (scenario.snap("09:52"), scenario.snap("10:06"), scenario.snap("10:31"))
    evaluation = scenario.run(*snapshots)
    as_of = {
        snapshot_id: step.as_of
        for snapshot_id, step in zip(snapshots, evaluation.steps, strict=True)
    }
    for event in evaluation.lifecycle.events:
        assert event.evidence.reference in as_of
        assert event.evaluated_at == as_of[event.evidence.reference]


# ---------------------------------------------------------------------------
# G. Defensive refusals reached with constructed, structurally valid snapshots
#
# Each snapshot below starts from a real EI-2B snapshot and changes exactly one
# thing through ``dataclasses.replace``, so the real snapshot constructor
# re-validates it and re-derives its evidence class, manifest and identity. It
# is planted in the test snapshot store at the read-port boundary. A control
# evaluation of the unchanged snapshot shows the fixture is otherwise valid, so
# each refusal comes from the intended predicate, not an unrelated failure.
# ---------------------------------------------------------------------------

_INTERVAL = snapshot_module.EquityCandleSnapshotInterval
_EI2A = snapshot_module.EquityCandleIntervalOutcome
_Snapshot = snapshot_module.EquityCandleEvidenceSnapshot


def _plant(scenario: _Scenario, snapshot: _Snapshot) -> str:
    scenario.world.snapshots.issue(snapshot)
    return snapshot.snapshot_id


def _rebuilt(snapshot: _Snapshot, intervals: tuple[_INTERVAL, ...], **changes: object) -> _Snapshot:
    """Return ``snapshot`` with other intervals, re-validated by the real constructor."""
    return replace(
        snapshot, intervals=intervals, coverage=snapshot_module._coverage(intervals), **changes
    )


def _without_interval_evidence(snapshot: _Snapshot, interval: _INTERVAL) -> dict[str, object]:
    """Return observations and finalities without those of ``interval``'s candles."""
    removed = {
        o.record.source_identity
        for o in snapshot.observations
        if o.sequence in interval.observation_sequences
    }
    return {
        "observations": tuple(
            o for o in snapshot.observations if o.sequence not in interval.observation_sequences
        ),
        "finalities": tuple(
            f for f in snapshot.finalities if f.record.source_identity not in removed
        ),
    }


def _assert_refused_without_effect(
    scenario: _Scenario, request: EquitySyntheticLifecycleRequest, reason: Refusal
) -> None:
    """The exact typed refusal, with nothing written and no stored evidence changed."""
    documents = dict(scenario.world.snapshots.documents)
    stored = dict(scenario.store.dispositions)
    assert scenario.refused(request) is reason
    assert scenario.world.snapshots.documents == documents
    assert scenario.store.dispositions == stored and scenario.store.writes == 0


def test_r1_a_structurally_valid_production_class_snapshot_is_refused() -> None:
    scenario = _scenario()
    real = scenario.world.issue(_ist("09:51"))
    assert scenario.run(real.snapshot_id).state is S.WAITING_FOR_ENTRY  # control
    emptied = tuple(
        _INTERVAL(
            i.opens_at,
            i.closes_at,
            i.outcome if i.outcome is _EI2A.NOT_ENDED else _EI2A.MISSING,
            (),
            (),
        )
        for i in real.intervals
    )
    accepted = snapshot_module.EquityReferenceProvenanceKind.ACCEPTED_PRODUCTION
    production = _rebuilt(
        real,
        emptied,
        observations=(),
        finalities=(),
        unmatched_sequences=(),
        venue_provenance=accepted,
        applicability_provenance=accepted,
        finality_policy=NO_APPROVED_EQUITY_CANDLE_FINALITY,
    )
    assert production.evidence_class is snapshot_module.EquityCandleEvidenceClass.PRODUCTION
    assert (production.listing, production.trading_date, production.venue_reference) == (
        real.listing,
        real.trading_date,
        real.venue_reference,
    )
    request = EquitySyntheticLifecycleRequest(_LABEL, scenario.key, (_plant(scenario, production),))
    _assert_refused_without_effect(scenario, request, Refusal.SNAPSHOT_NOT_SYNTHETIC)


@pytest.mark.parametrize(
    "change",
    [
        {"window_policy_version": "synthetic-another-research-window/1"},
        {"window_closes_at": ei4c._pit(_ist("15:05"))},
    ],
    ids=["window-version", "window-bounds"],
)
def test_r2_a_snapshot_of_another_research_window_is_refused(change: dict[str, object]) -> None:
    scenario = _scenario()
    real = scenario.world.issue(_ist("09:51"))
    assert scenario.run(real.snapshot_id).state is S.WAITING_FOR_ENTRY  # control
    other = replace(real, **change)
    # Every earlier check still passes: synthetic, same listing, same session and venue.
    assert other.evidence_class is snapshot_module.EquityCandleEvidenceClass.SYNTHETIC
    assert (other.listing, other.trading_date, other.venue_reference) == (
        real.listing,
        real.trading_date,
        real.venue_reference,
    )
    request = EquitySyntheticLifecycleRequest(_LABEL, scenario.key, (_plant(scenario, other),))
    _assert_refused_without_effect(scenario, request, Refusal.SNAPSHOT_WINDOW_MISMATCH)


def test_r3_a_snapshot_with_a_gap_in_its_interval_grid_is_refused() -> None:
    scenario = _scenario()
    scenario.fill("09:40", "10:20")
    real = scenario.world.issue(_ist("10:21"))
    assert scenario.run(real.snapshot_id).steps[0].status is Status.AWAITING_EVIDENCE  # control
    holed = _rebuilt(real, tuple(i for i in real.intervals if i.opens_at != _pit("10:30")))
    assert len(holed.intervals) == len(real.intervals) - 1
    request = EquitySyntheticLifecycleRequest(_LABEL, scenario.key, (_plant(scenario, holed),))
    _assert_refused_without_effect(scenario, request, Refusal.SNAPSHOT_EVIDENCE_INCONSISTENT)


def test_r3_acceptable_candles_that_disagree_on_one_range_are_refused() -> None:
    scenario = _scenario()
    scenario.fill("09:40", "10:00")
    real = scenario.world.issue(_ist("10:01"))
    assert scenario.run(real.snapshot_id).state is S.WAITING_FOR_ENTRY  # control
    target = next(i for i in real.intervals if i.opens_at == _pit("09:45"))
    (original,) = (o for o in real.observations if o.sequence in target.acceptable_sequences)
    low, high = Price(Decimal("97"), ei4c._INR), Price(Decimal("98"), ei4c._INR)
    sequence = max(o.sequence for o in real.observations) + 1
    assert sequence <= real.watermark  # an unused observation sequence under the watermark
    bar = original.record.observation.bar
    twin = replace(
        original,
        sequence=sequence,
        record=replace(
            original.record,
            ingestion_id=f"{original.record.ingestion_id}-twin",
            observation=replace(
                original.record.observation,
                bar=replace(bar, open=low, high=high, low=low, close=high),
            ),
        ),
    )
    disagreeing = _INTERVAL(
        target.opens_at,
        target.closes_at,
        _EI2A.ACCEPTABLE,
        (*target.observation_sequences, sequence),
        (*target.acceptable_sequences, sequence),
    )
    forged = _rebuilt(
        real,
        tuple(disagreeing if i is target else i for i in real.intervals),
        observations=(*real.observations, twin),
    )
    request = EquitySyntheticLifecycleRequest(_LABEL, scenario.key, (_plant(scenario, forged),))
    _assert_refused_without_effect(scenario, request, Refusal.SNAPSHOT_EVIDENCE_INCONSISTENT)


def test_r3_a_later_snapshot_lacking_a_cited_interval_is_refused() -> None:
    scenario = _scenario()
    active = _active(scenario)
    scenario.fill("09:50", "10:00")
    real = scenario.world.issue(_ist("10:01"))
    assert _then(scenario, active, real.snapshot_id).state is S.ACTIVE  # control
    cited = active.lifecycle.events[-1].evidence
    target = next(i for i in real.intervals if i.opens_at == cited.opens_at)
    lacking = _rebuilt(
        real,
        tuple(i for i in real.intervals if i is not target),
        **_without_interval_evidence(real, target),
    )
    request = EquitySyntheticLifecycleRequest.following(active, (_plant(scenario, lacking),))
    _assert_refused_without_effect(scenario, request, Refusal.SNAPSHOT_EVIDENCE_INCONSISTENT)


@pytest.mark.parametrize("contradiction", ["MISSING", "UNFINALIZED", "NOT_ENDED"])
def test_r4_a_later_snapshot_contradicting_a_cited_candle_is_refused(contradiction: str) -> None:
    scenario = _scenario()
    active = _active(scenario)
    history = active.lifecycle.events
    activation = history[-1]
    scenario.fill("09:50", "10:00")
    real = scenario.world.issue(_ist("10:01"))
    control = _then(scenario, active, real.snapshot_id)  # the true later snapshot agrees
    assert control.state is S.ACTIVE and control.new_events == ()
    cited = next(i for i in real.intervals if i.opens_at == activation.evidence.opens_at)
    if contradiction == "UNFINALIZED":
        # The cited candle is still listed, but no longer final.
        contradicted = _INTERVAL(
            cited.opens_at, cited.closes_at, _EI2A.UNFINALIZED, cited.observation_sequences, ()
        )
        evidence = _without_interval_evidence(real, cited)
        changes: dict[str, object] = {
            "observations": tuple(
                replace(o, finality_sequences=())
                if o.sequence in cited.observation_sequences
                else o
                for o in real.observations
            ),
            "finalities": evidence["finalities"],
        }
    else:
        kind = _EI2A.MISSING if contradiction == "MISSING" else _EI2A.NOT_ENDED
        contradicted = _INTERVAL(cited.opens_at, cited.closes_at, kind, (), ())
        changes = _without_interval_evidence(real, cited)
    forged = _rebuilt(
        real, tuple(contradicted if i is cited else i for i in real.intervals), **changes
    )
    assert forged.as_of == real.as_of and forged.watermark == real.watermark
    request = EquitySyntheticLifecycleRequest.following(active, (_plant(scenario, forged),))
    _assert_refused_without_effect(scenario, request, Refusal.SNAPSHOT_CONTRADICTS_CITED_EVIDENCE)
    # History and provenance are untouched: the activation still cites the original
    # snapshot and candle, and no revision event was fabricated to hide the contradiction.
    assert active.lifecycle.events == history
    assert activation.evidence.reference not in (forged.snapshot_id, real.snapshot_id)
    assert activation.evidence.observations[0].split(":")[2] == f"evt-{_index('09:45')}"
    assert all(e.kind is not K.SOURCE_REVISION_RECORDED for e in active.lifecycle.events)


def test_r4_is_distinct_from_an_ordinary_revision_which_is_recorded() -> None:
    scenario = _scenario()
    active = _active(scenario)
    scenario.put("09:45", ("101.6", "102.3"), revision="r2")  # a genuine later correction
    scenario.fill("09:50", "10:00")
    revised = _then(scenario, active, scenario.snap("10:01"))
    assert [e.kind for e in revised.new_events] == [K.SOURCE_REVISION_RECORDED]
    assert revised.state is S.ACTIVE
