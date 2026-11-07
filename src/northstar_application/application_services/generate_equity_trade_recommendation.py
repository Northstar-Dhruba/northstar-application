"""Synthetic equity intraday recommendation eligibility and generation (EI-4C).

GenerateEquityTradeRecommendationUseCase evaluates the first authoritative
EI-3C research CANDIDATE of one strategy configuration, listing and trading
date against a versioned synthetic policy set, and -- when every implemented
check passes -- invokes the merged EI-4B Core formation to produce a
synthetic, non-authoritative proposal. It applies the owner decisions
recorded on 2026-11-07 in the EI-4 Decision Register (northstar-docs,
product/Equity-Intraday-V1-EI4-Decision-Register.md), ADR-020 and ADR-021:
EI4-D1, EI0-D7 (synthetic structure only), EI4-D2, EI0-D14, EI4-D3, EI4-D4,
EI4-D6, EI4-D7, EI4-D8 and EI4-D9. Nothing is issued, persisted, scanned,
filled or traded, and no clock, provider, broker or network is read.

It works in two parts:

1. **Gathering** reads the ports and freezes every input into one
   EquityRecommendationInputs envelope (equity_recommendation_generation_
   evidence).
2. **Assessment**, ``assess_equity_recommendation_inputs``, is pure: it reads
   only the frozen envelope. Historical replay calls the same function with a
   stored envelope, so a replayed assessment is the original assessment,
   never one built from today's reference answers.

Every attempt returns an EquityRecommendationEligibilityAudit, whose status
separates a refused or unresolved opportunity from an integrity failure, an
unavailable dependency and a proposal (equity_trade_recommendation_
eligibility). Checks run in this order, and a check runs only when every
earlier step passed:

1. AUTHORITATIVE_CANDIDATE. The timeline is loaded from the EI-3C store and
   rehydrated through ``rehydrate_equity_intraday_research_timeline`` -- whole
   or not at all (EI3-D13). Only its CANDIDATE is considered (EI4-D2). Its
   evidence is re-checked: the snapshot loads and is the one named,
   synthetic, at the decision instant; the signal bar is the decision's latest
   required interval and ended before it; and the snapshot's grid is exactly
   the research window's five-minute grid. The signal close and opening-range
   low are the EI-3A decision's own values, in the snapshot's currency.
2. SYNTHETIC_REFERENCES. The policy set's five components are loaded by their
   exact versions, then the venue-day reference, the listing eligibility and
   -- as of the modeled issuance, which is the decision instant plus the
   timing policy's versioned latency (EI4-D9-C) -- the issued-history record.
   It must be complete through that instant and states its recording
   watermark (EI4-D9-H). Every missing, unknown, incomplete, inconsistent or
   unanswerable input is reported; none is defaulted.
3. SESSION_TIMING (EI4-D1, EI0-D7 structure), LISTING_ELIGIBILITY (EI4-D7)
   and OPEN_RECOMMENDATIONS (EI4-D8) are evaluated together, from the frozen
   evidence only:

   - only a NORMAL venue session is covered, and one that closes before the
     modeled exit boundary is refused, never re-timed;
   - the signal bar must end by the policy's latest signal-bar end, and a
     decision after the research window closed is historical research only;
   - the earliest permissible entry is the first research-window grid bar
     that *opens* at or after the modeled issuance -- a bar that opened before
     the decision was available never evidences a full-bar entry -- and it
     must open no later than the latest permissible entry-bar opening and
     leave the minimum remaining time before the modeled exit boundary;
   - every eligibility criterion must be ELIGIBLE;
   - every recommendation in the as-of issued-history evidence is treated as
     open (EI4-D8). EI4-D8 approves no exclusivity rule or exposure limit, so
     whether the candidate may coexist with one is unresolved:
     EXCLUSIVITY_POLICY_UNRESOLVED for the same listing,
     EXPOSURE_POLICY_UNRESOLVED for another. Both fail closed and neither is
     reported as a breach.
4. CORE_FORMATION. ``form_equity_recommendation`` runs on the candidate's
   research prices and the policy set, and its exact reasons are kept. A
   confirmed Core rejection is CORE_FORMATION_REJECTED. A result Core could
   not establish -- SIZING_SEARCH_LIMIT_EXCEEDED -- is
   CORE_FORMATION_UNRESOLVED, fails closed, and is never reported as a failed
   reward-to-risk test.

SAME_LISTING_EXCLUSIVITY, PORTFOLIO_EXPOSURE, DAILY_LOSS_LIMIT,
PRE_ENTRY_INVALIDATION, CORPORATE_ACTIONS, EXCHANGE_SURVEILLANCE and
CANDLE_REVISION_MATERIALITY are always NOT_ASSESSED: their policies or inputs
do not exist, and a proposal names them as blocking actionable issuance.

Failure semantics
-----------------
Expected outages of the injected ports -- the timeline store, the snapshot
store, the synthetic reference sources and the issued-history source -- become
DEPENDENCY_UNAVAILABLE, and so does an answer for another version, listing,
date or session, which is never substituted. Verification failures of stored
history become RESEARCH_INTEGRITY_FAILURE. Programming errors -- a wrong
argument type, an unexpected exception -- raise and produce no audit.
Identical inputs and stored history always give an identical assessment: the
same status, reasons, key, levels, costs, sizing and fingerprint.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timezone

from northstar_core.domain.value_objects import ListingReference
from northstar_core.foundation.exceptions.validation import ValidationError
from northstar_core.foundation.value_objects import PointInTime, Price
from northstar_core.recommendations.equity import (
    EquityIssuanceKind,
    EquityPriceIncrement,
    EquityRecommendationDefinition,
    EquityRecommendationTiming,
    EquityResearchCandidateReference,
    EquitySyntheticCostSchedule,
    EquitySyntheticGenerationPolicy,
    EquitySyntheticRiskPolicy,
    EquitySyntheticVersion,
    form_equity_recommendation,
)

from northstar_application.application_services.advance_equity_intraday_research_timeline import (
    rehydrate_equity_intraday_research_timeline,
)
from northstar_application.application_services.equity_candle_evidence_snapshot import (
    EquityCandleEvidenceClass,
)
from northstar_application.application_services.equity_recommendation_generation_evidence import (
    EquityIssuedHistoryEvidence,
    EquityRecommendationCandidateProvenance,
    EquityRecommendationInputs,
    EquityResolvedSyntheticPolicySet,
    EquityVenueDayEvidence,
    InvalidEquityRecommendationEvidenceError,
    equity_modeled_issuance,
)
from northstar_application.application_services.equity_trade_recommendation_eligibility import (
    EQUITY_CORE_UNRESOLVED_REASONS,
    EQUITY_RECOMMENDATION_UNASSESSED_CONTROLS,
    EquityModeledEntryWindow,
    EquityRecommendationCheck,
    EquityRecommendationCheckResult,
    EquityRecommendationCheckStatus,
    EquityRecommendationEligibilityAudit,
    EquityRecommendationEligibilityReason,
    EquityTradeRecommendationProposal,
    EquityTradeRecommendationRequest,
    ordered_equity_recommendation_reasons,
)
from northstar_application.application_services.schedule_equity_intraday_research import (
    EquityIntradayResearchTimelineIdentity,
)
from northstar_application.ports.equity_candle_evidence_journal import (
    EquityCandleEvidenceSnapshotStore,
    EquityCandleEvidenceUnavailableError,
)
from northstar_application.ports.equity_intraday_research_timeline_store import (
    EquityIntradayResearchDecisionNotReproducedError,
    EquityIntradayResearchHistoricalEvaluationUnavailableError,
    EquityIntradayResearchTimelineConfigurationConflictError,
    EquityIntradayResearchTimelineStore,
    EquityIntradayResearchTimelineUnavailableError,
    InvalidStoredEquityIntradayResearchTimelineError,
)
from northstar_application.ports.equity_recommendation_references import (
    EquityRecommendationHistoryUnavailableError,
    EquitySessionRecommendationHistory,
    EquitySessionRecommendationHistorySource,
    EquitySyntheticEligibilityVerdict,
    EquitySyntheticEntryTimingPolicy,
    EquitySyntheticListingEligibility,
    EquitySyntheticListingEligibilitySource,
    EquitySyntheticRecommendationReferenceSource,
    EquitySyntheticReferenceUnavailableError,
)
from northstar_application.ports.equity_research_window_reference import (
    EquityVenueDayClassification,
    EquityVenueDayReference,
    EquityVenueDayReferenceSource,
)

_R = EquityRecommendationEligibilityReason
_CHECK = EquityRecommendationCheck
_STATUS = EquityRecommendationCheckStatus
_UNANSWERED_DAYS = frozenset(
    {EquityVenueDayClassification.NOT_STATED, EquityVenueDayClassification.OUTSIDE_COVERAGE}
)
_Found = list[tuple[EquityRecommendationEligibilityReason, str]]


@dataclass(frozen=True, slots=True)
class _Failed:
    reasons: frozenset[EquityRecommendationEligibilityReason]
    detail: str


def _failed(reason: EquityRecommendationEligibilityReason, detail: str) -> _Failed:
    return _Failed(frozenset({reason}), detail)


def _result(
    check: EquityRecommendationCheck, outcome: _Failed | str
) -> EquityRecommendationCheckResult:
    if isinstance(outcome, _Failed):
        reasons = ordered_equity_recommendation_reasons(set(outcome.reasons))
        return EquityRecommendationCheckResult(check, _STATUS.FAILED, reasons, outcome.detail)
    return EquityRecommendationCheckResult(check, _STATUS.PASSED, (), outcome)


def _collect(found: _Found) -> _Failed:
    return _Failed(frozenset(r for r, _ in found), " ".join(detail for _, detail in found))


def _instant(value: PointInTime) -> datetime:
    return datetime.fromisoformat(value.value.replace("Z", "+00:00")).astimezone(UTC)


def _point(value: datetime) -> PointInTime:
    return PointInTime(value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"))


def _audit(
    request: EquityTradeRecommendationRequest,
    results: dict[EquityRecommendationCheck, EquityRecommendationCheckResult],
    *,
    candidate: EquityRecommendationCandidateProvenance | None = None,
    inputs: EquityRecommendationInputs | None = None,
    entry_window: EquityModeledEntryWindow | None = None,
    core_outcome: object = None,
    proposal: EquityTradeRecommendationProposal | None = None,
    observed_timeline_head: str | None = None,
) -> EquityRecommendationEligibilityAudit:
    unassessed = dict(EQUITY_RECOMMENDATION_UNASSESSED_CONTROLS)
    checks = []
    for check in EquityRecommendationCheck:
        if check in results:
            checks.append(results[check])
        elif check in unassessed:
            checks.append(
                EquityRecommendationCheckResult(check, _STATUS.NOT_ASSESSED, (), unassessed[check])
            )
        else:
            checks.append(
                EquityRecommendationCheckResult(
                    check, _STATUS.NOT_EVALUATED, (), "Not evaluated: an earlier step failed."
                )
            )
    return EquityRecommendationEligibilityAudit(
        request=request,
        checks=tuple(checks),
        candidate=candidate,
        inputs=inputs,
        entry_window=entry_window,
        core_outcome=core_outcome,  # type: ignore[arg-type]
        proposal=proposal,
        observed_timeline_head=observed_timeline_head,
    )


@dataclass(frozen=True, slots=True)
class _Candidate:
    provenance: EquityRecommendationCandidateProvenance
    timeline_head: str


# ---------------------------------------------------------------------------
# Gathering
# ---------------------------------------------------------------------------


class GenerateEquityTradeRecommendationUseCase:
    """Evaluates one authoritative research candidate into a synthetic proposal or refusal."""

    def __init__(
        self,
        timelines: EquityIntradayResearchTimelineStore,
        snapshots: EquityCandleEvidenceSnapshotStore,
        venue_days: EquityVenueDayReferenceSource,
        references: EquitySyntheticRecommendationReferenceSource,
        listing_eligibility: EquitySyntheticListingEligibilitySource,
        session_history: EquitySessionRecommendationHistorySource,
    ) -> None:
        for value, kind, name in (
            (timelines, EquityIntradayResearchTimelineStore, "timelines"),
            (snapshots, EquityCandleEvidenceSnapshotStore, "snapshots"),
            (venue_days, EquityVenueDayReferenceSource, "venue_days"),
            (references, EquitySyntheticRecommendationReferenceSource, "references"),
            (listing_eligibility, EquitySyntheticListingEligibilitySource, "listing_eligibility"),
            (session_history, EquitySessionRecommendationHistorySource, "session_history"),
        ):
            if not isinstance(value, kind):
                raise TypeError(f"{name} must be an {kind.__name__}.")
        self._timelines = timelines
        self._snapshots = snapshots
        self._venue_days = venue_days
        self._references = references
        self._eligibility = listing_eligibility
        self._history = session_history

    def execute(
        self, request: EquityTradeRecommendationRequest
    ) -> EquityRecommendationEligibilityAudit:
        """Evaluate ``request`` and return its eligibility audit, with a proposal when PROPOSED."""
        if not isinstance(request, EquityTradeRecommendationRequest):
            raise TypeError("request must be an EquityTradeRecommendationRequest.")
        candidate = self._candidate(request)
        if isinstance(candidate, _Failed):
            return _audit(
                request,
                {
                    _CHECK.AUTHORITATIVE_CANDIDATE: _result(
                        _CHECK.AUTHORITATIVE_CANDIDATE, candidate
                    )
                },
            )
        inputs = self._inputs(request, candidate.provenance)
        if isinstance(inputs, _Failed):
            return _audit(
                request,
                {
                    _CHECK.AUTHORITATIVE_CANDIDATE: _result(
                        _CHECK.AUTHORITATIVE_CANDIDATE, _candidate_detail(candidate.provenance)
                    ),
                    _CHECK.SYNTHETIC_REFERENCES: _result(_CHECK.SYNTHETIC_REFERENCES, inputs),
                },
                candidate=candidate.provenance,
                observed_timeline_head=candidate.timeline_head,
            )
        return assess_equity_recommendation_inputs(
            request, inputs, observed_timeline_head=candidate.timeline_head
        )

    # ------------------------------------------------------------------
    # 1. Authoritative candidate
    # ------------------------------------------------------------------

    def _candidate(self, request: EquityTradeRecommendationRequest) -> _Candidate | _Failed:
        identity = EquityIntradayResearchTimelineIdentity(
            request.listing, request.trading_date, request.configuration
        )
        try:
            stored = self._timelines.load(identity.strategy, request.listing, request.trading_date)
        except EquityIntradayResearchTimelineUnavailableError as exc:
            return _failed(_R.RESEARCH_TIMELINE_UNAVAILABLE, f"The timeline store failed: {exc}")
        if stored is None:
            return _failed(
                _R.NO_AUTHORITATIVE_CANDIDATE,
                "No research timeline entry was ever committed for this strategy, listing and "
                "date (EI3-D11).",
            )
        try:
            timeline = rehydrate_equity_intraday_research_timeline(
                identity, stored, self._snapshots
            )
        except EquityIntradayResearchTimelineConfigurationConflictError as exc:
            return _failed(_R.RESEARCH_CONFIGURATION_MISMATCH, str(exc))
        except EquityIntradayResearchDecisionNotReproducedError as exc:
            return _failed(_R.RESEARCH_DECISION_NOT_REPRODUCED, str(exc))
        except EquityIntradayResearchHistoricalEvaluationUnavailableError as exc:
            return _failed(_R.HISTORICAL_EVALUATION_UNAVAILABLE, str(exc))
        except InvalidStoredEquityIntradayResearchTimelineError as exc:
            return _failed(_R.RESEARCH_TIMELINE_INVALID, str(exc))
        except EquityCandleEvidenceUnavailableError as exc:
            return _failed(_R.EVIDENCE_SNAPSHOT_UNAVAILABLE, f"The snapshot store failed: {exc}")
        entry = timeline.candidate
        if entry is None:
            return _failed(
                _R.NO_AUTHORITATIVE_CANDIDATE,
                f"The verified timeline of {len(timeline.entries)} entries holds no CANDIDATE.",
            )
        head = timeline.head_fingerprint
        assert head is not None

        def invalid(detail: str) -> _Failed:
            return _failed(
                _R.CANDIDATE_EVIDENCE_INVALID, f"CANDIDATE entry {entry.sequence}: {detail}"
            )

        decision, values = entry.decision, entry.decision.values
        evidence = decision.evidence
        if (
            evidence is None
            or values is None
            or values.signal_close is None
            or values.opening_range_low is None
            or values.signal_interval_end is None
        ):
            return invalid("the decision lacks its signal close, opening-range low or signal bar.")
        if evidence.evidence_class is not EquityCandleEvidenceClass.SYNTHETIC:
            return _failed(
                _R.CANDIDATE_EVIDENCE_NOT_SYNTHETIC, "EI-4C generates from synthetic evidence only."
            )
        try:
            snapshot = self._snapshots.get_snapshot(evidence.snapshot_id)
        except EquityCandleEvidenceUnavailableError as exc:
            return _failed(_R.EVIDENCE_SNAPSHOT_UNAVAILABLE, f"The snapshot store failed: {exc}")
        if snapshot is None:
            return invalid(f"its source snapshot {evidence.snapshot_id} is unavailable.")
        if (
            snapshot.snapshot_id,
            snapshot.manifest_fingerprint,
            snapshot.as_of,
            snapshot.listing,
            snapshot.trading_date,
            snapshot.evidence_class,
            snapshot.window_policy_version,
        ) != (
            evidence.snapshot_id,
            evidence.manifest_fingerprint,
            evidence.as_of,
            request.listing,
            request.trading_date,
            evidence.evidence_class,
            evidence.window_policy_version,
        ):
            return invalid(f"it does not match its source snapshot {evidence.snapshot_id}.")
        required = evidence.required_interval_ends
        if not required or required[-1] != values.signal_interval_end:
            return invalid("its signal bar is not the latest interval its decision required.")
        try:
            provenance = EquityRecommendationCandidateProvenance(
                source=EquityResearchCandidateReference(
                    strategy=request.configuration.strategy,
                    research_configuration_version=EquitySyntheticVersion(
                        timeline.configuration.version
                    ),
                    listing=request.listing,
                    trading_date=request.trading_date,
                    decision_as_of=evidence.as_of,
                    research_decision_fingerprint=decision.fingerprint,
                    candidate_entry_fingerprint=entry.fingerprint,
                    evidence_snapshot_id=evidence.snapshot_id,
                ),
                manifest_fingerprint=evidence.manifest_fingerprint,
                candidate_sequence=entry.sequence,
                signal_interval_end=values.signal_interval_end,
                signal_close=Price(values.signal_close, snapshot.currency),
                opening_range_low=Price(values.opening_range_low, snapshot.currency),
                research_window_policy_version=snapshot.window_policy_version,
                research_window_opens_at=snapshot.window_opens_at,
                research_window_closes_at=snapshot.window_closes_at,
                venue_reference=snapshot.venue_reference,
                venue_provenance=snapshot.venue_provenance,
            )
        except (ValidationError, InvalidEquityRecommendationEvidenceError) as exc:
            return invalid(f"its research evidence cannot be referenced: {exc}")
        grid = tuple(interval.opens_at for interval in snapshot.intervals)
        ends = tuple(interval.closes_at for interval in snapshot.intervals)
        if grid != provenance.entry_bar_openings() or frozenset(ends) != provenance.bar_ends():
            return invalid("its snapshot's grid is not the research window's five-minute grid.")
        return _Candidate(provenance, head)

    # ------------------------------------------------------------------
    # 2. Synthetic references, frozen
    # ------------------------------------------------------------------

    def _inputs(
        self,
        request: EquityTradeRecommendationRequest,
        candidate: EquityRecommendationCandidateProvenance,
    ) -> EquityRecommendationInputs | _Failed:
        found: _Found = []
        sources = self._references
        policy = request.policy_set
        listing, day = request.listing, request.trading_date

        def versioned(
            label: str,
            version: EquitySyntheticVersion,
            missing: EquityRecommendationEligibilityReason,
            kind: type,
            fetch: object,
        ) -> object:
            try:
                value = fetch()  # type: ignore[operator]
            except EquitySyntheticReferenceUnavailableError as exc:
                found.append((missing, f"The {label} source is unavailable: {exc}."))
                return None
            if value is None:
                found.append((missing, f"No {label} {version} is held."))
                return None
            if not isinstance(value, kind) or value.version != version:
                found.append(
                    (_R.POLICY_REFERENCE_MISMATCH, f"The {label} source did not answer {version}.")
                )
                return None
            return value

        timing = versioned(
            "timing policy",
            policy.timing_policy_version,
            _R.TIMING_POLICY_UNAVAILABLE,
            EquitySyntheticEntryTimingPolicy,
            lambda: sources.timing_policy(policy.timing_policy_version),
        )
        generation = versioned(
            "generation policy",
            policy.generation_policy_version,
            _R.GENERATION_POLICY_UNAVAILABLE,
            EquitySyntheticGenerationPolicy,
            lambda: sources.generation_policy(policy.generation_policy_version),
        )
        risk = versioned(
            "risk policy",
            policy.risk_policy_version,
            _R.RISK_POLICY_UNAVAILABLE,
            EquitySyntheticRiskPolicy,
            lambda: sources.risk_policy(policy.risk_policy_version),
        )
        costs = versioned(
            "cost schedule",
            policy.cost_schedule_version,
            _R.COST_SCHEDULE_UNAVAILABLE,
            EquitySyntheticCostSchedule,
            lambda: sources.cost_schedule(policy.cost_schedule_version),
        )
        increment = versioned(
            "price increment",
            policy.price_increment_version,
            _R.PRICE_INCREMENT_UNAVAILABLE,
            EquityPriceIncrement,
            lambda: sources.price_increment(policy.price_increment_version, listing, day),
        )
        venue = self._venue_day(request, candidate, found)
        eligibility = self._listing_reference(request, found)
        history = None
        if isinstance(timing, EquitySyntheticEntryTimingPolicy):
            as_of = equity_modeled_issuance(
                candidate.source.decision_as_of, timing.modeled_issuance_latency
            )
            history = self._session_history(request, as_of, found)
        if found:
            return _collect(found)
        assert isinstance(timing, EquitySyntheticEntryTimingPolicy) and history is not None
        return EquityRecommendationInputs(
            candidate=candidate,
            policy_set=EquityResolvedSyntheticPolicySet(
                reference=policy,
                generation_policy=generation,  # type: ignore[arg-type]
                risk_policy=risk,  # type: ignore[arg-type]
                cost_schedule=costs,  # type: ignore[arg-type]
                price_increment=increment,  # type: ignore[arg-type]
                timing_policy=timing,
            ),
            listing_eligibility=eligibility,  # type: ignore[arg-type]
            venue_day=venue,  # type: ignore[arg-type]
            issued_history=history,
        )

    def _venue_day(
        self,
        request: EquityTradeRecommendationRequest,
        candidate: EquityRecommendationCandidateProvenance,
        found: _Found,
    ) -> EquityVenueDayEvidence | None:
        listing, day = request.listing, request.trading_date
        venue = self._venue_days.venue_day(listing.exchange_code, day)
        if not isinstance(venue, EquityVenueDayReference) or (
            venue.exchange_code,
            venue.trading_date,
        ) != (listing.exchange_code, day):
            found.append(
                (_R.SESSION_REFERENCE_MISMATCH, "The venue reference answered another session.")
            )
            return None
        if venue.classification in _UNANSWERED_DAYS:
            found.append(
                (
                    _R.SESSION_REFERENCE_UNAVAILABLE,
                    f"The venue reference cannot resolve the session: {venue.classification}.",
                )
            )
            return None
        if venue.reference != candidate.venue_reference:
            found.append(
                (
                    _R.SESSION_REFERENCE_MISMATCH,
                    f"The venue reference {venue.reference} is not the candidate evidence's "
                    f"{candidate.venue_reference}.",
                )
            )
            return None
        return EquityVenueDayEvidence.from_reference(venue)

    def _listing_reference(
        self, request: EquityTradeRecommendationRequest, found: _Found
    ) -> EquitySyntheticListingEligibility | None:
        version = request.listing_eligibility_version
        try:
            answer = self._eligibility.eligibility(version, request.listing, request.trading_date)
        except EquitySyntheticReferenceUnavailableError as exc:
            found.append(
                (_R.LISTING_REFERENCE_UNAVAILABLE, f"The eligibility source is unavailable: {exc}.")
            )
            return None
        if answer is None:
            found.append(
                (_R.LISTING_REFERENCE_UNAVAILABLE, f"No synthetic eligibility {version} is held.")
            )
            return None
        if not isinstance(answer, EquitySyntheticListingEligibility) or (
            answer.version,
            answer.listing,
            answer.trading_date,
        ) != (version, request.listing, request.trading_date):
            found.append(
                (
                    _R.LISTING_REFERENCE_MISMATCH,
                    "The eligibility source answered another version, listing or date.",
                )
            )
            return None
        unknown = answer.with_verdict(EquitySyntheticEligibilityVerdict.UNKNOWN)
        if unknown:
            found.append(
                (
                    _R.LISTING_ELIGIBILITY_UNKNOWN,
                    "Synthetic eligibility is unknown for: "
                    + ", ".join(criterion.value for criterion in unknown)
                    + ".",
                )
            )
            return None
        return answer

    def _session_history(
        self, request: EquityTradeRecommendationRequest, as_of: PointInTime, found: _Found
    ) -> EquityIssuedHistoryEvidence | None:
        exchange, day = request.listing.exchange_code, request.trading_date
        try:
            history = self._history.session_history(exchange, day, as_of)
        except EquityRecommendationHistoryUnavailableError as exc:
            found.append(
                (
                    _R.OPEN_RECOMMENDATION_STATE_UNAVAILABLE,
                    f"The session recommendation record is unavailable: {exc}.",
                )
            )
            return None
        if history is None:
            found.append(
                (
                    _R.OPEN_RECOMMENDATION_STATE_UNAVAILABLE,
                    "No authoritative session recommendation record is established; an "
                    "empty session is never assumed (EI4-D8).",
                )
            )
            return None
        if not isinstance(history, EquitySessionRecommendationHistory) or (
            history.exchange_code,
            history.trading_date,
        ) != (exchange, day):
            found.append(
                (
                    _R.OPEN_RECOMMENDATION_STATE_UNAVAILABLE,
                    "The session recommendation record answered another session.",
                )
            )
            return None
        if not history.covers(as_of):
            found.append(
                (
                    _R.RECOMMENDATION_HISTORY_INCOMPLETE,
                    f"The session recommendation record {history.reference} is complete only "
                    f"through {history.complete_through}, before the modeled issuance {as_of}.",
                )
            )
            return None
        return EquityIssuedHistoryEvidence.as_of_view(history, as_of)


# ---------------------------------------------------------------------------
# Pure assessment
# ---------------------------------------------------------------------------


def _candidate_detail(candidate: EquityRecommendationCandidateProvenance) -> str:
    return (
        f"Verified CANDIDATE entry {candidate.candidate_sequence} of the authoritative "
        f"timeline, decided at {candidate.source.decision_as_of}."
    )


def assess_equity_recommendation_inputs(
    request: EquityTradeRecommendationRequest,
    inputs: EquityRecommendationInputs,
    *,
    observed_timeline_head: str | None = None,
) -> EquityRecommendationEligibilityAudit:
    """Assess ``request`` from the frozen ``inputs`` alone, with no port and no clock.

    The use case calls it with the inputs it gathered; historical replay calls
    it with stored inputs. Equal inputs always give an equal assessment and
    assessment fingerprint. ``observed_timeline_head`` is diagnostics only.
    """
    if not isinstance(request, EquityTradeRecommendationRequest):
        raise TypeError("request must be an EquityTradeRecommendationRequest.")
    if not isinstance(inputs, EquityRecommendationInputs):
        raise TypeError("inputs must be EquityRecommendationInputs.")
    candidate = inputs.candidate
    history = inputs.issued_history
    results = {
        _CHECK.AUTHORITATIVE_CANDIDATE: _result(
            _CHECK.AUTHORITATIVE_CANDIDATE, _candidate_detail(candidate)
        ),
        _CHECK.SYNTHETIC_REFERENCES: _result(
            _CHECK.SYNTHETIC_REFERENCES,
            f"Every input of policy set {inputs.policy_set.reference.version} was found under "
            f"its version; the issued-history record {history.reference} is complete through "
            f"{history.complete_through} at recording watermark {history.recording_watermark}.",
        ),
    }

    def audit(**found: object) -> EquityRecommendationEligibilityAudit:
        return _audit(
            request,
            results,
            candidate=candidate,
            inputs=inputs,
            observed_timeline_head=observed_timeline_head,
            **found,  # type: ignore[arg-type]
        )

    timing = _timing(request.trading_date, inputs)
    window = timing if isinstance(timing, EquityModeledEntryWindow) else None
    results[_CHECK.SESSION_TIMING] = _result(
        _CHECK.SESSION_TIMING,
        timing
        if isinstance(timing, _Failed)
        else (
            f"Earliest full entry bar opens {timing.timing.eligible_from}, at or after the "
            f"modeled issuance {timing.timing.issued_at}; no entry bar may open after "
            f"{timing.latest_entry_bar_opens_at}; modeled exit {timing.modeled_exit_at}."
        ),
    )
    results[_CHECK.LISTING_ELIGIBILITY] = _result(
        _CHECK.LISTING_ELIGIBILITY, _listing(inputs.listing_eligibility)
    )
    results[_CHECK.OPEN_RECOMMENDATIONS] = _result(
        _CHECK.OPEN_RECOMMENDATIONS, _open_recommendations(request.listing, inputs)
    )
    if any(result.status is _STATUS.FAILED for result in results.values()):
        return audit(entry_window=window)
    assert window is not None

    policy = inputs.policy_set
    core = form_equity_recommendation(
        candidate.source,
        candidate.signal_close,
        candidate.opening_range_low,
        policy.generation_policy,
        policy.risk_policy,
        policy.price_increment,
        policy.cost_schedule,
    )
    if not isinstance(core, EquityRecommendationDefinition):
        stated = ", ".join(reason.value for reason in core.reasons)
        outcome = (
            _failed(
                _R.CORE_FORMATION_UNRESOLVED,
                f"Core could not establish a result ({stated}): its exact search reached its "
                "bound, so no quantity was shown to pass or fail. Nothing is proposed.",
            )
            if EQUITY_CORE_UNRESOLVED_REASONS & set(core.reasons)
            else _failed(_R.CORE_FORMATION_REJECTED, f"Core formation rejected it: {stated}.")
        )
        results[_CHECK.CORE_FORMATION] = _result(_CHECK.CORE_FORMATION, outcome)
        return audit(entry_window=window, core_outcome=core)
    sizing = core.sizing
    results[_CHECK.CORE_FORMATION] = _result(
        _CHECK.CORE_FORMATION,
        f"Core formed a synthetic setup of {sizing.quantity} shares, net risk "
        f"{sizing.net_risk} and net reward {sizing.net_reward} {sizing.currency.value}.",
    )
    proposal = EquityTradeRecommendationProposal(
        generation_key=inputs.generation_key,
        definition=core,
        entry_window=window,
        listing_eligibility_version=inputs.listing_eligibility.version,
        universe_version=inputs.listing_eligibility.universe_version,
        not_assessed=tuple(check for check, _ in EQUITY_RECOMMENDATION_UNASSESSED_CONTROLS),
    )
    return audit(entry_window=window, core_outcome=core, proposal=proposal)


def _timing(
    trading_date: date, inputs: EquityRecommendationInputs
) -> EquityModeledEntryWindow | _Failed:
    policy = inputs.policy_set.timing_policy
    venue, candidate = inputs.venue_day, inputs.candidate
    if venue.classification is not EquityVenueDayClassification.NORMAL_SESSION:
        return _failed(
            _R.SESSION_NOT_SUPPORTED,
            f"The session is {venue.classification}: only normal sessions are covered, and "
            "a special or shortened session is refused, never re-timed.",
        )
    assert venue.utc_offset is not None and venue.common_closes_at is not None
    offset = timezone(venue.utc_offset)

    def local(value: time) -> PointInTime:
        return _point(datetime.combine(trading_date, value, tzinfo=offset))

    signal_cutoff = local(policy.latest_signal_bar_end)
    latest_opening = local(policy.latest_entry_bar_opening)
    exit_at = local(policy.modeled_exit_boundary)
    openings = candidate.entry_bar_openings()

    if exit_at.compare(venue.common_closes_at) > 0:
        return _failed(
            _R.SESSION_NOT_SUPPORTED,
            f"The session closes at {venue.common_closes_at}, before the modeled exit "
            f"boundary {exit_at}; it is refused, never re-timed.",
        )
    if (
        policy.exchange_code != candidate.source.listing.exchange_code
        or policy.research_window_policy_version != candidate.research_window_policy_version
        or latest_opening not in openings
        or exit_at not in candidate.bar_ends()
    ):
        return _failed(
            _R.TIMING_POLICY_NOT_APPLICABLE,
            f"Timing policy {policy.version} does not fit the "
            f"{candidate.research_window_policy_version} research-window grid of this session.",
        )

    decision = candidate.source.decision_as_of
    issued = inputs.modeled_issued_at
    found: _Found = []
    signal_end = candidate.signal_interval_end
    if signal_end.compare(signal_cutoff) > 0:
        found.append(
            (
                _R.AFTER_SIGNAL_CUTOFF,
                f"The signal bar ended {signal_end}, after the latest signal-bar end "
                f"{signal_cutoff}.",
            )
        )
    if decision.compare(candidate.research_window_closes_at) > 0:
        found.append(
            (
                _R.POST_WINDOW_DECISION,
                f"The decision at {decision} came after the research window closed at "
                f"{candidate.research_window_closes_at}: historical research, never actionable.",
            )
        )
    earliest = next((o for o in openings if o.compare(issued) >= 0), None)
    if earliest is None or earliest.compare(latest_opening) > 0:
        found.append(
            (
                _R.NO_PERMISSIBLE_ENTRY_BAR,
                f"No full entry bar opens at or after the modeled issuance {issued} and by "
                f"the latest permissible entry-bar opening {latest_opening}.",
            )
        )
    elif _instant(exit_at) - _instant(earliest) < policy.minimum_remaining_session:
        found.append(
            (
                _R.INSUFFICIENT_SESSION_TIME,
                f"From {earliest} to the modeled exit boundary {exit_at} is less than the "
                "minimum remaining session time.",
            )
        )
    if found:
        return _collect(found)
    assert earliest is not None
    return EquityModeledEntryWindow(
        timing=EquityRecommendationTiming(decision, EquityIssuanceKind.MODELED, issued, earliest),
        latest_entry_bar_opens_at=latest_opening,
        modeled_exit_at=exit_at,
        timing_policy_version=policy.version,
    )


def _listing(eligibility: EquitySyntheticListingEligibility) -> _Failed | str:
    ineligible = eligibility.with_verdict(EquitySyntheticEligibilityVerdict.INELIGIBLE)
    if ineligible:
        return _failed(
            _R.LISTING_NOT_ELIGIBLE,
            f"Synthetic eligibility {eligibility.version} states the listing is not eligible "
            "for: " + ", ".join(criterion.value for criterion in ineligible) + ".",
        )
    return (
        f"Synthetic eligibility {eligibility.version} (universe "
        f"{eligibility.universe_version}) states every criterion ELIGIBLE; it is not "
        "production authority."
    )


def _open_recommendations(
    listing: ListingReference, inputs: EquityRecommendationInputs
) -> _Failed | str:
    history = inputs.issued_history
    as_of = history.as_of
    open_now = history.recommendations  # in (issued_at, key) order
    same = [r.recommendation_key for r in open_now if r.listing == listing]
    other = [r.recommendation_key for r in open_now if r.listing != listing]
    found: _Found = []
    if same:
        found.append(
            (
                _R.EXCLUSIVITY_POLICY_UNRESOLVED,
                f"{len(same)} recommendation(s) for {listing} issued by {as_of} are "
                "treated as open (EI4-D8): " + ", ".join(same) + ". No exclusivity rule is "
                "approved, so whether another may coexist is unresolved; this is not a "
                "breach.",
            )
        )
    if other:
        found.append(
            (
                _R.EXPOSURE_POLICY_UNRESOLVED,
                f"{len(other)} recommendation(s) for other listings issued by {as_of} are "
                "treated as open (EI4-D8): " + ", ".join(other) + ". No exposure or "
                "simultaneous-open limit is approved, so the combined exposure is "
                "unresolved; this is not a breach.",
            )
        )
    if found:
        return _collect(found)
    return (
        f"Record {history.reference} at recording watermark {history.recording_watermark} "
        f"lists no recommendation issued by {as_of}."
    )
