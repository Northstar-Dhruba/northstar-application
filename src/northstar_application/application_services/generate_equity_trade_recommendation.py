"""Synthetic equity intraday recommendation eligibility and generation (EI-4C).

GenerateEquityTradeRecommendationUseCase evaluates the first authoritative
EI-3C research CANDIDATE of one strategy configuration, listing and trading
date against explicitly versioned synthetic policies, and -- when every
implemented check passes -- invokes the merged EI-4B Core formation to produce
a synthetic, non-authoritative proposal. It applies the owner decisions
recorded on 2026-11-07 in the EI-4 Decision Register (northstar-docs,
product/Equity-Intraday-V1-EI4-Decision-Register.md) and ADR-020: EI4-D1,
EI0-D7 (synthetic structure only), EI4-D2, EI0-D14, EI4-D3, EI4-D4, EI4-D6,
EI4-D7 and EI4-D8. Nothing is issued, persisted, scanned, filled or traded,
and no clock, provider, broker or network is read.

Every attempt returns an EquityRecommendationEligibilityAudit, whose status
separates a refused opportunity from an integrity failure, an unavailable
dependency and a proposal (equity_trade_recommendation_eligibility). Checks
run in this order, and a check runs only when every earlier step passed:

1. AUTHORITATIVE_CANDIDATE. The timeline is loaded from the EI-3C store and
   rehydrated through ``rehydrate_equity_intraday_research_timeline`` -- whole
   or not at all (EI3-D13). An uncommitted or in-memory timeline is never
   used. Only its CANDIDATE is considered (EI4-D2): REPEAT_QUALIFYING,
   NO_SIGNAL and REJECTED entries never generate, and no later entry
   replaces the candidate. Its evidence is then re-checked: the snapshot
   loads and is the one named, synthetic, at the decision instant; the
   signal bar is the snapshot's latest required interval and ended before the
   decision; and the research-window grid tiles the window. The signal close
   and opening-range low are the EI-3A decision's own values, priced in the
   snapshot's currency; nothing is recomputed from later candles.
2. SYNTHETIC_REFERENCES. Each input is loaded by its exact version: timing,
   generation, risk (with model capital), cost and increment references, the
   venue-day reference, the synthetic listing eligibility and the authoritative
   session recommendation record, read as of the modeled issuance instant. The
   record must be complete through that instant; the point-in-time view is
   the recommendations issued at or before it, so a later issuance never
   changes an earlier assessment. Every missing, unknown, incomplete,
   inconsistent or unanswerable input is reported; none is defaulted.
3. SESSION_TIMING (EI4-D1, EI0-D7 structure), LISTING_ELIGIBILITY (EI4-D7)
   and OPEN_RECOMMENDATIONS (EI4-D8) are evaluated together:

   - only a NORMAL venue session is covered, and one that closes before the
     modeled exit boundary is refused, never re-timed;
   - the modeled issuance may not precede the decision instant;
   - the signal bar must end by the policy's latest signal-bar end, and a
     decision after the research window closed is historical research only;
   - the earliest permissible entry is the first research-window grid bar
     that *opens* at or after the modeled issuance -- a bar that opened before
     the decision was available never evidences a full-bar entry -- and it
     must open no later than the latest permissible entry-bar opening and
     leave the minimum remaining time before the modeled exit boundary;
   - every eligibility criterion must be ELIGIBLE;
   - every recommendation in the point-in-time view is treated as open,
     whatever its levels might have touched (EI4-D8). EI4-D8 approves no
     exclusivity rule or exposure limit, so whether the candidate may coexist
     with one is unresolved: EXCLUSIVITY_POLICY_UNRESOLVED for the same
     listing, EXPOSURE_POLICY_UNRESOLVED for another. Both fail closed and
     neither is reported as a breach.
4. CORE_FORMATION. ``form_equity_recommendation`` runs on the candidate's
   research prices and the loaded policies, and its exact reasons are kept
   in the audit. A confirmed Core rejection is CORE_FORMATION_REJECTED. A
   result Core could not establish -- SIZING_SEARCH_LIMIT_EXCEEDED, where the
   exact search reached its bound -- is CORE_FORMATION_UNRESOLVED, fails
   closed, and is never reported as a failed reward-to-risk test.

SAME_LISTING_EXCLUSIVITY, PORTFOLIO_EXPOSURE, DAILY_LOSS_LIMIT,
PRE_ENTRY_INVALIDATION, CORPORATE_ACTIONS, EXCHANGE_SURVEILLANCE and
CANDLE_REVISION_MATERIALITY are always NOT_ASSESSED: their policies or inputs
do not exist, and a proposal names them as blocking actionable issuance.

Failure semantics
-----------------
Expected outages of the injected ports -- the timeline store, the snapshot
store, the synthetic reference sources and the session state -- become
DEPENDENCY_UNAVAILABLE with the source's message, and so does an answer for
another version, listing, date or session, which is never substituted.
Verification failures of stored history become RESEARCH_INTEGRITY_FAILURE.
Programming errors -- a wrong argument type, an unexpected exception -- raise
and produce no audit. Identical inputs and stored history always give an
identical audit: the same status, reasons, key, levels, costs and sizing.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, time, timezone

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
    EquityCandleEvidenceSnapshot,
)
from northstar_application.application_services.equity_trade_recommendation_eligibility import (
    EQUITY_CORE_UNRESOLVED_REASONS,
    EQUITY_RECOMMENDATION_UNASSESSED_CONTROLS,
    EquityModeledEntryWindow,
    EquityRecommendationCandidateProvenance,
    EquityRecommendationCheck,
    EquityRecommendationCheckResult,
    EquityRecommendationCheckStatus,
    EquityRecommendationEligibilityAudit,
    EquityRecommendationEligibilityReason,
    EquityRecommendationInputs,
    EquityTradeRecommendationProposal,
    EquityTradeRecommendationRequest,
    equity_trade_recommendation_proposal_key,
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
    EquityPriorSessionRecommendation,
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


def _collect(found: list[tuple[EquityRecommendationEligibilityReason, str]]) -> _Failed:
    return _Failed(frozenset(r for r, _ in found), " ".join(detail for _, detail in found))


def _instant(value: PointInTime) -> datetime:
    return datetime.fromisoformat(value.value.replace("Z", "+00:00")).astimezone(UTC)


def _point(value: datetime) -> PointInTime:
    return PointInTime(value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"))


@dataclass(frozen=True, slots=True)
class _Candidate:
    provenance: EquityRecommendationCandidateProvenance
    snapshot: EquityCandleEvidenceSnapshot
    signal_close: Price
    opening_range_low: Price


@dataclass(frozen=True, slots=True)
class _References:
    timing: EquitySyntheticEntryTimingPolicy
    generation: EquitySyntheticGenerationPolicy
    risk: EquitySyntheticRiskPolicy
    costs: EquitySyntheticCostSchedule
    increment: EquityPriceIncrement
    venue: EquityVenueDayReference
    eligibility: EquitySyntheticListingEligibility
    history: EquitySessionRecommendationHistory
    open_recommendations: tuple[EquityPriorSessionRecommendation, ...]

    def inputs(self) -> EquityRecommendationInputs:
        return EquityRecommendationInputs(
            timing_policy=self.timing,
            generation_policy=self.generation,
            risk_policy=self.risk,
            cost_schedule=self.costs,
            price_increment=self.increment,
            listing_eligibility=self.eligibility,
            venue_reference=self.venue.reference,
            history_reference=self.history.reference,
            history_complete_through=self.history.complete_through,
            open_recommendations=self.open_recommendations,
        )


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
        results: dict[EquityRecommendationCheck, EquityRecommendationCheckResult] = {}

        def audit(**found: object) -> EquityRecommendationEligibilityAudit:
            return self._audit(request, results, **found)  # type: ignore[arg-type]

        candidate = self._candidate(request)
        if isinstance(candidate, _Failed):
            results[_CHECK.AUTHORITATIVE_CANDIDATE] = _result(
                _CHECK.AUTHORITATIVE_CANDIDATE, candidate
            )
            return audit()
        results[_CHECK.AUTHORITATIVE_CANDIDATE] = _result(
            _CHECK.AUTHORITATIVE_CANDIDATE,
            f"Verified CANDIDATE entry {candidate.provenance.candidate_sequence} of the "
            f"authoritative timeline (head {candidate.provenance.timeline_head_fingerprint}).",
        )
        provenance = candidate.provenance

        references = self._load_references(request, candidate)
        if isinstance(references, _Failed):
            results[_CHECK.SYNTHETIC_REFERENCES] = _result(_CHECK.SYNTHETIC_REFERENCES, references)
            return audit(candidate=provenance)
        results[_CHECK.SYNTHETIC_REFERENCES] = _result(
            _CHECK.SYNTHETIC_REFERENCES,
            "Every synthetic reference was found under its requested version, and the "
            f"recommendation record {references.history.reference} is complete through "
            f"{references.history.complete_through}.",
        )
        inputs = references.inputs()

        timing = self._timing(request, candidate, references)
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
            _CHECK.LISTING_ELIGIBILITY, self._listing(references.eligibility)
        )
        results[_CHECK.OPEN_RECOMMENDATIONS] = _result(
            _CHECK.OPEN_RECOMMENDATIONS, self._open_recommendations(request, references)
        )
        if any(result.status is _STATUS.FAILED for result in results.values()):
            return audit(candidate=provenance, inputs=inputs, entry_window=window)
        assert window is not None

        core = form_equity_recommendation(
            provenance.source,
            candidate.signal_close,
            candidate.opening_range_low,
            references.generation,
            references.risk,
            references.increment,
            references.costs,
        )
        if not isinstance(core, EquityRecommendationDefinition):
            stated = ", ".join(reason.value for reason in core.reasons)
            outcome = (
                _failed(
                    _R.CORE_FORMATION_UNRESOLVED,
                    f"Core could not establish a result ({stated}): its exact search reached "
                    "its bound, so no quantity was shown to pass or fail. Nothing is proposed.",
                )
                if EQUITY_CORE_UNRESOLVED_REASONS & set(core.reasons)
                else _failed(_R.CORE_FORMATION_REJECTED, f"Core formation rejected it: {stated}.")
            )
            results[_CHECK.CORE_FORMATION] = _result(_CHECK.CORE_FORMATION, outcome)
            return audit(
                candidate=provenance, inputs=inputs, entry_window=window, core_outcome=core
            )
        sizing = core.sizing
        results[_CHECK.CORE_FORMATION] = _result(
            _CHECK.CORE_FORMATION,
            f"Core formed a synthetic setup of {sizing.quantity} shares, net risk "
            f"{sizing.net_risk} and net reward {sizing.net_reward} {sizing.currency.value}.",
        )
        proposal = EquityTradeRecommendationProposal(
            proposal_key=equity_trade_recommendation_proposal_key(
                provenance.source, references.generation.version
            ),
            definition=core,
            entry_window=window,
            listing_eligibility_version=references.eligibility.version,
            universe_version=references.eligibility.universe_version,
            not_assessed=tuple(check for check, _ in EQUITY_RECOMMENDATION_UNASSESSED_CONTROLS),
        )
        return audit(
            candidate=provenance,
            inputs=inputs,
            entry_window=window,
            core_outcome=core,
            proposal=proposal,
        )

    # ------------------------------------------------------------------
    # Audit assembly
    # ------------------------------------------------------------------

    @staticmethod
    def _audit(
        request: EquityTradeRecommendationRequest,
        results: dict[EquityRecommendationCheck, EquityRecommendationCheckResult],
        candidate: EquityRecommendationCandidateProvenance | None = None,
        inputs: EquityRecommendationInputs | None = None,
        entry_window: EquityModeledEntryWindow | None = None,
        core_outcome: object = None,
        proposal: EquityTradeRecommendationProposal | None = None,
    ) -> EquityRecommendationEligibilityAudit:
        unassessed = dict(EQUITY_RECOMMENDATION_UNASSESSED_CONTROLS)
        checks = []
        for check in EquityRecommendationCheck:
            if check in results:
                checks.append(results[check])
            elif check in unassessed:
                checks.append(
                    EquityRecommendationCheckResult(
                        check, _STATUS.NOT_ASSESSED, (), unassessed[check]
                    )
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
                _R.CANDIDATE_EVIDENCE_NOT_SYNTHETIC,
                "EI-4C generates from synthetic evidence only.",
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
        signal_end = values.signal_interval_end
        required = evidence.required_interval_ends
        if not required or required[-1] != signal_end:
            return invalid("its signal bar is not the latest interval its decision required.")
        if signal_end.compare(snapshot.as_of) >= 0:
            return invalid("its decision instant is not after its signal bar ended.")
        intervals = snapshot.intervals
        tiled = (
            intervals[0].opens_at == snapshot.window_opens_at
            and intervals[-1].closes_at == snapshot.window_closes_at
            and all(
                a.closes_at == b.opens_at for a, b in zip(intervals, intervals[1:], strict=False)
            )
        )
        if not tiled or signal_end not in {interval.closes_at for interval in intervals}:
            return invalid(
                "its research-window grid does not tile the window around the signal bar."
            )
        try:
            source = EquityResearchCandidateReference(
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
            )
            signal_close = Price(values.signal_close, snapshot.currency)
            opening_range_low = Price(values.opening_range_low, snapshot.currency)
        except ValidationError as exc:
            return invalid(f"its research values cannot be referenced: {exc}")
        return _Candidate(
            provenance=EquityRecommendationCandidateProvenance(
                source=source,
                manifest_fingerprint=evidence.manifest_fingerprint,
                candidate_sequence=entry.sequence,
                timeline_head_fingerprint=head,
                signal_interval_end=signal_end,
            ),
            snapshot=snapshot,
            signal_close=signal_close,
            opening_range_low=opening_range_low,
        )

    # ------------------------------------------------------------------
    # 2. Synthetic references
    # ------------------------------------------------------------------

    def _load_references(
        self, request: EquityTradeRecommendationRequest, candidate: _Candidate
    ) -> _References | _Failed:
        found: list[tuple[EquityRecommendationEligibilityReason, str]] = []
        sources = self._references
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
            request.timing_policy_version,
            _R.TIMING_POLICY_UNAVAILABLE,
            EquitySyntheticEntryTimingPolicy,
            lambda: sources.timing_policy(request.timing_policy_version),
        )
        generation = versioned(
            "generation policy",
            request.generation_policy_version,
            _R.GENERATION_POLICY_UNAVAILABLE,
            EquitySyntheticGenerationPolicy,
            lambda: sources.generation_policy(request.generation_policy_version),
        )
        risk = versioned(
            "risk policy",
            request.risk_policy_version,
            _R.RISK_POLICY_UNAVAILABLE,
            EquitySyntheticRiskPolicy,
            lambda: sources.risk_policy(request.risk_policy_version),
        )
        costs = versioned(
            "cost schedule",
            request.cost_schedule_version,
            _R.COST_SCHEDULE_UNAVAILABLE,
            EquitySyntheticCostSchedule,
            lambda: sources.cost_schedule(request.cost_schedule_version),
        )
        increment = versioned(
            "price increment",
            request.price_increment_version,
            _R.PRICE_INCREMENT_UNAVAILABLE,
            EquityPriceIncrement,
            lambda: sources.price_increment(request.price_increment_version, listing, day),
        )

        venue = self._venue_days.venue_day(listing.exchange_code, day)
        if not isinstance(venue, EquityVenueDayReference) or (
            venue.exchange_code,
            venue.trading_date,
        ) != (listing.exchange_code, day):
            found.append(
                (_R.SESSION_REFERENCE_MISMATCH, "The venue reference answered another session.")
            )
        elif venue.classification in _UNANSWERED_DAYS:
            found.append(
                (
                    _R.SESSION_REFERENCE_UNAVAILABLE,
                    f"The venue reference cannot resolve the session: {venue.classification}.",
                )
            )
        elif venue.reference != candidate.snapshot.venue_reference:
            found.append(
                (
                    _R.SESSION_REFERENCE_MISMATCH,
                    f"The venue reference {venue.reference} is not the candidate evidence's "
                    f"{candidate.snapshot.venue_reference}.",
                )
            )

        eligibility = self._listing_reference(request, found)
        history = self._session_history(request, found)
        if found or history is None:
            return _collect(found)
        return _References(
            timing=timing,  # type: ignore[arg-type]
            generation=generation,  # type: ignore[arg-type]
            risk=risk,  # type: ignore[arg-type]
            costs=costs,  # type: ignore[arg-type]
            increment=increment,  # type: ignore[arg-type]
            venue=venue,
            eligibility=eligibility,  # type: ignore[arg-type]
            history=history,
            open_recommendations=history.issued_by(request.modeled_issued_at),
        )

    def _listing_reference(
        self,
        request: EquityTradeRecommendationRequest,
        found: list[tuple[EquityRecommendationEligibilityReason, str]],
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
        self,
        request: EquityTradeRecommendationRequest,
        found: list[tuple[EquityRecommendationEligibilityReason, str]],
    ) -> EquitySessionRecommendationHistory | None:
        exchange, day = request.listing.exchange_code, request.trading_date
        as_of = request.modeled_issued_at
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
        return history

    # ------------------------------------------------------------------
    # 3. Timing, listing eligibility and open recommendations
    # ------------------------------------------------------------------

    @staticmethod
    def _timing(
        request: EquityTradeRecommendationRequest,
        candidate: _Candidate,
        references: _References,
    ) -> EquityModeledEntryWindow | _Failed:
        policy, venue, snapshot = references.timing, references.venue, candidate.snapshot
        if venue.classification is not EquityVenueDayClassification.NORMAL_SESSION:
            return _failed(
                _R.SESSION_NOT_SUPPORTED,
                f"The session is {venue.classification}: only normal sessions are covered, and "
                "a special or shortened session is refused, never re-timed.",
            )
        assert venue.utc_offset is not None and venue.common_closes_at is not None
        offset = timezone(venue.utc_offset)

        def local(value: time) -> PointInTime:
            return _point(datetime.combine(request.trading_date, value, tzinfo=offset))

        signal_cutoff = local(policy.latest_signal_bar_end)
        latest_opening = local(policy.latest_entry_bar_opening)
        exit_at = local(policy.modeled_exit_boundary)
        openings = tuple(interval.opens_at for interval in snapshot.intervals)
        ends = frozenset(interval.closes_at for interval in snapshot.intervals)

        if exit_at.compare(venue.common_closes_at) > 0:
            return _failed(
                _R.SESSION_NOT_SUPPORTED,
                f"The session closes at {venue.common_closes_at}, before the modeled exit "
                f"boundary {exit_at}; it is refused, never re-timed.",
            )
        if (
            policy.exchange_code != request.listing.exchange_code
            or policy.research_window_policy_version != snapshot.window_policy_version
            or latest_opening not in openings
            or exit_at not in ends
        ):
            return _failed(
                _R.TIMING_POLICY_NOT_APPLICABLE,
                f"Timing policy {policy.version} does not fit the {snapshot.window_policy_version} "
                "research-window grid of this session.",
            )

        decision = candidate.provenance.source.decision_as_of
        issued = request.modeled_issued_at
        if issued.compare(decision) < 0:
            return _failed(
                _R.MODELED_ISSUANCE_BEFORE_DECISION,
                f"Modeled issuance {issued} precedes the decision instant {decision}; nothing "
                "is backdated.",
            )
        found: list[tuple[EquityRecommendationEligibilityReason, str]] = []
        signal_end = candidate.provenance.signal_interval_end
        if signal_end.compare(signal_cutoff) > 0:
            found.append(
                (
                    _R.AFTER_SIGNAL_CUTOFF,
                    f"The signal bar ended {signal_end}, after the latest signal-bar end "
                    f"{signal_cutoff}.",
                )
            )
        if decision.compare(snapshot.window_closes_at) > 0:
            found.append(
                (
                    _R.POST_WINDOW_DECISION,
                    f"The decision at {decision} came after the research window closed at "
                    f"{snapshot.window_closes_at}: historical research, never actionable.",
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
            timing=EquityRecommendationTiming(
                decision, EquityIssuanceKind.MODELED, issued, earliest
            ),
            latest_entry_bar_opens_at=latest_opening,
            modeled_exit_at=exit_at,
            timing_policy_version=policy.version,
        )

    @staticmethod
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

    @staticmethod
    def _open_recommendations(
        request: EquityTradeRecommendationRequest, references: _References
    ) -> _Failed | str:
        as_of = request.modeled_issued_at
        open_now = references.open_recommendations  # in (issued_at, key) order
        same = [r.recommendation_key for r in open_now if r.listing == request.listing]
        other = [r.recommendation_key for r in open_now if r.listing != request.listing]
        found: list[tuple[EquityRecommendationEligibilityReason, str]] = []
        if same:
            found.append(
                (
                    _R.EXCLUSIVITY_POLICY_UNRESOLVED,
                    f"{len(same)} recommendation(s) for {request.listing} issued by {as_of} are "
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
        later = len(references.history.recommendations) - len(open_now)
        return (
            f"Record {references.history.reference} lists no recommendation issued by {as_of}; "
            f"{later} issued later do not affect this assessment."
        )
