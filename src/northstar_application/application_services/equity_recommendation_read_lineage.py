"""Exact generation lineage of an authoritative issued recommendation (EI-7C, EI7-CF4).

An EI-7 read never re-evaluates a generation (EI-7 R-6). It proves, from
stored facts only, that a recorded MODELED issuance and the generation
disposition stored under that issuance's exact ``generation_key`` are one
lineage. ``verify_equity_issued_generation_lineage`` applies the ten required
comparisons of the EI-7 Decision Register, section 13.4, in order:

1. the disposition exists under the issuance's generation key, has that key,
   is PROPOSED and carries its proposal;
2. strategy, 3. listing and 4. trading date of the research candidate equal
   the issuance's;
5. policy and version identity: the policy set the assessment applied is the
   one the request named; the proposal's generation, risk, cost, increment
   and timing versions are that policy set's; its listing-eligibility and
   universe versions are the frozen eligibility evidence's and the request's;
   and the research configuration version is the requested configuration's;
6. the proposal was formed from exactly the disposition's research candidate;
7. levels, direction and modeled timing of the issuance are the proposal's;
8. the issuance binds the proposal (its fingerprint) that holds the sizing and
   risk definition;
9. evaluator compatibility: issuance accepts only evaluator ``/2`` (ADR-025);
10. ``equity_issued_recommendation(disposition)`` reproduces the recorded
    issued recommendation exactly, identity included.

Any failure refuses with PERSISTED_AUTHORITY_INVALID: an issued
recommendation's missing or inconsistent generation is corruption of required
authority, never UNKNOWN. The disposition record already verified its own
documents, fingerprints and evaluator pairing on construction
(EquityRecommendationGenerationDisposition); nothing here re-implements that.

Two provenance limits are carried, never hidden (EI-7 register, section
13.4): the historical publication of the policy set cannot be established
from persisted facts, and the research candidate is the frozen reference
bound by the generation key -- its authenticity against the timeline needs
EI3-D13 rehydration, which only the research query performs.

Nothing here reads a clock, writes, or calls a provider.
"""

from __future__ import annotations

from dataclasses import dataclass

from northstar_core.recommendations.equity import (
    EquityRecommendationDefinition,
    EquityResearchCandidateReference,
    EquitySyntheticSizing,
    EquitySyntheticVersion,
)

from northstar_application.application_services.equity_read_query_errors import (
    EquityReadQueryRefusal,
    refuse_equity_read_query,
)
from northstar_application.application_services.equity_recommendation_generation_evidence import (
    EQUITY_RECOMMENDATION_EVALUATOR_V2,
    EquityRecommendationCandidateProvenance,
    EquitySyntheticPolicySetReference,
)
from northstar_application.application_services.equity_recommendation_generation_records import (
    EquityRecommendationDispositionKind,
    EquityRecommendationGenerationDisposition,
)
from northstar_application.application_services.equity_recommendation_issuance_records import (
    EquityRecordedIssuance,
    InvalidEquityIssuedRecommendationError,
    equity_issued_recommendation,
)
from northstar_application.application_services.equity_trade_recommendation_eligibility import (
    EquityModeledEntryWindow,
    EquityTradeRecommendationProposal,
)
from northstar_application.application_services.evaluate_equity_intraday_research import (
    EquityOpeningRangeBreakoutConfiguration,
)

_INVALID = EquityReadQueryRefusal.PERSISTED_AUTHORITY_INVALID


class _LineageMismatch(Exception):
    """Internal: one named comparison failed."""

    def __init__(self, check: str) -> None:
        super().__init__(check)
        self.check = check


def _require(condition: bool, check: str) -> None:
    if not condition:
        raise _LineageMismatch(check)


@dataclass(frozen=True, slots=True)
class EquityVerifiedGenerationLineage:
    """The verified generation facts of one authoritative issued recommendation.

    Built only by ``verify_equity_issued_generation_lineage``. Every field is
    a reused, immutable domain value from the stored disposition; no raw
    stored document is held. ``policy_publication_established`` is always
    False: publication history is not stored. ``research_candidate_rehydrated``
    is always False here: ``candidate`` is the frozen reference bound by the
    generation key.
    """

    issuance: EquityRecordedIssuance
    generation_key: str
    evaluator_version: str
    policy_set: EquitySyntheticPolicySetReference
    policy_set_fingerprint: str
    inputs_fingerprint: str
    research_configuration: EquityOpeningRangeBreakoutConfiguration
    candidate: EquityRecommendationCandidateProvenance
    proposal: EquityTradeRecommendationProposal
    listing_eligibility_version: EquitySyntheticVersion
    universe_version: EquitySyntheticVersion

    def __post_init__(self) -> None:
        if not isinstance(self.issuance, EquityRecordedIssuance):
            raise TypeError("issuance must be an EquityRecordedIssuance.")
        if not isinstance(self.proposal, EquityTradeRecommendationProposal):
            raise TypeError("proposal must be an EquityTradeRecommendationProposal.")
        if not isinstance(self.candidate, EquityRecommendationCandidateProvenance):
            raise TypeError("candidate must be an EquityRecommendationCandidateProvenance.")
        issued = self.issuance.recommendation
        if (issued.generation_key, issued.proposal_fingerprint) != (
            self.generation_key,
            self.proposal.fingerprint,
        ) or self.proposal.generation_key != self.generation_key:
            raise ValueError("A verified lineage binds one generation key and one proposal.")
        if self.evaluator_version != EQUITY_RECOMMENDATION_EVALUATOR_V2:
            raise ValueError("An issued recommendation's generation is evaluator /2 only.")

    @property
    def research_reference(self) -> EquityResearchCandidateReference:
        """Return the frozen research candidate reference bound by the generation key."""
        return self.candidate.source

    @property
    def definition(self) -> EquityRecommendationDefinition:
        """Return Core's recommendation definition: levels, sizing and policy versions."""
        return self.proposal.definition

    @property
    def sizing(self) -> EquitySyntheticSizing:
        """Return the synthetic sizing and modeled cost, risk and reward estimates."""
        return self.proposal.definition.sizing

    @property
    def entry_window(self) -> EquityModeledEntryWindow:
        """Return the modeled timing and entry-eligibility bounds."""
        return self.proposal.entry_window

    @property
    def policy_publication_established(self) -> bool:
        """Return False: no persisted fact establishes when the policy set was published."""
        return False

    @property
    def research_candidate_rehydrated(self) -> bool:
        """Return False: the candidate is the frozen reference, not a rehydrated entry."""
        return False


def _compare(
    recorded: EquityRecordedIssuance, disposition: EquityRecommendationGenerationDisposition | None
) -> EquityVerifiedGenerationLineage:
    issued = recorded.recommendation
    # 1. The exact-key disposition exists, has the key, is PROPOSED and carries a proposal.
    _require(disposition is not None, "1: the issuance's generation disposition is missing")
    assert disposition is not None
    _require(disposition.generation_key == issued.generation_key, "1: generation key")
    _require(
        disposition.kind is EquityRecommendationDispositionKind.PROPOSED
        and disposition.proposal is not None,
        "1: the disposition is not PROPOSED",
    )
    proposal = disposition.proposal
    assert proposal is not None
    # 9. Issuance accepts only evaluator /2 (checked before reconstruction relies on it).
    _require(
        disposition.evaluator_version == EQUITY_RECOMMENDATION_EVALUATOR_V2,
        "9: evaluator compatibility",
    )
    inputs = disposition.inputs
    source = disposition.candidate.source
    request = disposition.request
    # 2-4. Strategy, listing and trading date.
    _require(source.strategy == issued.strategy, "2: strategy")
    _require(source.listing == issued.listing, "3: listing")
    _require(source.trading_date == issued.trading_date, "4: trading date")
    # 5. Policy and version identity.
    policy = inputs.policy_set.reference
    definition = proposal.definition
    window = proposal.entry_window
    eligibility = inputs.listing_eligibility
    _require(request.policy_set == policy, "5: applied policy set is the requested one")
    _require(
        (
            definition.generation_policy_version,
            definition.risk_policy_version,
            definition.cost_schedule_version,
            definition.price_increment.version,
            window.timing_policy_version,
        )
        == (
            policy.generation_policy_version,
            policy.risk_policy_version,
            policy.cost_schedule_version,
            policy.price_increment_version,
            policy.timing_policy_version,
        ),
        "5: policy component versions",
    )
    _require(
        proposal.listing_eligibility_version
        == eligibility.version
        == request.listing_eligibility_version,
        "5: listing eligibility version",
    )
    _require(proposal.universe_version == eligibility.universe_version, "5: universe version")
    _require(
        request.configuration.strategy == source.strategy
        and request.configuration.version == source.research_configuration_version.value,
        "5: research configuration version",
    )
    # 6. The proposal was formed from exactly this research candidate.
    _require(definition.source == source, "6: research candidate reference")
    # 7. Levels, direction and modeled timing.
    _require(
        (issued.direction, issued.levels) == (definition.direction, definition.levels),
        "7: levels and direction",
    )
    timing = window.timing
    _require(
        (
            issued.decision_as_of,
            issued.issued_at,
            issued.eligible_from,
            issued.latest_entry_bar_opens_at,
            issued.modeled_exit_at,
            issued.timing_policy_version,
        )
        == (
            timing.decision_as_of,
            timing.issued_at,
            timing.eligible_from,
            window.latest_entry_bar_opens_at,
            window.modeled_exit_at,
            window.timing_policy_version,
        ),
        "7: modeled timing",
    )
    # 8. The issuance binds the proposal that holds the sizing and risk definition.
    _require(
        (issued.proposal_fingerprint, issued.assessment_fingerprint)
        == (proposal.fingerprint, disposition.assessment_fingerprint),
        "8: proposal and assessment binding",
    )
    # 10. Canonical reconstruction reproduces the recorded issuance exactly.
    try:
        reconstructed = equity_issued_recommendation(disposition)
    except InvalidEquityIssuedRecommendationError as exc:
        raise _LineageMismatch("10: the disposition does not reconstruct an issuance") from exc
    _require(reconstructed == issued, "10: reconstructed issuance equivalence")
    return EquityVerifiedGenerationLineage(
        issuance=recorded,
        generation_key=disposition.generation_key,
        evaluator_version=disposition.evaluator_version,
        policy_set=policy,
        policy_set_fingerprint=disposition.policy_set_fingerprint,
        inputs_fingerprint=disposition.inputs_fingerprint,
        research_configuration=request.configuration,
        candidate=disposition.candidate,
        proposal=proposal,
        listing_eligibility_version=proposal.listing_eligibility_version,
        universe_version=proposal.universe_version,
    )


def verify_equity_issued_generation_lineage(
    recorded: EquityRecordedIssuance, disposition: EquityRecommendationGenerationDisposition | None
) -> EquityVerifiedGenerationLineage:
    """Return the verified lineage, or refuse with PERSISTED_AUTHORITY_INVALID.

    ``disposition`` is what the generation reader returned for the issuance's
    exact ``generation_key`` (None when nothing is stored there).
    """
    if not isinstance(recorded, EquityRecordedIssuance):
        raise TypeError("recorded must be an EquityRecordedIssuance.")
    if disposition is not None and not isinstance(
        disposition, EquityRecommendationGenerationDisposition
    ):
        raise refuse_equity_read_query(
            _INVALID, "the generation reader answered something other than a disposition"
        )
    try:
        return _compare(recorded, disposition)
    except _LineageMismatch as exc:
        raise refuse_equity_read_query(
            _INVALID, f"generation lineage comparison failed ({exc.check})"
        ) from exc
