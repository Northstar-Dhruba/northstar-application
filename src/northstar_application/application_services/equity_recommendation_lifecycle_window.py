"""The authoritative lifecycle window of an equity recommendation (EI-6B design, section 6).

A Core ``EquityLifecycleWindow`` needs the entry-window end, which neither the
proposal nor the issued record states: it is the close of the bar that opens
at ``latest_entry_bar_opens_at`` on the candidate's research-window grid. The
window is therefore always derived from verified source evidence, never
supplied by a caller.

- ``equity_lifecycle_window(proposal, candidate)`` is the grid derivation,
  shared with EI-5C's synthetic evaluator (one rule set): the latest entry-bar
  opening must be a grid opening and the modeled exit a grid close.
- ``equity_issued_lifecycle_window(issuance, disposition)`` is the governed
  derivation for an authoritative issuance. The disposition must be the
  PROPOSED, evaluator-``/2`` disposition the issued record re-derives from
  exactly; the window comes from its proposal and candidate grid, and must
  agree with the issued record's levels, eligibility, latest entry-bar opening
  and modeled exit.

Missing, inconsistent or undecodable provenance raises
InvalidEquityLifecycleWindowProvenanceError. There is no fallback window.
Nothing here reads a clock or calls out.
"""

from __future__ import annotations

from northstar_core.recommendations.equity.lifecycle import (
    EquityLifecycleWindow,
    InvalidEquityLifecycleWindowError,
)

from northstar_application.application_services.equity_recommendation_generation_evidence import (
    EQUITY_RECOMMENDATION_EVALUATOR_V2,
    EquityRecommendationCandidateProvenance,
)
from northstar_application.application_services.equity_recommendation_generation_records import (
    EquityRecommendationDispositionKind,
    EquityRecommendationGenerationDisposition,
)
from northstar_application.application_services.equity_recommendation_issuance_records import (
    EquityIssuedRecommendation,
    InvalidEquityIssuedRecommendationError,
    equity_issued_recommendation,
)
from northstar_application.application_services.equity_trade_recommendation_eligibility import (
    EquityTradeRecommendationProposal,
)

OFF_GRID = "The proposal's entry window does not lie on its research-window grid."


class InvalidEquityLifecycleWindowProvenanceError(ValueError):
    """Raised when a lifecycle window cannot be derived from authoritative provenance."""


def _refuse(message: str) -> InvalidEquityLifecycleWindowProvenanceError:
    return InvalidEquityLifecycleWindowProvenanceError(message)


def equity_lifecycle_window(
    proposal: EquityTradeRecommendationProposal,
    candidate: EquityRecommendationCandidateProvenance,
) -> EquityLifecycleWindow:
    """Return the Core window from a verified proposal and its candidate's grid."""
    entry = proposal.entry_window
    openings = candidate.entry_bar_openings()
    latest = entry.latest_entry_bar_opens_at
    if latest not in openings or entry.modeled_exit_at not in candidate.bar_ends():
        raise _refuse(OFF_GRID)
    index = openings.index(latest)
    ends = openings[index + 1] if index + 1 < len(openings) else candidate.research_window_closes_at
    return EquityLifecycleWindow(
        levels=proposal.definition.levels,
        eligible_from=entry.timing.eligible_from,
        latest_entry_bar_opens_at=latest,
        entry_window_ends_at=ends,
        modeled_exit_at=entry.modeled_exit_at,
    )


def equity_issued_lifecycle_window(
    issuance: EquityIssuedRecommendation,
    disposition: EquityRecommendationGenerationDisposition,
) -> EquityLifecycleWindow:
    """Return the lifecycle window of an authoritative issuance, from its verified source.

    The caller passes the issued record and the disposition it was issued
    from; nothing else enters the window. See the module docstring.
    """
    if not isinstance(issuance, EquityIssuedRecommendation):
        raise _refuse("A lifecycle window needs the authoritative EquityIssuedRecommendation.")
    if not isinstance(disposition, EquityRecommendationGenerationDisposition):
        raise _refuse("A lifecycle window needs the issuance's verified disposition.")
    if disposition.kind is not EquityRecommendationDispositionKind.PROPOSED:
        raise _refuse(f"The disposition is {disposition.kind}, not PROPOSED.")
    if disposition.evaluator_version != EQUITY_RECOMMENDATION_EVALUATOR_V2:
        raise _refuse(f"The disposition is from {disposition.evaluator_version}, not evaluator /2.")
    if disposition.generation_key != issuance.generation_key:
        raise _refuse("The disposition is not the one the issuance names.")
    try:
        derived = equity_issued_recommendation(disposition)
    except InvalidEquityIssuedRecommendationError as exc:
        raise _refuse(f"The disposition does not derive an issued record: {exc}") from exc
    if derived != issuance:
        raise _refuse("The issued record does not re-derive from its disposition.")
    proposal = disposition.proposal
    assert proposal is not None
    try:
        window = equity_lifecycle_window(proposal, disposition.candidate)
    except InvalidEquityLifecycleWindowError as exc:
        raise _refuse(f"The source does not form a lifecycle window: {exc}") from exc
    if (
        window.levels,
        window.eligible_from,
        window.latest_entry_bar_opens_at,
        window.modeled_exit_at,
    ) != (
        issuance.levels,
        issuance.eligible_from,
        issuance.latest_entry_bar_opens_at,
        issuance.modeled_exit_at,
    ):
        raise _refuse("The derived window disagrees with the issued record.")
    return window
