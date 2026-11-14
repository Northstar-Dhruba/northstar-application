"""EI-5E: synthetic end to end, from a verified proposal through issuance to lifecycle.

Layer E of the EI-5 design acceptance matrix (IS-01, IS-02, IS-03 and IS-17
end to end; HW-10; NB-04; RR-02), delivered with EI-5D in one Application PR
(D5D-2). Every component is the real merged one, in memory:

- EI-2B snapshots issued by the real use case over a synthetic journal, the
  EI-3C timeline committed by the real advancement (``ei4c`` fixtures);
- the labelled in-memory issuance authority (``ei5d._Authority``) and the
  completeness declaration use case;
- EI-4C version-2 generation, reading the authoritative history source;
- the verified EI-4D recorder and store double, and re-verification at levels
  A to C;
- the EI-5D issuer;
- the EI-5C synthetic lifecycle evaluator over the EI-5B Core lifecycle.

Nothing bypasses source verification, no exposure or exclusivity rule is
relaxed (EI4-D8), and nothing is filled, positioned or traded. The authority
is a test double: no durability, concurrency or restart is claimed (EI-6).
"""

from __future__ import annotations

from datetime import timedelta

import test_evaluate_synthetic_equity_recommendation_lifecycle as lifecycle
import test_generate_equity_trade_recommendation as ei4c
import test_issue_equity_trade_recommendation as ei5d
import test_record_equity_recommendation_generation as ei4d
from northstar_core.recommendations.equity.lifecycle import (
    EquityLifecycleState,
    EquityRecommendationLifecycle,
)

from northstar_application.application_services import (
    AuthoritativeEquitySessionRecommendationHistorySource,
    EquityRecommendationIssuanceRefusal,
    EquityRecommendationIssuanceRefusedError,
    EvaluateSyntheticEquityRecommendationLifecycleUseCase,
    equity_issued_recommendation,
    equity_recommendation_disposition,
    equity_recommendation_issuance_identity,
)
from northstar_application.ports import (
    EquityIssuanceDeclarationOutcome,
    EquityRecommendationIssuanceOutcome,
)

S = EquityLifecycleState
_O = EquityRecommendationIssuanceOutcome
_D = EquityIssuanceDeclarationOutcome
_LATE = "timing/ei5e-later"  # a later modeled issuance for the same candidate: 10:06 IST
# The end-to-end issuance identity, pinned (deterministic: no clock, sequence or process).
_E2E_IDENTITY = "eiris1-sha256:17c29950c0b14faf04b5acd66f7ffc5b046500f7b05dc739d879d0b3ea51f178"


def _chain() -> tuple[ei5d._World, lifecycle._Scenario]:
    """Steps 1 to 6: authority, declaration through b, v2 generation, recording, issuance."""
    world = ei4c._World({index: ei4c._MORNING[index] for index in range(1, 6)})
    world.commit(ei4c._ist(9, 31), ei4c._ist(9, 36), ei4c._ist(9, 41))
    issuance = ei5d._World(ei4c._Ports(world), ei5d._Authority(), ei4d._Store())
    # 1-2. The labelled in-memory authority, declared complete through b = t - 1 us.
    assert issuance.declare(ei5d._B).outcome is _D.DECLARED
    # 3. Version-2 generation reading the authoritative history (complete exactly through b).
    audit = issuance.generate()
    assert audit.status is ei4c._S.PROPOSED and audit.evaluator_version == ei5d._V2
    # 4. Recorded through the verified EI-4D recorder.
    issuance.record(audit)
    disposition = equity_recommendation_disposition(audit)
    assert disposition is not None
    use_case = EvaluateSyntheticEquityRecommendationLifecycleUseCase(
        issuance.verifier(), world.snapshots
    )
    scenario = lifecycle._Scenario(world, issuance.ports, disposition, issuance.store, use_case)
    return issuance, scenario


def test_e2e_a_verified_v2_proposal_is_issued_listed_in_history_and_starts_recorded() -> None:
    world, scenario = _chain()
    key = scenario.key
    # 5. Re-verified at levels A to C before anything treats it as issuable.
    verified = world.verifier().execute(key)
    assert verified is not None
    assert (verified.source_integrity_verified, verified.evaluation_reproduced) == (True, True)
    assert verified.disposition.inputs.issued_history.reference == ei5d._REF
    # 6. Issued: MODELED, at the authority's next recording sequence (IS-01).
    result = world.issue(key)
    assert result.outcome is _O.ISSUED and result.recording_sequence == 2
    issued = result.issuance
    assert issued == equity_issued_recommendation(verified.disposition)
    # 7. The authoritative history: at watermark 2 the session is complete only
    #    through b, so the issuance at t is not yet listed; declaring further lists it.
    source = AuthoritativeEquitySessionRecommendationHistorySource(world.authority)
    at_two = source.session_history(ei4c._NSE, ei4c._DAY, ei5d._B)
    assert at_two is not None and at_two.recommendations == ()
    assert world.declare(ei5d._ist(10, 0)).outcome is _D.DECLARED
    now = source.session_history(ei4c._NSE, ei4c._DAY, ei5d._ist(10, 0))
    assert now is not None and now.recording_watermark == 3
    assert now.recommendations == (issued.history_record(2),)
    assert world.authority.session_view_at(ei4c._NSE, ei4c._DAY, 2) == at_two  # stable
    # 8-9. The EI-5C lifecycle of the issued recommendation starts RECORDED (EI5-D2).
    evaluation = scenario.run(scenario.snap("09:41"))
    window = evaluation.window
    assert EquityRecommendationLifecycle(window, ()).state is S.RECORDED
    assert (window.levels, window.eligible_from) == (issued.levels, issued.eligible_from)
    assert (window.latest_entry_bar_opens_at, window.modeled_exit_at) == (
        issued.latest_entry_bar_opens_at,
        issued.modeled_exit_at,
    )
    assert evaluation.generation_key == issued.generation_key
    assert evaluation.proposal_fingerprint == issued.proposal_fingerprint
    assert evaluation.state is S.WAITING_FOR_ENTRY  # the first eligible snapshot (EI5-D2)


def test_e2e_replay_identities_and_retries_stay_stable_after_issuance() -> None:
    world, scenario = _chain()
    key = scenario.key
    first = world.issue(key)
    world.declare(ei5d._ist(10, 0))
    # IS-02 end to end: an identical retry after a later declaration is EQUIVALENT.
    again = world.issue(key)
    assert again.outcome is _O.EQUIVALENT and again.authoritative == first.authoritative
    # Replay: the stored proposal re-verifies and re-derives the same issued record.
    verified = world.verifier().execute(key)
    assert verified is not None
    assert equity_issued_recommendation(verified.disposition) == first.issuance
    # The EI-5C world journals only bars 1-5, so its proposal -- and identity -- is its own.
    assert (
        first.issuance.issuance_id
        == _E2E_IDENTITY
        == equity_recommendation_issuance_identity(
            key, first.issuance.proposal_fingerprint, first.issuance.issuance_kind
        )
    )
    # IS-17 end to end: the EI-4D records are unchanged by issuance.
    assert world.store.dispositions[key] == scenario.disposition
    # RR-02: the same snapshot sequence from RECORDED gives identical events.
    scenario.fill("09:40", "09:50", {"09:45": lifecycle._ENTRY})
    snapshots = (scenario.snap("09:41"), scenario.snap("09:51"))
    one, two = scenario.run(*snapshots), scenario.run(*snapshots)
    assert one.lifecycle.events == two.lifecycle.events and one.state is S.ACTIVE


def test_e2e_an_active_recommendation_creates_no_position_or_record() -> None:
    world, scenario = _chain()
    world.issue(scenario.key)
    issuances, dispositions = list(world.authority.issuances), dict(world.store.dispositions)
    attempts = dict(world.store.attempts)
    active = lifecycle._active(scenario)  # NB-04
    assert active.state is S.ACTIVE
    assert (active.issuance_established, active.is_authoritative) == (False, False)
    assert world.authority.issuances == issuances
    assert (world.store.dispositions, world.store.attempts) == (dispositions, attempts)


def test_e2e_a_second_policy_set_for_the_same_session_is_refused() -> None:
    world, scenario = _chain()
    assert world.issue(scenario.key).outcome is _O.ISSUED
    audit = world.generate(policy_set_version="set/2")  # same instant: not yet visible
    assert audit.status is ei4c._S.PROPOSED
    world.record(audit)
    result = world.issue(audit.generation_key)  # type: ignore[arg-type]
    assert result.outcome is _O.SESSION_STRATEGY_ALREADY_ISSUED  # IS-03 end to end


def test_e2e_hw10_an_issuance_closed_by_its_lifecycle_still_makes_later_proposals_unresolved() -> (
    None
):
    world, scenario = _chain()
    issued = world.issue(scenario.key).issuance
    # The issued recommendation's lifecycle closes at its stop (in memory only).
    active = lifecycle._active(scenario)
    scenario.put("09:50", ("98.5", "100"))
    closed = lifecycle._then(scenario, active, scenario.snap("09:56"))
    assert closed.state is S.CLOSED_STOP
    # A later modeled issuance for the same candidate, under another policy set.
    world.ports.references.hold("timing", ei4c._timing_policy(_LATE, latency=timedelta(minutes=30)))
    later_b = ei5d.PointInTime("2026-08-12T04:35:59.999999Z")  # 10:06 IST - 1 us
    assert world.declare(later_b).outcome is _D.DECLARED
    audit = world.generate(timing_policy_version=ei4c._v(_LATE))
    # EI4-D8, unchanged: the authority's history lists the earlier issuance, which
    # stays open whatever its lifecycle did, so exclusivity is unresolved.
    assert audit.status is ei4c._S.EVALUATION_UNRESOLVED
    assert audit.reasons == (ei4c._R.EXCLUSIVITY_POLICY_UNRESOLVED,)
    assert audit.inputs is not None
    listed = audit.inputs.issued_history.recommendations
    assert [r.recommendation_key for r in listed] == [issued.issuance_id]
    world.record(audit)
    try:
        world.issue(audit.generation_key)  # type: ignore[arg-type]
    except EquityRecommendationIssuanceRefusedError as refused:
        assert refused.reason is EquityRecommendationIssuanceRefusal.DISPOSITION_NOT_PROPOSED
    else:
        raise AssertionError("an UNRESOLVED disposition must never be issued")
    assert len(world.authority.issuances) == 1
