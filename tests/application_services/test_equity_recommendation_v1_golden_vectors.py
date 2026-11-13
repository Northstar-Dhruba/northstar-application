"""EI-4C compatibility slice 1: the version-1 golden-vector freeze (ADR-025, GV-01 to GV-06).

ADR-025 makes version-1 golden vectors a mandatory prerequisite of evaluator
version 2. Before this slice only ``eirs1``, ``eirk1`` and ``eipd1`` were
pinned. These tests pin ``eiri1``, ``eira1``, ``eirp1`` and ``eirat1`` as well,
and prove that pinned version-1 records still replay through levels A to C.

How the goldens were made
-------------------------
They were derived once, outside the test suite, from the unchanged production
code (Application ``75c6a35``, Core ``ccff26e``), using the existing synthetic
EI-4C scenarios of test_generate_equity_trade_recommendation and
test_record_equity_recommendation_generation. Four fresh processes with
different ``PYTHONHASHSEED`` values produced identical results. The documents
are stored under ``golden/equity_recommendation_v1`` as pretty-printed JSON
(two-space indent, sorted keys) so they can be reviewed line by line. The
fingerprints are literals below. Nothing in this module writes or regenerates
a golden: a vector that changes fails until a separately approved
compatibility review authorizes the change.

How a golden is compared
------------------------
For every golden, three things must hold independently:

1. ``_canonical`` -- a test-local restatement of the EI4-D9-C encoding (sorted
   keys, no whitespace, ASCII escapes, no NaN) -- turns the fixture into its
   canonical text, and ``hashlib`` over that text's UTF-8 bytes, under the
   literal prefix, equals the literal fingerprint. This pins the fixture to its
   fingerprint without any production helper.
2. The production path, re-run now, produces exactly that canonical text.
3. The production fingerprint equals the literal fingerprint.

A change to the hash preimage, the document shape, any field value, the
encoding or the prefix therefore fails at least one assertion.

Invariant map (slice 1 instructions, section 8)
-----------------------------------------------
==  ============================================  ==========================================
#   Invariant                                     Test
==  ============================================  ==========================================
1   Canonical format string                       test_neg01_*
2   Fingerprint domain namespace (prefix)         test_neg02_*, test_gv05_*
3   Canonical serialized content                  test_neg03_*, every test_gv0*
4   Field omission                                test_neg04_*
5   Ordering where it is semantic                 test_neg03_* (keys), test_neg05_* (lists)
6   Frozen-input history instant                  test_gv01_*_inclusively_*, test_neg06_*
7   Assessment reason or status                   test_neg07_*
8   Proposal timing                               test_neg08_*
9   Generation-attempt status                     test_neg09_*
10  Version-1 evaluator label                     test_gv05_*_labels_*
11  Embedded reference fingerprints               test_neg11_*
12  Stored provenance                             test_gv06_level_b_*, test_neg12_*
13  Stored version label                          test_neg13_*
14  Fingerprint tampering                         test_neg14_*
==  ============================================  ==========================================

Everything is synthetic and in memory: no clock, randomness, network, provider
or database is read. The version-1 history boundary is inclusive through the
modeled issuance ``t``; the version-2 strict-before boundary does not exist
here and is not introduced.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
import test_equity_synthetic_policy_catalog as catalog_tests
import test_generate_equity_trade_recommendation as ei4c
import test_record_equity_recommendation_generation as ei4d

from northstar_application.application_services import (
    EQUITY_RECOMMENDATION_ASSESSMENT_FORMAT,
    EQUITY_RECOMMENDATION_ATTEMPT_FORMAT,
    EQUITY_RECOMMENDATION_CORE_OUTCOME_FORMAT,
    EQUITY_RECOMMENDATION_DOCUMENT_PREFIXES,
    EQUITY_RECOMMENDATION_EVALUATOR_VERSION,
    EQUITY_RECOMMENDATION_GENERATION_KEY_NAMESPACE,
    EQUITY_RECOMMENDATION_GENERATION_KEY_PREFIX,
    EQUITY_RECOMMENDATION_INPUTS_FORMAT,
    EQUITY_RECOMMENDATION_POLICY_SET_FORMAT,
    EQUITY_RECOMMENDATION_PROPOSAL_FORMAT,
    SUPPORTED_EQUITY_RECOMMENDATION_EVALUATOR_VERSIONS,
    EquityAuthoritativeCandidate,
    EquityRecommendationDispositionKind,
    EquityRecommendationEligibilityAudit,
    EquityRecommendationGenerationAttempt,
    EquityRecommendationGenerationDisposition,
    EquityRecommendationGenerationVerifier,
    EquityRecommendationReplayMismatchError,
    EquityRecommendationSourceUnavailableError,
    EquityRecommendationSourceVerificationError,
    IncompatibleEquityRecommendationRecordError,
    InvalidEquityRecommendationRecordError,
    VerifyStoredEquityRecommendationGenerationUseCase,
    assess_equity_recommendation_inputs,
    canonical_equity_recommendation_json,
    decode_equity_recommendation_assessment,
    decode_equity_recommendation_inputs,
    equity_recommendation_attempt,
    equity_recommendation_disposition,
    equity_recommendation_fingerprint,
    establish_equity_recommendation_candidate,
)
from northstar_application.application_services import (
    equity_recommendation_generation_decoding as decoding_module,
)

_GOLDEN_DIR = Path(__file__).parent / "golden" / "equity_recommendation_v1"
_K = EquityRecommendationDispositionKind
_S = ei4c._S
_R = ei4c._R
Disposition = EquityRecommendationGenerationDisposition
Attempt = EquityRecommendationGenerationAttempt

# ----------------------------------------------------------------------
# The frozen version-1 contract, as literals (GV-05)
# ----------------------------------------------------------------------

_EVALUATOR_V1 = "northstar.equity-recommendation-evaluator/1"
_POLICY_SET_V1 = "northstar.equity-recommendation-policy-set/1"
_INPUTS_V1 = "northstar.equity-recommendation-generation-inputs/1"
_ASSESSMENT_V1 = "northstar.equity-recommendation-assessment/1"
_PROPOSAL_V1 = "northstar.equity-recommendation-proposal/1"
_ATTEMPT_V1 = "northstar.equity-recommendation-generation-attempt/1"
_CORE_OUTCOME_V1 = "northstar.equity-recommendation-core-outcome/1"
_KEY_NAMESPACE_V1 = "northstar.equity-recommendation-generation-key/1"
_KEY_PREFIX_V1 = "eirk1-sha256:"

_PREFIX = {
    _POLICY_SET_V1: "eirs1-sha256:",
    _INPUTS_V1: "eiri1-sha256:",
    _ASSESSMENT_V1: "eira1-sha256:",
    _PROPOSAL_V1: "eirp1-sha256:",
    _ATTEMPT_V1: "eirat1-sha256:",
}

# The existing goldens, cross-referenced -- never duplicated or replaced.
_EIRS1 = catalog_tests._GOLDEN_EIRS1
_EIRK1 = catalog_tests._GOLDEN_EIRK1

# The new goldens. Each fixture file is ``<name>.json``.
_GOLDEN = {
    # GV-01: eiri1
    "proposed.generation-inputs": (
        _INPUTS_V1,
        "eiri1-sha256:5bc0ea495b86ca2c60d2870242709531bb35b0230e1f5fda6699c0693790f58d",
    ),
    "fractional-latency.generation-inputs": (
        _INPUTS_V1,
        "eiri1-sha256:50fdef63087eaefea22a7eaa55c5a014939861db11c3b96dad5c8f3bd250c48f",
    ),
    "issued-at-t.generation-inputs": (
        _INPUTS_V1,
        "eiri1-sha256:cad1c27141efecd13ecb974050e0fecbc942de23fa1e9ca78e28c2dfeb343618",
    ),
    "non-ascii-reference.generation-inputs": (
        _INPUTS_V1,
        "eiri1-sha256:edebe16f37626d3aa0da3bfc25d88a1fde5d969f41a59e7ec2194e0a2adb3c5d",
    ),
    # GV-02: eira1
    "proposed.assessment": (
        _ASSESSMENT_V1,
        "eira1-sha256:13634b3a4c8d7f2511ee8ed8837910e12c542abc3124e53cdc13f7750c63e1bd",
    ),
    "not-eligible.assessment": (
        _ASSESSMENT_V1,
        "eira1-sha256:b0e78fcac774b02b60bcc174a80615f10e80de16665dbcc43159e8e6fc62a618",
    ),
    "unresolved.assessment": (
        _ASSESSMENT_V1,
        "eira1-sha256:0292e2ffffd7f078fa0d78c3d0ce15d7773e9612c7235afa6b2367ef6254a937",
    ),
    # GV-03: eirp1
    "proposed.proposal": (
        _PROPOSAL_V1,
        "eirp1-sha256:678242e1c71544d99abba7882eb68c3645eda2e656cc06e08b5b6fe38eb2d5fd",
    ),
    "entry-at-t.proposal": (
        _PROPOSAL_V1,
        "eirp1-sha256:89a98c3753b43cf7ab60616ba28bccf42264c2dc39bab7eaa7c1019ed6536110",
    ),
    # GV-04: eirat1, keyed and keyless
    "proposed.attempt": (
        _ATTEMPT_V1,
        "eirat1-sha256:fa89c56132e21745c9022f88373afd99b4f9faef6dc925aa57f03adb60b48238",
    ),
    "history-incomplete.attempt": (
        _ATTEMPT_V1,
        "eirat1-sha256:7ac11181e0aa2e25dcac43ae90bb5be9da4ad8675dc5d6a4917200d51534a1ce",
    ),
}


def _fp(name: str) -> str:
    return _GOLDEN[name][1]


# Embedded identities the goldens state, as literals.
_NOT_ELIGIBLE_EIRK1 = (
    "eirk1-sha256:31682368a74c72e32f7c568d21a89c5694205228b60c98953a31ca4ac32bc8e3"
)
_NOT_ELIGIBLE_EIRI1 = (
    "eiri1-sha256:9e7c517e3298ca7b0d05daaf2d27176fc90940ce829c41fd5ec31e697a3ea87c"
)
_UNRESOLVED_EIRI1 = "eiri1-sha256:9310d462346dd06d2e8d857622392d101d6f98bb8e32beb4c1e6457eeacf14d6"
_LATENCY_EIRS1 = "eirs1-sha256:d7d0acfea01f08e879c1ab72e69a3ddca8030d2242f31690f25e0adc179d705d"
_LATENCY_EIRK1 = "eirk1-sha256:2340414b0b5aa96869a6a345d3534109f014d58787145da64770b19f0f16bf4e"
_CANDIDATE_ENTRY = "eirt1-sha256:cb9aed19dbf0a097845f3e0e01912b6b83b2c94b432847a4c82af0717295bf76"
_DECISION = "eir1-sha256:9a047a8c43af5cc0086d6a5c6af3cbafa483b7b6f8bada7f08214a9d15cc2d28"
_SNAPSHOT = "ecs1-sha256:a7695d1abc0ff506fbce773680f03141c2d334e2a9c2a62303e06b04eecb1a0a"
_MANIFEST = "ecm1-sha256:5e3ba7f461fbf5e27465402a00c9f35ad3cfbd91c6732363383ceba4e827eb57"

# The modeled issuance t of the existing scenario (09:36 IST, zero latency),
# and of the microsecond-precision scenario (latency 90.123450 s).
_T = "2026-08-12T04:06:00Z"
_T_FRACTIONAL = "2026-08-12T04:07:30.12345Z"


# ----------------------------------------------------------------------
# Test-local canonical encoding and fingerprinting (independent of production)
# ----------------------------------------------------------------------


def _canonical(document: dict[str, Any]) -> str:
    """The EI4-D9-C encoding, restated: sorted keys, no whitespace, ASCII, no NaN."""
    return json.dumps(
        document, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )


def _sha(document: dict[str, Any]) -> str:
    """The namespaced SHA-256 of a version-1 document, from the literal prefix table."""
    return _PREFIX[document["format"]] + hashlib.sha256(_canonical(document).encode()).hexdigest()


def _golden(name: str) -> dict[str, Any]:
    """Load a golden document. Line endings and indentation are not part of it."""
    with (_GOLDEN_DIR / f"{name}.json").open(encoding="utf-8") as handle:
        return json.load(handle)


def _golden_text(name: str) -> str:
    return _canonical(_golden(name))


def _assert_golden(name: str, document: dict[str, Any], fingerprint: str) -> None:
    """Compare a document and fingerprint produced now with the frozen golden."""
    expected_format, expected_fingerprint = _GOLDEN[name]
    golden = _golden(name)
    text = _canonical(golden)
    # 1. The fixture and the literal fingerprint agree, independently of production.
    assert golden["format"] == expected_format
    assert text.isascii()
    assert _PREFIX[expected_format] + hashlib.sha256(text.encode()).hexdigest() == (
        expected_fingerprint
    )
    # 2. Production re-creates exactly the canonical text.
    assert canonical_equity_recommendation_json(document) == text
    assert document == golden
    # 3. Production re-creates exactly the fingerprint.
    assert fingerprint == expected_fingerprint
    assert equity_recommendation_fingerprint(document) == expected_fingerprint


# ----------------------------------------------------------------------
# The synthetic scenarios, each built fresh through the real production path
# ----------------------------------------------------------------------


def _proposed() -> tuple[ei4c._Ports, EquityRecommendationEligibilityAudit]:
    return ei4c._proposed()


def _not_eligible() -> tuple[ei4c._Ports, EquityRecommendationEligibilityAudit]:
    ports_ = ei4c._ports()
    return ports_, ei4d._not_eligible(ports_)


def _unresolved() -> tuple[ei4c._Ports, EquityRecommendationEligibilityAudit]:
    ports_ = ei4c._ports()
    return ports_, ei4d._unresolved(ports_)


def _history_incomplete() -> tuple[ei4c._Ports, EquityRecommendationEligibilityAudit]:
    """The issued history is complete only through 09:35:59 IST, before t = 09:36 IST."""
    ports_ = ei4c._ports()
    ports_.state.answer = ei4c._history(through=ei4c._at(9, 35, 59))
    return ports_, ports_.run()


def _fractional_latency() -> tuple[ei4c._Ports, EquityRecommendationEligibilityAudit]:
    """The same candidate under a timing policy whose latency has microseconds."""
    ports_ = ei4c._ports()
    latency = timedelta(seconds=90, microseconds=123450)
    ports_.references.hold("timing", ei4c._timing_policy("timing/latency", latency=latency))
    return ports_, ports_.run(timing_policy_version=ei4c._v("timing/latency"))


def _issued_at_t() -> tuple[ei4c._Ports, EquityRecommendationEligibilityAudit]:
    """Two records issued exactly at t = 09:36 IST: version 1 includes both (inclusive)."""
    ports_ = ei4c._ports()
    ports_.state.answer = ei4c._history(
        ei4c._issued("a-key", ei4c._OTHER, ei4c._at(9, 36), sequence=2),
        ei4c._issued("b-key", ei4c._LISTING, ei4c._at(9, 36), sequence=1),
    )
    return ports_, ports_.run()


def _entry_at_t() -> tuple[ei4c._Ports, EquityRecommendationEligibilityAudit]:
    """A 4-minute latency puts t on a bar opening (09:40 IST): that bar is the first entry."""
    ports_ = ei4c._ports()
    latency = timedelta(minutes=4)
    ports_.references.hold("timing", ei4c._timing_policy("timing/bar-opening", latency=latency))
    return ports_, ports_.run(timing_policy_version=ei4c._v("timing/bar-opening"))


def _non_ascii_reference() -> tuple[ei4c._Ports, EquityRecommendationEligibilityAudit]:
    """A free-text reference holding a non-ASCII character, escaped canonically as \\u00fc."""
    ports_ = ei4c._ports()
    ports_.state.answer = ei4c._history(reference="test-history/über-1")
    return ports_, ports_.run()


_DISPOSITIONS = {
    # golden assessment: (scenario, kind, generation key, inputs fingerprint)
    "proposed.assessment": (_proposed, _K.PROPOSED, _EIRK1, _fp("proposed.generation-inputs")),
    "not-eligible.assessment": (
        _not_eligible,
        _K.NOT_ELIGIBLE,
        _NOT_ELIGIBLE_EIRK1,
        _NOT_ELIGIBLE_EIRI1,
    ),
    "unresolved.assessment": (_unresolved, _K.UNRESOLVED, _EIRK1, _UNRESOLVED_EIRI1),
}


def _stored(name: str, **changes: Any) -> Disposition:
    """A version-1 disposition built only from golden text and literal fingerprints."""
    _, kind, key, _ = _DISPOSITIONS[name]
    proposed = kind is _K.PROPOSED
    stored: dict[str, Any] = {
        "generation_key": key,
        "kind": kind,
        "evaluator_version": _EVALUATOR_V1,
        "assessment_fingerprint": _fp(name),
        "assessment_document": _golden_text(name),
        "proposal_fingerprint": _fp("proposed.proposal") if proposed else None,
        "proposal_document": _golden_text("proposed.proposal") if proposed else None,
    }
    stored.update(changes)
    return Disposition(**stored)


def _produced_fingerprints() -> dict[str, str]:
    """Every pinned identity, produced now by the production path (also run in a subprocess)."""
    _, proposed = _proposed()
    _, not_eligible = _not_eligible()
    _, unresolved = _unresolved()
    _, incomplete = _history_incomplete()
    _, latency = _fractional_latency()
    _, at_t = _issued_at_t()
    _, entry = _entry_at_t()
    _, non_ascii = _non_ascii_reference()
    assert proposed.inputs is not None and proposed.proposal is not None
    assert latency.inputs is not None and at_t.inputs is not None
    assert entry.proposal is not None and non_ascii.inputs is not None
    return {
        "eirs1": proposed.inputs.policy_set.fingerprint,
        "eirk1": proposed.generation_key,  # type: ignore[dict-item]
        "proposed.generation-inputs": proposed.inputs.fingerprint,
        "fractional-latency.generation-inputs": latency.inputs.fingerprint,
        "issued-at-t.generation-inputs": at_t.inputs.fingerprint,
        "non-ascii-reference.generation-inputs": non_ascii.inputs.fingerprint,
        "entry-at-t.proposal": entry.proposal.fingerprint,
        "proposed.assessment": proposed.assessment_fingerprint,
        "not-eligible.assessment": not_eligible.assessment_fingerprint,
        "unresolved.assessment": unresolved.assessment_fingerprint,
        "proposed.proposal": proposed.proposal.fingerprint,
        "proposed.attempt": equity_recommendation_attempt(proposed).attempt_identity,
        "history-incomplete.attempt": equity_recommendation_attempt(incomplete).attempt_identity,
    }


def _expected_fingerprints() -> dict[str, str]:
    return {"eirs1": _EIRS1, "eirk1": _EIRK1, **{name: fp for name, (_, fp) in _GOLDEN.items()}}


def _verifier(ports_: ei4c._Ports) -> EquityRecommendationGenerationVerifier:
    return ei4d._verifier(ports_)


def _resealed(document: dict[str, Any]) -> tuple[str, str]:
    """Test-only tampering: the canonical text of ``document`` and a fingerprint that matches."""
    return _canonical(document), _sha(document)


def _reseal_assessment(document: dict[str, Any]) -> None:
    """Test-only tampering: re-derive the embedded inputs fingerprint after an edit."""
    if document["inputs"] is not None:
        document["inputs_fingerprint"] = _sha(document["inputs"])


# ======================================================================
# Golden fixtures
# ======================================================================


def test_the_golden_fixture_set_is_closed_and_canonically_sorted() -> None:
    assert sorted(path.name for path in _GOLDEN_DIR.iterdir()) == sorted(
        f"{name}.json" for name in _GOLDEN
    )
    for name in _GOLDEN:
        raw = (_GOLDEN_DIR / f"{name}.json").read_text(encoding="utf-8")
        document = json.loads(raw)
        # The reviewable form is exactly the sorted, two-space rendering of the document.
        expected = json.dumps(document, sort_keys=True, indent=2, ensure_ascii=True) + "\n"
        assert raw.replace("\r\n", "\n") == expected, name


# ======================================================================
# GV-05 -- the version-1 labels, formats, prefixes and existing goldens
# ======================================================================


def test_gv05_version1_labels_formats_prefixes_and_key_namespace_are_pinned() -> None:
    assert EQUITY_RECOMMENDATION_EVALUATOR_VERSION == _EVALUATOR_V1
    assert SUPPORTED_EQUITY_RECOMMENDATION_EVALUATOR_VERSIONS == frozenset({_EVALUATOR_V1})
    assert (
        EQUITY_RECOMMENDATION_POLICY_SET_FORMAT,
        EQUITY_RECOMMENDATION_INPUTS_FORMAT,
        EQUITY_RECOMMENDATION_ASSESSMENT_FORMAT,
        EQUITY_RECOMMENDATION_PROPOSAL_FORMAT,
        EQUITY_RECOMMENDATION_ATTEMPT_FORMAT,
        EQUITY_RECOMMENDATION_CORE_OUTCOME_FORMAT,
    ) == (
        _POLICY_SET_V1,
        _INPUTS_V1,
        _ASSESSMENT_V1,
        _PROPOSAL_V1,
        _ATTEMPT_V1,
        _CORE_OUTCOME_V1,
    )
    assert EQUITY_RECOMMENDATION_GENERATION_KEY_NAMESPACE == _KEY_NAMESPACE_V1
    assert EQUITY_RECOMMENDATION_GENERATION_KEY_PREFIX == _KEY_PREFIX_V1
    prefixes = dict(EQUITY_RECOMMENDATION_DOCUMENT_PREFIXES)
    assert prefixes.pop("northstar.equity-synthetic-policy-definition/1") == "eipd1-sha256:"
    assert prefixes == _PREFIX


def test_gv05_existing_eirs1_and_eirk1_goldens_are_unchanged() -> None:
    assert _EIRS1 == (
        "eirs1-sha256:7602c7204df699f0c18eed863b11e835d0eb1ecb720ab6ee979b9b950e959986"
    )
    assert _EIRK1 == (
        "eirk1-sha256:4161d4d3fa27fcf1ceffb413a2428e04cfa9bb8154aad02935778cc217a1ece6"
    )
    _, audit = _proposed()
    assert audit.inputs is not None
    assert (audit.inputs.policy_set.fingerprint, audit.generation_key) == (_EIRS1, _EIRK1)
    # The frozen inputs golden embeds the policy set whose eirs1 is the existing golden.
    inputs = _golden("proposed.generation-inputs")
    assert inputs["policy_set_fingerprint"] == _EIRS1 == _sha(inputs["policy_set"])
    assert inputs["policy_set"]["format"] == _POLICY_SET_V1


def _independent_generation_key(inputs: dict[str, Any]) -> str:
    """Re-derive eirk1 from a frozen inputs document: length-prefixed parts, SHA-256."""
    source = inputs["candidate"]["source"]
    parts = (
        _KEY_NAMESPACE_V1,
        source["strategy"],
        source["research_configuration_version"],
        source["listing"]["symbol"],
        source["listing"]["exchange_code"],
        source["trading_date"],
        source["candidate_entry_fingerprint"],
        inputs["policy_set"]["reference"]["version"],
        inputs["policy_set_fingerprint"],
    )
    encoded = b"".join(str(len(p.encode())).encode() + b":" + p.encode() for p in parts)
    return _KEY_PREFIX_V1 + hashlib.sha256(encoded).hexdigest()


def test_gv05_generation_keys_re_derive_from_the_frozen_inputs_goldens() -> None:
    assert _independent_generation_key(_golden("proposed.generation-inputs")) == _EIRK1
    latency = _golden("fractional-latency.generation-inputs")
    assert latency["policy_set_fingerprint"] == _LATENCY_EIRS1 == _sha(latency["policy_set"])
    assert _independent_generation_key(latency) == _LATENCY_EIRK1
    not_eligible = _golden("not-eligible.assessment")
    assert _independent_generation_key(not_eligible["inputs"]) == _NOT_ELIGIBLE_EIRK1
    assert not_eligible["generation_key"] == _NOT_ELIGIBLE_EIRK1


# ======================================================================
# GV-01 -- eiri1: frozen generation inputs
# ======================================================================


@pytest.mark.parametrize(
    ("name", "scenario"),
    [
        ("proposed.generation-inputs", _proposed),
        ("fractional-latency.generation-inputs", _fractional_latency),
    ],
)
def test_gv01_eiri1_frozen_inputs_match_the_golden_vector(name, scenario) -> None:  # type: ignore[no-untyped-def]
    _, audit = scenario()
    assert audit.status is _S.PROPOSED and audit.inputs is not None
    _assert_golden(name, audit.inputs.document(), audit.inputs.fingerprint)
    # Decoding the golden and re-encoding it gives the same text and identity.
    decoded = decode_equity_recommendation_inputs(_golden(name))
    assert decoded == audit.inputs
    assert canonical_equity_recommendation_json(decoded.document()) == _golden_text(name)
    assert decoded.fingerprint == _fp(name)


def test_gv01_version1_history_is_observed_inclusively_at_the_modeled_issuance() -> None:
    # The version-1 contract: issued_history.as_of == modeled_issued_at == t (never t - 1 us).
    for name, scenario, t in (
        ("proposed.generation-inputs", _proposed, _T),
        ("fractional-latency.generation-inputs", _fractional_latency, _T_FRACTIONAL),
    ):
        golden = _golden(name)
        assert golden["modeled_issued_at"] == golden["issued_history"]["as_of"] == t
        assert "history_boundary" not in golden["issued_history"]
        ports_, audit = scenario()
        assert audit.inputs is not None
        assert audit.inputs.modeled_issued_at.value == audit.inputs.issued_history.as_of.value == t
        assert [instant.value for instant in ports_.state.asked] == [t]  # asked at t itself
    # Microseconds are kept exactly, with the trailing zero of .123450 dropped canonically.
    assert _T_FRACTIONAL.endswith(".12345Z")


def test_gv01_records_issued_exactly_at_t_are_frozen_into_version1_history() -> None:
    ports_, audit = _issued_at_t()
    assert audit.status is _S.EVALUATION_UNRESOLVED and audit.inputs is not None
    assert audit.reasons == (_R.EXCLUSIVITY_POLICY_UNRESOLVED, _R.EXPOSURE_POLICY_UNRESOLVED)
    _assert_golden(
        "issued-at-t.generation-inputs", audit.inputs.document(), audit.inputs.fingerprint
    )
    history = _golden("issued-at-t.generation-inputs")["issued_history"]
    assert history["as_of"] == _T and history["recording_watermark"] == 2
    # Inclusive through t, in (issued_at, key) order: a strict-before view would hold neither.
    assert [(r["recommendation_key"], r["issued_at"]) for r in history["recommendations"]] == [
        ("a-key", _T),
        ("b-key", _T),
    ]
    assert [instant.value for instant in ports_.state.asked] == [_T]


def test_gv01_non_ascii_text_is_escaped_in_the_canonical_encoding() -> None:
    _, audit = _non_ascii_reference()
    assert audit.status is _S.PROPOSED and audit.inputs is not None
    _assert_golden(
        "non-ascii-reference.generation-inputs", audit.inputs.document(), audit.inputs.fingerprint
    )
    text = _golden_text("non-ascii-reference.generation-inputs")
    assert '"reference":"test-history/\\u00fcber-1"' in text  # escaped, never raw UTF-8
    assert text.isascii()


def test_gv01_frozen_inputs_name_their_provenance_policy_set_and_history_evidence() -> None:
    inputs = _golden("proposed.generation-inputs")
    assert sorted(inputs) == [
        "candidate",
        "format",
        "issued_history",
        "listing_eligibility",
        "modeled_issued_at",
        "policy_set",
        "policy_set_fingerprint",
        "venue_day",
    ]
    candidate = inputs["candidate"]
    assert candidate["source"] == {
        "candidate_entry_fingerprint": _CANDIDATE_ENTRY,
        "decision_as_of": _T,
        "evidence_snapshot_id": _SNAPSHOT,
        "listing": {"exchange_code": "NSE", "symbol": "M&M"},
        "research_configuration_version": "synthetic-ei4c-research/1",
        "research_decision_fingerprint": _DECISION,
        "strategy": "ei4c-orb-vwap-volume",
        "trading_date": "2026-08-12",
    }
    assert (candidate["manifest_fingerprint"], candidate["candidate_sequence"]) == (_MANIFEST, 2)
    assert (candidate["signal_close"], candidate["opening_range_low"]) == ("102.5", "99")
    assert candidate["venue_reference"] == inputs["venue_day"]["reference"] == "test-venue/1"
    assert inputs["policy_set"]["reference"] == {
        "cost_schedule_version": "synthetic-ei4c-costs/1",
        "generation_policy_version": "synthetic-ei4c-generation/1",
        "price_increment_version": "synthetic-ei4c-increment/1",
        "risk_policy_version": "synthetic-ei4c-risk/1",
        "timing_policy_version": "synthetic-ei4c-timing/1",
        "version": "synthetic-ei4c-set/1",
    }
    assert inputs["issued_history"] == {
        "as_of": _T,
        "complete_through": "2026-08-12T10:00:00Z",  # the completeness boundary
        "exchange_code": "NSE",
        "recommendations": [],
        "recording_watermark": 0,
        "reference": "test-history/1",
        "trading_date": "2026-08-12",
    }


# ======================================================================
# GV-02 -- eira1: assessments for PROPOSED, NOT_ELIGIBLE and UNRESOLVED
# ======================================================================

_ASSESSMENT_OUTCOMES = {
    "proposed.assessment": ("PROPOSED", []),
    "not-eligible.assessment": ("NOT_ELIGIBLE", ["CORE_FORMATION_REJECTED"]),
    "unresolved.assessment": ("EVALUATION_UNRESOLVED", ["EXCLUSIVITY_POLICY_UNRESOLVED"]),
}


@pytest.mark.parametrize("name", list(_DISPOSITIONS))
def test_gv02_eira1_assessments_match_the_golden_vectors(name: str) -> None:
    scenario, kind, key, inputs_fingerprint = _DISPOSITIONS[name]
    _, audit = scenario()
    _assert_golden(name, audit.assessment_document(), audit.assessment_fingerprint)
    golden = _golden(name)
    status, reasons = _ASSESSMENT_OUTCOMES[name]
    assert (golden["status"], golden["reasons"]) == (status, reasons)
    assert (golden["generation_key"], golden["inputs_fingerprint"]) == (key, inputs_fingerprint)
    assert _sha(golden["inputs"]) == inputs_fingerprint  # the embedded eiri1 re-derives
    assert (golden["synthetic"], golden["authoritative"]) == (True, False)
    # The evaluator label is not part of the assessment's hash preimage.
    assert "evaluator" not in _golden_text(name)
    # The checks: every control once, in order, with exactly the failed ones carrying reasons.
    failed = {c["check"]: c["reasons"] for c in golden["checks"] if c["status"] == "FAILED"}
    assert (
        failed
        == {
            "proposed.assessment": {},
            "not-eligible.assessment": {"CORE_FORMATION": ["CORE_FORMATION_REJECTED"]},
            "unresolved.assessment": {"OPEN_RECOMMENDATIONS": ["EXCLUSIVITY_POLICY_UNRESOLVED"]},
        }[name]
    )
    assert [c["check"] for c in golden["checks"]] == [c.value for c in ei4c._C]
    assert kind is _K.PROPOSED or golden["proposal"] is None


def test_gv02_assessments_embed_the_frozen_core_outcome_and_proposal_references() -> None:
    proposed = _golden("proposed.assessment")
    assert proposed["inputs"] == _golden("proposed.generation-inputs")
    assert proposed["proposal"] == _golden("proposed.proposal")
    assert proposed["proposal_fingerprint"] == _fp("proposed.proposal")
    assert proposed["core_outcome"] == proposed["proposal"]["definition"]
    assert proposed["entry_window"] == proposed["proposal"]["entry_window"]
    not_eligible = _golden("not-eligible.assessment")
    assert not_eligible["core_outcome"] == {
        "format": _CORE_OUTCOME_V1,
        "kind": "REJECTION",
        "outcome": {
            "generation_policy_version": "synthetic-ei4c-generation/strict",
            "reasons": ["NET_REWARD_TO_RISK_BELOW_MINIMUM"],
            "source": proposed["inputs"]["candidate"]["source"],
        },
    }
    unresolved = _golden("unresolved.assessment")
    assert unresolved["core_outcome"] is None  # Core never ran
    assert [
        r["recommendation_key"] for r in unresolved["inputs"]["issued_history"]["recommendations"]
    ] == ["prior-1"]
    # All three record the same authoritative candidate.
    assert proposed["candidate"] == not_eligible["candidate"] == unresolved["candidate"]


# ======================================================================
# GV-03 -- eirp1: the proposal
# ======================================================================


def test_gv03_eirp1_proposal_matches_the_golden_vector() -> None:
    _, audit = _proposed()
    assert audit.proposal is not None
    _assert_golden("proposed.proposal", audit.proposal.document(), audit.proposal.fingerprint)
    proposal = _golden("proposed.proposal")
    assert sorted(proposal) == [
        "actionable",
        "authoritative",
        "definition",
        "entry_window",
        "format",
        "generation_key",
        "issued",
        "listing_eligibility_version",
        "not_assessed",
        "synthetic",
        "universe_version",
    ]
    assert proposal["generation_key"] == _EIRK1
    flags = [proposal[k] for k in ("synthetic", "authoritative", "issued", "actionable")]
    assert flags == [True, False, False, False]
    assert proposal["entry_window"] == {
        "latest_entry_bar_opens_at": "2026-08-12T05:15:00Z",
        "modeled_exit_at": "2026-08-12T08:30:00Z",
        "timing": {
            "decision_as_of": _T,
            "eligible_from": "2026-08-12T04:10:00Z",
            "issuance_kind": "MODELED",
            "issued_at": _T,
        },
        "timing_policy_version": "synthetic-ei4c-timing/1",
    }
    definition = proposal["definition"]
    assert (definition["format"], definition["kind"]) == (_CORE_OUTCOME_V1, "DEFINITION")
    assert definition["outcome"]["direction"] == "LONG"
    assert definition["outcome"]["levels"] == {
        "currency": "INR",
        "entry_zone": {"high": "102.5", "low": "101.5"},
        "stop": "99",
        "target": "109.5",
    }
    assert definition["outcome"]["sizing"]["quantity"] == 195
    assert proposal["not_assessed"] == [
        "SAME_LISTING_EXCLUSIVITY",
        "PORTFOLIO_EXPOSURE",
        "DAILY_LOSS_LIMIT",
        "PRE_ENTRY_INVALIDATION",
        "CORPORATE_ACTIONS",
        "EXCHANGE_SURVEILLANCE",
        "CANDLE_REVISION_MATERIALITY",
    ]


def test_gv03_a_bar_opening_exactly_at_t_is_the_first_entry_bar() -> None:
    # Version 1: the earliest entry bar is the first that opens at or after t (inclusive).
    _, audit = _entry_at_t()
    assert audit.status is _S.PROPOSED and audit.proposal is not None
    _assert_golden("entry-at-t.proposal", audit.proposal.document(), audit.proposal.fingerprint)
    timing = _golden("entry-at-t.proposal")["entry_window"]["timing"]
    assert timing == {
        "decision_as_of": _T,
        "eligible_from": "2026-08-12T04:10:00Z",
        "issuance_kind": "MODELED",
        "issued_at": "2026-08-12T04:10:00Z",
    }


# ======================================================================
# GV-04 -- eirat1: keyed and keyless attempts
# ======================================================================


def test_gv04_keyed_attempt_matches_the_golden_vector() -> None:
    _, audit = _proposed()
    attempt = equity_recommendation_attempt(audit)
    _assert_golden(
        "proposed.attempt", json.loads(attempt.attempt_document), attempt.attempt_identity
    )
    assert attempt.attempt_document == _golden_text("proposed.attempt")
    golden = _golden("proposed.attempt")
    assert (
        golden["status"],
        golden["reasons"],
        golden["generation_key"],
        golden["disposition_kind"],
        golden["assessment_fingerprint"],
    ) == ("PROPOSED", [], _EIRK1, "PROPOSED", _fp("proposed.assessment"))
    rebuilt = Attempt(_fp("proposed.attempt"), _golden_text("proposed.attempt"))
    assert rebuilt == attempt


def test_gv04_keyless_history_incomplete_attempt_matches_the_golden_vector() -> None:
    _, audit = _history_incomplete()
    assert audit.status is _S.DEPENDENCY_UNAVAILABLE
    assert audit.reasons == (_R.RECOMMENDATION_HISTORY_INCOMPLETE,)
    assert audit.generation_key is None and audit.inputs is None
    assert equity_recommendation_disposition(audit) is None  # attempt-only
    attempt = equity_recommendation_attempt(audit)
    _assert_golden(
        "history-incomplete.attempt",
        json.loads(attempt.attempt_document),
        attempt.attempt_identity,
    )
    golden = _golden("history-incomplete.attempt")
    assert (
        golden["status"],
        golden["reasons"],
        golden["generation_key"],
        golden["disposition_kind"],
        golden["assessment_fingerprint"],
    ) == ("DEPENDENCY_UNAVAILABLE", ["RECOMMENDATION_HISTORY_INCOMPLETE"], None, None, None)
    failed = [(c["check"], c["reasons"]) for c in golden["checks"] if c["status"] == "FAILED"]
    assert failed == [("SYNTHETIC_REFERENCES", ["RECOMMENDATION_HISTORY_INCOMPLETE"])]
    rebuilt = Attempt(_fp("history-incomplete.attempt"), _golden_text("history-incomplete.attempt"))
    assert rebuilt == attempt
    assert (rebuilt.generation_key, rebuilt.disposition_kind, rebuilt.assessment_fingerprint) == (
        None,
        None,
        None,
    )


def test_gv04_v1_history_complete_exactly_at_t_is_proposed_not_incomplete() -> None:
    # The version-1 boundary is inclusive: complete_through == t is enough.
    ports_ = ei4c._ports()
    ports_.state.answer = ei4c._history(through=ei4c._at(9, 36))
    audit = ports_.run()
    assert audit.status is _S.PROPOSED and audit.inputs is not None
    assert audit.inputs.issued_history.complete_through.value == _T


# ======================================================================
# GV-06 -- the pinned version-1 records replay through levels A to C
# ======================================================================


@pytest.mark.parametrize("name", list(_DISPOSITIONS))
def test_gv06_level_a_golden_records_decode_re_encode_and_re_derive(name: str) -> None:
    scenario, kind, key, inputs_fingerprint = _DISPOSITIONS[name]
    stored = _stored(name)  # construction decodes and verifies every document
    _, audit = scenario()
    assert stored == equity_recommendation_disposition(audit)
    assert (stored.kind, stored.generation_key, stored.evaluator_version) == (
        kind,
        key,
        _EVALUATOR_V1,
    )
    assert stored.assessment_fingerprint == _fp(name)
    assert stored.inputs_fingerprint == inputs_fingerprint
    assert stored.policy_set_fingerprint == _golden(name)["inputs"]["policy_set_fingerprint"]
    # Decoding then re-encoding is the identity on the canonical text.
    assert canonical_equity_recommendation_json(stored.assessment.document) == _golden_text(name)
    assert stored.inputs_document == _canonical(_golden(name)["inputs"])
    decoded = decode_equity_recommendation_assessment(_golden_text(name), _fp(name))
    assert decoded.fingerprint == _fp(name) and decoded.inputs == stored.inputs
    if kind is _K.PROPOSED:
        assert stored.proposal_document == _golden_text("proposed.proposal")
        assert stored.proposal is not None
        assert stored.proposal.fingerprint == _fp("proposed.proposal")


@pytest.mark.parametrize("name", list(_DISPOSITIONS))
def test_gv06_levels_b_and_c_golden_records_verify_against_a_rebuilt_world(name: str) -> None:
    stored = _stored(name)
    world = ei4c._ports()  # an independently rebuilt synthetic research world
    verified = _verifier(world).verify(stored)
    assert (
        verified.document_integrity_verified,
        verified.source_integrity_verified,
        verified.evaluation_reproduced,
        verified.issuance_established,
    ) == (True, True, True, False)
    replayed = verified.replayed
    assert replayed.assessment_fingerprint == _fp(name)
    assert canonical_equity_recommendation_json(replayed.assessment_document()) == (
        _golden_text(name)
    )
    expected_proposal = _fp("proposed.proposal") if stored.kind is _K.PROPOSED else None
    assert (None if replayed.proposal is None else replayed.proposal.fingerprint) == (
        expected_proposal
    )
    # And through the stored-record use case, as a reload would.
    store = ei4d._Store()
    store.dispositions[stored.generation_key] = stored
    reloaded = VerifyStoredEquityRecommendationGenerationUseCase(store, _verifier(world)).execute(
        stored.generation_key
    )
    assert reloaded is not None and reloaded.disposition == stored
    assert reloaded.replayed.assessment_fingerprint == _fp(name)


def test_gv06_level_b_source_provenance_is_stable() -> None:
    world = ei4c._ports()
    stored = _stored("proposed.assessment")
    found = establish_equity_recommendation_candidate(
        stored.request, world.world.timelines, world.world.snapshots
    )
    assert isinstance(found, EquityAuthoritativeCandidate)
    assert found.provenance == stored.candidate
    assert found.provenance.document() == _golden("proposed.generation-inputs")["candidate"]
    source = found.provenance.source
    assert (
        source.candidate_entry_fingerprint,
        source.research_decision_fingerprint,
        source.evidence_snapshot_id,
        found.provenance.manifest_fingerprint,
    ) == (_CANDIDATE_ENTRY, _DECISION, _SNAPSHOT, _MANIFEST)
    assert _SNAPSHOT in world.world.snapshots.documents


def test_gv06_level_b_unavailable_or_mismatched_source_fails_closed() -> None:
    stored = _stored("proposed.assessment")
    world = ei4c._ports()
    world.world.timelines.records.clear()  # the authoritative timeline is gone
    with pytest.raises(EquityRecommendationSourceVerificationError) as missing:
        _verifier(world).verify(stored)
    assert missing.value.reason is _R.NO_AUTHORITATIVE_CANDIDATE
    world.world.timelines.unavailable = True
    with pytest.raises(EquityRecommendationSourceUnavailableError):
        _verifier(world).verify(stored)

    world = ei4c._ports()
    saved = world.world.snapshots.documents.pop(_SNAPSHOT)  # the source snapshot is gone
    with pytest.raises(EquityRecommendationSourceVerificationError) as gone:
        _verifier(world).verify(stored)
    assert gone.value.reason is _R.RESEARCH_TIMELINE_INVALID
    world.world.snapshots.documents[_SNAPSHOT] = saved
    world.world.snapshots.unavailable = True
    with pytest.raises(EquityRecommendationSourceUnavailableError):
        _verifier(world).verify(stored)

    # Another authoritative history: its CANDIDATE was decided at 09:35:30 IST.
    other = ei4c._ports(ei4c._ist(9, 31), ei4c._ist(9, 35, 30))
    with pytest.raises(EquityRecommendationSourceVerificationError, match="source"):
        _verifier(other).verify(stored)


@pytest.mark.parametrize("name", list(_DISPOSITIONS))
def test_gv06_level_c_replay_is_pure_and_deterministic(name: str) -> None:
    stored = _stored(name)
    # The pure assessment, on the decoded golden inputs alone: no port, no world, no clock.
    first = assess_equity_recommendation_inputs(stored.request, stored.inputs)
    again = assess_equity_recommendation_inputs(stored.request, stored.inputs)
    assert first.assessment_document() == again.assessment_document()
    assert first.assessment_fingerprint == again.assessment_fingerprint == _fp(name)
    assert _canonical(first.assessment_document()) == _golden_text(name)
    world = ei4c._ports()
    one, two = _verifier(world).verify(stored), _verifier(world).verify(stored)
    assert one == two


def test_gv06_repeated_production_runs_are_identical() -> None:
    assert _produced_fingerprints() == _produced_fingerprints() == _expected_fingerprints()


def test_gv06_a_fresh_process_reproduces_every_golden_identity() -> None:
    """A separate interpreter, with another hash seed, rebuilds every pinned identity."""
    here = Path(__file__).parent
    code = (
        "import json, test_equity_recommendation_v1_golden_vectors as gv; "
        "print(json.dumps(gv._produced_fingerprints(), sort_keys=True))"
    )
    env = dict(os.environ, PYTHONHASHSEED="4242", PYTHONPATH=str(here))
    completed = subprocess.run(  # the test's own interpreter, running fixed code only
        [sys.executable, "-c", code],
        cwd=here,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout.strip().splitlines()[-1]) == _expected_fingerprints()


# ======================================================================
# Negative compatibility: the frozen references detect drift
# ======================================================================


def test_neg01_a_changed_format_string_is_refused_and_moves_the_identity() -> None:
    # An unapproved version within a known family is incompatible; never migrated.
    for name, family in (
        ("proposed.assessment", "northstar.equity-recommendation-assessment/9"),
        ("proposed.attempt", "northstar.equity-recommendation-generation-attempt/9"),
    ):
        document = dict(_golden(name), format=family)
        assert _canonical(document) != _golden_text(name)
        with pytest.raises(IncompatibleEquityRecommendationRecordError, match="migrated"):
            if name.endswith("attempt"):
                Attempt(_fp(name), _canonical(document))
            else:
                decode_equity_recommendation_assessment(_canonical(document), _fp(name))
    nested = _golden("proposed.generation-inputs")
    nested["format"] = "northstar.equity-recommendation-generation-inputs/9"
    with pytest.raises(IncompatibleEquityRecommendationRecordError):
        decode_equity_recommendation_inputs(nested)
    proposal = dict(
        _golden("proposed.proposal"), format="northstar.equity-recommendation-proposal/9"
    )
    with pytest.raises(IncompatibleEquityRecommendationRecordError):
        _stored("proposed.assessment", proposal_document=_canonical(proposal))
    # Another family entirely is corrupt, not incompatible.
    stranger = dict(_golden("proposed.assessment"), format="northstar.something-else/1")
    with pytest.raises(InvalidEquityRecommendationRecordError) as raised:
        decode_equity_recommendation_assessment(_canonical(stranger), _fp("proposed.assessment"))
    assert not isinstance(raised.value, IncompatibleEquityRecommendationRecordError)


def test_neg02_another_fingerprint_namespace_is_refused() -> None:
    digest = _fp("proposed.assessment").removeprefix("eira1-sha256:")
    for prefix in ("eirp1-sha256:", "eiri1-sha256:", "eira2-sha256:", "sha256:", ""):
        with pytest.raises(InvalidEquityRecommendationRecordError, match="fingerprint"):
            _stored("proposed.assessment", assessment_fingerprint=prefix + digest)
    attempt = _fp("proposed.attempt").removeprefix("eirat1-sha256:")
    with pytest.raises(InvalidEquityRecommendationRecordError, match="fingerprint"):
        Attempt("eira1-sha256:" + attempt, _golden_text("proposed.attempt"))
    # The generation key namespace is part of the eirk1 preimage.
    inputs = _golden("proposed.generation-inputs")
    source = inputs["candidate"]["source"]
    parts = (
        "northstar.equity-recommendation-generation-key/2",
        source["strategy"],
        source["research_configuration_version"],
        source["listing"]["symbol"],
        source["listing"]["exchange_code"],
        source["trading_date"],
        source["candidate_entry_fingerprint"],
        inputs["policy_set"]["reference"]["version"],
        inputs["policy_set_fingerprint"],
    )
    encoded = b"".join(str(len(p.encode())).encode() + b":" + p.encode() for p in parts)
    assert _KEY_PREFIX_V1 + hashlib.sha256(encoded).hexdigest() != _EIRK1


def test_neg03_non_canonical_encodings_of_the_same_document_are_refused() -> None:
    golden = _golden("proposed.assessment")
    text = _golden_text("proposed.assessment")
    variants = {
        "indented": json.dumps(golden, sort_keys=True, indent=2),
        "spaced": json.dumps(golden, sort_keys=True),
        "reordered-keys": json.dumps(dict(reversed(list(golden.items()))), separators=(",", ":")),
        "trailing-newline": text + "\n",
    }
    for label, variant in variants.items():
        assert json.loads(variant) == golden, label
        assert variant != text, label
        with pytest.raises(InvalidEquityRecommendationRecordError, match="canonical"):
            _stored("proposed.assessment", assessment_document=variant)
    attempt = _golden("proposed.attempt")
    with pytest.raises(InvalidEquityRecommendationRecordError, match="canonical"):
        Attempt(_fp("proposed.attempt"), json.dumps(attempt, indent=1, sort_keys=True))


def test_neg04_an_omitted_field_is_refused_even_when_resealed() -> None:
    for path in (("inputs", "modeled_issued_at"), ("inputs", "issued_history", "as_of")):
        document = _golden("proposed.assessment")
        container = document
        for part in path[:-1]:
            container = container[part]
        del container[path[-1]]
        _reseal_assessment(document)
        text, fingerprint = _resealed(document)
        assert fingerprint != _fp("proposed.assessment")
        with pytest.raises(InvalidEquityRecommendationRecordError):
            decode_equity_recommendation_assessment(text, fingerprint)
    for field in ("checks", "assessment_fingerprint", "request"):
        attempt = _golden("proposed.attempt")
        del attempt[field]
        text, fingerprint = _resealed(attempt)
        with pytest.raises(InvalidEquityRecommendationRecordError):
            Attempt(fingerprint, text)


def test_neg05_reordered_semantic_lists_are_refused_even_when_resealed() -> None:
    assessment = _golden("proposed.assessment")
    assessment["checks"] = list(reversed(assessment["checks"]))
    with pytest.raises(InvalidEquityRecommendationRecordError, match="order"):
        decode_equity_recommendation_assessment(*_resealed(assessment))
    proposal = _golden("proposed.proposal")
    proposal["not_assessed"] = list(reversed(proposal["not_assessed"]))
    text, fingerprint = _resealed(proposal)
    with pytest.raises(InvalidEquityRecommendationRecordError):
        _stored("proposed.assessment", proposal_document=text, proposal_fingerprint=fingerprint)
    attempt = _golden("history-incomplete.attempt")
    attempt["checks"] = attempt["checks"][1:] + attempt["checks"][:1]
    text, fingerprint = _resealed(attempt)
    with pytest.raises(InvalidEquityRecommendationRecordError, match="order"):
        Attempt(fingerprint, text)


@pytest.mark.parametrize(
    "as_of",
    [
        "2026-08-12T04:05:59.999999Z",  # t - 1 us: the version-2 boundary, never version 1's
        "2026-08-12T04:06:00.000001Z",  # t + 1 us
        "2026-08-12T04:05:00Z",
    ],
)
def test_neg06_a_changed_history_instant_is_refused_for_version1(as_of: str) -> None:
    inputs = _golden("proposed.generation-inputs")
    inputs["issued_history"]["as_of"] = as_of
    assert _sha(inputs) != _fp("proposed.generation-inputs")
    with pytest.raises(InvalidEquityRecommendationRecordError, match="modeled issuance"):
        decode_equity_recommendation_inputs(inputs)
    # Moving t and the history together is refused too: t derives from the candidate.
    moved = _golden("proposed.generation-inputs")
    moved["issued_history"]["as_of"] = moved["modeled_issued_at"] = as_of
    with pytest.raises(InvalidEquityRecommendationRecordError):
        decode_equity_recommendation_inputs(moved)


def test_neg07_a_changed_assessment_status_or_reason_is_refused() -> None:
    # A status that does not follow from the reasons: refused at level A.
    document = _golden("not-eligible.assessment")
    document["status"] = "PROPOSED"
    text, fingerprint = _resealed(document)
    with pytest.raises(InvalidEquityRecommendationRecordError, match="status"):
        decode_equity_recommendation_assessment(text, fingerprint)
    # A structurally sound, historically false Core reason: refused at level C.
    document = _golden("not-eligible.assessment")
    document["core_outcome"]["outcome"]["reasons"] = ["INSUFFICIENT_RISK_BUDGET"]
    text, fingerprint = _resealed(document)
    assert fingerprint != _fp("not-eligible.assessment")
    altered = _stored(
        "not-eligible.assessment", assessment_document=text, assessment_fingerprint=fingerprint
    )
    with pytest.raises(EquityRecommendationReplayMismatchError):
        _verifier(ei4c._ports()).verify(altered)


def _proposed_with(change: Any) -> tuple[Disposition | None, Exception | None, str]:
    """Apply ``change`` to the proposal everywhere the assessment repeats it, then reseal."""
    proposal = _golden("proposed.proposal")
    change(proposal["entry_window"])
    proposal_text, proposal_fp = _resealed(proposal)
    assessment = _golden("proposed.assessment")
    assessment["proposal"] = proposal
    assessment["proposal_fingerprint"] = proposal_fp
    assessment["entry_window"] = proposal["entry_window"]
    text, fingerprint = _resealed(assessment)
    try:
        stored = _stored(
            "proposed.assessment",
            assessment_document=text,
            assessment_fingerprint=fingerprint,
            proposal_document=proposal_text,
            proposal_fingerprint=proposal_fp,
        )
    except InvalidEquityRecommendationRecordError as exc:
        return None, exc, proposal_fp
    return stored, None, proposal_fp


def test_neg08_changed_proposal_timing_is_refused() -> None:
    # The modeled issuance in the proposal must be the frozen t: refused at level A.
    def issued(window: dict[str, Any]) -> None:
        window["timing"]["issued_at"] = "2026-08-12T04:05:59.999999Z"

    stored, error, fingerprint = _proposed_with(issued)
    assert stored is None and error is not None and fingerprint != _fp("proposed.proposal")

    # A later, well-formed entry eligibility: decodes, and fails replay at level C.
    def later(window: dict[str, Any]) -> None:
        window["timing"]["eligible_from"] = "2026-08-12T04:15:00Z"

    stored, error, fingerprint = _proposed_with(later)
    assert error is None and stored is not None and fingerprint != _fp("proposed.proposal")
    with pytest.raises(EquityRecommendationReplayMismatchError):
        _verifier(ei4c._ports()).verify(stored)


def test_neg09_a_changed_attempt_status_is_refused() -> None:
    attempt = _golden("history-incomplete.attempt")
    attempt["status"] = "NOT_ELIGIBLE"
    text, fingerprint = _resealed(attempt)
    assert fingerprint != _fp("history-incomplete.attempt")
    with pytest.raises(InvalidEquityRecommendationRecordError, match="status"):
        Attempt(fingerprint, text)
    with pytest.raises(InvalidEquityRecommendationRecordError, match="fingerprint"):
        Attempt(_fp("history-incomplete.attempt"), text)  # unresealed
    keyed = _golden("proposed.attempt")
    keyed["disposition_kind"] = "UNRESOLVED"
    with pytest.raises(InvalidEquityRecommendationRecordError, match="disposition kind"):
        Attempt(*reversed(_resealed(keyed)))


def test_neg11_changed_embedded_reference_fingerprints_are_refused() -> None:
    edits = {
        "inputs_fingerprint": lambda d: d.__setitem__("inputs_fingerprint", _UNRESOLVED_EIRI1),
        "proposal_fingerprint": lambda d: d.__setitem__(
            "proposal_fingerprint", "eirp1-sha256:" + "0" * 64
        ),
        "generation_key": lambda d: d.__setitem__("generation_key", _NOT_ELIGIBLE_EIRK1),
        "policy_set_fingerprint": lambda d: d["inputs"].__setitem__(
            "policy_set_fingerprint", _LATENCY_EIRS1
        ),
    }
    for label, edit in edits.items():
        document = _golden("proposed.assessment")
        edit(document)
        text, fingerprint = _resealed(document)
        assert fingerprint != _fp("proposed.assessment"), label
        with pytest.raises(InvalidEquityRecommendationRecordError):
            decode_equity_recommendation_assessment(text, fingerprint)
    # An attempt naming another assessment is still a well-formed attempt, but a different
    # logical attempt: its identity no longer matches the golden. (The recorder and the
    # EI4-D11 authorization bind an attempt to its disposition; an attempt alone cannot.)
    attempt = _golden("proposed.attempt")
    attempt["assessment_fingerprint"] = _fp("unresolved.assessment")
    text, fingerprint = _resealed(attempt)
    assert Attempt(fingerprint, text).attempt_identity != _fp("proposed.attempt")


def test_neg12_changed_stored_provenance_is_refused_at_level_b() -> None:
    for field, value in (
        ("research_decision_fingerprint", "eir1-sha256:" + "e" * 64),
        ("evidence_snapshot_id", "ecs1-sha256:" + "e" * 64),
    ):
        document = _golden("unresolved.assessment")  # Core never ran, so one copy per place
        document["candidate"]["source"][field] = value
        document["inputs"]["candidate"]["source"][field] = value
        _reseal_assessment(document)
        text, fingerprint = _resealed(document)
        relinked = _stored(
            "unresolved.assessment", assessment_document=text, assessment_fingerprint=fingerprint
        )
        assert relinked.generation_key == _EIRK1  # provenance outside the key, so level A passes
        with pytest.raises(EquityRecommendationSourceVerificationError, match="source"):
            _verifier(ei4c._ports()).verify(relinked)


def test_neg13_a_changed_stored_version_label_is_refused() -> None:
    assert _stored("proposed.assessment").evaluator_version == _EVALUATOR_V1
    # An unknown label is incompatible, never migrated or replayed.
    with pytest.raises(IncompatibleEquityRecommendationRecordError, match="not supported"):
        _stored("proposed.assessment", evaluator_version=ei4d._FUTURE)
    # Version-1 documents under the approved-but-unimplemented /2 label are refused. Only the
    # base record error is asserted: slice 3a's pairing rule may reclassify it as Invalid.
    with pytest.raises(InvalidEquityRecommendationRecordError):
        _stored("proposed.assessment", evaluator_version=_EVALUATOR_V1[:-1] + "2")
    for corrupt in ("", None, 1, _EVALUATOR_V1.encode()):
        with pytest.raises(InvalidEquityRecommendationRecordError) as raised:
            _stored("proposed.assessment", evaluator_version=corrupt)
        assert not isinstance(raised.value, IncompatibleEquityRecommendationRecordError)
    # The stored kind is a label of the assessment's status too.
    with pytest.raises(InvalidEquityRecommendationRecordError, match="kind"):
        _stored("unresolved.assessment", kind=_K.NOT_ELIGIBLE)


def test_neg14_a_tampered_fingerprint_is_refused() -> None:
    digest = _fp("proposed.assessment").removeprefix("eira1-sha256:")
    for tampered in (
        "eira1-sha256:" + "0" * 64,
        "eira1-sha256:" + digest.upper(),
        "eira1-sha256:" + digest[:-1],
        "eira1-sha256:" + digest[1:] + digest[0],
        _fp("unresolved.assessment"),
    ):
        with pytest.raises(InvalidEquityRecommendationRecordError, match="fingerprint"):
            _stored("proposed.assessment", assessment_fingerprint=tampered)
    # A stored proposal fingerprint is first checked against the one the assessment names.
    with pytest.raises(InvalidEquityRecommendationRecordError, match="another proposal"):
        _stored("proposed.assessment", proposal_fingerprint="eirp1-sha256:" + "0" * 64)
    with pytest.raises(InvalidEquityRecommendationRecordError, match="fingerprint"):
        Attempt(_fp("proposed.attempt"), _golden_text("history-incomplete.attempt"))


def test_neg_the_unsupported_test_label_stays_unsupported_during_this_module() -> None:
    # Guards the module's own premise: nothing here widened the supported set.
    assert decoding_module.SUPPORTED_EQUITY_RECOMMENDATION_EVALUATOR_VERSIONS == frozenset(
        {_EVALUATOR_V1}
    )
