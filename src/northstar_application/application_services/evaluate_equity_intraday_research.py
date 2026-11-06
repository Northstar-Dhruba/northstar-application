"""Synthetic equity intraday research evaluation (EI-3A).

Turns one immutable EI-2B evidence snapshot into one explainable research
decision, under the owner decisions recorded on 2026-11-05 in the EI-3
Decision Register (northstar-docs, product/Equity-Intraday-V1-EI3-Decision-
Register.md): EI3-D1 to EI3-D5, EI3-D8, EI0-D3 and EI0-D6. It is synthetic
research only. A decision is not a recommendation, order, fill, position or
P&L, and no result here is evidence of profitability or significance.

Decision instant (EI3-D1)
-------------------------
The decision instant is the snapshot's ``as_of``. Only what the snapshot
froze at that instant -- observations journaled by its watermark, received by
``as_of``, final under its finality policy -- contributes. A candle's interval
end is its identity (EI0-D6), never its availability.

Evidence (EI3-D2)
-----------------
Required intervals are every interval of the approved 09:15 to 15:00 IST
research window (CE-D17) that had ended at ``as_of``; intervals that had not
ended are not required and are never treated as missing. Every required
interval must be EI-2A ACCEPTABLE. Otherwise the decision is REJECTED with
every category present, in this order: UNRESOLVED_SOURCE_REVISION,
CONFLICTING_CANDLE_VALUES, UNFINALIZED_CANDLE, MISSING_CANDLE, then
UNMATCHED_SESSION_EVIDENCE when the snapshot holds any observation EI-2A
could not place in an interval. Nothing is interpolated, carried forward or
chosen: an acceptable interval's acceptable observations are exact repeats of
one source identity, so its bar is unique.

Before any of that, the snapshot itself must be trustworthy and in scope. The
first failure below rejects the decision on its own, in this precedence:
SNAPSHOT_UNAVAILABLE, SNAPSHOT_REFERENCE_MISMATCH, PROVENANCE_UNKNOWN (a
reference kind that is not SYNTHETIC or ACCEPTED_PRODUCTION),
SNAPSHOT_INTEGRITY_FAILURE (it does not re-derive through the EI-2B canonical
codec, or an acceptable interval's observations disagree),
EVIDENCE_NOT_SYNTHETIC (this evaluator accepts SYNTHETIC evidence only),
LISTING_MISMATCH, TRADING_DATE_MISMATCH, UNSUPPORTED_TIMEFRAME,
RESEARCH_WINDOW_POLICY_MISMATCH and INVALID_INTERVAL_GRID.

Strategy (EI3-D4, EI3-D5, EI3-D8, EI0-D3)
-----------------------------------------
An opening-range breakout with a session VWAP approximation and volume
confirmation, under an explicit, versioned, synthetic configuration. With the
ended session bars B1..Bk in interval order and ``n`` the configured opening-
range length:

- the opening range is B1..Bn: its high is max(high), its low min(low);
- the signal bar is Bk, the latest ended interval -- never an older one and
  never a later one. It must be after the opening range (k > n). Whether a
  listing may signal more than once a session is EI3-D7, which is pending, and
  is not decided here;
- breakout: close(Bk) > opening-range high, strictly; equality is not one;
- VWAP approximation over B1..Bk, the signal bar included:
  sum(((high + low + close) / 3) * volume) / sum(volume). It is an
  approximation from 5-minute bars, not the exchange's traded VWAP. It is
  unavailable when sum(volume) is 0. Filter: close(Bk) > VWAP, strictly;
- volume baseline: the arithmetic mean of volume over B1..B(k-1) -- every
  earlier bar of the session, never the signal bar and never a later bar.
  It needs at least ``minimum_baseline_bars`` bars; a mean of 0 is unusable
  rather than a threshold every bar passes. Confirmation:
  volume(Bk) >= mean * volume_multiplier.

Every quantity is computed with exact rational arithmetic from the Decimal
prices and volumes -- no float, no rounding -- and recorded exactly: as
decimal text when it terminates, otherwise as ``numerator/denominator``.

LONG_CANDIDATE needs every condition. Otherwise, for trustworthy evidence, the
outcome is NO_SIGNAL with every reason that applies, among them
BEARISH_RESEARCH_NOT_SUPPORTED when the signal bar closes below the opening-
range low: research is LONG only, and a breakdown never becomes a short.

Nothing here computes an entry, stop, target, exit, size, cost or P&L; the
opening-range low is research data, not a protective stop. Nothing reads a
clock, persists anything, or calls a provider, broker or network.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from enum import StrEnum
from fractions import Fraction
from typing import Any

from northstar_core.domain.value_objects import ListingReference
from northstar_core.foundation.value_objects import PointInTime, Timeframe
from northstar_core.market_data import HistoricalOHLCVBar
from northstar_core.strategy import ExplanationReason, StrategyIdentity

from northstar_application.application_services.equity_candle_evidence_journal import (
    canonical_equity_candle_json,
)
from northstar_application.application_services.equity_candle_evidence_snapshot import (
    EquityCandleEvidenceClass,
    EquityCandleEvidenceSnapshot,
    equity_candle_snapshot_from_document,
)
from northstar_application.application_services.equity_intraday_candle_validation import (
    EquityCandleCoverage,
    EquityCandleIntervalOutcome,
)
from northstar_application.application_services.equity_research_window import (
    NSE_EQUITY_INTRADAY_V1_RESEARCH_WINDOW,
)
from northstar_application.ports.equity_candle_evidence_journal import (
    EquityCandleEvidenceSnapshotStore,
)
from northstar_application.ports.equity_research_reference_provenance import (
    EquityReferenceProvenanceKind,
)

EQUITY_INTRADAY_RESEARCH_DECISION_PREFIX = "eir1-sha256:"
_FORMAT = "northstar.equity-intraday-research-decision/1"
_FIVE_MINUTES = Timeframe("5m")
_SYNTHETIC_VERSION_PREFIX = "synthetic-"


class InvalidEquityIntradayResearchConfigurationError(ValueError):
    """Raised when a research configuration is malformed or not visibly synthetic."""


class InvalidEquityIntradayResearchDecisionError(ValueError):
    """Raised when a research decision is internally inconsistent."""


def _fail(message: str) -> None:
    raise InvalidEquityIntradayResearchConfigurationError(message)


@dataclass(frozen=True, slots=True)
class EquityOpeningRangeBreakoutConfiguration:
    """An explicit, versioned, synthetic opening-range breakout configuration (EI3-D5).

    ``version`` must begin ``synthetic-``: no value here is an approved
    production parameter, and any changed value is a new version.
    """

    strategy: StrategyIdentity
    version: str
    opening_range_bars: int
    volume_multiplier: Decimal
    minimum_baseline_bars: int
    timeframe: Timeframe = _FIVE_MINUTES
    window_policy_version: str = NSE_EQUITY_INTRADAY_V1_RESEARCH_WINDOW.version

    def __post_init__(self) -> None:
        if not isinstance(self.strategy, StrategyIdentity):
            _fail("strategy must be a StrategyIdentity.")
        if not isinstance(self.version, str) or not self.version.startswith(
            _SYNTHETIC_VERSION_PREFIX
        ):
            _fail("version must be text beginning 'synthetic-': the calibration is synthetic.")
        if len(self.version.strip()) <= len(_SYNTHETIC_VERSION_PREFIX):
            _fail("version must name the calibration after 'synthetic-'.")
        for name in ("opening_range_bars", "minimum_baseline_bars"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                _fail(f"{name} must be an integer of at least 1.")
        multiplier = self.volume_multiplier
        if not isinstance(multiplier, Decimal) or not multiplier.is_finite() or multiplier <= 0:
            _fail("volume_multiplier must be a finite, positive Decimal.")
        if self.timeframe != _FIVE_MINUTES:
            _fail("Only five-minute candles are supported.")
        if not isinstance(self.window_policy_version, str) or not self.window_policy_version:
            _fail("window_policy_version must be non-empty text.")

    @property
    def is_synthetic(self) -> bool:
        """Return True: every EI-3A configuration is synthetic calibration."""
        return True

    def document(self) -> dict[str, Any]:
        """Return the configuration's canonical content."""
        return {
            "strategy": self.strategy.identity,
            "version": self.version,
            "opening_range_bars": self.opening_range_bars,
            "volume_multiplier": _decimal_text(self.volume_multiplier),
            "minimum_baseline_bars": self.minimum_baseline_bars,
            "timeframe": self.timeframe.value,
            "window_policy_version": self.window_policy_version,
        }


class EquityIntradayResearchOutcome(StrEnum):
    """The research outcome (EI3-D3). None of them is an order or a recommendation."""

    LONG_CANDIDATE = "LONG_CANDIDATE"
    NO_SIGNAL = "NO_SIGNAL"
    REJECTED = "REJECTED"


class EquityIntradayResearchRejection(StrEnum):
    """Why evidence could not be researched, in precedence order."""

    SNAPSHOT_UNAVAILABLE = "SNAPSHOT_UNAVAILABLE"
    SNAPSHOT_REFERENCE_MISMATCH = "SNAPSHOT_REFERENCE_MISMATCH"
    PROVENANCE_UNKNOWN = "PROVENANCE_UNKNOWN"
    SNAPSHOT_INTEGRITY_FAILURE = "SNAPSHOT_INTEGRITY_FAILURE"
    EVIDENCE_NOT_SYNTHETIC = "EVIDENCE_NOT_SYNTHETIC"
    LISTING_MISMATCH = "LISTING_MISMATCH"
    TRADING_DATE_MISMATCH = "TRADING_DATE_MISMATCH"
    UNSUPPORTED_TIMEFRAME = "UNSUPPORTED_TIMEFRAME"
    RESEARCH_WINDOW_POLICY_MISMATCH = "RESEARCH_WINDOW_POLICY_MISMATCH"
    INVALID_INTERVAL_GRID = "INVALID_INTERVAL_GRID"
    UNRESOLVED_SOURCE_REVISION = "UNRESOLVED_SOURCE_REVISION"
    CONFLICTING_CANDLE_VALUES = "CONFLICTING_CANDLE_VALUES"
    UNFINALIZED_CANDLE = "UNFINALIZED_CANDLE"
    MISSING_CANDLE = "MISSING_CANDLE"
    UNMATCHED_SESSION_EVIDENCE = "UNMATCHED_SESSION_EVIDENCE"


class EquityIntradayNoSignalReason(StrEnum):
    """Why trustworthy evidence produced no LONG candidate."""

    OPENING_RANGE_INCOMPLETE = "OPENING_RANGE_INCOMPLETE"
    NO_POST_OPENING_RANGE_INTERVAL = "NO_POST_OPENING_RANGE_INTERVAL"
    CLOSE_NOT_ABOVE_OPENING_RANGE_HIGH = "CLOSE_NOT_ABOVE_OPENING_RANGE_HIGH"
    BEARISH_RESEARCH_NOT_SUPPORTED = "BEARISH_RESEARCH_NOT_SUPPORTED"
    VWAP_APPROXIMATION_UNAVAILABLE = "VWAP_APPROXIMATION_UNAVAILABLE"
    CLOSE_NOT_ABOVE_VWAP_APPROXIMATION = "CLOSE_NOT_ABOVE_VWAP_APPROXIMATION"
    INSUFFICIENT_VOLUME_BASELINE = "INSUFFICIENT_VOLUME_BASELINE"
    ZERO_VOLUME_BASELINE = "ZERO_VOLUME_BASELINE"
    VOLUME_NOT_CONFIRMED = "VOLUME_NOT_CONFIRMED"


_REJECTION_ORDER = tuple(EquityIntradayResearchRejection)
_NO_SIGNAL_ORDER = tuple(EquityIntradayNoSignalReason)
_UNTRUSTED = frozenset(
    {
        EquityIntradayResearchRejection.SNAPSHOT_UNAVAILABLE,
        EquityIntradayResearchRejection.SNAPSHOT_REFERENCE_MISMATCH,
        EquityIntradayResearchRejection.SNAPSHOT_INTEGRITY_FAILURE,
        EquityIntradayResearchRejection.PROVENANCE_UNKNOWN,
    }
)
_EVIDENCE_REJECTIONS = {
    EquityCandleIntervalOutcome.UNRESOLVED_REVISIONS: (
        EquityIntradayResearchRejection.UNRESOLVED_SOURCE_REVISION
    ),
    EquityCandleIntervalOutcome.CONFLICTING_VALUES: (
        EquityIntradayResearchRejection.CONFLICTING_CANDLE_VALUES
    ),
    EquityCandleIntervalOutcome.UNFINALIZED: EquityIntradayResearchRejection.UNFINALIZED_CANDLE,
    EquityCandleIntervalOutcome.MISSING: EquityIntradayResearchRejection.MISSING_CANDLE,
}


# ---------------------------------------------------------------------------
# Exact values
# ---------------------------------------------------------------------------


def _decimal_text(value: Decimal) -> str:
    return format(value, "f")


def exact_text(value: Fraction) -> str:
    """Return ``value`` exactly: terminating decimal text, otherwise ``numerator/denominator``."""
    numerator, denominator = value.numerator, value.denominator
    rest, twos, fives = denominator, 0, 0
    while rest % 2 == 0:
        rest, twos = rest // 2, twos + 1
    while rest % 5 == 0:
        rest, fives = rest // 5, fives + 1
    if rest != 1:
        return f"{numerator}/{denominator}"
    places = max(twos, fives)
    scaled = abs(numerator) * (10**places // denominator)
    digits = str(scaled).rjust(places + 1, "0")
    whole, fraction = digits[: len(digits) - places], digits[len(digits) - places :]
    fraction = fraction.rstrip("0")
    text = f"{whole}.{fraction}" if fraction else whole
    return f"-{text}" if numerator < 0 else text


def _fraction(value: Decimal) -> Fraction:
    return Fraction(value)


# ---------------------------------------------------------------------------
# Decision contract
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EquityIntradayResearchEvidence:
    """What a decision rests on: its snapshot, provenance and the bars it cited."""

    snapshot_id: str
    manifest_fingerprint: str
    watermark: int
    as_of: PointInTime
    evidence_class: EquityCandleEvidenceClass
    coverage: EquityCandleCoverage
    finality_policy_version: str
    window_policy_version: str
    venue_reference: str
    venue_provenance: EquityReferenceProvenanceKind
    applicability_reference: str
    applicability_provenance: EquityReferenceProvenanceKind
    required_interval_ends: tuple[PointInTime, ...]
    cited_sequences: tuple[int, ...]

    def document(self) -> dict[str, Any]:
        """Return the evidence's canonical content."""
        return {
            "snapshot_id": self.snapshot_id,
            "manifest_fingerprint": self.manifest_fingerprint,
            "watermark": self.watermark,
            "as_of": self.as_of.value,
            "evidence_class": self.evidence_class.value,
            "coverage": self.coverage.value,
            "finality_policy_version": self.finality_policy_version,
            "window_policy_version": self.window_policy_version,
            "venue_reference": self.venue_reference,
            "venue_provenance": self.venue_provenance.value,
            "applicability_reference": self.applicability_reference,
            "applicability_provenance": self.applicability_provenance.value,
            "required_interval_ends": [end.value for end in self.required_interval_ends],
            "cited_sequences": list(self.cited_sequences),
        }


@dataclass(frozen=True, slots=True)
class EquityOpeningRangeBreakoutValues:
    """The values the strategy computed, as far as the evidence allowed.

    Prices and volumes are the candles' own Decimals; derived values are exact
    rationals. A value is None when it could not be computed.
    """

    ended_bars: int
    opening_range_high: Decimal | None = None
    opening_range_low: Decimal | None = None
    signal_interval_end: PointInTime | None = None
    signal_close: Decimal | None = None
    signal_volume: Decimal | None = None
    vwap_approximation: Fraction | None = None
    cumulative_volume: Decimal | None = None
    baseline_bars: int | None = None
    baseline_mean_volume: Fraction | None = None
    volume_threshold: Fraction | None = None

    def document(self) -> dict[str, Any]:
        """Return the values' canonical content."""

        def decimal(value: Decimal | None) -> str | None:
            return None if value is None else _decimal_text(value)

        def rational(value: Fraction | None) -> str | None:
            return None if value is None else exact_text(value)

        return {
            "ended_bars": self.ended_bars,
            "opening_range_high": decimal(self.opening_range_high),
            "opening_range_low": decimal(self.opening_range_low),
            "signal_interval_end": (
                None if self.signal_interval_end is None else self.signal_interval_end.value
            ),
            "signal_close": decimal(self.signal_close),
            "signal_volume": decimal(self.signal_volume),
            "vwap_approximation": rational(self.vwap_approximation),
            "cumulative_volume": decimal(self.cumulative_volume),
            "baseline_bars": self.baseline_bars,
            "baseline_mean_volume": rational(self.baseline_mean_volume),
            "volume_threshold": rational(self.volume_threshold),
        }


EquityIntradayResearchReason = EquityIntradayResearchRejection | EquityIntradayNoSignalReason


@dataclass(frozen=True, slots=True)
class EquityIntradayResearchDecision:
    """One immutable research decision for one listing, date and snapshot.

    ``evidence`` is None only when the snapshot could not be trusted (its
    rejection is one of SNAPSHOT_UNAVAILABLE, SNAPSHOT_REFERENCE_MISMATCH,
    PROVENANCE_UNKNOWN or SNAPSHOT_INTEGRITY_FAILURE); then there is no decision
    instant. ``fingerprint`` covers every structured field and no prose.
    """

    listing: ListingReference
    trading_date: date
    requested_snapshot_id: str
    configuration: EquityOpeningRangeBreakoutConfiguration
    outcome: EquityIntradayResearchOutcome
    reasons: tuple[EquityIntradayResearchReason, ...]
    evidence: EquityIntradayResearchEvidence | None
    values: EquityOpeningRangeBreakoutValues | None
    explanation: tuple[ExplanationReason, ...]
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        outcome, reasons = self.outcome, self.reasons
        if not isinstance(outcome, EquityIntradayResearchOutcome):
            raise InvalidEquityIntradayResearchDecisionError("outcome is not a research outcome.")
        if not isinstance(reasons, tuple):
            raise InvalidEquityIntradayResearchDecisionError("reasons must be a tuple.")
        expected: type
        if outcome is EquityIntradayResearchOutcome.LONG_CANDIDATE:
            if reasons:
                raise InvalidEquityIntradayResearchDecisionError("A candidate has no reasons.")
        else:
            expected = (
                EquityIntradayResearchRejection
                if outcome is EquityIntradayResearchOutcome.REJECTED
                else EquityIntradayNoSignalReason
            )
            order = list(expected)
            if not reasons or not all(isinstance(r, expected) for r in reasons):
                raise InvalidEquityIntradayResearchDecisionError(
                    f"A {outcome.value} decision needs {expected.__name__} reasons."
                )
            if sorted(reasons, key=order.index) != list(reasons) or len(set(reasons)) != len(
                reasons
            ):
                raise InvalidEquityIntradayResearchDecisionError(
                    "Reasons must be distinct and in precedence order."
                )
        untrusted = bool(reasons) and reasons[0] in _UNTRUSTED
        if (self.evidence is None) != untrusted:
            raise InvalidEquityIntradayResearchDecisionError(
                "Evidence is absent exactly when the snapshot could not be trusted."
            )
        if outcome is not EquityIntradayResearchOutcome.REJECTED and self.values is None:
            raise InvalidEquityIntradayResearchDecisionError("A researched decision has values.")
        object.__setattr__(
            self,
            "fingerprint",
            EQUITY_INTRADAY_RESEARCH_DECISION_PREFIX
            + hashlib.sha256(canonical_equity_candle_json(self.document()).encode()).hexdigest(),
        )

    @property
    def decision_instant(self) -> PointInTime | None:
        """Return the snapshot's ``as_of`` (EI3-D1), or None when it could not be trusted."""
        return None if self.evidence is None else self.evidence.as_of

    @property
    def is_synthetic(self) -> bool:
        """Return True: EI-3A researches synthetic evidence only."""
        return True

    def document(self) -> dict[str, Any]:
        """Return the decision's canonical content -- the fingerprint's input."""
        return {
            "format": _FORMAT,
            "listing": {
                "symbol": self.listing.symbol.value,
                "exchange_code": self.listing.exchange_code.value,
            },
            "trading_date": self.trading_date.isoformat(),
            "requested_snapshot_id": self.requested_snapshot_id,
            "configuration": self.configuration.document(),
            "outcome": self.outcome.value,
            "reasons": [reason.value for reason in self.reasons],
            "evidence": None if self.evidence is None else self.evidence.document(),
            "values": None if self.values is None else self.values.document(),
        }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _instant(value: PointInTime) -> datetime:
    return datetime.fromisoformat(value.value.replace("Z", "+00:00")).astimezone(UTC)


def _evidence(
    snapshot: EquityCandleEvidenceSnapshot,
    required: tuple[PointInTime, ...],
    cited: tuple[int, ...],
) -> EquityIntradayResearchEvidence:
    return EquityIntradayResearchEvidence(
        snapshot_id=snapshot.snapshot_id,
        manifest_fingerprint=snapshot.manifest_fingerprint,
        watermark=snapshot.watermark,
        as_of=snapshot.as_of,
        evidence_class=snapshot.evidence_class,
        coverage=snapshot.coverage,
        finality_policy_version=snapshot.finality_policy.version,
        window_policy_version=snapshot.window_policy_version,
        venue_reference=snapshot.venue_reference,
        venue_provenance=snapshot.venue_provenance,
        applicability_reference=snapshot.applicability_reference,
        applicability_provenance=snapshot.applicability_provenance,
        required_interval_ends=required,
        cited_sequences=cited,
    )


class EquityOpeningRangeBreakoutEvaluator:
    """Evaluates one snapshot under one synthetic opening-range breakout configuration."""

    def __init__(self, configuration: EquityOpeningRangeBreakoutConfiguration) -> None:
        if not isinstance(configuration, EquityOpeningRangeBreakoutConfiguration):
            raise TypeError("configuration must be an EquityOpeningRangeBreakoutConfiguration.")
        self._configuration = configuration

    def evaluate(
        self,
        listing: ListingReference,
        trading_date: date,
        snapshot: EquityCandleEvidenceSnapshot | None,
        requested_snapshot_id: str | None = None,
    ) -> EquityIntradayResearchDecision:
        """Return the research decision for ``listing`` on ``trading_date`` from ``snapshot``.

        ``requested_snapshot_id`` names the snapshot the caller asked for; a
        missing snapshot, or one under another identity, is rejected.
        """
        if not isinstance(listing, ListingReference):
            raise TypeError("listing must be a ListingReference.")
        if not isinstance(trading_date, date) or isinstance(trading_date, datetime):
            raise TypeError("trading_date must be a plain date.")
        if snapshot is not None and not isinstance(snapshot, EquityCandleEvidenceSnapshot):
            raise TypeError("snapshot must be an EquityCandleEvidenceSnapshot or None.")
        rejection = EquityIntradayResearchRejection
        requested = requested_snapshot_id
        if requested is None:
            if snapshot is None:
                raise ValueError("Name the requested snapshot when none is supplied.")
            requested = snapshot.snapshot_id

        def decide(
            outcome: EquityIntradayResearchOutcome,
            reasons: tuple[EquityIntradayResearchReason, ...],
            evidence: EquityIntradayResearchEvidence | None,
            values: EquityOpeningRangeBreakoutValues | None,
            explanation: tuple[ExplanationReason, ...],
        ) -> EquityIntradayResearchDecision:
            return EquityIntradayResearchDecision(
                listing=listing,
                trading_date=trading_date,
                requested_snapshot_id=requested,  # type: ignore[arg-type]
                configuration=self._configuration,
                outcome=outcome,
                reasons=reasons,
                evidence=evidence,
                values=values,
                explanation=explanation,
            )

        def reject(
            reasons: tuple[EquityIntradayResearchRejection, ...],
            detail: str,
            evidence: EquityIntradayResearchEvidence | None = None,
        ) -> EquityIntradayResearchDecision:
            return decide(
                EquityIntradayResearchOutcome.REJECTED,
                reasons,
                evidence,
                None,
                (ExplanationReason(detail, tuple(r.value for r in reasons)),),
            )

        if snapshot is None:
            return reject(
                (rejection.SNAPSHOT_UNAVAILABLE,), f"No snapshot is issued as {requested}."
            )
        if snapshot.snapshot_id != requested:
            return reject(
                (rejection.SNAPSHOT_REFERENCE_MISMATCH,),
                f"Snapshot {snapshot.snapshot_id} is not the requested {requested}.",
            )
        for kind in (snapshot.venue_provenance, snapshot.applicability_provenance):
            if not isinstance(kind, EquityReferenceProvenanceKind):
                return reject(
                    (rejection.PROVENANCE_UNKNOWN,), "The snapshot names an unknown provenance."
                )
        if not self._trustworthy(snapshot):
            return reject(
                (rejection.SNAPSHOT_INTEGRITY_FAILURE,),
                "The snapshot does not re-derive through the EI-2B canonical codec.",
            )

        ended = tuple(
            i for i in snapshot.intervals if i.outcome is not EquityCandleIntervalOutcome.NOT_ENDED
        )
        required = tuple(i.closes_at for i in ended)
        scope = _evidence(snapshot, required, ())
        structural = self._structural(listing, trading_date, snapshot)
        if structural is not None:
            return reject((structural[0],), structural[1], scope)

        problems = {
            _EVIDENCE_REJECTIONS[i.outcome]
            for i in ended
            if i.outcome is not EquityCandleIntervalOutcome.ACCEPTABLE
        }
        if snapshot.unmatched_sequences:
            problems.add(rejection.UNMATCHED_SESSION_EVIDENCE)
        if problems:
            ordered = tuple(r for r in _REJECTION_ORDER if r in problems)
            failing = [
                f"{i.closes_at.value}:{i.outcome.value}"
                for i in ended
                if i.outcome is not EquityCandleIntervalOutcome.ACCEPTABLE
            ]
            return reject(
                ordered,
                "Required session evidence is not acceptable: "
                + ", ".join(failing or [f"{len(snapshot.unmatched_sequences)} unplaced"])
                + ". Nothing is interpolated, carried forward or chosen.",
                scope,
            )

        records = {o.sequence: o.record for o in snapshot.observations}
        bars: list[HistoricalOHLCVBar] = []
        cited: list[int] = []
        for interval in ended:
            sequence = interval.acceptable_sequences[0]
            bar = records[sequence].observation.bar
            if any(
                records[s].observation.values != records[sequence].observation.values
                for s in interval.acceptable_sequences
            ):
                return reject(
                    (rejection.SNAPSHOT_INTEGRITY_FAILURE,),
                    f"Acceptable observations at {interval.closes_at} disagree.",
                )
            bars.append(bar)
            cited.append(sequence)
        evidence = _evidence(snapshot, required, tuple(cited))
        return self._strategy(tuple(bars), evidence, decide)

    @staticmethod
    def _trustworthy(snapshot: EquityCandleEvidenceSnapshot) -> bool:
        try:
            rebuilt = equity_candle_snapshot_from_document(snapshot.document())
        except (ValueError, TypeError, KeyError, AttributeError, ArithmeticError):
            return False
        return rebuilt == snapshot and rebuilt.snapshot_id == snapshot.snapshot_id

    def _structural(
        self,
        listing: ListingReference,
        trading_date: date,
        snapshot: EquityCandleEvidenceSnapshot,
    ) -> tuple[EquityIntradayResearchRejection, str] | None:
        rejection = EquityIntradayResearchRejection
        configuration = self._configuration
        if snapshot.evidence_class is not EquityCandleEvidenceClass.SYNTHETIC:
            return (
                rejection.EVIDENCE_NOT_SYNTHETIC,
                f"EI-3A researches SYNTHETIC evidence only, not {snapshot.evidence_class.value}.",
            )
        if snapshot.listing != listing:
            return (
                rejection.LISTING_MISMATCH,
                f"Snapshot is for {snapshot.listing}, not {listing}.",
            )
        if snapshot.trading_date != trading_date:
            return (
                rejection.TRADING_DATE_MISMATCH,
                f"Snapshot is for {snapshot.trading_date.isoformat()}, not "
                f"{trading_date.isoformat()}.",
            )
        if snapshot.timeframe != configuration.timeframe:
            return (
                rejection.UNSUPPORTED_TIMEFRAME,
                f"Snapshot timeframe {snapshot.timeframe} is not {configuration.timeframe}.",
            )
        if snapshot.window_policy_version != configuration.window_policy_version:
            return (
                rejection.RESEARCH_WINDOW_POLICY_MISMATCH,
                f"Snapshot window policy {snapshot.window_policy_version} is not "
                f"{configuration.window_policy_version}.",
            )
        step = timedelta(minutes=5)
        expected_open = _instant(snapshot.window_opens_at)
        for interval in snapshot.intervals:
            opens, closes = _instant(interval.opens_at), _instant(interval.closes_at)
            if opens != expected_open or closes - opens != step:
                return (
                    rejection.INVALID_INTERVAL_GRID,
                    f"Interval ending {interval.closes_at} breaks the contiguous 5-minute grid.",
                )
            expected_open = closes
        if expected_open != _instant(snapshot.window_closes_at):
            return (rejection.INVALID_INTERVAL_GRID, "The grid does not reach the window close.")
        return None

    def _strategy(self, bars, evidence, decide):  # type: ignore[no-untyped-def]
        configuration = self._configuration
        no = EquityIntradayNoSignalReason
        n = configuration.opening_range_bars
        if len(bars) < n:
            return decide(
                EquityIntradayResearchOutcome.NO_SIGNAL,
                (no.OPENING_RANGE_INCOMPLETE,),
                evidence,
                EquityOpeningRangeBreakoutValues(ended_bars=len(bars)),
                (
                    ExplanationReason(
                        f"Only {len(bars)} of the {n} opening-range bars had ended.",
                        (no.OPENING_RANGE_INCOMPLETE.value,),
                    ),
                ),
            )
        opening = bars[:n]
        range_high = max(bar.high.amount for bar in opening)
        range_low = min(bar.low.amount for bar in opening)
        if len(bars) == n:
            return decide(
                EquityIntradayResearchOutcome.NO_SIGNAL,
                (no.NO_POST_OPENING_RANGE_INTERVAL,),
                evidence,
                EquityOpeningRangeBreakoutValues(
                    ended_bars=n, opening_range_high=range_high, opening_range_low=range_low
                ),
                (
                    ExplanationReason(
                        "The opening range is complete; no later interval had ended.",
                        (no.NO_POST_OPENING_RANGE_INTERVAL.value,),
                    ),
                ),
            )
        signal = bars[-1]
        close, volume = signal.close.amount, signal.volume.value
        reasons: list[EquityIntradayNoSignalReason] = []
        explanation: list[ExplanationReason] = []

        def note(reason: EquityIntradayNoSignalReason | None, text: str, *signals: str) -> None:
            if reason is not None:
                reasons.append(reason)
            explanation.append(
                ExplanationReason(text, signals or ((reason.value,) if reason else ("pass",)))
            )

        if close > range_high:
            note(None, "Close is strictly above the opening-range high.", f"close={close}")
        else:
            note(
                no.CLOSE_NOT_ABOVE_OPENING_RANGE_HIGH,
                f"Close {close} is not strictly above the opening-range high {range_high}.",
            )
        if close < range_low:
            note(
                no.BEARISH_RESEARCH_NOT_SUPPORTED,
                f"Close {close} is below the opening-range low {range_low}; research is LONG "
                "only and a breakdown never becomes a short.",
            )

        cumulative = sum((bar.volume.value for bar in bars), Decimal(0))
        vwap: Fraction | None = None
        if cumulative == 0:
            note(
                no.VWAP_APPROXIMATION_UNAVAILABLE,
                "The session VWAP approximation is unavailable: cumulative volume is 0.",
            )
        else:
            weighted = sum(
                (
                    (_fraction(b.high.amount) + _fraction(b.low.amount) + _fraction(b.close.amount))
                    / 3
                    * _fraction(b.volume.value)
                    for b in bars
                ),
                Fraction(0),
            )
            vwap = weighted / _fraction(cumulative)
            if _fraction(close) > vwap:
                note(
                    None,
                    "Close is strictly above the session VWAP approximation.",
                    f"vwap_approximation={exact_text(vwap)}",
                )
            else:
                note(
                    no.CLOSE_NOT_ABOVE_VWAP_APPROXIMATION,
                    f"Close {close} is not strictly above the session VWAP approximation "
                    f"{exact_text(vwap)}.",
                )

        prior = bars[:-1]
        mean: Fraction | None = None
        threshold: Fraction | None = None
        if len(prior) < configuration.minimum_baseline_bars:
            note(
                no.INSUFFICIENT_VOLUME_BASELINE,
                f"{len(prior)} prior bars are fewer than the "
                f"{configuration.minimum_baseline_bars} the baseline needs.",
            )
        else:
            mean = sum((_fraction(b.volume.value) for b in prior), Fraction(0)) / len(prior)
            threshold = mean * _fraction(configuration.volume_multiplier)
            if mean == 0:
                note(
                    no.ZERO_VOLUME_BASELINE,
                    "The prior-bar mean volume is 0, so no volume can be confirmed against it.",
                )
            elif _fraction(volume) >= threshold:
                note(
                    None,
                    "Volume meets the configured multiple of the prior-bar mean.",
                    f"volume_threshold={exact_text(threshold)}",
                )
            else:
                note(
                    no.VOLUME_NOT_CONFIRMED,
                    f"Volume {volume} is below {exact_text(threshold)}, the configured multiple "
                    "of the prior-bar mean.",
                )

        values = EquityOpeningRangeBreakoutValues(
            ended_bars=len(bars),
            opening_range_high=range_high,
            opening_range_low=range_low,
            signal_interval_end=signal.point_in_time,
            signal_close=close,
            signal_volume=volume,
            vwap_approximation=vwap,
            cumulative_volume=cumulative,
            baseline_bars=len(prior),
            baseline_mean_volume=mean,
            volume_threshold=threshold,
        )
        ordered = tuple(r for r in _NO_SIGNAL_ORDER if r in reasons)
        outcome = (
            EquityIntradayResearchOutcome.NO_SIGNAL
            if ordered
            else EquityIntradayResearchOutcome.LONG_CANDIDATE
        )
        return decide(outcome, ordered, evidence, values, tuple(explanation))


class EvaluateEquityIntradayResearchUseCase:
    """Loads one issued snapshot and evaluates it; nothing is persisted."""

    def __init__(
        self,
        snapshots: EquityCandleEvidenceSnapshotStore,
        configuration: EquityOpeningRangeBreakoutConfiguration,
    ) -> None:
        if not isinstance(snapshots, EquityCandleEvidenceSnapshotStore):
            raise TypeError("snapshots must be an EquityCandleEvidenceSnapshotStore.")
        self._snapshots = snapshots
        self._evaluator = EquityOpeningRangeBreakoutEvaluator(configuration)

    def execute(
        self, listing: ListingReference, trading_date: date, snapshot_id: str
    ) -> EquityIntradayResearchDecision:
        """Evaluate snapshot ``snapshot_id`` for ``listing`` on ``trading_date``.

        A storage failure propagates; it is operational, not a research result.
        """
        if not isinstance(snapshot_id, str) or not snapshot_id:
            raise TypeError("snapshot_id must be non-empty text.")
        snapshot = self._snapshots.get_snapshot(snapshot_id)
        return self._evaluator.evaluate(listing, trading_date, snapshot, snapshot_id)
