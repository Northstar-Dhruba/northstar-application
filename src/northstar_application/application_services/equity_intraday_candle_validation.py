"""Validation and finality gating of 5-minute equity candles (EI-2A).

Given an EI-1C-3 research-window evaluation for one listing and date, an
explicit observation instant, and the candle observations a source has
supplied, this module answers -- separately, never as one boolean:

1. Which intervals are expected (the window's intervals).
2. Which of them had ended at the observation instant.
3. Which have a candle received by then.
4. Whether each candle is structurally valid: guaranteed by the Core
   ``HistoricalOHLCVBar`` it is carried in (prices non-negative and finite,
   one currency, high and low bracketing open and close, volume a
   non-negative finite Quantity). That validation is reused, not repeated.
5. Whether each candle matches its expected interval: listing, timeframe,
   currency, raw (unadjusted) values and an interval end that is one of the
   window's expected interval ends.
6. Whether it carries acceptable source-finality evidence.
7. Whether the interval's candle is therefore acceptable for later accepted
   persistence (EI-2B). Nothing here persists anything.
8. Which intervals are missing, unfinalized, conflicting or unresolved.

Finality is evidence, never inferred
------------------------------------
A candle is final only when finality evidence from its own source, for its own
source event and revision, was confirmed at or before the observation instant
and after the interval ended, and that source is approved by the supplied
``EquityCandleFinalityPolicy``. Finality is never inferred from elapsed time,
from the absence of a correction, from the interval being the last one, or
from storage. No real NSE or provider finality source is approved: the
production policy approves none and refuses synthetic evidence, so every real
request fails closed. Synthetic evidence is accepted only under a policy that
says it is synthetic.

Conflicts are never resolved here
---------------------------------
Observations for the same listing, timeframe and interval end are compared.
Only exact repeats of one identity -- the same source, source event and
revision -- are repeated evidence, and only they can be acceptable. Different
values under that one identity are CONFLICTING_VALUES. More than one source,
source-event or revision identity, or any correction reference, is
UNRESOLVED_REVISIONS even when the values match, because no revision ordering
or selection is approved yet. Matching prices never merge distinct records.
Nothing is sorted, merged, overwritten or preferred by recency, volume or
position, and every observation stays in the result with its provenance. The
append-only revision policy, materiality and restated evaluation belong to
EI-2B.

Completeness
------------
Coverage is judged only over intervals that had ended at the observation
instant, so an early snapshot is not incomplete merely because later
intervals have not ended. The whole window is complete only when every
expected interval has ended and is acceptable.

Nothing reads a clock, provider, file or network. An ended interval is a time
fact, and an acceptable candle is not a trade, fill or profit.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import date
from enum import StrEnum

from northstar_core.domain.value_objects import ListingReference
from northstar_core.foundation.value_objects import Currency, PointInTime
from northstar_core.market_data import HistoricalOHLCVBar

from northstar_application.application_services.equity_research_window import (
    NSE_EQUITY_INTRADAY_V1_RESEARCH_WINDOW,
    EquityResearchWindowEvaluation,
    EquityResearchWindowPolicy,
)
from northstar_application.application_services.equity_session_interval_grid import (
    EquitySessionInterval,
)


class InvalidEquityCandleEvidenceError(ValueError):
    """Raised when an observation, finality evidence or policy value is malformed."""


def _require_text(value: object, subject: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise InvalidEquityCandleEvidenceError(f"{subject} must be a non-empty string.")


def _require_instant(value: object, subject: str) -> None:
    if not isinstance(value, PointInTime):
        raise InvalidEquityCandleEvidenceError(f"{subject} must be a PointInTime.")


@dataclass(frozen=True, slots=True)
class EquityCandleFinalityEvidence:
    """A source's statement that one as-issued candle revision is final.

    It names the source, source event and revision it confirms, when the
    source confirmed it, and a reference to the evidence. ``is_synthetic``
    marks invented test evidence.
    """

    source_id: str
    source_event_id: str
    source_revision: str
    confirmed_at: PointInTime
    evidence_reference: str
    is_synthetic: bool

    def __post_init__(self) -> None:
        _require_text(self.source_id, "Finality source_id")
        _require_text(self.source_event_id, "Finality source_event_id")
        _require_text(self.source_revision, "Finality source_revision")
        _require_instant(self.confirmed_at, "Finality confirmed_at")
        _require_text(self.evidence_reference, "Finality evidence_reference")
        if not isinstance(self.is_synthetic, bool):
            raise InvalidEquityCandleEvidenceError("Finality is_synthetic must be a bool.")


@dataclass(frozen=True, slots=True)
class EquityCandleObservation:
    """One as-issued candle as received from one source.

    ``source_event_id`` and ``source_revision`` identify the as-issued record
    immutably, so EI-2B can store it append-only. ``corrects_revision`` names
    the revision this one claims to correct, if any; it is recorded as
    correction evidence, never applied. ``received_at`` is when Northstar
    received the record; a record received after the observation instant is
    invisible at that instant.
    """

    bar: HistoricalOHLCVBar
    source_id: str
    source_event_id: str
    source_revision: str
    received_at: PointInTime
    finality: EquityCandleFinalityEvidence | None
    corrects_revision: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.bar, HistoricalOHLCVBar):
            raise InvalidEquityCandleEvidenceError("Observation bar must be a HistoricalOHLCVBar.")
        _require_text(self.source_id, "Observation source_id")
        _require_text(self.source_event_id, "Observation source_event_id")
        _require_text(self.source_revision, "Observation source_revision")
        _require_instant(self.received_at, "Observation received_at")
        if self.finality is not None and not isinstance(
            self.finality, EquityCandleFinalityEvidence
        ):
            raise InvalidEquityCandleEvidenceError(
                "Observation finality must be EquityCandleFinalityEvidence or None."
            )
        if self.corrects_revision is not None:
            _require_text(self.corrects_revision, "Observation corrects_revision")
            if self.corrects_revision == self.source_revision:
                raise InvalidEquityCandleEvidenceError(
                    "Observation cannot correct its own revision."
                )

    @property
    def values(self) -> tuple[object, ...]:
        """Return the candle's as-issued market values, for comparison."""
        bar = self.bar
        return (bar.open, bar.high, bar.low, bar.close, bar.volume, bar.adjusted_close)


@dataclass(frozen=True, slots=True)
class EquityCandleFinalityPolicy:
    """Which sources' finality evidence is approved, and whether synthetic evidence is.

    A policy that permits synthetic evidence is a test policy and must be
    versioned ``synthetic-...``.
    """

    version: str
    approved_source_ids: frozenset[str]
    permits_synthetic: bool

    def __post_init__(self) -> None:
        _require_text(self.version, "Finality policy version")
        if not isinstance(self.approved_source_ids, frozenset):
            raise InvalidEquityCandleEvidenceError(
                "Finality policy approved_source_ids must be a frozenset."
            )
        for source_id in self.approved_source_ids:
            _require_text(source_id, "Finality policy approved source")
        if not isinstance(self.permits_synthetic, bool):
            raise InvalidEquityCandleEvidenceError(
                "Finality policy permits_synthetic must be a bool."
            )
        if self.permits_synthetic and not self.version.startswith("synthetic-"):
            raise InvalidEquityCandleEvidenceError(
                "A finality policy that permits synthetic evidence must be versioned synthetic-."
            )


NO_APPROVED_EQUITY_CANDLE_FINALITY = EquityCandleFinalityPolicy(
    version="no-approved-equity-candle-finality/1",
    approved_source_ids=frozenset(),
    permits_synthetic=False,
)


class EquityCandleFinding(StrEnum):
    """What was found about one observation, in the order the checks run."""

    WINDOW_NOT_ELIGIBLE = "WINDOW_NOT_ELIGIBLE"
    WRONG_LISTING = "WRONG_LISTING"
    WRONG_TIMEFRAME = "WRONG_TIMEFRAME"
    WRONG_CURRENCY = "WRONG_CURRENCY"
    ADJUSTED_VALUE_NOT_ACCEPTED = "ADJUSTED_VALUE_NOT_ACCEPTED"
    OUTSIDE_RESEARCH_WINDOW = "OUTSIDE_RESEARCH_WINDOW"
    MISALIGNED_INTERVAL_END = "MISALIGNED_INTERVAL_END"
    INTERVAL_NOT_ENDED = "INTERVAL_NOT_ENDED"
    NOT_RECEIVED_AS_OF = "NOT_RECEIVED_AS_OF"
    RECEIVED_BEFORE_INTERVAL_END = "RECEIVED_BEFORE_INTERVAL_END"
    FINALITY_MISSING = "FINALITY_MISSING"
    FINALITY_SOURCE_MISMATCH = "FINALITY_SOURCE_MISMATCH"
    FINALITY_EVENT_MISMATCH = "FINALITY_EVENT_MISMATCH"
    FINALITY_REVISION_MISMATCH = "FINALITY_REVISION_MISMATCH"
    SYNTHETIC_FINALITY_NOT_PERMITTED = "SYNTHETIC_FINALITY_NOT_PERMITTED"
    FINALITY_SOURCE_NOT_APPROVED = "FINALITY_SOURCE_NOT_APPROVED"
    FINALITY_BEFORE_INTERVAL_END = "FINALITY_BEFORE_INTERVAL_END"
    FINALITY_NOT_CONFIRMED_AS_OF = "FINALITY_NOT_CONFIRMED_AS_OF"
    ACCEPTABLE = "ACCEPTABLE"


_UNMATCHED = frozenset(
    {
        EquityCandleFinding.WINDOW_NOT_ELIGIBLE,
        EquityCandleFinding.WRONG_LISTING,
        EquityCandleFinding.WRONG_TIMEFRAME,
        EquityCandleFinding.WRONG_CURRENCY,
        EquityCandleFinding.ADJUSTED_VALUE_NOT_ACCEPTED,
        EquityCandleFinding.OUTSIDE_RESEARCH_WINDOW,
        EquityCandleFinding.MISALIGNED_INTERVAL_END,
    }
)


class EquityCandleIntervalOutcome(StrEnum):
    """What the evidence establishes for one expected interval."""

    NOT_ENDED = "NOT_ENDED"
    MISSING = "MISSING"
    ACCEPTABLE = "ACCEPTABLE"
    UNFINALIZED = "UNFINALIZED"
    CONFLICTING_VALUES = "CONFLICTING_VALUES"
    UNRESOLVED_REVISIONS = "UNRESOLVED_REVISIONS"


class EquityCandleCoverage(StrEnum):
    """Snapshot completeness over the intervals that had ended at the observation instant."""

    WINDOW_NOT_ELIGIBLE = "WINDOW_NOT_ELIGIBLE"
    NO_INTERVALS_ENDED = "NO_INTERVALS_ENDED"
    INCOMPLETE_AS_OF = "INCOMPLETE_AS_OF"
    COMPLETE_AS_OF = "COMPLETE_AS_OF"
    WINDOW_COMPLETE = "WINDOW_COMPLETE"


@dataclass(frozen=True, slots=True)
class EquityCandleObservationAssessment:
    """One observation and what was found about it."""

    observation: EquityCandleObservation
    finding: EquityCandleFinding
    detail: str


@dataclass(frozen=True, slots=True)
class EquityCandleIntervalAssessment:
    """One expected interval, its outcome and every visible observation for it.

    ``acceptable`` lists every acceptable observation; when there are several,
    they are exact repeats of one source, event and revision identity, and
    none is preferred over another.
    """

    interval: EquitySessionInterval
    outcome: EquityCandleIntervalOutcome
    observations: tuple[EquityCandleObservationAssessment, ...]
    acceptable: tuple[EquityCandleObservation, ...]


@dataclass(frozen=True, slots=True)
class EquityCandleValidationResult:
    """The validation of one listing and date at one observation instant.

    ``intervals`` covers every expected interval in order. ``unmatched`` holds
    observations that match no expected interval of this listing. Nothing has
    been persisted, and an acceptable interval is only eligible for later
    accepted persistence.
    """

    listing: ListingReference
    trading_date: date
    as_of: PointInTime
    window_policy_version: str
    finality_policy_version: str
    coverage: EquityCandleCoverage
    intervals: tuple[EquityCandleIntervalAssessment, ...]
    unmatched: tuple[EquityCandleObservationAssessment, ...]

    def _with(self, *outcomes: EquityCandleIntervalOutcome) -> tuple[EquitySessionInterval, ...]:
        return tuple(a.interval for a in self.intervals if a.outcome in outcomes)

    @property
    def ended_intervals(self) -> tuple[EquitySessionInterval, ...]:
        """Return the expected intervals that had ended at ``as_of``."""
        return tuple(
            a.interval
            for a in self.intervals
            if a.outcome is not EquityCandleIntervalOutcome.NOT_ENDED
        )

    @property
    def acceptable_intervals(self) -> tuple[EquitySessionInterval, ...]:
        """Return the intervals whose candle is acceptable for later accepted persistence."""
        return self._with(EquityCandleIntervalOutcome.ACCEPTABLE)

    @property
    def missing_intervals(self) -> tuple[EquitySessionInterval, ...]:
        """Return the ended intervals with no candle received by ``as_of``."""
        return self._with(EquityCandleIntervalOutcome.MISSING)

    @property
    def unfinalized_intervals(self) -> tuple[EquitySessionInterval, ...]:
        """Return the ended intervals with a received candle but no acceptable finality."""
        return self._with(EquityCandleIntervalOutcome.UNFINALIZED)

    @property
    def conflicted_intervals(self) -> tuple[EquitySessionInterval, ...]:
        """Return the ended intervals whose evidence conflicts and is unresolved."""
        return self._with(
            EquityCandleIntervalOutcome.CONFLICTING_VALUES,
            EquityCandleIntervalOutcome.UNRESOLVED_REVISIONS,
        )


class ValidateEquityIntradayCandlesUseCase:
    """Validates candle observations against one research-window evaluation and finality policy."""

    def __init__(
        self,
        currency: Currency,
        finality_policy: EquityCandleFinalityPolicy = NO_APPROVED_EQUITY_CANDLE_FINALITY,
        window_policy: EquityResearchWindowPolicy = NSE_EQUITY_INTRADAY_V1_RESEARCH_WINDOW,
    ) -> None:
        if not isinstance(currency, Currency):
            raise TypeError("currency must be a Currency.")
        if not isinstance(finality_policy, EquityCandleFinalityPolicy):
            raise TypeError("finality_policy must be an EquityCandleFinalityPolicy.")
        if not isinstance(window_policy, EquityResearchWindowPolicy):
            raise TypeError("window_policy must be an EquityResearchWindowPolicy.")
        self._currency = currency
        self._finality = finality_policy
        self._window = window_policy

    def execute(
        self,
        window: EquityResearchWindowEvaluation,
        observations: Iterable[EquityCandleObservation],
        as_of: PointInTime,
    ) -> EquityCandleValidationResult:
        """Validate ``observations`` for ``window`` as known at ``as_of``."""
        if not isinstance(window, EquityResearchWindowEvaluation):
            raise TypeError("window must be an EquityResearchWindowEvaluation.")
        if not isinstance(as_of, PointInTime):
            raise TypeError("as_of must be a PointInTime.")
        if window.as_of != as_of:
            raise ValueError(
                "as_of must equal the research-window evaluation's as_of; one observation "
                "instant governs the whole assessment."
            )
        if window.policy_version != self._window.version:
            raise ValueError(
                f"The research-window evaluation used policy {window.policy_version}, not "
                f"{self._window.version}."
            )
        supplied = tuple(observations)
        for observation in supplied:
            if not isinstance(observation, EquityCandleObservation):
                raise TypeError("observations must be EquityCandleObservation values.")

        def result(
            coverage: EquityCandleCoverage,
            intervals: tuple[EquityCandleIntervalAssessment, ...],
            unmatched: tuple[EquityCandleObservationAssessment, ...],
        ) -> EquityCandleValidationResult:
            return EquityCandleValidationResult(
                listing=window.listing,
                trading_date=window.trading_date,
                as_of=as_of,
                window_policy_version=self._window.version,
                finality_policy_version=self._finality.version,
                coverage=coverage,
                intervals=intervals,
                unmatched=unmatched,
            )

        if not window.is_eligible:
            return result(
                EquityCandleCoverage.WINDOW_NOT_ELIGIBLE,
                (),
                tuple(
                    EquityCandleObservationAssessment(
                        o,
                        EquityCandleFinding.WINDOW_NOT_ELIGIBLE,
                        f"The research window is not eligible: {window.rejection}.",
                    )
                    for o in supplied
                ),
            )

        by_end = {interval.closes_at: interval for interval in window.intervals}
        unmatched: list[EquityCandleObservationAssessment] = []
        per_interval: dict[PointInTime, list[EquityCandleObservationAssessment]] = {
            end: [] for end in by_end
        }
        for observation in supplied:
            assessment = self._assess(observation, window, by_end, as_of)
            if assessment.finding in _UNMATCHED:
                unmatched.append(assessment)
            else:
                per_interval[observation.bar.point_in_time].append(assessment)

        assessments = tuple(
            self._interval(interval, tuple(per_interval[interval.closes_at]), as_of)
            for interval in window.intervals
        )
        return result(self._coverage(assessments), assessments, tuple(unmatched))

    def _assess(
        self,
        observation: EquityCandleObservation,
        window: EquityResearchWindowEvaluation,
        by_end: dict[PointInTime, EquitySessionInterval],
        as_of: PointInTime,
    ) -> EquityCandleObservationAssessment:
        bar = observation.bar
        finding = EquityCandleFinding

        def found(kind: EquityCandleFinding, detail: str) -> EquityCandleObservationAssessment:
            return EquityCandleObservationAssessment(observation, kind, detail)

        if bar.symbol != window.listing.symbol or bar.exchange_code != window.listing.exchange_code:
            return found(
                finding.WRONG_LISTING,
                f"Candle is for {bar.symbol}@{bar.exchange_code}, not {window.listing}.",
            )
        if bar.timeframe != self._window.timeframe:
            return found(
                finding.WRONG_TIMEFRAME,
                f"Candle timeframe {bar.timeframe} is not {self._window.timeframe}.",
            )
        if bar.open.currency != self._currency:
            return found(
                finding.WRONG_CURRENCY,
                f"Candle currency {bar.open.currency} is not {self._currency}.",
            )
        if bar.adjusted_close is not None:
            return found(
                finding.ADJUSTED_VALUE_NOT_ACCEPTED,
                "An as-issued intraday candle carries raw values only; an adjusted close is "
                "not accepted.",
            )
        end = bar.point_in_time
        interval = by_end.get(end)
        if interval is None:
            opens, closes = window.window_opens_at, window.window_closes_at
            inside = opens.compare(end) < 0 and end.compare(closes) <= 0  # type: ignore[union-attr]
            if inside:
                return found(
                    finding.MISALIGNED_INTERVAL_END,
                    f"Candle end {end} is inside the window but is not an expected interval end.",
                )
            return found(
                finding.OUTSIDE_RESEARCH_WINDOW,
                f"Candle end {end} is outside the research window {opens} to {closes} of "
                f"{window.trading_date.isoformat()} (another date, pre-open, auction or "
                "excluded time).",
            )
        if end.compare(as_of) > 0:
            return found(
                finding.INTERVAL_NOT_ENDED,
                f"Interval ending {end} had not ended at {as_of}.",
            )
        if observation.received_at.compare(as_of) > 0:
            return found(
                finding.NOT_RECEIVED_AS_OF,
                f"Candle was received at {observation.received_at}, after {as_of}.",
            )
        if observation.received_at.compare(end) < 0:
            return found(
                finding.RECEIVED_BEFORE_INTERVAL_END,
                f"Candle was received at {observation.received_at}, before its interval ended "
                f"at {end}; it cannot be a completed candle.",
            )
        return self._finality_finding(observation, end, as_of, found)

    def _finality_finding(
        self,
        observation: EquityCandleObservation,
        end: PointInTime,
        as_of: PointInTime,
        found: Callable[[EquityCandleFinding, str], EquityCandleObservationAssessment],
    ) -> EquityCandleObservationAssessment:
        finding = EquityCandleFinding
        evidence = observation.finality
        if evidence is None:
            return found(finding.FINALITY_MISSING, "No source-finality evidence was supplied.")
        if evidence.source_id != observation.source_id:
            return found(
                finding.FINALITY_SOURCE_MISMATCH,
                f"Finality is from {evidence.source_id}, not the candle's source "
                f"{observation.source_id}.",
            )
        if evidence.source_event_id != observation.source_event_id:
            return found(
                finding.FINALITY_EVENT_MISMATCH,
                "Finality names a different source event.",
            )
        if evidence.source_revision != observation.source_revision:
            return found(
                finding.FINALITY_REVISION_MISMATCH,
                f"Finality confirms revision {evidence.source_revision}, not "
                f"{observation.source_revision}.",
            )
        if evidence.is_synthetic and not self._finality.permits_synthetic:
            return found(
                finding.SYNTHETIC_FINALITY_NOT_PERMITTED,
                f"Synthetic finality evidence is not permitted by {self._finality.version}.",
            )
        if evidence.source_id not in self._finality.approved_source_ids:
            return found(
                finding.FINALITY_SOURCE_NOT_APPROVED,
                f"Finality source {evidence.source_id} is not approved by "
                f"{self._finality.version}.",
            )
        if evidence.confirmed_at.compare(end) < 0:
            return found(
                finding.FINALITY_BEFORE_INTERVAL_END,
                f"Finality was confirmed at {evidence.confirmed_at}, before the interval "
                f"ended at {end}.",
            )
        if evidence.confirmed_at.compare(as_of) > 0:
            return found(
                finding.FINALITY_NOT_CONFIRMED_AS_OF,
                f"Finality was confirmed at {evidence.confirmed_at}, after {as_of}.",
            )
        return found(
            finding.ACCEPTABLE,
            f"Final under {self._finality.version} by {evidence.evidence_reference}.",
        )

    @staticmethod
    def _interval(
        interval: EquitySessionInterval,
        assessments: tuple[EquityCandleObservationAssessment, ...],
        as_of: PointInTime,
    ) -> EquityCandleIntervalAssessment:
        outcome = EquityCandleIntervalOutcome

        def assessed(kind: EquityCandleIntervalOutcome) -> EquityCandleIntervalAssessment:
            acceptable = tuple(
                a.observation for a in assessments if a.finding is EquityCandleFinding.ACCEPTABLE
            )
            if kind is not outcome.ACCEPTABLE:
                acceptable = ()
            return EquityCandleIntervalAssessment(interval, kind, assessments, acceptable)

        if interval.closes_at.compare(as_of) > 0:
            return assessed(outcome.NOT_ENDED)
        visible = [
            a for a in assessments if a.finding is not EquityCandleFinding.NOT_RECEIVED_AS_OF
        ]
        if not visible:
            return assessed(outcome.MISSING)
        # No revision ordering or selection is approved (EI-2B): a correction, or
        # more than one source/event/revision identity, is unresolved even when
        # the values match. Only exact repeats of one identity can be accepted.
        if any(a.observation.corrects_revision is not None for a in visible):
            return assessed(outcome.UNRESOLVED_REVISIONS)
        identities = {
            (
                a.observation.source_id,
                a.observation.source_event_id,
                a.observation.source_revision,
            )
            for a in visible
        }
        if len(identities) > 1:
            return assessed(outcome.UNRESOLVED_REVISIONS)
        if len({a.observation.values for a in visible}) > 1:
            return assessed(outcome.CONFLICTING_VALUES)
        if any(a.finding is EquityCandleFinding.ACCEPTABLE for a in visible):
            return assessed(outcome.ACCEPTABLE)
        return assessed(outcome.UNFINALIZED)

    @staticmethod
    def _coverage(
        assessments: tuple[EquityCandleIntervalAssessment, ...],
    ) -> EquityCandleCoverage:
        ended = [a for a in assessments if a.outcome is not EquityCandleIntervalOutcome.NOT_ENDED]
        if not ended:
            return EquityCandleCoverage.NO_INTERVALS_ENDED
        if any(a.outcome is not EquityCandleIntervalOutcome.ACCEPTABLE for a in ended):
            return EquityCandleCoverage.INCOMPLETE_AS_OF
        if len(ended) == len(assessments):
            return EquityCandleCoverage.WINDOW_COMPLETE
        return EquityCandleCoverage.COMPLETE_AS_OF
