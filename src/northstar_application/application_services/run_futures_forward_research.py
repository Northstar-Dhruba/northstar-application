"""Application orchestration for one futures forward research monitoring run.

A forward run re-measures every frozen decision for one contract as of one
explicit evidence cutoff. It creates no decision and freezes nothing: decisions
are frozen individually beforehand, and a run is a derived view over them. It
records factual measurement attempts only -- it never interprets BUY, SELL or
HOLD, aggregates no metric, persists nothing, reads no clock, and models no
order, position, margin or profit.

What a run at one cutoff does and does not guarantee
----------------------------------------------------
``available_through`` bounds both which decisions a run includes and what each
is measured against. Market bars stored after the cutoff therefore cannot
change the run, and neither can decisions frozen after it. A later cutoff is a
different view: the same frozen decision can read PENDING at one cutoff and
MEASURED or UNAVAILABLE at a later one, and that evolution is the point of
forward research. The decision itself never changes.

Two things can change a re-run at the same cutoff, and both are accepted:

- a decision frozen later with a decision instant at or before the cutoff joins
  the run, because the run reads whatever decisions are frozen when it runs;
- a market bar back-filled between a decision and its selected horizon
  observation can shift which observation a horizon selects.

Nothing here snapshots the record set or persists measurements to prevent
either; a forward run is re-derived every time.
"""

from __future__ import annotations

from dataclasses import dataclass

from northstar_core.foundation.value_objects import PointInTime
from northstar_core.strategy import ResearchHorizon

from northstar_application.application_services.futures_forward_research_record import (
    FuturesForwardResearchRecord,
)
from northstar_application.application_services.measure_futures_forward_research_record import (
    FuturesForwardResearchMeasurement,
    MeasureFuturesForwardResearchRecordUseCase,
)
from northstar_application.application_services.record_forward_research_decision import (
    ForwardResearchContractViolationError,
)
from northstar_application.application_services.run_futures_historical_research import (
    _validate_horizons,
)
from northstar_application.ports import (
    FuturesForwardResearchRecordQuery,
    FuturesForwardResearchRecordRepository,
    FuturesHistoricalMarketDataRepository,
)


def _canonical_order(
    earlier: FuturesForwardResearchRecord, later: FuturesForwardResearchRecord
) -> bool:
    """Return whether two records are in canonical order.

    Decision instants compare semantically; records sharing an instant order by
    strategy identity text, the rule equity forward research uses.
    """
    instant = earlier.decision_instant.compare(later.decision_instant)
    return instant < 0 or (
        instant == 0 and earlier.strategy_identity.identity < later.strategy_identity.identity
    )


@dataclass(frozen=True, slots=True)
class FuturesForwardResearchRun:
    """Complete factual measurement set for one futures forward monitoring run.

    ``records`` holds the frozen decisions visible at ``available_through`` for
    the query's contract, in canonical order, possibly from several strategies.
    ``measurements`` holds one measurement for every record and horizon pair,
    record first and horizon second in the caller's horizon order. Pending and
    unavailable measurements remain present, so no frozen decision is ever
    silently dropped from a run.
    """

    query: FuturesForwardResearchRecordQuery
    horizons: tuple[ResearchHorizon, ...]
    available_through: PointInTime
    records: tuple[FuturesForwardResearchRecord, ...]
    measurements: tuple[FuturesForwardResearchMeasurement, ...]

    def __post_init__(self) -> None:
        subject = "FuturesForwardResearchRun"
        if not isinstance(self.query, FuturesForwardResearchRecordQuery):
            raise TypeError(f"{subject} query must be a FuturesForwardResearchRecordQuery.")
        _validate_horizons(self.horizons, subject)
        if not isinstance(self.available_through, PointInTime):
            raise TypeError(f"{subject} available-through must be a PointInTime.")
        if not isinstance(self.records, tuple):
            raise TypeError(f"{subject} records must be a tuple.")
        if not all(isinstance(record, FuturesForwardResearchRecord) for record in self.records):
            raise TypeError(f"{subject} records must contain FuturesForwardResearchRecord values.")
        if not isinstance(self.measurements, tuple):
            raise TypeError(f"{subject} measurements must be a tuple.")
        if not all(
            isinstance(measurement, FuturesForwardResearchMeasurement)
            for measurement in self.measurements
        ):
            raise TypeError(
                f"{subject} measurements must contain FuturesForwardResearchMeasurement values."
            )

        self._validate_records()
        self._validate_measurements()

    def _validate_records(self) -> None:
        seen: set[tuple] = set()
        previous: FuturesForwardResearchRecord | None = None
        for record in self.records:
            if record.contract != self.query.contract or record.timeframe != self.query.timeframe:
                raise ValueError(
                    "FuturesForwardResearchRun records must match the run query contract "
                    "and timeframe."
                )
            if record.decision_instant.compare(self.available_through) > 0:
                raise ValueError(
                    f"FuturesForwardResearchRun record decided at {record.decision_instant} is "
                    f"after the run cutoff {self.available_through}."
                )
            if record.natural_key in seen:
                raise ValueError("FuturesForwardResearchRun records cannot share a natural key.")
            seen.add(record.natural_key)
            if previous is not None and not _canonical_order(previous, record):
                raise ValueError(
                    "FuturesForwardResearchRun records must be ordered by decision instant, "
                    "then strategy identity."
                )
            previous = record

    def _validate_measurements(self) -> None:
        if len(self.measurements) != len(self.records) * len(self.horizons):
            raise ValueError(
                "FuturesForwardResearchRun must contain one measurement for every record and "
                "horizon pair."
            )
        pairs = ((record, horizon) for record in self.records for horizon in self.horizons)
        for measurement, (record, horizon) in zip(self.measurements, pairs, strict=True):
            if measurement.record != record:
                raise ValueError(
                    "FuturesForwardResearchRun measurements must follow record order, measuring "
                    "each record's own frozen decision."
                )
            if measurement.horizon != horizon:
                raise ValueError(
                    "FuturesForwardResearchRun measurements must follow horizon order within "
                    "each record."
                )
            evaluation = measurement.outcome.evaluation_instant
            if evaluation is not None and evaluation.compare(self.available_through) > 0:
                raise ValueError(
                    f"FuturesForwardResearchRun measurement evaluated at {evaluation} is after "
                    f"the run cutoff {self.available_through}."
                )


class RunFuturesForwardResearchUseCase:
    """Re-measure every frozen futures decision for one contract as of one cutoff.

    Measurement is delegated to MeasureFuturesForwardResearchRecordUseCase, so
    horizon selection, look-ahead safety and state semantics stay defined in one
    place. Repository output is checked against its contract before anything is
    measured and is never reordered or repaired.
    """

    def __init__(
        self,
        forward_repository: FuturesForwardResearchRecordRepository,
        market_repository: FuturesHistoricalMarketDataRepository,
    ) -> None:
        if not isinstance(forward_repository, FuturesForwardResearchRecordRepository):
            raise TypeError(
                "RunFuturesForwardResearchUseCase forward_repository "
                "must be a FuturesForwardResearchRecordRepository."
            )
        if not isinstance(market_repository, FuturesHistoricalMarketDataRepository):
            raise TypeError(
                "RunFuturesForwardResearchUseCase market_repository "
                "must be a FuturesHistoricalMarketDataRepository."
            )
        self._forward_repository = forward_repository
        self._measure = MeasureFuturesForwardResearchRecordUseCase(market_repository)

    def execute(
        self,
        query: FuturesForwardResearchRecordQuery,
        horizons: tuple[ResearchHorizon, ...],
        available_through: PointInTime,
    ) -> FuturesForwardResearchRun:
        """Measure every frozen decision visible at the cutoff, at every horizon."""
        subject = "RunFuturesForwardResearchUseCase"
        if not isinstance(query, FuturesForwardResearchRecordQuery):
            raise TypeError(f"{subject} query must be a FuturesForwardResearchRecordQuery.")
        _validate_horizons(horizons, subject)
        if not isinstance(available_through, PointInTime):
            raise TypeError(f"{subject} available-through must be a PointInTime.")

        stored = self._forward_repository.get_records(query)
        self._validate_repository_output(stored, query)

        # Records decided after the cutoff are valid repository output: the
        # query has no cutoff. Excluding them is this run's as-of view.
        records = tuple(
            record for record in stored if record.decision_instant.compare(available_through) <= 0
        )
        measurements = tuple(
            self._measure.execute(record, horizon, available_through)
            for record in records
            for horizon in horizons
        )
        return FuturesForwardResearchRun(
            query=query,
            horizons=horizons,
            available_through=available_through,
            records=records,
            measurements=measurements,
        )

    @staticmethod
    def _validate_repository_output(
        records: object, query: FuturesForwardResearchRecordQuery
    ) -> None:
        if not isinstance(records, tuple):
            raise ForwardResearchContractViolationError(
                "FuturesForwardResearchRecordRepository must return a tuple of "
                "FuturesForwardResearchRecord."
            )
        seen: set[tuple] = set()
        previous: FuturesForwardResearchRecord | None = None
        for index, record in enumerate(records):
            if not isinstance(record, FuturesForwardResearchRecord):
                raise ForwardResearchContractViolationError(
                    f"FuturesForwardResearchRecordRepository record {index} must be a "
                    "FuturesForwardResearchRecord."
                )
            if record.contract != query.contract or record.timeframe != query.timeframe:
                raise ForwardResearchContractViolationError(
                    f"FuturesForwardResearchRecordRepository record {index} is for "
                    f"{record.contract} {record.timeframe}, not the queried "
                    f"{query.contract} {query.timeframe}."
                )
            if record.natural_key in seen:
                raise ForwardResearchContractViolationError(
                    f"FuturesForwardResearchRecordRepository record {index} repeats a natural key."
                )
            seen.add(record.natural_key)
            if previous is not None and not _canonical_order(previous, record):
                raise ForwardResearchContractViolationError(
                    "FuturesForwardResearchRecordRepository records must be ordered by decision "
                    f"instant, then strategy identity; record {index} is out of order."
                )
            previous = record
