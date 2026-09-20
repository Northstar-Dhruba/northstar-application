"""Application orchestration for one forward research monitoring run.

A forward run re-measures every frozen decision for one observation series at
every requested horizon, as of one data boundary. It records factual
measurement attempts only: it never interprets BUY, SELL or HOLD, aggregates no
metric, persists no state, reads no clock, and models no order, fill, position
or profit and loss.
"""

from __future__ import annotations

from dataclasses import dataclass

from northstar_core.foundation.value_objects import PointInTime
from northstar_core.strategy import ResearchHorizon

from northstar_application.application_services.ingest_historical_market_data import (
    HistoricalDataContractViolationError,
)
from northstar_application.application_services.measure_forward_research_record import (
    ForwardResearchMeasurement,
    MeasureForwardResearchRecordUseCase,
)
from northstar_application.application_services.record_forward_research_decision import (
    ForwardResearchRecord,
)
from northstar_application.ports import (
    ForwardResearchRecordQuery,
    ForwardResearchRecordRepository,
)


@dataclass(frozen=True, slots=True)
class ForwardResearchRun:
    """Complete factual measurement set for one forward monitoring run.

    ``records`` holds the frozen decisions that were visible at
    ``available_through``; decisions frozen after that boundary are excluded so
    a run remains reproducible for a given as-of point. ``measurements`` holds
    one measurement for every record and horizon pair, ordered by record first
    and horizon second. Pending and unavailable measurements remain present as
    explicit entries so no frozen decision is ever silently dropped.
    """

    query: ForwardResearchRecordQuery
    horizons: tuple[ResearchHorizon, ...]
    available_through: PointInTime
    records: tuple[ForwardResearchRecord, ...]
    measurements: tuple[ForwardResearchMeasurement, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.query, ForwardResearchRecordQuery):
            raise TypeError("ForwardResearchRun query must be a ForwardResearchRecordQuery.")
        if not isinstance(self.available_through, PointInTime):
            raise TypeError("ForwardResearchRun available-through must be a PointInTime.")
        if not isinstance(self.horizons, tuple):
            raise TypeError("ForwardResearchRun horizons must be a tuple.")
        if not all(isinstance(horizon, ResearchHorizon) for horizon in self.horizons):
            raise TypeError("ForwardResearchRun horizons must contain ResearchHorizon values.")
        if not isinstance(self.records, tuple):
            raise TypeError("ForwardResearchRun records must be a tuple.")
        if not all(isinstance(record, ForwardResearchRecord) for record in self.records):
            raise TypeError("ForwardResearchRun records must contain ForwardResearchRecord values.")
        if not isinstance(self.measurements, tuple):
            raise TypeError("ForwardResearchRun measurements must be a tuple.")
        if not all(
            isinstance(measurement, ForwardResearchMeasurement) for measurement in self.measurements
        ):
            raise TypeError(
                "ForwardResearchRun measurements must contain ForwardResearchMeasurement values."
            )
        self._validate_horizons()
        self._validate_records()
        if len(self.measurements) != len(self.records) * len(self.horizons):
            raise ValueError(
                "ForwardResearchRun must contain one measurement for every record and horizon pair."
            )

        expected_pairs = ((record, horizon) for record in self.records for horizon in self.horizons)
        for measurement, (record, horizon) in zip(self.measurements, expected_pairs, strict=True):
            if measurement.record != record:
                raise ValueError(
                    "ForwardResearchRun measurements must follow record order, measuring "
                    "each record's own frozen decision."
                )
            if measurement.horizon != horizon:
                raise ValueError(
                    "ForwardResearchRun measurements must follow horizon order within each record."
                )

    def _validate_horizons(self) -> None:
        """Require a non-empty, duplicate-free horizon set."""
        if not self.horizons:
            raise ValueError("ForwardResearchRun requires at least one horizon.")

        seen: set[int] = set()
        for horizon in self.horizons:
            if horizon.observations in seen:
                raise ValueError("ForwardResearchRun horizons cannot contain duplicates.")
            seen.add(horizon.observations)

    def _validate_records(self) -> None:
        """Require records to match the query, respect the as-of boundary and be ordered."""
        seen_identities: set[tuple[str, str, str, str, str]] = set()
        previous: ForwardResearchRecord | None = None
        for record in self.records:
            if (
                record.listing_reference.symbol != self.query.symbol
                or record.listing_reference.exchange_code != self.query.exchange_code
                or record.timeframe != self.query.timeframe
            ):
                raise ValueError(
                    "ForwardResearchRun records must match the run query listing and timeframe."
                )
            if record.decision_instant.compare(self.available_through) > 0:
                raise ValueError(
                    "ForwardResearchRun records cannot be frozen after the run "
                    "available-through boundary."
                )

            identity = (
                record.listing_reference.symbol.value,
                record.listing_reference.exchange_code.value,
                record.timeframe.value,
                record.decision_instant.value,
                record.strategy_identity.identity,
            )
            if identity in seen_identities:
                raise ValueError(
                    "ForwardResearchRun records cannot contain duplicate natural keys."
                )
            seen_identities.add(identity)

            if previous is not None:
                instant = previous.decision_instant.compare(record.decision_instant)
                if instant > 0 or (
                    instant == 0
                    and previous.strategy_identity.identity > record.strategy_identity.identity
                ):
                    raise ValueError(
                        "ForwardResearchRun records must be ordered by decision instant, "
                        "then strategy identity."
                    )
            previous = record


class RunForwardResearchUseCase:
    """Re-measure every frozen decision for one series as of one data boundary.

    Measurement is delegated to MeasureForwardResearchRecordUseCase so horizon
    selection, look-ahead safety and state semantics stay defined in exactly one
    place. Repository output is validated defensively before it is measured.
    """

    def __init__(
        self,
        repository: ForwardResearchRecordRepository,
        measure_forward_research_record: MeasureForwardResearchRecordUseCase,
    ) -> None:
        if repository is None:
            raise TypeError("RunForwardResearchUseCase repository cannot be None.")
        if not isinstance(repository, ForwardResearchRecordRepository):
            raise TypeError(
                "RunForwardResearchUseCase repository must be a ForwardResearchRecordRepository."
            )
        if measure_forward_research_record is None:
            raise TypeError(
                "RunForwardResearchUseCase measure_forward_research_record cannot be None."
            )
        if not isinstance(measure_forward_research_record, MeasureForwardResearchRecordUseCase):
            raise TypeError(
                "RunForwardResearchUseCase measure_forward_research_record must be a "
                "MeasureForwardResearchRecordUseCase."
            )
        self._repository = repository
        self._measure_forward_research_record = measure_forward_research_record

    def execute(
        self,
        query: ForwardResearchRecordQuery,
        horizons: tuple[ResearchHorizon, ...],
        available_through: PointInTime,
    ) -> ForwardResearchRun:
        """Measure every visible frozen decision against every requested horizon."""
        if not isinstance(query, ForwardResearchRecordQuery):
            raise TypeError("RunForwardResearchUseCase query must be a ForwardResearchRecordQuery.")
        self._validate_horizons(horizons)
        if not isinstance(available_through, PointInTime):
            raise TypeError("RunForwardResearchUseCase available-through must be a PointInTime.")

        stored = self._repository.get_records(query)
        self._validate_records(stored, query)
        records = tuple(
            record for record in stored if record.decision_instant.compare(available_through) <= 0
        )

        measurements = tuple(
            self._measure_forward_research_record.execute(record, horizon, available_through)
            for record in records
            for horizon in horizons
        )
        return ForwardResearchRun(
            query=query,
            horizons=horizons,
            available_through=available_through,
            records=records,
            measurements=measurements,
        )

    @staticmethod
    def _validate_horizons(horizons: object) -> None:
        if horizons is None:
            raise TypeError("RunForwardResearchUseCase horizons cannot be None.")
        if not isinstance(horizons, tuple):
            raise TypeError("RunForwardResearchUseCase horizons must be a tuple.")
        if not horizons:
            raise ValueError("RunForwardResearchUseCase requires at least one horizon.")

        seen: set[int] = set()
        for horizon in horizons:
            if not isinstance(horizon, ResearchHorizon):
                raise TypeError(
                    "RunForwardResearchUseCase horizons must contain ResearchHorizon values."
                )
            if horizon.observations in seen:
                raise ValueError("RunForwardResearchUseCase horizons cannot contain duplicates.")
            seen.add(horizon.observations)

    @staticmethod
    def _validate_records(records: object, query: ForwardResearchRecordQuery) -> None:
        if not isinstance(records, tuple):
            raise HistoricalDataContractViolationError(
                "ForwardResearchRecordRepository must return a tuple of ForwardResearchRecord."
            )

        seen_identities: set[tuple[str, str, str, str, str]] = set()
        previous: ForwardResearchRecord | None = None
        for record in records:
            if not isinstance(record, ForwardResearchRecord):
                raise HistoricalDataContractViolationError(
                    "Forward research repository record must be a ForwardResearchRecord instance."
                )
            if record.listing_reference.symbol != query.symbol:
                raise HistoricalDataContractViolationError(
                    f"Forward research repository record symbol {record.listing_reference.symbol} "
                    f"does not match query symbol {query.symbol}."
                )
            if record.listing_reference.exchange_code != query.exchange_code:
                raise HistoricalDataContractViolationError(
                    "Forward research repository record exchange "
                    f"{record.listing_reference.exchange_code} does not match query exchange "
                    f"{query.exchange_code}."
                )
            if record.timeframe != query.timeframe:
                raise HistoricalDataContractViolationError(
                    f"Forward research repository record timeframe {record.timeframe} "
                    f"does not match query timeframe {query.timeframe}."
                )

            identity = (
                record.listing_reference.symbol.value,
                record.listing_reference.exchange_code.value,
                record.timeframe.value,
                record.decision_instant.value,
                record.strategy_identity.identity,
            )
            if identity in seen_identities:
                raise HistoricalDataContractViolationError(
                    "Forward research repository records cannot contain duplicate natural keys."
                )
            seen_identities.add(identity)

            if previous is not None:
                instant = previous.decision_instant.compare(record.decision_instant)
                if instant > 0 or (
                    instant == 0
                    and previous.strategy_identity.identity > record.strategy_identity.identity
                ):
                    raise HistoricalDataContractViolationError(
                        "Forward research repository records must be ordered by decision "
                        "instant, then strategy identity."
                    )
            previous = record
