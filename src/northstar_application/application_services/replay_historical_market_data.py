"""Application orchestration for deterministic historical replay."""

from __future__ import annotations

from northstar_core.market_data import HistoricalOHLCVBar, HistoricalReplaySnapshot

from northstar_application.application_services.ingest_historical_market_data import (
    HistoricalDataContractViolationError,
)
from northstar_application.ports import (
    HistoricalMarketDataQuery,
    HistoricalMarketDataRepository,
)


class ReplayHistoricalMarketDataUseCase:
    """Build cumulative, look-ahead-safe replay snapshots from stored history."""

    def __init__(self, repository: HistoricalMarketDataRepository) -> None:
        if repository is None:
            raise TypeError("ReplayHistoricalMarketDataUseCase repository cannot be None.")
        if not isinstance(repository, HistoricalMarketDataRepository):
            raise TypeError(
                "ReplayHistoricalMarketDataUseCase repository must be a "
                "HistoricalMarketDataRepository."
            )
        self._repository = repository

    def execute(self, query: HistoricalMarketDataQuery) -> tuple[HistoricalReplaySnapshot, ...]:
        """Return one cumulative snapshot for each unique completion instant."""
        if query is None:
            raise TypeError("ReplayHistoricalMarketDataUseCase query cannot be None.")
        if not isinstance(query, HistoricalMarketDataQuery):
            raise TypeError(
                "ReplayHistoricalMarketDataUseCase query must be a HistoricalMarketDataQuery."
            )

        observations = self._repository.get_history(query)
        self._validate_observations(observations, query)
        if not observations:
            return ()

        snapshots: list[HistoricalReplaySnapshot] = []
        visible: list[HistoricalOHLCVBar] = []
        index = 0
        while index < len(observations):
            replay_instant = observations[index].point_in_time
            group_end = index + 1
            while (
                group_end < len(observations)
                and observations[group_end].point_in_time.compare(replay_instant) == 0
            ):
                group_end += 1

            visible.extend(observations[index:group_end])
            snapshots.append(HistoricalReplaySnapshot(replay_instant, tuple(visible)))
            index = group_end

        return tuple(snapshots)

    @staticmethod
    def _validate_observations(observations: object, query: HistoricalMarketDataQuery) -> None:
        if not isinstance(observations, tuple):
            raise HistoricalDataContractViolationError(
                "HistoricalMarketDataRepository must return a tuple of HistoricalOHLCVBar."
            )

        seen_identities: set[tuple[str, str, str, str]] = set()
        for index, bar in enumerate(observations):
            if not isinstance(bar, HistoricalOHLCVBar):
                raise HistoricalDataContractViolationError(
                    "Historical repository observation must be a HistoricalOHLCVBar instance."
                )
            if bar.symbol != query.symbol:
                raise HistoricalDataContractViolationError(
                    f"Historical repository observation symbol {bar.symbol} "
                    f"does not match query symbol {query.symbol}."
                )
            if bar.exchange_code != query.exchange_code:
                raise HistoricalDataContractViolationError(
                    f"Historical repository observation exchange {bar.exchange_code} "
                    f"does not match query exchange {query.exchange_code}."
                )
            if bar.timeframe != query.timeframe:
                raise HistoricalDataContractViolationError(
                    f"Historical repository observation timeframe {bar.timeframe} "
                    f"does not match query timeframe {query.timeframe}."
                )
            if (
                query.start.compare(bar.point_in_time) > 0
                or query.end.compare(bar.point_in_time) < 0
            ):
                raise HistoricalDataContractViolationError(
                    f"Historical repository observation point in time {bar.point_in_time} "
                    f"is outside query bounds [{query.start}, {query.end}]."
                )

            identity = (
                bar.symbol.value,
                bar.exchange_code.value,
                bar.timeframe.value,
                bar.point_in_time.value,
            )
            if identity in seen_identities:
                raise HistoricalDataContractViolationError(
                    "Historical repository observations cannot contain duplicate logical bars."
                )
            seen_identities.add(identity)

            if index and observations[index - 1].point_in_time.compare(bar.point_in_time) > 0:
                raise HistoricalDataContractViolationError(
                    "Historical repository observations must be ordered oldest to newest."
                )
