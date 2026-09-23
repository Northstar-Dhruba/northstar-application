"""Tests for the reproducible futures historical research run.

The repository here behaves as a real one must -- it answers each query with the
stored bars of that contract and timeframe inside the inclusive window -- and
records every query. That makes the cutoff observable: every read the run makes
is visible, so "every measurement used the run's cutoff" is checked rather than
assumed.
"""

from __future__ import annotations

import ast
import dataclasses
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

import pytest
from northstar_core.derivatives import ExpirationDate, QuoteValue
from northstar_core.foundation.value_objects import (
    ExchangeCode,
    PointInTime,
    Quantity,
    Symbol,
    Timeframe,
)
from northstar_core.futures import FuturesContract, FuturesOHLCVBar, FuturesProductReference
from northstar_core.strategy import (
    FuturesAssetAnalysisGenerator,
    FuturesRecommendationOutcome,
    FuturesRecommendationOutcomeUnavailableReason,
    ResearchHorizon,
    Strategy,
    StrategyIdentity,
)

from northstar_application.application_services import (
    FuturesHistoricalDataContractViolationError,
    FuturesHistoricalResearchRun,
    RunFuturesHistoricalResearchUseCase,
    UnsupportedFuturesReplayTimeframeError,
)
from northstar_application.ports import (
    FuturesHistoricalMarketDataQuery,
    FuturesHistoricalMarketDataRepository,
)

_CME = ExchangeCode("CME")
_ES = FuturesProductReference(Symbol("ES"), _CME)
_ES_DEC = FuturesContract(_ES, ExpirationDate("2026-12-18"))
_ES_MAR = FuturesContract(_ES, ExpirationDate("2027-03-19"))
_MES_DEC = FuturesContract(
    FuturesProductReference(Symbol("MES"), _CME), ExpirationDate("2026-12-18")
)
_DAILY = Timeframe("1d")
_STRATEGY = Strategy(StrategyIdentity("futures-directional"))
_GENERATOR = FuturesAssetAnalysisGenerator()

_INSUFFICIENT = FuturesRecommendationOutcomeUnavailableReason.INSUFFICIENT_FUTURE_OBSERVATIONS
_H1, _H2, _H3 = ResearchHorizon(1), ResearchHorizon(2), ResearchHorizon(3)


def _instant(session: int) -> str:
    """Completion of the ``session``-th stored session, one calendar day apart."""
    return f"{date(2026, 6, 1) + timedelta(days=session - 1)}T21:00:00Z"


def _bar(
    instant: str,
    close: str,
    *,
    contract: FuturesContract = _ES_DEC,
    timeframe: Timeframe = _DAILY,
    volume: str = "1000",
) -> FuturesOHLCVBar:
    quote = Decimal(close)
    return FuturesOHLCVBar(
        contract=contract,
        point_in_time=PointInTime(instant),
        timeframe=timeframe,
        open=QuoteValue(quote),
        high=QuoteValue(quote + 1),
        low=QuoteValue(quote - 1),
        close=QuoteValue(quote),
        volume=Quantity(Decimal(volume)),
    )


def _history(count: int, *, contract: FuturesContract = _ES_DEC) -> tuple[FuturesOHLCVBar, ...]:
    """``count`` sessions whose closes wander, so decisions and outcomes differ."""
    return tuple(
        _bar(_instant(session), str(100 + (session * 7) % 13), contract=contract)
        for session in range(1, count + 1)
    )


class ConformingRepository(FuturesHistoricalMarketDataRepository):
    def __init__(self, bars: tuple[FuturesOHLCVBar, ...] = ()) -> None:
        self.bars = bars
        self.queries: list[FuturesHistoricalMarketDataQuery] = []

    def get_bars(self, query):
        self.queries.append(query)
        return tuple(
            bar
            for bar in self.bars
            if bar.contract == query.contract
            and bar.timeframe == query.timeframe
            and query.covers(bar.point_in_time)
        )


class StubRepository(FuturesHistoricalMarketDataRepository):
    """Returns what it is given, whatever the query."""

    def __init__(self, bars: object) -> None:
        self.bars = bars

    def get_bars(self, query):
        return self.bars


def _run(
    repository: FuturesHistoricalMarketDataRepository,
    through: str,
    horizons: tuple[ResearchHorizon, ...] = (_H1,),
    *,
    contract: FuturesContract = _ES_DEC,
    strategy: Strategy = _STRATEGY,
) -> FuturesHistoricalResearchRun:
    return RunFuturesHistoricalResearchUseCase(repository, _GENERATOR).execute(
        contract, _DAILY, strategy, horizons, PointInTime(through)
    )


def _decisions(run: FuturesHistoricalResearchRun) -> list[PointInTime]:
    return [result.recommendation.point_in_time for result in run.analysis_results]


# ---------------------------------------------------------------------------
# No data, warm-up and the first decisions
# ---------------------------------------------------------------------------


def test_no_stored_history_is_a_valid_empty_run_that_still_names_its_subject() -> None:
    run = _run(ConformingRepository(), _instant(30), (_H1, _H2))

    assert run.analysis_results == ()
    assert run.outcomes == ()
    assert run.contract == _ES_DEC
    assert run.timeframe == _DAILY
    assert run.strategy_identity == _STRATEGY.strategy_identity
    assert run.available_through == PointInTime(_instant(30))
    assert run.horizons == (_H1, _H2)


def test_nineteen_stored_bars_are_all_warm_up() -> None:
    repository = ConformingRepository(_history(19))

    run = _run(repository, _instant(19))

    assert run.analysis_results == ()
    assert run.outcomes == ()
    assert len(repository.queries) == 1  # the replay read happened; nothing was measured


def test_the_twentieth_bar_is_the_first_decision() -> None:
    run = _run(ConformingRepository(_history(20)), _instant(20))

    assert _decisions(run) == [PointInTime(_instant(20))]
    assert len(run.outcomes) == 1


def test_each_later_bar_adds_one_cumulative_decision() -> None:
    run = _run(ConformingRepository(_history(25)), _instant(25))

    assert _decisions(run) == [PointInTime(_instant(session)) for session in range(20, 26)]
    for result, session in zip(run.analysis_results, range(20, 26), strict=True):
        context = result.market_observation_context
        assert context.observed_at == PointInTime(_instant(session))
        assert context.recent_closes == tuple(bar.close for bar in _history(session)[-20:])


# ---------------------------------------------------------------------------
# Horizons and canonical ordering
# ---------------------------------------------------------------------------


def test_one_horizon_gives_one_outcome_per_decision() -> None:
    run = _run(ConformingRepository(_history(24)), _instant(24), (_H2,))

    assert len(run.outcomes) == len(run.analysis_results) == 5
    assert all(outcome.horizon == _H2 for outcome in run.outcomes)


def test_outcomes_are_ordered_by_decision_then_by_the_callers_horizon_order() -> None:
    horizons = (_H3, _H1, _H2)  # deliberately not sorted

    run = _run(ConformingRepository(_history(26)), _instant(26), horizons)

    expected = [
        (result.recommendation, horizon) for result in run.analysis_results for horizon in horizons
    ]
    assert [(outcome.recommendation, outcome.horizon) for outcome in run.outcomes] == expected
    assert run.horizons == horizons


def test_each_outcome_is_the_measurement_of_its_decision_at_its_horizon() -> None:
    history = _history(24)
    run = _run(ConformingRepository(history), _instant(24), (_H1, _H2))

    first_decision = run.analysis_results[0]
    one, two = run.outcomes[0], run.outcomes[1]
    assert one.decision_quote == first_decision.market_observation_context.latest_quote
    assert one.evaluation_instant == history[20].point_in_time  # session 21
    assert two.evaluation_instant == history[21].point_in_time  # session 22
    assert one.evaluation_quote == history[20].close


@pytest.mark.parametrize(
    ("horizons", "error", "message"),
    [
        pytest.param((_H1, _H1), ValueError, "duplicates", id="duplicate"),
        pytest.param((_H1, _H2, ResearchHorizon(1)), ValueError, "duplicates", id="rebuilt-dup"),
        pytest.param((), ValueError, "at least one horizon", id="empty"),
        pytest.param([_H1], TypeError, "must be a tuple", id="list"),
        pytest.param((_H1, 2), TypeError, "ResearchHorizon values", id="non-horizon"),
    ],
)
def test_invalid_horizons_are_refused_before_any_read(
    horizons: object, error: type[Exception], message: str
) -> None:
    repository = ConformingRepository(_history(25))

    with pytest.raises(error, match=message):
        _run(repository, _instant(25), horizons)  # type: ignore[arg-type]

    assert repository.queries == []


# ---------------------------------------------------------------------------
# The evidence cutoff
# ---------------------------------------------------------------------------


def test_every_read_the_run_makes_ends_at_its_cutoff() -> None:
    repository = ConformingRepository(_history(30))
    cutoff = PointInTime(_instant(27))

    run = _run(repository, _instant(27), (_H1, _H3))

    assert len(repository.queries) == 1 + len(run.analysis_results) * 2
    assert all(query.end == cutoff for query in repository.queries)
    replay, *measurements = repository.queries
    assert replay.start is None
    assert [query.start for query in measurements] == [
        decision for decision in _decisions(run) for _ in range(2)
    ]


def test_no_decision_or_evaluation_is_after_the_cutoff() -> None:
    run = _run(ConformingRepository(_history(40)), _instant(30), (_H1, _H3, ResearchHorizon(5)))

    cutoff = run.available_through
    assert all(decision.compare(cutoff) <= 0 for decision in _decisions(run))
    evaluated = [outcome for outcome in run.outcomes if outcome.evaluation_instant is not None]
    assert evaluated
    assert all(outcome.evaluation_instant.compare(cutoff) <= 0 for outcome in evaluated)


@pytest.mark.parametrize(
    ("through_session", "decision_count"),
    [
        pytest.param(19, 0, id="before-the-first-decision"),
        pytest.param(20, 1, id="at-the-first-decision"),
        pytest.param(23, 4, id="after-several-decisions"),
    ],
)
def test_the_cutoff_bounds_which_decisions_exist(through_session: int, decision_count: int) -> None:
    run = _run(ConformingRepository(_history(30)), _instant(through_session))

    assert len(run.analysis_results) == decision_count


def test_a_decision_at_the_cutoff_can_only_be_insufficient() -> None:
    run = _run(ConformingRepository(_history(30)), _instant(20), (_H1, _H2))

    assert [outcome.unavailable_reason for outcome in run.outcomes] == [_INSUFFICIENT] * 2


def test_the_same_cutoff_reproduces_the_run_after_later_history_is_stored() -> None:
    """The load-bearing reproducibility regression."""
    repository = ConformingRepository(_history(24))
    cutoff = _instant(24)
    horizons = (_H1, _H3)

    first = _run(repository, cutoff, horizons)
    last_decision_h3 = first.outcomes[-1]
    assert last_decision_h3.horizon == _H3
    assert last_decision_h3.unavailable_reason == _INSUFFICIENT

    # Later acquisition stores wildly divergent sessions, all after the cutoff.
    repository.bars = (
        *repository.bars,
        *(
            _bar(_instant(session), close, volume="99999")
            for session, close in (
                (25, "-5000"),
                (26, "90000"),
                (27, "0"),
            )
        ),
    )
    rerun = _run(repository, cutoff, horizons)

    assert rerun == first

    # Only a deliberately later cutoff lets the new evidence in.
    later = _run(repository, _instant(27), horizons)

    assert later != first
    resolved = later.outcomes[len(first.outcomes) - 1]
    assert resolved.recommendation == last_decision_h3.recommendation
    assert resolved.horizon == _H3
    assert resolved.unavailable_reason is None
    assert resolved.evaluation_instant == PointInTime(_instant(27))
    assert later.analysis_results[: len(first.analysis_results)] == first.analysis_results


def test_divergent_bars_after_the_cutoff_would_have_changed_the_run() -> None:
    """Guard: the reproducibility test is load-bearing only if this holds."""
    calm = ConformingRepository(_history(27))
    wild = ConformingRepository(
        (
            *_history(24),
            _bar(_instant(25), "-5000"),
            _bar(_instant(26), "90000"),
            _bar(_instant(27), "0"),
        )
    )

    assert _run(calm, _instant(27), (_H1, _H3)) != _run(wild, _instant(27), (_H1, _H3))


# ---------------------------------------------------------------------------
# Semantic timestamps
# ---------------------------------------------------------------------------


def test_an_offset_spelled_cutoff_is_the_same_cutoff() -> None:
    repository = ConformingRepository(_history(25))

    z = _run(repository, _instant(22))
    offset = _run(repository, "2026-06-22T16:00:00-05:00")

    assert _instant(22) == "2026-06-22T21:00:00Z"
    assert offset == z


def test_a_cutoff_half_a_second_after_a_bar_includes_it() -> None:
    """Lexically ``...00.5Z`` < ``...00Z``; a text bound would exclude session 20."""
    half_second_after = "2026-06-20T21:00:00.5Z"
    assert half_second_after < _instant(20)

    run = _run(ConformingRepository(_history(25)), half_second_after)

    assert _decisions(run) == [PointInTime(_instant(20))]


def test_a_cutoff_half_a_second_before_a_bar_excludes_it() -> None:
    run = _run(ConformingRepository(_history(25)), "2026-06-20T20:59:59.5Z")

    assert run.analysis_results == ()


# ---------------------------------------------------------------------------
# Sparse history, contract isolation and strategy identity
# ---------------------------------------------------------------------------


def test_sparse_history_counts_stored_sessions_not_calendar_days() -> None:
    dates = [date(2026, 6, 1) + timedelta(days=offset * 3) for offset in range(24)]
    bars = tuple(_bar(f"{day}T21:00:00Z", str(100 + index % 5)) for index, day in enumerate(dates))

    run = _run(ConformingRepository(bars), f"{dates[-1]}T21:00:00Z", (_H2,))

    assert len(run.analysis_results) == 5
    first = run.outcomes[0]
    assert first.evaluation_instant == PointInTime(f"{dates[21]}T21:00:00Z")


def test_other_contracts_in_the_store_never_enter_the_run() -> None:
    mixed = (*_history(25), *_history(25, contract=_ES_MAR), *_history(25, contract=_MES_DEC))

    run = _run(ConformingRepository(mixed), _instant(25), (_H1,))
    alone = _run(ConformingRepository(_history(25)), _instant(25), (_H1,))

    assert run == alone
    assert all(result.recommendation.contract == _ES_DEC for result in run.analysis_results)


def test_a_repository_returning_another_contract_is_a_contract_violation() -> None:
    with pytest.raises(FuturesHistoricalDataContractViolationError, match="not the queried"):
        _run(StubRepository(_history(25, contract=_MES_DEC)), _instant(25))


def test_the_run_contract_is_the_requested_one_with_no_substitution() -> None:
    run = _run(ConformingRepository(_history(25, contract=_ES_MAR)), _instant(25), contract=_ES_DEC)

    assert run.contract == _ES_DEC
    assert run.analysis_results == ()


def test_every_decision_belongs_to_the_run_strategy() -> None:
    strategy = Strategy(StrategyIdentity("another-strategy"))

    run = _run(ConformingRepository(_history(25)), _instant(25), strategy=strategy)

    assert run.strategy_identity == StrategyIdentity("another-strategy")
    assert {result.recommendation.strategy_identity for result in run.analysis_results} == {
        StrategyIdentity("another-strategy")
    }


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_repeated_runs_are_equal() -> None:
    repository = ConformingRepository(_history(30))

    first = _run(repository, _instant(28), (_H1, _H3))
    second = _run(repository, _instant(28), (_H1, _H3))
    fresh = _run(ConformingRepository(_history(30)), _instant(28), (_H1, _H3))

    assert first == second == fresh
    assert repr(first) == repr(fresh)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("timeframe", [Timeframe("1m"), Timeframe("1h")], ids=["1m", "1h"])
def test_a_non_daily_run_is_refused_before_any_read(timeframe: Timeframe) -> None:
    repository = ConformingRepository(_history(25))

    with pytest.raises(UnsupportedFuturesReplayTimeframeError, match="session-daily"):
        RunFuturesHistoricalResearchUseCase(repository, _GENERATOR).execute(
            _ES_DEC, timeframe, _STRATEGY, (_H1,), PointInTime(_instant(25))
        )

    assert repository.queries == []


@pytest.mark.parametrize("position", [0, 1, 2, 4])
def test_every_execute_input_is_type_checked(position: int) -> None:
    arguments: list[object] = [_ES_DEC, _DAILY, _STRATEGY, (_H1,), PointInTime(_instant(25))]
    arguments[position] = "not an input"
    repository = ConformingRepository(_history(25))

    with pytest.raises(TypeError):
        RunFuturesHistoricalResearchUseCase(repository, _GENERATOR).execute(*arguments)  # type: ignore[arg-type]

    assert repository.queries == []


@pytest.mark.parametrize("position", [0, 1])
def test_both_dependencies_are_type_checked(position: int) -> None:
    dependencies: list[object] = [ConformingRepository(), _GENERATOR]
    dependencies[position] = "not a dependency"

    with pytest.raises(TypeError):
        RunFuturesHistoricalResearchUseCase(*dependencies)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# FuturesHistoricalResearchRun coherence
# ---------------------------------------------------------------------------


def _valid_run() -> FuturesHistoricalResearchRun:
    return _run(ConformingRepository(_history(25)), _instant(25), (_H1, _H2))


def test_a_run_for_another_contract_rejects_its_results() -> None:
    with pytest.raises(ValueError, match="does not belong to the run contract"):
        dataclasses.replace(_valid_run(), contract=_ES_MAR)


def test_a_run_for_another_strategy_rejects_its_results() -> None:
    with pytest.raises(ValueError, match="does not belong to the run strategy"):
        dataclasses.replace(_valid_run(), strategy_identity=StrategyIdentity("impostor"))


def test_a_decision_after_the_cutoff_is_rejected() -> None:
    """One decision, measured insufficient: only the decision itself can breach."""
    run = _run(ConformingRepository(_history(20)), _instant(20), (_H1,))
    assert run.outcomes[0].evaluation_instant is None

    with pytest.raises(ValueError, match="decision at .* is after the run cutoff"):
        dataclasses.replace(run, available_through=PointInTime(_instant(19)))


def test_an_evaluation_after_the_cutoff_is_rejected() -> None:
    run = _run(ConformingRepository(_history(25)), _instant(25), (_H1,))

    # Drop the last decision so every remaining decision precedes the new cutoff,
    # leaving the session-24 decision's session-25 evaluation after it.
    with pytest.raises(ValueError, match="evaluated at .* after the run cutoff"):
        dataclasses.replace(
            run,
            available_through=PointInTime(_instant(24)),
            analysis_results=run.analysis_results[:-1],
            outcomes=run.outcomes[:-1],
        )


def test_outcomes_out_of_decision_order_are_rejected() -> None:
    run = _valid_run()

    with pytest.raises(ValueError, match="follow result order"):
        dataclasses.replace(run, outcomes=tuple(reversed(run.outcomes)))


def test_an_outcome_for_an_unrequested_horizon_is_rejected() -> None:
    with pytest.raises(ValueError, match="horizon order"):
        dataclasses.replace(_valid_run(), horizons=(_H1, _H3))


def test_a_duplicated_decision_is_rejected() -> None:
    run = _valid_run()

    with pytest.raises(ValueError, match="strictly chronological"):
        dataclasses.replace(
            run,
            analysis_results=(run.analysis_results[0], run.analysis_results[0]),
            outcomes=run.outcomes[:2] * 2,
        )


def test_a_duplicated_outcome_pair_is_rejected() -> None:
    run = _valid_run()

    with pytest.raises(ValueError, match="one outcome for every"):
        dataclasses.replace(run, outcomes=(*run.outcomes, run.outcomes[-1]))


def test_an_outcome_measured_from_other_evidence_is_rejected() -> None:
    run = _valid_run()
    first = run.outcomes[0]
    forged = FuturesRecommendationOutcome(
        first.recommendation, first.horizon, QuoteValue(Decimal("1")), None, None
    )

    with pytest.raises(ValueError, match="decision evidence"):
        dataclasses.replace(run, outcomes=(forged, *run.outcomes[1:]))


def test_a_non_daily_run_value_is_rejected() -> None:
    with pytest.raises(UnsupportedFuturesReplayTimeframeError):
        dataclasses.replace(_valid_run(), timeframe=Timeframe("1m"))


def test_the_run_value_refuses_invalid_horizons() -> None:
    with pytest.raises(ValueError, match="duplicates"):
        dataclasses.replace(_valid_run(), horizons=(_H1, _H1))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("contract", "ES"),
        ("timeframe", "1d"),
        ("strategy_identity", "futures-directional"),
        ("available_through", _instant(25)),
        ("analysis_results", []),
        ("outcomes", []),
    ],
)
def test_every_run_field_is_type_checked(field: str, value: object) -> None:
    with pytest.raises(TypeError):
        dataclasses.replace(_valid_run(), **{field: value})


def test_the_run_is_immutable() -> None:
    with pytest.raises(AttributeError):
        _valid_run().available_through = PointInTime(_instant(30))  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Boundaries and exports
# ---------------------------------------------------------------------------


def _module_tree() -> ast.Module:
    import northstar_application.application_services.run_futures_historical_research as module

    return ast.parse(Path(module.__file__).read_text(encoding="utf-8"))


def test_the_run_reads_persisted_history_through_the_repository_only() -> None:
    tree = _module_tree()
    modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    } | {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    names = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }

    for forbidden in (
        "FuturesHistoricalMarketDataSource",
        "FuturesHistoricalMarketDataStore",
        "FuturesTradingSessionResolver",
        "AcquireFuturesDailyHistoryUseCase",
        "AggregateFuturesDailySessionBarUseCase",
    ):
        assert forbidden not in names
    for module in modules:
        assert module.split(".")[0] not in {
            "databento",
            "exchange_calendars",
            "sqlite3",
            "northstar_infrastructure",
            "requests",
            "urllib",
            "socket",
            "time",
            "datetime",
            "random",
            "decimal",
        }
        assert "paper_trading" not in module
        assert "execution" not in module


def test_the_run_reads_no_clock_no_action_and_does_no_arithmetic() -> None:
    for node in ast.walk(_module_tree()):
        if isinstance(node, ast.Attribute):
            assert node.attr not in {"action", "now", "today", "utcnow", "value"}
        # ``X | None`` annotations are BitOr, not arithmetic.
        if isinstance(node, ast.BinOp) and not isinstance(node.op, ast.BitOr):
            # ``len(results) * len(horizons)`` is the only arithmetic: a count.
            assert isinstance(node.op, ast.Mult)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in {"sorted", "sum", "Decimal"}


def test_the_run_and_use_case_are_exported() -> None:
    import northstar_application.application_services as services

    for name in ("FuturesHistoricalResearchRun", "RunFuturesHistoricalResearchUseCase"):
        assert name in services.__all__
    for private in ("_validate_horizons", "_require_daily", "_SUPPORTED_TIMEFRAME"):
        assert private not in services.__all__
        assert not hasattr(services, private)
