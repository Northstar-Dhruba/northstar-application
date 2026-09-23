"""Tests for analysing one futures replay snapshot.

The service arranges facts and delegates every decision, so these tests pin the
arrangement -- which bars become which context fields, and that only the
latest twenty do -- and prove through the real 9.7b replay that nothing after
the snapshot's replay instant can reach the context, the analysis or the view.
"""

from __future__ import annotations

import ast
import copy
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
from northstar_core.futures import (
    FuturesContract,
    FuturesOHLCVBar,
    FuturesProductReference,
    FuturesReplaySnapshot,
)
from northstar_core.strategy import (
    FuturesAssetAnalysis,
    FuturesAssetAnalysisGenerator,
    FuturesMarketObservationContext,
    FuturesRecommendation,
    RecommendationAction,
    Strategy,
    StrategyIdentity,
)

from northstar_application.application_services import (
    AnalyzeFuturesReplaySnapshotService,
    FuturesAnalysisResult,
    ReplayFuturesHistoricalMarketDataUseCase,
    UnsupportedFuturesReplayTimeframeError,
)
from northstar_application.ports import (
    FuturesHistoricalMarketDataQuery,
    FuturesHistoricalMarketDataRepository,
)

_ES_DEC = FuturesContract(
    FuturesProductReference(Symbol("ES"), ExchangeCode("CME")), ExpirationDate("2026-12-18")
)
_DAILY = Timeframe("1d")
_STRATEGY = Strategy(StrategyIdentity("futures-directional"))

_BUY = RecommendationAction("BUY")
_SELL = RecommendationAction("SELL")
_HOLD = RecommendationAction("HOLD")


def _instant(day: int) -> str:
    """Session completion ``day`` days after 2026-08-31, at 21:00Z."""
    month, day_of_month = (9, day) if day <= 30 else (10, day - 30)
    return f"2026-{month:02d}-{day_of_month:02d}T21:00:00Z"


def _bar(
    day: int,
    close: str,
    volume: str = "1000",
    *,
    timeframe: Timeframe = _DAILY,
    completion: str | None = None,
) -> FuturesOHLCVBar:
    """A bar whose open is its close, inside a band one point either side."""
    quote = Decimal(close)
    return FuturesOHLCVBar(
        contract=_ES_DEC,
        point_in_time=PointInTime(completion if completion is not None else _instant(day)),
        timeframe=timeframe,
        open=QuoteValue(quote),
        high=QuoteValue(quote + 1),
        low=QuoteValue(quote - 1),
        close=QuoteValue(quote),
        volume=Quantity(Decimal(volume)),
    )


def _bars(closes: list[str], volumes: list[str] | None = None) -> tuple[FuturesOHLCVBar, ...]:
    volumes = volumes if volumes is not None else ["1000"] * len(closes)
    return tuple(
        _bar(day, close, volume)
        for day, (close, volume) in enumerate(zip(closes, volumes, strict=True), start=1)
    )


def _snapshot(bars: tuple[FuturesOHLCVBar, ...]) -> FuturesReplaySnapshot:
    return FuturesReplaySnapshot(_ES_DEC, bars[0].timeframe, bars[-1].point_in_time, bars)


def _service(
    strategy: Strategy = _STRATEGY,
    generator: FuturesAssetAnalysisGenerator | None = None,
) -> AnalyzeFuturesReplaySnapshotService:
    return AnalyzeFuturesReplaySnapshotService(
        strategy, generator if generator is not None else FuturesAssetAnalysisGenerator()
    )


def _analyze(bars: tuple[FuturesOHLCVBar, ...]) -> FuturesAnalysisResult | None:
    return _service().execute(_snapshot(bars))


def _required(result: FuturesAnalysisResult | None) -> FuturesAnalysisResult:
    assert result is not None
    return result


# Twenty closes that read strongly bullish: a rising last five over a flat base.
_RISING = ["100"] * 15 + ["120", "125", "130", "135", "140"]
_FALLING = ["140"] * 15 + ["120", "115", "110", "105", "100"]
_FLAT = ["110"] * 20


class RecordingGenerator(FuturesAssetAnalysisGenerator):
    def __init__(self) -> None:
        self.contexts: list[FuturesMarketObservationContext] = []

    def generate(self, context):
        self.contexts.append(context)
        return super().generate(context)


class RecordingStrategy(Strategy):
    def __init__(self, identity: StrategyIdentity) -> None:
        super().__init__(identity)
        self.analyses: list[FuturesAssetAnalysis] = []

    def evaluate_futures(self, asset_analysis):
        self.analyses.append(asset_analysis)
        return super().evaluate_futures(asset_analysis)


class StubRepository(FuturesHistoricalMarketDataRepository):
    def __init__(self, bars: tuple[FuturesOHLCVBar, ...]) -> None:
        self.bars = bars

    def get_bars(self, query):
        return self.bars


def _replay(bars: tuple[FuturesOHLCVBar, ...]) -> tuple[FuturesReplaySnapshot, ...]:
    return ReplayFuturesHistoricalMarketDataUseCase(StubRepository(bars)).execute(
        FuturesHistoricalMarketDataQuery(_ES_DEC, _DAILY)
    )


# ---------------------------------------------------------------------------
# Warm-up
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("count", [1, 2, 5, 19])
def test_fewer_than_twenty_observations_is_warm_up_and_returns_none(count: int) -> None:
    generator = RecordingGenerator()

    result = _service(generator=generator).execute(_snapshot(_bars(_RISING[:count])))

    assert result is None
    assert generator.contexts == []


def test_nineteen_observations_are_not_analysable_and_twenty_are() -> None:
    history = _bars(_RISING)

    assert _analyze(history[:19]) is None
    first = _required(_analyze(history[:20]))
    assert type(first) is FuturesAnalysisResult


def test_every_snapshot_of_a_replay_is_absent_until_the_twentieth() -> None:
    snapshots = _replay(_bars(_RISING + ["145", "150", "155"]))

    results = [_service().execute(snapshot) for snapshot in snapshots]

    assert results[:19] == [None] * 19
    assert all(result is not None for result in results[19:])
    assert len(results) == 23


# ---------------------------------------------------------------------------
# Timeframe gate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("timeframe", [Timeframe("1m"), Timeframe("1h")], ids=["1m", "1h"])
@pytest.mark.parametrize("count", [3, 20])
def test_an_intraday_snapshot_is_rejected_before_any_analysis(
    timeframe: Timeframe, count: int
) -> None:
    """Rejected even when warm-up alone would have returned None."""
    bars = tuple(
        _bar(0, "100", timeframe=timeframe, completion=f"2026-09-15T21:{minute:02d}:00Z")
        for minute in range(count)
    )
    generator = RecordingGenerator()

    with pytest.raises(UnsupportedFuturesReplayTimeframeError, match="session-daily") as raised:
        _service(generator=generator).execute(_snapshot(bars))

    assert str(timeframe) in str(raised.value)
    assert generator.contexts == []


def test_the_replay_timeframe_error_is_reused_not_redefined() -> None:
    import northstar_application.application_services.analyze_futures_replay_snapshot as module
    import northstar_application.application_services.replay_futures_historical_market_data as rp

    assert (
        module.UnsupportedFuturesReplayTimeframeError is rp.UnsupportedFuturesReplayTimeframeError
    )
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    assert not [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name.endswith("Error")
    ]


# ---------------------------------------------------------------------------
# Snapshot -> context mapping
# ---------------------------------------------------------------------------


def test_the_context_is_built_from_exactly_the_latest_observations() -> None:
    bars = _bars(_RISING, [str(1000 + day) for day in range(20)])
    snapshot = _snapshot(bars)

    context = _required(_service().execute(snapshot)).market_observation_context

    latest = bars[-1]
    assert context.contract == snapshot.contract
    assert context.timeframe == snapshot.timeframe
    assert context.observed_at == latest.point_in_time
    assert context.latest_quote == latest.close
    assert context.previous_close == bars[-2].close
    assert context.latest_volume == latest.volume
    assert context.session_high == latest.high
    assert context.session_low == latest.low
    assert context.recent_closes == tuple(bar.close for bar in bars)
    assert context.recent_volumes == tuple(bar.volume for bar in bars)


def test_at_twenty_observations_the_previous_close_is_the_nineteenth() -> None:
    bars = _bars(_RISING)

    context = _required(_analyze(bars)).market_observation_context

    assert context.previous_close == bars[18].close
    assert context.previous_close.value == Decimal("135")


def test_the_context_is_aligned_with_the_replay_boundary() -> None:
    for snapshot in _replay(_bars(_RISING + ["145", "150"]))[19:]:
        context = _required(_service().execute(snapshot)).market_observation_context

        assert context.observed_at.compare(snapshot.replay_instant) == 0
        assert context.observed_at.compare(snapshot.latest_observation.point_in_time) == 0


# ---------------------------------------------------------------------------
# Exactly the latest twenty
# ---------------------------------------------------------------------------

# Five wildly divergent sessions followed by the twenty that decide the view.
_ANCIENT = ["-99999", "99999", "-50000", "50000", "0"]
_ANCIENT_VOLUMES = ["99999999", "1", "99999999", "1", "99999999"]


def test_only_the_latest_twenty_observations_enter_the_context() -> None:
    bars = _bars(_ANCIENT + _RISING, _ANCIENT_VOLUMES + ["1000"] * 20)

    context = _required(_analyze(bars)).market_observation_context

    assert len(context.recent_closes) == 20
    assert context.recent_closes == tuple(bar.close for bar in bars[-20:])
    assert context.recent_volumes == tuple(bar.volume for bar in bars[-20:])


def test_observations_older_than_twenty_do_not_affect_the_view() -> None:
    windowed = _required(_analyze(_bars(_RISING)))
    long = _required(_analyze(_bars(_ANCIENT + _RISING, _ANCIENT_VOLUMES + ["1000"] * 20)))

    assert long.market_observation_context.recent_closes == (
        windowed.market_observation_context.recent_closes
    )
    assert long.market_observation_context.recent_volumes == (
        windowed.market_observation_context.recent_volumes
    )
    assert long.recommendation.asset_analysis.summarized_signals == ("strong bullish",)
    assert long.recommendation.action == windowed.recommendation.action == _BUY


def test_the_ancient_observations_would_change_the_view_if_they_were_read() -> None:
    """Guard: the window test above is only meaningful if this holds."""
    ancient_window = _ANCIENT + _RISING[:15]
    assert _required(_analyze(_bars(ancient_window))).recommendation.action != _BUY


def test_changing_one_of_the_latest_twenty_can_change_the_view() -> None:
    altered = list(_RISING)
    altered[5] = "999"  # inside the long window: lifts the long average above the short

    base = _required(_analyze(_bars(_ANCIENT + _RISING)))
    changed = _required(_analyze(_bars(_ANCIENT + altered)))

    assert base.recommendation.action == _BUY
    assert changed.recommendation.action == _HOLD
    assert changed.market_observation_context.recent_closes[5].value == Decimal("999")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def test_the_generator_and_strategy_are_each_called_once_with_the_right_values() -> None:
    generator = RecordingGenerator()
    strategy = RecordingStrategy(StrategyIdentity("recorded"))

    result = _required(_service(strategy, generator).execute(_snapshot(_bars(_RISING))))

    assert generator.contexts == [result.market_observation_context]
    assert len(strategy.analyses) == 1
    assert strategy.analyses[0] is result.recommendation.asset_analysis
    assert result.recommendation.strategy_identity == StrategyIdentity("recorded")


def test_the_result_is_exactly_what_core_produces_from_the_context() -> None:
    result = _required(_analyze(_bars(_RISING)))
    context = result.market_observation_context

    expected = _STRATEGY.evaluate_futures(FuturesAssetAnalysisGenerator().generate(context))

    assert result.recommendation == expected
    assert type(result.recommendation) is FuturesRecommendation


@pytest.mark.parametrize(
    ("closes", "action", "signal"),
    [
        (_RISING, _BUY, "strong bullish"),
        (_FALLING, _SELL, "strong bearish"),
        (_FLAT, _HOLD, "neutral trend"),
    ],
)
def test_each_direction_flows_through_to_the_recommendation(
    closes: list[str], action: RecommendationAction, signal: str
) -> None:
    result = _required(_analyze(_bars(closes)))

    analysis = result.recommendation.asset_analysis
    assert analysis.summarized_signals == (signal,)
    assert result.recommendation.action == action
    assert analysis.contract == result.market_observation_context.contract
    assert analysis.point_in_time == result.market_observation_context.observed_at
    assert result.recommendation.contract == _ES_DEC


# ---------------------------------------------------------------------------
# Negative and zero quotations
# ---------------------------------------------------------------------------


def _shift(closes: list[str], offset: str) -> list[str]:
    return [str(Decimal(close) + Decimal(offset)) for close in closes]


@pytest.mark.parametrize(
    ("closes", "action"),
    [
        pytest.param(_RISING, _BUY, id="positive"),
        pytest.param(_shift(_RISING, "-10000"), _BUY, id="all-negative"),
        pytest.param(_shift(_RISING, "-120"), _BUY, id="crossing-zero"),
        pytest.param(_shift(_RISING, "-140"), _BUY, id="latest-close-zero"),
        pytest.param(_shift(_FALLING, "-10000"), _SELL, id="all-negative-falling"),
        pytest.param(_shift(_FALLING, "-120"), _SELL, id="crossing-zero-falling"),
    ],
)
def test_the_service_imposes_no_positivity(closes: list[str], action: RecommendationAction) -> None:
    result = _required(_analyze(_bars(closes)))

    assert result.recommendation.action == action
    assert result.market_observation_context.latest_quote.value == Decimal(closes[-1])


def test_a_latest_close_of_zero_is_an_ordinary_observation() -> None:
    closes = _shift(_RISING, "-140")
    assert closes[-1] == "0"

    context = _required(_analyze(_bars(closes))).market_observation_context

    assert context.latest_quote.value == 0
    assert context.session_low.value == -1
    assert context.session_high.value == 1


# ---------------------------------------------------------------------------
# Look-ahead protection through the real replay
# ---------------------------------------------------------------------------

# B1..B20 read strongly bullish at B20. B21..B25 collapse on heavy volume: had
# they reached the B20 context, the latest-twenty window would read bearish.
_B1_TO_B20 = _bars(_RISING)
_B21_TO_B25 = tuple(
    _bar(day, close, "5000")
    for day, close in zip(range(21, 26), ["60", "40", "20", "0", "-20"], strict=True)
)


def test_the_divergent_future_really_would_change_the_view() -> None:
    """Guard: the look-ahead test is load-bearing only if this holds."""
    everything = _replay(_B1_TO_B20 + _B21_TO_B25)

    at_b25 = _required(_service().execute(everything[-1]))

    assert at_b25.recommendation.action == _SELL


def test_future_observations_never_reach_the_b20_result() -> None:
    before = _replay(_B1_TO_B20)
    after = _replay(_B1_TO_B20 + _B21_TO_B25)

    result_before = _required(_service().execute(before[19]))
    result_after = _required(_service().execute(after[19]))

    assert result_after == result_before
    assert result_after.recommendation.action == _BUY

    context = result_after.market_observation_context
    future_instants = {bar.point_in_time for bar in _B21_TO_B25}
    assert context.observed_at == _B1_TO_B20[-1].point_in_time
    assert all(close in {bar.close for bar in _B1_TO_B20} for close in context.recent_closes)
    assert context.observed_at not in future_instants
    for future in _B21_TO_B25:
        assert future.point_in_time.compare(context.observed_at) > 0


# ---------------------------------------------------------------------------
# Determinism and immutability
# ---------------------------------------------------------------------------


def test_the_same_inputs_give_an_equal_result_and_nothing_is_mutated() -> None:
    snapshot = _snapshot(_bars(_ANCIENT + _RISING))
    snapshot_before = copy.deepcopy(snapshot)
    strategy_identity_before = _STRATEGY.strategy_identity
    service = _service()

    first = service.execute(snapshot)
    second = service.execute(snapshot)
    fresh = _service(Strategy(StrategyIdentity("futures-directional"))).execute(snapshot)

    assert first == second == fresh
    assert snapshot == snapshot_before
    assert snapshot.observations == snapshot_before.observations
    assert _STRATEGY.strategy_identity == strategy_identity_before


# ---------------------------------------------------------------------------
# FuturesAnalysisResult
# ---------------------------------------------------------------------------


def _context(
    *, contract: FuturesContract = _ES_DEC, observed_at: str = "2026-09-20T21:00:00Z"
) -> FuturesMarketObservationContext:
    quote = QuoteValue(Decimal("100"))
    return FuturesMarketObservationContext(
        contract=contract,
        timeframe=_DAILY,
        observed_at=PointInTime(observed_at),
        latest_quote=quote,
        previous_close=quote,
        latest_volume=Quantity(Decimal("1000")),
        session_high=quote,
        session_low=quote,
        recent_closes=(quote,) * 20,
        recent_volumes=(Quantity(Decimal("1000")),) * 20,
    )


def _recommendation(
    *, contract: FuturesContract = _ES_DEC, point_in_time: str = "2026-09-20T21:00:00Z"
) -> FuturesRecommendation:
    analysis = FuturesAssetAnalysis(contract, PointInTime(point_in_time), ("neutral trend",))
    return _STRATEGY.evaluate_futures(analysis)


def test_a_coherent_result_holds_exactly_its_two_views() -> None:
    result = FuturesAnalysisResult(_recommendation(), _context())

    assert set(FuturesAnalysisResult.__slots__) == {
        "recommendation",
        "market_observation_context",
    }
    with pytest.raises(AttributeError):
        result.recommendation = _recommendation()  # type: ignore[misc]


@pytest.mark.parametrize(
    "other",
    [
        FuturesContract(
            FuturesProductReference(Symbol("ES"), ExchangeCode("CME")), ExpirationDate("2027-03-19")
        ),
        FuturesContract(
            FuturesProductReference(Symbol("MES"), ExchangeCode("CME")),
            ExpirationDate("2026-12-18"),
        ),
    ],
    ids=["other-expiry", "other-product"],
)
def test_a_recommendation_for_another_contract_is_rejected(other: FuturesContract) -> None:
    with pytest.raises(ValueError, match="does not describe the observed contract"):
        FuturesAnalysisResult(_recommendation(contract=other), _context())


def test_a_recommendation_at_another_instant_is_rejected() -> None:
    with pytest.raises(ValueError, match="must equal the observed instant"):
        FuturesAnalysisResult(_recommendation(point_in_time="2026-09-21T21:00:00Z"), _context())


def test_a_sub_second_instant_difference_is_rejected() -> None:
    with pytest.raises(ValueError, match="must equal the observed instant"):
        FuturesAnalysisResult(_recommendation(point_in_time="2026-09-20T21:00:00.5Z"), _context())


def test_a_rebuilt_equal_contract_is_accepted() -> None:
    rebuilt = FuturesContract(
        FuturesProductReference(Symbol("ES"), ExchangeCode("CME")), ExpirationDate("2026-12-18")
    )
    assert rebuilt is not _ES_DEC

    FuturesAnalysisResult(_recommendation(contract=rebuilt), _context())


def test_an_offset_equivalent_instant_is_accepted() -> None:
    FuturesAnalysisResult(
        _recommendation(point_in_time="2026-09-20T16:00:00-05:00"),
        _context(observed_at="2026-09-20T21:00:00Z"),
    )


@pytest.mark.parametrize("position", [0, 1])
def test_the_result_type_checks_both_views(position: int) -> None:
    members: list[object] = [_recommendation(), _context()]
    members[position] = "not a view"

    with pytest.raises(TypeError):
        FuturesAnalysisResult(*members)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Construction, boundaries and exports
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("position", [0, 1])
def test_both_dependencies_are_type_checked(position: int) -> None:
    dependencies: list[object] = [_STRATEGY, FuturesAssetAnalysisGenerator()]
    dependencies[position] = "not a dependency"

    with pytest.raises(TypeError):
        AnalyzeFuturesReplaySnapshotService(*dependencies)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [None, "snapshot", _B1_TO_B20])
def test_a_foreign_snapshot_is_rejected(value: object) -> None:
    with pytest.raises(TypeError, match="FuturesReplaySnapshot"):
        _service().execute(value)  # type: ignore[arg-type]


def _module_tree() -> ast.Module:
    import northstar_application.application_services.analyze_futures_replay_snapshot as module

    return ast.parse(Path(module.__file__).read_text(encoding="utf-8"))


def test_the_service_reaches_no_evidence_source() -> None:
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
    names = (
        {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
        }
        | {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        | {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    )

    for forbidden in (
        "FuturesHistoricalMarketDataRepository",
        "FuturesHistoricalMarketDataSource",
        "FuturesHistoricalMarketDataStore",
        "FuturesHistoricalMarketDataQuery",
        "AcquireFuturesDailyHistoryUseCase",
        "ReplayFuturesHistoricalMarketDataUseCase",
        "FuturesTradingSessionResolver",
    ):
        assert forbidden not in names

    for module in modules:
        root = module.split(".")[0]
        assert root not in {
            "databento",
            "northstar_infrastructure",
            "sqlite3",
            "exchange_calendars",
            "requests",
            "urllib",
            "http",
            "socket",
            "time",
            "datetime",
            "random",
            "os",
        }
        assert not module.startswith("northstar_application.ports")


_ARITHMETIC = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)


def test_the_service_performs_no_arithmetic_of_its_own() -> None:
    for node in ast.walk(_module_tree()):
        # ``X | None`` annotations are BitOr; negative slice bounds are unary.
        if isinstance(node, ast.BinOp) and isinstance(node.op, _ARITHMETIC):
            raise AssertionError(f"arithmetic at line {node.lineno}")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in {"sum", "Decimal", "float", "localcontext", "sorted"}


def test_the_service_and_result_are_exported() -> None:
    import northstar_application.application_services as services

    for name in ("AnalyzeFuturesReplaySnapshotService", "FuturesAnalysisResult"):
        assert name in services.__all__
    for private in ("_MINIMUM_HISTORY", "_SUPPORTED_TIMEFRAME", "_build_context"):
        assert private not in services.__all__
        assert not hasattr(services, private)
