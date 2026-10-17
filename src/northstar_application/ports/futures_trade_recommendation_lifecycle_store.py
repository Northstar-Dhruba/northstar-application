"""Application port for durably recording futures trade recommendation lifecycle progress.

A recorded recommendation's definition never changes. What changes is its F-2
lifecycle, and that is recorded as insert-only *steps*: one step per lifecycle
evaluation that F-2 actually performed.

- An observation step is one completed bar that ``advance`` evaluated, whether
  or not it produced an event.
- A time-expiry step is one TimeExpiryEvidence that ``expire`` applied.

Why progress names its steps
----------------------------
A lifecycle snapshot keeps only its latest evaluated bar. A bar that changes
nothing leaves no event, so from ``previous`` and ``current`` alone the bars
evaluated in between cannot be recovered: a lifecycle that evaluated D2, D3 and
D4 quietly equals one that evaluated only D4. Recording progress therefore
takes the evaluations themselves, in order, and ``current`` must be exactly
what Core produces by applying them to ``previous``. Nothing is inferred and no
quiet evaluation is invented or dropped.

An observation step also names the trading-session label its bar was matched
to. The bar alone does not carry it, and the ``fdc1`` candle fingerprint that
ties a step to its stored candle is defined over it. The label is a caller
statement; it is bound to the candle, not verified against a calendar here.

Which bars a lifecycle should have evaluated -- whether a session was skipped --
is coverage, decided by whoever supplies TimeExpiryEvidence (F-5C). This port
records what was evaluated; it never claims coverage was complete.

Semantics every implementation honours
--------------------------------------
- ``validate_futures_trade_recommendation_lifecycle_progress`` runs first and
  refuses a self-contradictory request with
  InvalidFuturesTradeRecommendationLifecycleProgressError.
- stored lifecycle equal to ``previous``: the steps are appended, all or
  nothing, and their count is returned.
- the request's steps already recorded right after ``previous``, so the stored
  lifecycle already equals ``current``: an idempotent success returning 0.
- anything else -- a stale ``previous``, other progress recorded first, or an
  unrecorded or differently-defined recommendation -- raises
  FuturesTradeRecommendationLifecycleConflictError and records nothing.
- an observation step whose candle the store's market data does not hold, or
  holds with other values, raises FuturesTradeRecommendationLifecycleEvidenceError.
  No other candle is substituted and no provider is asked.

Status and terminality are never written directly: they are derived by Core
from the recorded steps. A terminal lifecycle accepts no further step.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, datetime

from northstar_core.futures import FuturesOHLCVBar
from northstar_core.recommendations.futures import (
    FuturesTradeRecommendationLifecycle,
    InvalidFuturesTradeRecommendationExpiryError,
    InvalidFuturesTradeRecommendationLifecycleError,
    InvalidFuturesTradeRecommendationObservationError,
    TimeExpiryEvidence,
)


class InvalidFuturesTradeRecommendationLifecycleProgressError(ValueError):
    """Raised when lifecycle progress contradicts its own steps or F-2 rules."""


class FuturesTradeRecommendationLifecycleConflictError(ValueError):
    """Raised when progress does not continue the recorded lifecycle."""


class FuturesTradeRecommendationLifecycleEvidenceError(ValueError):
    """Raised when an observation step's candle is not the stored market data."""


@dataclass(frozen=True, slots=True)
class FuturesTradeRecommendationObservationStep:
    """One completed bar a lifecycle evaluated, and its trading-session label."""

    bar: FuturesOHLCVBar
    trading_date: date

    def __post_init__(self) -> None:
        if not isinstance(self.bar, FuturesOHLCVBar):
            raise InvalidFuturesTradeRecommendationLifecycleProgressError(
                "FuturesTradeRecommendationObservationStep bar must be a FuturesOHLCVBar."
            )
        # A datetime carries a clock; a session label has none.
        if isinstance(self.trading_date, datetime) or not isinstance(self.trading_date, date):
            raise InvalidFuturesTradeRecommendationLifecycleProgressError(
                "FuturesTradeRecommendationObservationStep trading date must be a plain date."
            )


@dataclass(frozen=True, slots=True)
class FuturesTradeRecommendationTimeExpiryStep:
    """One time expiry a lifecycle applied, with the evidence it was applied on."""

    evidence: TimeExpiryEvidence

    def __post_init__(self) -> None:
        if not isinstance(self.evidence, TimeExpiryEvidence):
            raise InvalidFuturesTradeRecommendationLifecycleProgressError(
                "FuturesTradeRecommendationTimeExpiryStep evidence must be TimeExpiryEvidence."
            )


type FuturesTradeRecommendationLifecycleStep = (
    FuturesTradeRecommendationObservationStep | FuturesTradeRecommendationTimeExpiryStep
)

_CORE_REFUSALS = (
    InvalidFuturesTradeRecommendationExpiryError,
    InvalidFuturesTradeRecommendationLifecycleError,
    InvalidFuturesTradeRecommendationObservationError,
)


def apply_futures_trade_recommendation_lifecycle_step(
    lifecycle: FuturesTradeRecommendationLifecycle,
    step: FuturesTradeRecommendationLifecycleStep,
) -> FuturesTradeRecommendationLifecycle:
    """Return ``lifecycle`` after one step, refusing a step F-2 would not evaluate.

    Core's ``advance`` and ``expire`` decide everything. A step they leave
    unchanged -- an ineligible or repeated bar, a terminal lifecycle, expiry
    before validity ends -- is not an evaluation, so it is not a step.
    """
    try:
        if isinstance(step, FuturesTradeRecommendationObservationStep):
            after = lifecycle.advance(step.bar)
        elif isinstance(step, FuturesTradeRecommendationTimeExpiryStep):
            after = lifecycle.expire(step.evidence)
        else:
            raise InvalidFuturesTradeRecommendationLifecycleProgressError(
                "A lifecycle step must be an observation step or a time-expiry step."
            )
    except _CORE_REFUSALS as exc:
        raise InvalidFuturesTradeRecommendationLifecycleProgressError(str(exc)) from exc
    if after == lifecycle:
        raise InvalidFuturesTradeRecommendationLifecycleProgressError(
            f"A step leaves recommendation {lifecycle.recommendation.identity} unchanged at "
            f"{lifecycle.status}, so F-2 did not evaluate it."
        )
    return after


def validate_futures_trade_recommendation_lifecycle_progress(
    previous: FuturesTradeRecommendationLifecycle,
    current: FuturesTradeRecommendationLifecycle,
    steps: tuple[FuturesTradeRecommendationLifecycleStep, ...],
) -> None:
    """Refuse progress that is not exactly ``steps`` applied to ``previous`` by Core.

    Every store implementation calls this before touching storage, so all of
    them refuse the same requests:

    - ``previous`` and ``current`` are lifecycles of the same recommendation,
      equal in its whole immutable definition, not only its identity;
    - ``steps`` is a non-empty tuple of step values;
    - each step is one F-2 evaluation of the lifecycle it is applied to;
    - applying every step in order to ``previous`` yields exactly ``current``:
      the same events, the same last observation, so the same status.

    Because Core only ever appends, ``current`` keeps every event of
    ``previous``; history cannot be removed or rewritten.
    """
    for name, value in (("previous", previous), ("current", current)):
        if not isinstance(value, FuturesTradeRecommendationLifecycle):
            raise InvalidFuturesTradeRecommendationLifecycleProgressError(
                f"Lifecycle progress {name} must be a FuturesTradeRecommendationLifecycle."
            )
    if previous.recommendation.identity != current.recommendation.identity:
        raise InvalidFuturesTradeRecommendationLifecycleProgressError(
            f"Lifecycle progress from {previous.recommendation.identity} cannot end at "
            f"{current.recommendation.identity}."
        )
    if previous.recommendation != current.recommendation:
        raise InvalidFuturesTradeRecommendationLifecycleProgressError(
            f"Lifecycle progress for {previous.recommendation.identity} names two different "
            "recommendation definitions."
        )
    if not isinstance(steps, tuple) or not all(
        isinstance(
            step,
            (FuturesTradeRecommendationObservationStep, FuturesTradeRecommendationTimeExpiryStep),
        )
        for step in steps
    ):
        raise InvalidFuturesTradeRecommendationLifecycleProgressError(
            "Lifecycle progress steps must be a tuple of lifecycle step values."
        )
    if not steps:
        raise InvalidFuturesTradeRecommendationLifecycleProgressError(
            "Lifecycle progress must name at least one evaluation step."
        )

    lifecycle = previous
    for step in steps:
        lifecycle = apply_futures_trade_recommendation_lifecycle_step(lifecycle, step)
    if lifecycle != current:
        raise InvalidFuturesTradeRecommendationLifecycleProgressError(
            f"Applying the steps to {previous} gives {lifecycle}, which is not the requested "
            "current lifecycle."
        )


class FuturesTradeRecommendationLifecycleStore(ABC):
    """Records futures trade recommendation lifecycle progress as insert-only steps.

    Implementations must call
    validate_futures_trade_recommendation_lifecycle_progress first, then honour
    the semantics in this module's documentation. A durable implementation must
    compare, verify and insert in one transaction, so that concurrent writers
    cannot leave a gap, a duplicated evaluation or two terminal outcomes.

    The abstract contract documents these obligations but cannot enforce them
    at runtime.
    """

    @abstractmethod
    def record_progress(
        self,
        previous: FuturesTradeRecommendationLifecycle,
        current: FuturesTradeRecommendationLifecycle,
        *,
        steps: tuple[FuturesTradeRecommendationLifecycleStep, ...],
    ) -> int:
        """Record ``steps`` taking ``previous`` to ``current``; return the count added."""
