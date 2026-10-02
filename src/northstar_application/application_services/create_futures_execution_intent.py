"""Application target-position policy for futures paper trading.

The policy turns one frozen futures research decision into at most one
execution intent. A research action names the exposure the strategy wants, not
the trade that gets there:

    BUY  -> target +N contracts
    SELL -> target -N contracts
    HOLD -> keep the current exposure; no order

The trade is the difference between that target and the portfolio's current
net exposure to the decided contract, so a BUY against an over-target long is
a SELL trade, a SELL against an over-target short is a BUY trade, and one trade
may cross through zero. Research action and trade side are separate concepts.

The frozen FuturesForwardResearchRecord is the execution boundary: only a
decision that has been frozen can be executed. The policy reads no clock and
no market data, creates no order or fill, and applies no economics. Sizing is
plain integer arithmetic.

Pre-expiry flatten
------------------
A caller may supply a FuturesExpiryWindowAssessment for the decision. When it
requires a flatten, the target exposure in the decided contract is zero
whatever the research action, so BUY, SELL and HOLD alike close an open
position in full -- SELL the whole long, BUY the whole short -- and never open,
increase or reverse it. An already-flat portfolio gets no intent, with reason
EXPIRY_FLATTEN_WINDOW. Without an assessment, or with one that does not yet
require a flatten, the policy is exactly the target policy above.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from northstar_core.paper_trading import (
    FuturesContractCount,
    FuturesExecutionIntent,
    FuturesPaperPortfolio,
    OrderSide,
)

from northstar_application.application_services.futures_expiry_flatten_guard import (
    FuturesExpiryWindowAssessment,
)
from northstar_application.application_services.futures_forward_research_record import (
    FuturesForwardResearchRecord,
)

_HOLD_ACTION = "HOLD"

# Direction of the target exposure, never the side of the trade.
_TARGET_SIGNS: dict[str, int] = {"BUY": 1, "SELL": -1}


class FuturesExecutionIntentNoIntentReason(StrEnum):
    """Closed vocabulary for why a frozen futures decision produced no intent.

    HOLD means the strategy advised keeping the current exposure.
    TARGET_ALREADY_MET means the portfolio already holds the target exposure.
    EXPIRY_FLATTEN_WINDOW means the pre-expiry guard governs the decision and
    the portfolio is already flat in the expiring contract.
    """

    HOLD = "HOLD"
    TARGET_ALREADY_MET = "TARGET_ALREADY_MET"
    EXPIRY_FLATTEN_WINDOW = "EXPIRY_FLATTEN_WINDOW"


@dataclass(frozen=True, slots=True)
class FuturesExecutionIntentDecision:
    """Outcome of applying the target-position policy to one frozen decision.

    Exactly one of ``intent`` and ``no_intent_reason`` is present, so a decision
    is never silently dropped and a deliberate non-action keeps its reason.

    ``expiry_flatten`` marks a decision governed by the pre-expiry guard rather
    than by the research action: its intent, if any, closes the position, and
    its only possible non-action is EXPIRY_FLATTEN_WINDOW.
    """

    intent: FuturesExecutionIntent | None = None
    no_intent_reason: FuturesExecutionIntentNoIntentReason | None = None
    expiry_flatten: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.expiry_flatten, bool):
            raise TypeError("FuturesExecutionIntentDecision expiry_flatten must be a bool.")
        if self.intent is not None and not isinstance(self.intent, FuturesExecutionIntent):
            raise TypeError(
                "FuturesExecutionIntentDecision intent must be a FuturesExecutionIntent or None."
            )
        if self.no_intent_reason is not None and not isinstance(
            self.no_intent_reason, FuturesExecutionIntentNoIntentReason
        ):
            raise TypeError(
                "FuturesExecutionIntentDecision no-intent reason must be a "
                "FuturesExecutionIntentNoIntentReason or None."
            )
        if (self.intent is None) == (self.no_intent_reason is None):
            raise ValueError(
                "FuturesExecutionIntentDecision must carry exactly one of intent or "
                "no-intent reason."
            )
        is_window_reason = (
            self.no_intent_reason is FuturesExecutionIntentNoIntentReason.EXPIRY_FLATTEN_WINDOW
        )
        if is_window_reason and not self.expiry_flatten:
            raise ValueError(
                "FuturesExecutionIntentDecision can decline as EXPIRY_FLATTEN_WINDOW only "
                "when the expiry guard governs it."
            )
        if self.expiry_flatten and self.no_intent_reason is not None and not is_window_reason:
            raise ValueError(
                "FuturesExecutionIntentDecision governed by the expiry guard can only decline "
                "as EXPIRY_FLATTEN_WINDOW."
            )

    @property
    def has_intent(self) -> bool:
        """Return whether the decision produced an intention to trade."""
        return self.intent is not None


class CreateFuturesExecutionIntentUseCase:
    """Apply the target-position policy to one frozen futures decision.

    The use case holds no dependencies: the frozen decision, the current
    portfolio and the caller-supplied target size are all it needs.
    """

    def execute(
        self,
        record: FuturesForwardResearchRecord,
        portfolio: FuturesPaperPortfolio,
        target_contracts: FuturesContractCount,
        expiry_window: FuturesExpiryWindowAssessment | None = None,
    ) -> FuturesExecutionIntentDecision:
        """Return the execution decision for one frozen futures decision.

        Only the portfolio's position in the decided contract is read. Other
        contracts, including other expiries of the same product, never affect
        the decision. ``expiry_window``, when supplied, must assess exactly this
        record's contract and decision instant.
        """
        self._validate_inputs(record, portfolio, target_contracts)
        if portfolio.strategy_identity != record.strategy_identity:
            raise ValueError(
                f"CreateFuturesExecutionIntentUseCase portfolio strategy "
                f"{portfolio.strategy_identity} does not match the decision strategy "
                f"{record.strategy_identity}."
            )
        if expiry_window is not None:
            self._validate_expiry_window(record, expiry_window)
            if expiry_window.flatten_required:
                return self._flatten(record, portfolio)

        action = record.result.recommendation.action.value
        if action == _HOLD_ACTION:
            return FuturesExecutionIntentDecision(
                no_intent_reason=FuturesExecutionIntentNoIntentReason.HOLD
            )
        sign = _TARGET_SIGNS.get(action)
        if sign is None:
            raise ValueError(
                f"CreateFuturesExecutionIntentUseCase cannot translate recommendation action "
                f"{action!r}."
            )

        position = portfolio.get_position(record.contract)
        current = 0 if position is None else position.net_contracts
        delta = sign * target_contracts.value - current
        if delta == 0:
            return FuturesExecutionIntentDecision(
                no_intent_reason=FuturesExecutionIntentNoIntentReason.TARGET_ALREADY_MET
            )

        return FuturesExecutionIntentDecision(
            intent=FuturesExecutionIntent(
                portfolio_identity=portfolio.identity,
                contract=record.contract,
                side=OrderSide.BUY if delta > 0 else OrderSide.SELL,
                contracts=FuturesContractCount(abs(delta)),
                strategy_identity=record.strategy_identity,
                decided_at=record.decision_instant,
            )
        )

    @staticmethod
    def _flatten(
        record: FuturesForwardResearchRecord, portfolio: FuturesPaperPortfolio
    ) -> FuturesExecutionIntentDecision:
        """Target zero exposure in the decided contract, whatever the action said."""
        position = portfolio.get_position(record.contract)
        if position is None:
            return FuturesExecutionIntentDecision(
                no_intent_reason=FuturesExecutionIntentNoIntentReason.EXPIRY_FLATTEN_WINDOW,
                expiry_flatten=True,
            )
        return FuturesExecutionIntentDecision(
            intent=FuturesExecutionIntent(
                portfolio_identity=portfolio.identity,
                contract=record.contract,
                side=OrderSide.SELL if position.is_long else OrderSide.BUY,
                contracts=FuturesContractCount(position.absolute_contracts),
                strategy_identity=record.strategy_identity,
                decided_at=record.decision_instant,
            ),
            expiry_flatten=True,
        )

    @staticmethod
    def _validate_expiry_window(
        record: FuturesForwardResearchRecord, expiry_window: FuturesExpiryWindowAssessment
    ) -> None:
        if not isinstance(expiry_window, FuturesExpiryWindowAssessment):
            raise TypeError(
                "CreateFuturesExecutionIntentUseCase expiry window must be a "
                "FuturesExpiryWindowAssessment or None."
            )
        if expiry_window.contract != record.contract:
            raise ValueError(
                f"CreateFuturesExecutionIntentUseCase expiry window assesses "
                f"{expiry_window.contract}, not the decided contract {record.contract}."
            )
        if expiry_window.decision_instant.compare(record.decision_instant) != 0:
            raise ValueError(
                "CreateFuturesExecutionIntentUseCase expiry window assesses another "
                "decision instant."
            )

    @staticmethod
    def _validate_inputs(
        record: FuturesForwardResearchRecord,
        portfolio: FuturesPaperPortfolio,
        target_contracts: FuturesContractCount,
    ) -> None:
        if record is None:
            raise TypeError("CreateFuturesExecutionIntentUseCase record cannot be None.")
        if not isinstance(record, FuturesForwardResearchRecord):
            raise TypeError(
                "CreateFuturesExecutionIntentUseCase record must be a FuturesForwardResearchRecord."
            )
        if portfolio is None:
            raise TypeError("CreateFuturesExecutionIntentUseCase portfolio cannot be None.")
        if not isinstance(portfolio, FuturesPaperPortfolio):
            raise TypeError(
                "CreateFuturesExecutionIntentUseCase portfolio must be a FuturesPaperPortfolio."
            )
        if target_contracts is None:
            raise TypeError("CreateFuturesExecutionIntentUseCase target contracts cannot be None.")
        if not isinstance(target_contracts, FuturesContractCount):
            raise TypeError(
                "CreateFuturesExecutionIntentUseCase target contracts must be a "
                "FuturesContractCount."
            )
