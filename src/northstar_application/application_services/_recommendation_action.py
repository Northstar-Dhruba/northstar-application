"""Shared translation from recommendation advice to paper-trading direction.

The Core recommendation vocabulary and the paper-trading direction vocabulary
are separate types whose members happen to share spellings, so the mapping
between them is stated once here rather than inferred from the shared text in
each service that needs it.

An advised HOLD has no direction at all. That is reported as the absence of a
side rather than a third member, because OrderSide deliberately has no HOLD and
callers differ in what a hold means to them: translation records it as a
deliberate non-action, while execution treats it as inexecutable.

This module is internal to the Application Services package.
"""

from __future__ import annotations

from northstar_core.paper_trading import OrderSide
from northstar_core.strategy import RecommendationAction

_HOLD_ACTION = "HOLD"

_ACTION_SIDES: dict[str, OrderSide] = {
    "BUY": OrderSide.BUY,
    "SELL": OrderSide.SELL,
}


def executable_side(action: RecommendationAction, subject: str) -> OrderSide | None:
    """Return the direction one recommendation executes as, or None for HOLD.

    An action outside the known vocabulary raises rather than returning None,
    so a future Core action can never be mistaken for a hold and silently do
    nothing.
    """
    if action.value == _HOLD_ACTION:
        return None

    side = _ACTION_SIDES.get(action.value)
    if side is None:
        raise ValueError(f"{subject} cannot translate recommendation action {action.value!r}.")
    return side
