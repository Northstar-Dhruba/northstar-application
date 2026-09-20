"""Deterministic return statistics shared by research metrics calculations.

Summary statistics over recorded forward movement must not depend on the
caller's ambient Decimal context: the same measured returns must summarise
identically wherever a calculation is invoked from. The arithmetic is therefore
performed inside one explicit local context, defined here once so historical
and forward metrics cannot drift apart.

This module is internal to the Application Services package. It expresses no
BUY, SELL or HOLD interpretation, no accuracy or success measure, and no profit
and loss meaning: it is arithmetic over already-measured movements.
"""

from __future__ import annotations

from decimal import ROUND_HALF_EVEN, Context, Decimal, localcontext

from northstar_core.foundation.value_objects import Percentage

_METRICS_CONTEXT = Context(prec=28, rounding=ROUND_HALF_EVEN)
_TWO = Decimal(2)


def average(returns: tuple[Percentage, ...]) -> Percentage | None:
    """Return the arithmetic mean of recorded movements, or None when empty."""
    if not returns:
        return None
    with localcontext(_METRICS_CONTEXT):
        total = Decimal(0)
        for forward_return in returns:
            total += forward_return.value
        return Percentage(total / Decimal(len(returns)))


def median(returns: tuple[Percentage, ...]) -> Percentage | None:
    """Return the median of ascending recorded movements, or None when empty."""
    if not returns:
        return None
    middle = len(returns) // 2
    if len(returns) % 2:
        return returns[middle]
    with localcontext(_METRICS_CONTEXT):
        lower = returns[middle - 1].value
        upper = returns[middle].value
        return Percentage((lower + upper) / _TWO)
