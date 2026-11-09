"""Strategy interface, built-in strategies and a name-based registry."""

from __future__ import annotations

from typing import Any

from backtest_lab.strategies.base import Strategy, ema, hold_for
from backtest_lab.strategies.builtin import DojiReversal, EngulfingTrend, SmaCrossover

STRATEGIES: dict[str, type[Strategy]] = {
    cls.name: cls for cls in (DojiReversal, EngulfingTrend, SmaCrossover)
}


def register_strategy(cls: type[Strategy]) -> type[Strategy]:
    """Class decorator that adds a custom strategy to :data:`STRATEGIES`."""
    if not cls.name:
        raise ValueError("strategy classes must define a non-empty `name`")
    STRATEGIES[cls.name] = cls
    return cls


def get_strategy(name: str, **params: Any) -> Strategy:
    """Instantiate a registered strategy by name.

    Raises:
        KeyError: if ``name`` is not registered.
    """
    try:
        cls = STRATEGIES[name]
    except KeyError:
        raise KeyError(f"unknown strategy '{name}'; available: {', '.join(STRATEGIES)}") from None
    return cls(**params)


__all__ = [
    "STRATEGIES",
    "DojiReversal",
    "EngulfingTrend",
    "SmaCrossover",
    "Strategy",
    "ema",
    "get_strategy",
    "hold_for",
    "register_strategy",
]
