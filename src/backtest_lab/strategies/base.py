"""Strategy interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

import numpy as np
import pandas as pd


class Strategy(ABC):
    """Base class for all strategies.

    A strategy turns OHLCV data into a *target direction* series (``+1`` long,
    ``-1`` short, ``0`` flat). The value at bar ``t`` must only use information
    available at the close of bar ``t``; the engine executes it at the next open.

    Subclasses declare a unique :attr:`name` and their :attr:`default_params`, and
    implement :meth:`generate_signals`. Unknown keyword parameters are rejected.
    """

    name: ClassVar[str] = ""
    description: ClassVar[str] = ""
    default_params: ClassVar[dict[str, Any]] = {}

    def __init__(self, **params: Any) -> None:
        unknown = set(params) - set(self.default_params)
        if unknown:
            raise ValueError(
                f"unknown parameter(s) for {self.name}: {', '.join(sorted(unknown))}; "
                f"valid: {', '.join(self.default_params) or 'none'}"
            )
        self.params: dict[str, Any] = {**self.default_params, **params}
        self.validate()

    def validate(self) -> None:  # noqa: B027 - optional hook
        """Check parameter values; override to raise ``ValueError`` on bad input."""

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Return target directions in ``{-1, 0, 1}`` aligned to ``data.index``."""

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self.params.items())
        return f"{type(self).__name__}({args})"


def hold_for(entries: pd.Series, bars: int) -> pd.Series:
    """Hold each non-zero entry signal for ``bars`` bars (including the signal bar).

    A new entry overrides an older one. Returns a float series of -1/0/+1.
    """
    if bars < 1:
        raise ValueError("bars must be >= 1")
    marked = entries.where(entries != 0, np.nan)
    held = marked.ffill(limit=bars - 1) if bars > 1 else marked
    return held.fillna(0.0).astype(float)


def ema(series: pd.Series, length: int) -> pd.Series:
    """Exponential moving average (``alpha = 2 / (length + 1)``)."""
    return series.ewm(span=length, adjust=False).mean()
