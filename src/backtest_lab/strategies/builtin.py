"""Built-in strategies."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import pandas as pd

from backtest_lab import patterns as P
from backtest_lab.strategies.base import Strategy, ema, hold_for


def _check_positive_ints(params: dict[str, object], *keys: str) -> None:
    for key in keys:
        value = params[key]
        if not isinstance(value, int | np.integer) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{key} must be a positive integer, got {value!r}")


class DojiReversal(Strategy):
    """Fade the prevailing trend after an indecision (doji) candle.

    * Long when a dragonfly doji appears, or a regular doji appears while the close is
      below its EMA (``trend_len``), i.e. after a decline.
    * Short when a gravestone doji appears, or a regular doji appears above the EMA.
    * Each signal is held for ``hold_bars`` bars.
    """

    name = "doji_reversal"
    description = "Doji / dragonfly / gravestone reversal against the EMA trend"
    default_params: ClassVar[dict[str, Any]] = {
        "trend_len": 20,
        "hold_bars": 5,
        "allow_short": True,
    }

    def validate(self) -> None:
        _check_positive_ints(self.params, "trend_len", "hold_bars")

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        pats = P.detect_patterns(data, ["doji", "dragonfly_doji", "gravestone_doji"])
        below = data["close"] < ema(data["close"], self.params["trend_len"])
        longs = pats["dragonfly_doji"] | (pats["doji"] & below)
        shorts = pats["gravestone_doji"] | (pats["doji"] & ~below)
        if not self.params["allow_short"]:
            shorts = shorts & False
        entries = pd.Series(np.where(longs, 1.0, np.where(shorts, -1.0, 0.0)), index=data.index)
        return hold_for(entries, self.params["hold_bars"]).rename("signal")


class EngulfingTrend(Strategy):
    """Trade engulfing candles only in the direction of the EMA trend.

    * Long on a bullish engulfing while ``close > EMA(trend_len)``.
    * Short on a bearish engulfing while ``close < EMA(trend_len)``.
    * Each signal is held for ``hold_bars`` bars.
    """

    name = "engulfing_trend"
    description = "Engulfing candles filtered by EMA trend direction"
    default_params: ClassVar[dict[str, Any]] = {
        "trend_len": 50,
        "hold_bars": 10,
        "allow_short": True,
    }

    def validate(self) -> None:
        _check_positive_ints(self.params, "trend_len", "hold_bars")

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        pats = P.detect_patterns(data, ["bullish_engulfing", "bearish_engulfing"])
        trend = ema(data["close"], self.params["trend_len"])
        longs = pats["bullish_engulfing"] & (data["close"] > trend)
        shorts = pats["bearish_engulfing"] & (data["close"] < trend)
        if not self.params["allow_short"]:
            shorts = shorts & False
        entries = pd.Series(np.where(longs, 1.0, np.where(shorts, -1.0, 0.0)), index=data.index)
        return hold_for(entries, self.params["hold_bars"]).rename("signal")


class SmaCrossover(Strategy):
    """Baseline trend follower: long when ``SMA(fast) > SMA(slow)``, else short/flat."""

    name = "sma_crossover"
    description = "Moving-average crossover baseline"
    default_params: ClassVar[dict[str, Any]] = {"fast": 20, "slow": 50, "allow_short": False}

    def validate(self) -> None:
        _check_positive_ints(self.params, "fast", "slow")
        if self.params["fast"] >= self.params["slow"]:
            raise ValueError("fast must be smaller than slow")

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        close = data["close"]
        fast = close.rolling(self.params["fast"]).mean()
        slow = close.rolling(self.params["slow"]).mean()
        down = -1.0 if self.params["allow_short"] else 0.0
        sig = np.where(fast > slow, 1.0, np.where(fast < slow, down, 0.0))
        sig[slow.isna().to_numpy()] = 0.0
        return pd.Series(sig, index=data.index, name="signal")
