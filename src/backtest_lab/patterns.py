"""Vectorized candlestick pattern detectors.

Every detector takes an OHLCV frame and returns a boolean :class:`pandas.Series`
aligned with the input index. A value of ``True`` at bar ``t`` means the pattern is
*completed* at the close of bar ``t``; detectors never read bars after ``t``.

The doji family follows the logic of the TradingView Pine Script indicator in
`ariaesm/Doji-indicator <https://github.com/ariaesm/Doji-indicator>`_:

* the candle body is compared with an EMA of body sizes (length 14);
* a doji body is at most 5% of the bar's high-low range;
* shadows are compared with each other (regular doji) or with the body
  (dragonfly / gravestone).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PatternParams:
    """Tunable thresholds shared by the detectors.

    Attributes:
        body_ema_len: EMA length for the average body size.
        doji_body_pct: maximum body size as a percentage of the bar range for a doji.
        shadow_pct: a shadow counts as "present" if it exceeds this % of the body.
        shadow_equals_pct: max relative difference (%) between shadows for a doji.
        hammer_factor: minimum long-shadow / body ratio for hammers and shooting stars.
    """

    body_ema_len: int = 14
    doji_body_pct: float = 5.0
    shadow_pct: float = 5.0
    shadow_equals_pct: float = 100.0
    hammer_factor: float = 2.0


DEFAULT_PARAMS = PatternParams()


def _ema(series: pd.Series, length: int) -> pd.Series:
    """Exponential moving average with ``alpha = 2 / (length + 1)`` (Pine ``ta.ema``)."""
    return series.ewm(span=length, adjust=False).mean()


@dataclass(frozen=True)
class _Candle:
    """Derived candle geometry used by all detectors."""

    open: pd.Series
    high: pd.Series
    low: pd.Series
    close: pd.Series
    body_hi: pd.Series
    body_lo: pd.Series
    body: pd.Series
    body_avg: pd.Series
    up_shadow: pd.Series
    dn_shadow: pd.Series
    range: pd.Series

    @property
    def small_body(self) -> pd.Series:
        return self.body < self.body_avg

    @property
    def long_body(self) -> pd.Series:
        return self.body > self.body_avg

    @property
    def white(self) -> pd.Series:
        return self.close > self.open

    @property
    def black(self) -> pd.Series:
        return self.close < self.open

    @property
    def body_mid(self) -> pd.Series:
        return (self.body_hi + self.body_lo) / 2


def _candle(df: pd.DataFrame, params: PatternParams) -> _Candle:
    o, h, lo, c = df["open"], df["high"], df["low"], df["close"]
    body_hi = np.maximum(o, c)
    body_lo = np.minimum(o, c)
    body = body_hi - body_lo
    return _Candle(
        open=o,
        high=h,
        low=lo,
        close=c,
        body_hi=body_hi,
        body_lo=body_lo,
        body=body,
        body_avg=_ema(body, params.body_ema_len),
        up_shadow=h - body_hi,
        dn_shadow=body_lo - lo,
        range=h - lo,
    )


def _doji_body(k: _Candle, p: PatternParams) -> pd.Series:
    """Body is tiny: <= ``doji_body_pct`` of range and smaller than the EMA body."""
    return (k.range > 0) & (k.body <= k.range * p.doji_body_pct / 100) & k.small_body


def _named(series: pd.Series, name: str) -> pd.Series:
    return series.fillna(False).astype(bool).rename(name)


def dragonfly_doji(df: pd.DataFrame, params: PatternParams = DEFAULT_PARAMS) -> pd.Series:
    """Doji with (almost) no upper shadow: ``up_shadow <= body``. Bullish hint."""
    k = _candle(df, params)
    return _named(_doji_body(k, params) & (k.up_shadow <= k.body), "dragonfly_doji")


def gravestone_doji(df: pd.DataFrame, params: PatternParams = DEFAULT_PARAMS) -> pd.Series:
    """Doji with (almost) no lower shadow: ``dn_shadow <= body``. Bearish hint."""
    k = _candle(df, params)
    return _named(_doji_body(k, params) & (k.dn_shadow <= k.body), "gravestone_doji")


def doji(df: pd.DataFrame, params: PatternParams = DEFAULT_PARAMS) -> pd.Series:
    """Regular doji: tiny body with roughly equal upper and lower shadows.

    Shadows are "equal" when they are identical or each differs from the other by
    less than ``shadow_equals_pct`` percent. Dragonfly and gravestone dojis are
    excluded, matching the Pine script.
    """
    k = _candle(df, params)
    up, dn = k.up_shadow, k.dn_shadow
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_up = (up - dn).abs() / dn * 100
        rel_dn = (dn - up).abs() / up * 100
    equal = (up == dn) | ((rel_up < params.shadow_equals_pct) & (rel_dn < params.shadow_equals_pct))
    body_ok = _doji_body(k, params)
    is_dragonfly = body_ok & (up <= k.body)
    is_gravestone = body_ok & (dn <= k.body)
    return _named(body_ok & equal & ~is_dragonfly & ~is_gravestone, "doji")


def hammer(df: pd.DataFrame, params: PatternParams = DEFAULT_PARAMS) -> pd.Series:
    """Small body in the upper half of the range with a long lower shadow."""
    k = _candle(df, params)
    hl2 = (k.high + k.low) / 2
    no_upper = k.up_shadow <= params.shadow_pct / 100 * k.body
    out = (
        k.small_body
        & (k.body > 0)
        & (k.body_lo > hl2)
        & (k.dn_shadow >= params.hammer_factor * k.body)
        & no_upper
    )
    return _named(out, "hammer")


def shooting_star(df: pd.DataFrame, params: PatternParams = DEFAULT_PARAMS) -> pd.Series:
    """Small body in the lower half of the range with a long upper shadow."""
    k = _candle(df, params)
    hl2 = (k.high + k.low) / 2
    no_lower = k.dn_shadow <= params.shadow_pct / 100 * k.body
    out = (
        k.small_body
        & (k.body > 0)
        & (k.body_hi < hl2)
        & (k.up_shadow >= params.hammer_factor * k.body)
        & no_lower
    )
    return _named(out, "shooting_star")


def bullish_engulfing(df: pd.DataFrame, params: PatternParams = DEFAULT_PARAMS) -> pd.Series:
    """A white body that engulfs the previous black body."""
    k = _candle(df, params)
    prev_black = k.black.shift(1, fill_value=False)
    po, pc = k.open.shift(1), k.close.shift(1)
    out = prev_black & k.white & (k.close >= po) & (k.open <= pc) & ((k.close > po) | (k.open < pc))
    return _named(out, "bullish_engulfing")


def bearish_engulfing(df: pd.DataFrame, params: PatternParams = DEFAULT_PARAMS) -> pd.Series:
    """A black body that engulfs the previous white body."""
    k = _candle(df, params)
    prev_white = k.white.shift(1, fill_value=False)
    po, pc = k.open.shift(1), k.close.shift(1)
    out = prev_white & k.black & (k.open >= pc) & (k.close <= po) & ((k.open > pc) | (k.close < po))
    return _named(out, "bearish_engulfing")


def morning_star(df: pd.DataFrame, params: PatternParams = DEFAULT_PARAMS) -> pd.Series:
    """Three-bar bullish reversal.

    Bar 1 is a long black candle, bar 2 a small body that sits below bar 1's body
    midpoint, and bar 3 a long white candle closing above bar 1's body midpoint.
    """
    k = _candle(df, params)
    long1 = k.long_body.shift(2, fill_value=False) & k.black.shift(2, fill_value=False)
    mid1 = k.body_mid.shift(2)
    small2 = k.small_body.shift(1, fill_value=False) & (k.body_hi.shift(1) < mid1)
    third = k.long_body & k.white & (k.close > mid1) & (k.body_lo > k.body_lo.shift(1))
    return _named(long1 & small2 & third, "morning_star")


def evening_star(df: pd.DataFrame, params: PatternParams = DEFAULT_PARAMS) -> pd.Series:
    """Three-bar bearish reversal, the mirror image of :func:`morning_star`."""
    k = _candle(df, params)
    long1 = k.long_body.shift(2, fill_value=False) & k.white.shift(2, fill_value=False)
    mid1 = k.body_mid.shift(2)
    small2 = k.small_body.shift(1, fill_value=False) & (k.body_lo.shift(1) > mid1)
    third = k.long_body & k.black & (k.close < mid1) & (k.body_hi < k.body_hi.shift(1))
    return _named(long1 & small2 & third, "evening_star")


PatternFn = Callable[[pd.DataFrame, PatternParams], pd.Series]

PATTERNS: dict[str, PatternFn] = {
    "doji": doji,
    "dragonfly_doji": dragonfly_doji,
    "gravestone_doji": gravestone_doji,
    "hammer": hammer,
    "shooting_star": shooting_star,
    "bullish_engulfing": bullish_engulfing,
    "bearish_engulfing": bearish_engulfing,
    "morning_star": morning_star,
    "evening_star": evening_star,
}

BULLISH: frozenset[str] = frozenset(
    {"dragonfly_doji", "hammer", "bullish_engulfing", "morning_star"}
)
BEARISH: frozenset[str] = frozenset(
    {"gravestone_doji", "shooting_star", "bearish_engulfing", "evening_star"}
)


def detect_patterns(
    df: pd.DataFrame,
    names: list[str] | None = None,
    params: PatternParams = DEFAULT_PARAMS,
) -> pd.DataFrame:
    """Run several detectors and return one boolean column per pattern.

    Args:
        df: OHLCV frame.
        names: subset of :data:`PATTERNS` to run (all when ``None``).
        params: detector thresholds.

    Raises:
        KeyError: if an unknown pattern name is requested.
    """
    selected = list(PATTERNS) if names is None else names
    unknown = [n for n in selected if n not in PATTERNS]
    if unknown:
        raise KeyError(f"unknown pattern(s): {', '.join(unknown)}; known: {', '.join(PATTERNS)}")
    return pd.DataFrame({n: PATTERNS[n](df, params) for n in selected}, index=df.index)
