"""Reproducible synthetic OHLCV data (geometric Brownian motion).

The generated data is **synthetic**: it has no relationship with any real market and
exists only to demonstrate and test the library.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_gbm_ohlcv(
    n_bars: int = 1000,
    *,
    start_price: float = 100.0,
    mu: float = 0.08,
    sigma: float = 0.25,
    seed: int = 42,
    start: str = "2020-01-01",
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """Generate daily OHLCV bars from a geometric Brownian motion.

    Closes follow ``S_t = S_{t-1} * exp((mu - sigma^2/2) dt + sigma sqrt(dt) Z_t)``.
    Each bar's open is the previous close plus a small overnight gap; highs and lows
    extend beyond the candle body by a random intrabar excursion, so the bars always
    satisfy ``low <= min(open, close) <= max(open, close) <= high``.

    Args:
        n_bars: number of bars to generate.
        start_price: initial price level.
        mu: annualised drift.
        sigma: annualised volatility.
        seed: RNG seed; identical seeds produce identical data.
        start: first business day of the series.
        periods_per_year: bars per year used to scale ``mu`` and ``sigma``.

    Returns:
        A frame indexed by ``timestamp`` with ``open, high, low, close, volume``.
    """
    if n_bars < 2:
        raise ValueError("n_bars must be >= 2")
    rng = np.random.default_rng(seed)
    dt = 1.0 / periods_per_year
    daily_sigma = sigma * np.sqrt(dt)

    # Split each day's log-return into an overnight gap and an intraday move.
    gap = rng.normal(0.0, 0.25 * daily_sigma, n_bars)
    intraday = rng.normal((mu - 0.5 * sigma**2) * dt, np.sqrt(0.9375) * daily_sigma, n_bars)

    log_close = np.log(start_price) + np.cumsum(gap + intraday)
    close = np.exp(log_close)
    prev_close = np.r_[start_price, close[:-1]]
    open_ = prev_close * np.exp(gap)

    wick_up = np.abs(rng.normal(0.0, 0.6 * daily_sigma, n_bars))
    wick_dn = np.abs(rng.normal(0.0, 0.6 * daily_sigma, n_bars))
    high = np.maximum(open_, close) * np.exp(wick_up)
    low = np.minimum(open_, close) * np.exp(-wick_dn)

    volume = np.round(rng.lognormal(mean=13.0, sigma=0.35, size=n_bars)).astype(np.int64)
    idx = pd.bdate_range(start=start, periods=n_bars, name="timestamp")
    return pd.DataFrame(
        {
            "open": open_.round(4),
            "high": high.round(4),
            "low": low.round(4),
            "close": close.round(4),
            "volume": volume,
        },
        index=idx,
    )
