"""Shared pytest fixtures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def make_ohlcv(
    rows: list[tuple[float, float, float, float]], start: str = "2024-01-01"
) -> pd.DataFrame:
    """Build an OHLCV frame from (open, high, low, close) tuples."""
    arr = np.asarray(rows, dtype=float)
    idx = pd.date_range(start, periods=len(arr), freq="D", name="timestamp")
    return pd.DataFrame(
        {
            "open": arr[:, 0],
            "high": arr[:, 1],
            "low": arr[:, 2],
            "close": arr[:, 3],
            "volume": 1000.0,
        },
        index=idx,
    )


@pytest.fixture
def ohlcv_factory():
    """Return :func:`make_ohlcv` so tests can build hand-crafted candles."""
    return make_ohlcv


@pytest.fixture
def random_ohlcv() -> pd.DataFrame:
    """A deterministic random-walk OHLCV frame with 400 bars."""
    rng = np.random.default_rng(7)
    n = 400
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    open_ = np.r_[100.0, close[:-1]] * (1 + rng.normal(0, 0.002, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.005, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.005, n)))
    idx = pd.date_range("2022-01-03", periods=n, freq="B", name="timestamp")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": 1e6}, index=idx
    )
