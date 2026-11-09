from __future__ import annotations

import pandas as pd

from backtest_lab.data import validate_ohlcv
from backtest_lab.synthetic import generate_gbm_ohlcv


def test_synthetic_is_reproducible_and_valid() -> None:
    a = generate_gbm_ohlcv(300, seed=1)
    b = generate_gbm_ohlcv(300, seed=1)
    pd.testing.assert_frame_equal(a, b)
    assert len(validate_ohlcv(a)) == 300


def test_synthetic_seed_changes_output() -> None:
    a = generate_gbm_ohlcv(50, seed=1)
    b = generate_gbm_ohlcv(50, seed=2)
    assert not a["close"].equals(b["close"])
