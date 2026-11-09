from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from backtest_lab.engine import run_backtest
from backtest_lab.patterns import detect_patterns
from backtest_lab.plotting import plot_equity, plot_patterns

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def test_plot_equity_writes_png(random_ohlcv: pd.DataFrame, tmp_path: Path) -> None:
    res = run_backtest(random_ohlcv, np.sign(random_ohlcv["close"].diff()).fillna(0))
    out = plot_equity(res, tmp_path / "sub" / "eq.png", benchmark=random_ohlcv["close"])
    assert out.read_bytes()[:8] == PNG_MAGIC


def test_plot_patterns_writes_png(random_ohlcv: pd.DataFrame, tmp_path: Path) -> None:
    out = plot_patterns(random_ohlcv, detect_patterns(random_ohlcv), tmp_path / "p.png", last=80)
    assert out.stat().st_size > 1000
    assert out.read_bytes()[:8] == PNG_MAGIC
