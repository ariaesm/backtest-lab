"""Minimal library usage: load data, detect patterns, backtest, plot.

Run from the repository root:
    PYTHONPATH=src python3 examples/quickstart.py
"""

from __future__ import annotations

from pathlib import Path

from backtest_lab.data import load_ohlcv
from backtest_lab.engine import BacktestConfig, run_backtest
from backtest_lab.metrics import format_metrics
from backtest_lab.patterns import detect_patterns
from backtest_lab.plotting import plot_equity
from backtest_lab.strategies import get_strategy

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    data = load_ohlcv(ROOT / "data" / "sample_ohlcv.csv")  # synthetic GBM data

    hits = detect_patterns(data).sum()
    print("pattern counts:\n" + hits.to_string() + "\n")

    strategy = get_strategy("engulfing_trend", trend_len=100, hold_bars=5)
    config = BacktestConfig(commission=0.0005, slippage=0.0005, stop_loss=0.05)
    result = run_backtest(data, strategy.generate_signals(data), config)

    print(repr(strategy))
    print(format_metrics(result.metrics))
    print("\nlast trades:\n" + result.trades.tail(3).to_string(index=False))

    out = plot_equity(result, ROOT / "examples" / "quickstart_equity.png", title=repr(strategy))
    print(f"\nchart saved to {out}")


if __name__ == "__main__":
    main()
