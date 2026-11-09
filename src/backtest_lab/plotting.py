"""Matplotlib charts saved to PNG (uses the non-interactive Agg backend)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from backtest_lab.engine import BacktestResult
from backtest_lab.metrics import drawdown_series
from backtest_lab.patterns import BEARISH, BULLISH

UP = "#1a9850"
DOWN = "#d73027"
NEUTRAL = "#4c72b0"


def plot_equity(
    result: BacktestResult,
    path: str | Path,
    title: str = "Equity curve",
    benchmark: pd.Series | None = None,
) -> Path:
    """Save a two-panel chart: equity (top) and drawdown (bottom).

    Args:
        result: backtest result.
        path: output PNG path.
        title: figure title.
        benchmark: optional price series rescaled to the initial equity (buy & hold).
    """
    out = Path(path)
    eq = result.equity
    dd = drawdown_series(eq) * 100
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    ax1.plot(eq.index, eq.values, color=NEUTRAL, lw=1.4, label="strategy")
    if benchmark is not None:
        bench = benchmark / benchmark.iloc[0] * eq.iloc[0]
        ax1.plot(bench.index, bench.values, color="#999999", lw=1.0, label="buy & hold")
    ax1.set_ylabel("equity")
    ax1.set_title(title)
    ax1.grid(alpha=0.3)
    ax1.legend(loc="upper left", frameon=False)
    ax2.fill_between(dd.index, dd.values, 0, color=DOWN, alpha=0.35, lw=0)
    ax2.set_ylabel("drawdown %")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def plot_patterns(
    data: pd.DataFrame,
    patterns: pd.DataFrame,
    path: str | Path,
    last: int = 150,
    title: str = "Candlestick patterns",
) -> Path:
    """Save a candlestick chart of the last ``last`` bars with pattern markers.

    Bullish patterns are marked below the low, bearish above the high, neutral
    (regular doji) with a diamond at the close.
    """
    out = Path(path)
    df = data.iloc[-last:]
    pats = patterns.loc[df.index]
    x = np.arange(len(df))
    o, h, lo, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
    colors = np.where(c >= o, UP, DOWN)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.vlines(x, lo, h, colors=colors, lw=0.8)
    ax.bar(x, np.maximum(np.abs(c - o), 1e-9), bottom=np.minimum(o, c), color=colors, width=0.6)

    pad = (h - lo).mean() * 0.6
    markers = {"bull": ("^", UP), "bear": ("v", DOWN), "neutral": ("D", "#7b3294")}
    for name in pats.columns:
        hit = pats[name].to_numpy()
        if not hit.any():
            continue
        if name in BULLISH:
            y, (m, col) = lo[hit] - pad, markers["bull"]
        elif name in BEARISH:
            y, (m, col) = h[hit] + pad, markers["bear"]
        else:
            y, (m, col) = c[hit], markers["neutral"]
        ax.scatter(x[hit], y, marker=m, color=col, s=36, label=name, zorder=3, edgecolors="k")

    step = max(len(df) // 8, 1)
    ax.set_xticks(x[::step])
    ax.set_xticklabels([f"{t:%Y-%m-%d}" for t in df.index[::step]], rotation=30, ha="right")
    ax.set_title(title)
    ax.set_ylabel("price")
    ax.grid(alpha=0.3)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="upper left", fontsize=8, ncol=3, frameon=False)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out
