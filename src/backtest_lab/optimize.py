"""Parameter grid search with a walk-forward (train/test) split.

Optimising on the full history and reporting the best in-sample score is the classic
way to overfit a backtest. :func:`walk_forward` instead selects parameters on the
*train* window only and reports their performance on the unseen *test* window.

Signals are always computed on the full history (strategies are causal, so this is
equivalent to computing them incrementally) and then sliced, which lets indicators
warm up before the test window starts without leaking future information.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

from backtest_lab.engine import BacktestConfig, BacktestResult, run_backtest
from backtest_lab.strategies import STRATEGIES, Strategy


def iter_grid(grid: Mapping[str, Sequence[Any]]) -> Iterator[dict[str, Any]]:
    """Yield every combination of a parameter grid as a dict."""
    if not grid:
        yield {}
        return
    keys = list(grid)
    for values in itertools.product(*(grid[k] for k in keys)):
        yield dict(zip(keys, values, strict=True))


def _resolve(strategy: str | type[Strategy]) -> type[Strategy]:
    if isinstance(strategy, str):
        if strategy not in STRATEGIES:
            raise KeyError(f"unknown strategy '{strategy}'")
        return STRATEGIES[strategy]
    return strategy


def _score(metrics: Mapping[str, float], metric: str) -> float:
    value = metrics[metric]
    return -math.inf if value is None or math.isnan(value) else float(value)


def backtest_window(
    data: pd.DataFrame,
    strategy: Strategy,
    config: BacktestConfig,
    start: int = 0,
    stop: int | None = None,
) -> BacktestResult:
    """Backtest ``strategy`` on rows ``[start, stop)`` using full-history signals."""
    signals = strategy.generate_signals(data)
    window = slice(start, stop)
    return run_backtest(data.iloc[window], signals.iloc[window], config)


def grid_search(
    data: pd.DataFrame,
    strategy: str | type[Strategy],
    grid: Mapping[str, Sequence[Any]],
    config: BacktestConfig | None = None,
    metric: str = "sharpe",
    start: int = 0,
    stop: int | None = None,
) -> pd.DataFrame:
    """Evaluate every parameter combination on rows ``[start, stop)``.

    Invalid combinations (those whose constructor raises ``ValueError``, such as
    ``fast >= slow``) are skipped.

    Returns:
        One row per combination with the parameters and all metrics, sorted by
        ``metric`` descending (NaN scores last).
    """
    cls = _resolve(strategy)
    cfg = config or BacktestConfig()
    rows: list[dict[str, Any]] = []
    for params in iter_grid(grid):
        try:
            strat = cls(**params)
        except ValueError:
            continue
        res = backtest_window(data, strat, cfg, start, stop)
        rows.append({**params, **res.metrics})
    if not rows:
        raise ValueError("no valid parameter combinations in grid")
    table = pd.DataFrame(rows)
    if metric not in table.columns:
        raise KeyError(f"unknown metric '{metric}'")
    order = table[metric].map(lambda v: _score({metric: v}, metric))
    return table.loc[order.sort_values(ascending=False, kind="stable").index].reset_index(drop=True)


@dataclass
class WalkForwardResult:
    """Outcome of :func:`walk_forward`.

    Attributes:
        best_params: parameters chosen on the train window.
        train_table: full grid-search table on the train window.
        train_metrics: metrics of ``best_params`` on the train window (in-sample).
        test_metrics: metrics of ``best_params`` on the test window (out-of-sample).
        test_result: full backtest result on the test window.
        split_time: first timestamp of the test window.
    """

    best_params: dict[str, Any]
    train_table: pd.DataFrame
    train_metrics: dict[str, float]
    test_metrics: dict[str, float]
    test_result: BacktestResult
    split_time: pd.Timestamp


def walk_forward(
    data: pd.DataFrame,
    strategy: str | type[Strategy],
    grid: Mapping[str, Sequence[Any]],
    config: BacktestConfig | None = None,
    metric: str = "sharpe",
    train_frac: float = 0.7,
) -> WalkForwardResult:
    """Pick the best parameters on the first ``train_frac`` of the data, test on the rest.

    Raises:
        ValueError: if ``train_frac`` leaves fewer than 2 bars on either side.
    """
    if not 0 < train_frac < 1:
        raise ValueError("train_frac must be between 0 and 1")
    split = int(len(data) * train_frac)
    if split < 2 or len(data) - split < 2:
        raise ValueError("not enough data for a train/test split")
    cls = _resolve(strategy)
    cfg = config or BacktestConfig()

    table = grid_search(data, cls, grid, cfg, metric, start=0, stop=split)
    param_keys = [k for k in grid if k in table.columns]
    best = {k: _py(table[k].iloc[0]) for k in param_keys}
    strat = cls(**best)
    train_res = backtest_window(data, strat, cfg, 0, split)
    test_res = backtest_window(data, strat, cfg, split, None)
    return WalkForwardResult(
        best_params=best,
        train_table=table,
        train_metrics=train_res.metrics,
        test_metrics=test_res.metrics,
        test_result=test_res,
        split_time=data.index[split],
    )


def _py(value: Any) -> Any:
    """Convert numpy scalars back to Python types for clean strategy params."""
    return value.item() if hasattr(value, "item") else value
