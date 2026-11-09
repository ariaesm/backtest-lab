from __future__ import annotations

import math

import pandas as pd
import pytest

from backtest_lab.engine import BacktestConfig
from backtest_lab.optimize import backtest_window, grid_search, iter_grid, walk_forward
from backtest_lab.strategies import SmaCrossover

GRID = {"fast": [5, 10, 30], "slow": [20, 40]}


def test_iter_grid() -> None:
    combos = list(iter_grid({"a": [1, 2], "b": ["x"]}))
    assert combos == [{"a": 1, "b": "x"}, {"a": 2, "b": "x"}]
    assert list(iter_grid({})) == [{}]


def test_grid_search_skips_invalid_and_sorts(random_ohlcv: pd.DataFrame) -> None:
    table = grid_search(random_ohlcv, "sma_crossover", GRID)
    # (30, 20) is invalid because fast >= slow
    assert len(table) == 5
    assert not ((table["fast"] == 30) & (table["slow"] == 20)).any()
    scores = table["sharpe"].tolist()
    assert scores == sorted(scores, reverse=True)


def test_grid_search_unknown_metric(random_ohlcv: pd.DataFrame) -> None:
    with pytest.raises(KeyError):
        grid_search(random_ohlcv, "sma_crossover", GRID, metric="magic")


def test_walk_forward_uses_train_window_only(random_ohlcv: pd.DataFrame) -> None:
    cfg = BacktestConfig()
    wf = walk_forward(random_ohlcv, "sma_crossover", GRID, cfg, train_frac=0.6)
    split = int(len(random_ohlcv) * 0.6)
    assert wf.split_time == random_ohlcv.index[split]
    assert wf.test_result.equity.index[0] == random_ohlcv.index[split]
    # best params are the top of the train-only grid search
    train_only = grid_search(random_ohlcv, "sma_crossover", GRID, cfg, stop=split)
    assert wf.best_params == {
        "fast": int(train_only.iloc[0]["fast"]),
        "slow": int(train_only.iloc[0]["slow"]),
    }
    # changing data in the test window must not change the selected parameters
    shocked = random_ohlcv.copy()
    shocked.iloc[split:, :4] *= 2.0
    wf2 = walk_forward(shocked, "sma_crossover", GRID, cfg, train_frac=0.6)
    assert wf2.best_params == wf.best_params
    assert math.isfinite(wf.test_metrics["total_return"])


def test_backtest_window_matches_signal_slice(random_ohlcv: pd.DataFrame) -> None:
    strat = SmaCrossover(fast=5, slow=20)
    res = backtest_window(random_ohlcv, strat, BacktestConfig(), 100, 200)
    assert len(res.equity) == 100
    expected = strat.generate_signals(random_ohlcv).iloc[100:200]
    assert (res.signals.to_numpy() == expected.to_numpy()).all()


def test_walk_forward_bad_fraction(random_ohlcv: pd.DataFrame) -> None:
    with pytest.raises(ValueError):
        walk_forward(random_ohlcv, "sma_crossover", GRID, train_frac=1.5)
