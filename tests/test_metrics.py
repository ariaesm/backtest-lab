"""Metric tests against values computed by hand."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from backtest_lab import metrics as M


def _eq(values: list[float]) -> pd.Series:
    return pd.Series(values, index=pd.date_range("2024-01-01", periods=len(values)))


def test_total_return_and_cagr() -> None:
    eq = _eq([100, 110, 121])
    assert M.total_return(eq) == pytest.approx(0.21)
    # 2 periods with periods_per_year=2 -> exactly one year -> CAGR == total return
    assert M.cagr(eq, periods_per_year=2) == pytest.approx(0.21)
    # 2 periods with periods_per_year=1 -> two years -> sqrt(1.21) - 1 = 10%
    assert M.cagr(eq, periods_per_year=1) == pytest.approx(0.10)


def test_volatility_and_sharpe_hand_computed() -> None:
    # returns: +10%, -10%, +10%, -10%
    eq = _eq([100, 110, 99, 108.9, 98.01])
    r = M.returns_from_equity(eq)
    assert list(r.round(10)) == [0.1, -0.1, 0.1, -0.1]
    # mean = 0, sample std = sqrt(4 * 0.01 / 3) = 0.11547
    sd = math.sqrt(0.04 / 3)
    assert M.annualized_volatility(r, periods_per_year=4) == pytest.approx(sd * 2)
    assert M.sharpe_ratio(r, periods_per_year=4) == pytest.approx(0.0, abs=1e-12)


def test_sharpe_positive_mean() -> None:
    r = pd.Series([0.02, 0.0, 0.01])
    # mean = 0.01, sample std = 0.01 -> per-period Sharpe 1, annualised sqrt(252)
    assert M.sharpe_ratio(r) == pytest.approx(math.sqrt(252))


def test_sortino_hand_computed() -> None:
    r = pd.Series([0.03, -0.01, 0.02, -0.02])
    # mean = 0.005; downside dev = sqrt((0.0001 + 0.0004) / 4) = sqrt(0.000125)
    expected = 0.005 / math.sqrt(0.000125) * math.sqrt(252)
    assert M.sortino_ratio(r) == pytest.approx(expected)
    assert M.sortino_ratio(pd.Series([0.01, 0.02])) == math.inf


def test_max_drawdown_and_calmar() -> None:
    eq = _eq([100, 120, 90, 130, 104])
    # peak 120 -> 90 = -25%; peak 130 -> 104 = -20%
    assert M.max_drawdown(eq) == pytest.approx(-0.25)
    dd = M.drawdown_series(eq)
    assert dd.iloc[4] == pytest.approx(-0.2)
    expected_cagr = (104 / 100) ** (4 / 4) - 1
    assert M.calmar_ratio(eq, periods_per_year=4) == pytest.approx(expected_cagr / 0.25)


def test_trade_metrics() -> None:
    trades = pd.DataFrame({"pnl": [100.0, -50.0, 30.0, -25.0]})
    assert M.win_rate(trades) == pytest.approx(0.5)
    assert M.profit_factor(trades) == pytest.approx(130 / 75)
    empty = pd.DataFrame({"pnl": []})
    assert math.isnan(M.win_rate(empty))
    assert math.isnan(M.profit_factor(empty))
    assert M.profit_factor(pd.DataFrame({"pnl": [5.0]})) == math.inf


def test_exposure_and_compute_metrics() -> None:
    pos = pd.Series([0, 1.0, 1.0, 0, -2.0])
    assert M.exposure(pos) == pytest.approx(0.6)
    eq = _eq([100, 101, 102, 101, 103])
    m = M.compute_metrics(eq, pd.DataFrame({"pnl": [1.0, -0.5]}), pos)
    assert list(m) == [
        "total_return",
        "cagr",
        "volatility",
        "sharpe",
        "sortino",
        "max_drawdown",
        "calmar",
        "win_rate",
        "profit_factor",
        "num_trades",
        "exposure",
    ]
    assert m["num_trades"] == 2
    text = M.format_metrics(m)
    assert "total_return" in text and "3.00%" in text
