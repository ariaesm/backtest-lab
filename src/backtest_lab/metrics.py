"""Performance metrics computed from an equity curve and a trade log.

Conventions:

* Returns are simple per-bar returns ``r_t = E_t / E_{t-1} - 1``.
* Annualisation uses ``periods_per_year`` (252 for daily bars).
* Volatility uses the sample standard deviation (``ddof=1``).
* Sortino uses the downside deviation ``sqrt(mean(min(r - target, 0)^2))`` over all
  periods.
* Max drawdown is reported as a negative fraction (``-0.25`` = 25% drawdown).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def returns_from_equity(equity: pd.Series) -> pd.Series:
    """Per-bar simple returns, first bar dropped."""
    return equity.pct_change().iloc[1:]


def total_return(equity: pd.Series) -> float:
    """``E_end / E_start - 1``."""
    return float(equity.iloc[-1] / equity.iloc[0] - 1)


def cagr(equity: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    """Compound annual growth rate over ``len(equity) - 1`` periods."""
    periods = len(equity) - 1
    if periods <= 0:
        return 0.0
    growth = equity.iloc[-1] / equity.iloc[0]
    if growth <= 0:
        return -1.0
    return float(growth ** (periods_per_year / periods) - 1)


def annualized_volatility(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    """Sample standard deviation of returns scaled by ``sqrt(periods_per_year)``."""
    if len(returns) < 2:
        return 0.0
    return float(returns.std(ddof=1) * math.sqrt(periods_per_year))


def sharpe_ratio(
    returns: pd.Series, risk_free: float = 0.0, periods_per_year: int = TRADING_DAYS
) -> float:
    """Annualised Sharpe ratio; ``risk_free`` is an annual rate. 0 if volatility is 0."""
    if len(returns) < 2:
        return 0.0
    excess = returns - risk_free / periods_per_year
    sd = excess.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return 0.0
    return float(excess.mean() / sd * math.sqrt(periods_per_year))


def sortino_ratio(
    returns: pd.Series, risk_free: float = 0.0, periods_per_year: int = TRADING_DAYS
) -> float:
    """Annualised Sortino ratio. ``inf`` if there are no losing periods and mean > 0."""
    if len(returns) == 0:
        return 0.0
    excess = returns - risk_free / periods_per_year
    downside = np.sqrt(np.mean(np.minimum(excess.to_numpy(), 0.0) ** 2))
    mean = excess.mean()
    if downside == 0:
        return math.inf if mean > 0 else 0.0
    return float(mean / downside * math.sqrt(periods_per_year))


def drawdown_series(equity: pd.Series) -> pd.Series:
    """Drawdown from the running peak at each bar (<= 0)."""
    return equity / equity.cummax() - 1


def max_drawdown(equity: pd.Series) -> float:
    """Most negative drawdown (e.g. ``-0.2`` for a 20% peak-to-trough loss)."""
    return float(drawdown_series(equity).min())


def calmar_ratio(equity: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    """CAGR divided by the absolute max drawdown (``inf`` with no drawdown)."""
    mdd = max_drawdown(equity)
    growth = cagr(equity, periods_per_year)
    if mdd == 0:
        return math.inf if growth > 0 else 0.0
    return float(growth / abs(mdd))


def win_rate(trades: pd.DataFrame) -> float:
    """Fraction of trades with positive P&L (NaN when there are no trades)."""
    if trades.empty:
        return math.nan
    return float((trades["pnl"] > 0).mean())


def profit_factor(trades: pd.DataFrame) -> float:
    """Gross profit / gross loss (``inf`` with no losers, NaN with no trades)."""
    if trades.empty:
        return math.nan
    gains = trades.loc[trades["pnl"] > 0, "pnl"].sum()
    losses = -trades.loc[trades["pnl"] < 0, "pnl"].sum()
    if losses == 0:
        return math.inf if gains > 0 else math.nan
    return float(gains / losses)


def exposure(position: pd.Series) -> float:
    """Fraction of bars ending with an open position."""
    if len(position) == 0:
        return 0.0
    return float((position != 0).mean())


def compute_metrics(
    equity: pd.Series,
    trades: pd.DataFrame,
    position: pd.Series,
    *,
    periods_per_year: int = TRADING_DAYS,
    risk_free: float = 0.0,
) -> dict[str, float]:
    """Compute every metric and return them in a stable, display-friendly order."""
    rets = returns_from_equity(equity)
    return {
        "total_return": total_return(equity),
        "cagr": cagr(equity, periods_per_year),
        "volatility": annualized_volatility(rets, periods_per_year),
        "sharpe": sharpe_ratio(rets, risk_free, periods_per_year),
        "sortino": sortino_ratio(rets, risk_free, periods_per_year),
        "max_drawdown": max_drawdown(equity),
        "calmar": calmar_ratio(equity, periods_per_year),
        "win_rate": win_rate(trades),
        "profit_factor": profit_factor(trades),
        "num_trades": float(len(trades)),
        "exposure": exposure(position),
    }


_PERCENT = {"total_return", "cagr", "volatility", "max_drawdown", "win_rate", "exposure"}


def format_metrics(metrics: dict[str, float]) -> str:
    """Render metrics as an aligned two-column text table."""
    width = max(len(k) for k in metrics)
    lines = []
    for key, value in metrics.items():
        if key == "num_trades":
            text = f"{int(value)}"
        elif isinstance(value, float) and math.isnan(value):
            text = "n/a"
        elif key in _PERCENT:
            text = f"{value * 100:.2f}%"
        else:
            text = f"{value:.2f}"
        lines.append(f"{key:<{width}}  {text:>10}")
    return "\n".join(lines)
