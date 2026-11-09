"""Event-driven backtest engine.

Timing model (no look-ahead bias):

* A signal is a *target direction* (``+1`` long, ``-1`` short, ``0`` flat) computed
  from information available at the **close** of bar ``t``.
* The engine acts on that signal at the **open of bar t + 1**. A signal on the last
  bar is therefore never executed.
* Stop-loss / take-profit levels are checked intrabar from bar ``t + 1`` onward using
  the bar's high and low. If both are touched in the same bar, the stop is assumed to
  fill first (conservative). Gaps through a level fill at the open.
* After a stop or target exit the engine stays flat until the signal changes value.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from backtest_lab.metrics import TRADING_DAYS, compute_metrics


@dataclass(frozen=True)
class BacktestConfig:
    """Execution and risk settings.

    Attributes:
        initial_cash: starting equity.
        fraction: fraction of current equity committed to each new position (0, 1].
        commission: commission as a fraction of traded notional, charged per side.
        slippage: adverse price slippage as a fraction of the fill price, per side.
        stop_loss: optional stop distance as a fraction of entry price (e.g. 0.05).
        take_profit: optional target distance as a fraction of entry price.
        allow_short: if ``False`` short signals are treated as flat.
    """

    initial_cash: float = 10_000.0
    fraction: float = 1.0
    commission: float = 0.0005
    slippage: float = 0.0005
    stop_loss: float | None = None
    take_profit: float | None = None
    allow_short: bool = True

    def __post_init__(self) -> None:
        if self.initial_cash <= 0:
            raise ValueError("initial_cash must be positive")
        if not 0 < self.fraction <= 1:
            raise ValueError("fraction must be in (0, 1]")
        if self.commission < 0 or self.slippage < 0:
            raise ValueError("commission and slippage must be non-negative")
        for name in ("stop_loss", "take_profit"):
            v = getattr(self, name)
            if v is not None and v <= 0:
                raise ValueError(f"{name} must be positive when set")


@dataclass
class BacktestResult:
    """Output of :func:`run_backtest`.

    Attributes:
        equity: mark-to-market equity at each bar close.
        position: signed quantity held at each bar close.
        trades: one row per closed trade.
        signals: the (clipped) target signals that were fed to the engine.
        config: the configuration used.
        metrics: summary statistics from :func:`backtest_lab.metrics.compute_metrics`.
    """

    equity: pd.Series
    position: pd.Series
    trades: pd.DataFrame
    signals: pd.Series
    config: BacktestConfig
    metrics: dict[str, float] = field(default_factory=dict)


TRADE_COLUMNS = [
    "entry_time",
    "exit_time",
    "side",
    "qty",
    "entry_price",
    "exit_price",
    "pnl",
    "return_pct",
    "bars_held",
    "exit_reason",
]


def _prepare_signals(signals: pd.Series, index: pd.Index, allow_short: bool) -> np.ndarray:
    if not isinstance(signals, pd.Series):
        raise TypeError("signals must be a pandas Series")
    if not signals.index.equals(index):
        raise ValueError("signals must share the data index")
    s = np.sign(signals.fillna(0).to_numpy(dtype=float))
    if not allow_short:
        s = np.where(s < 0, 0.0, s)
    return s


def run_backtest(
    data: pd.DataFrame,
    signals: pd.Series,
    config: BacktestConfig | None = None,
    *,
    periods_per_year: int = TRADING_DAYS,
) -> BacktestResult:
    """Simulate trading ``signals`` on ``data``.

    Args:
        data: validated OHLCV frame (see :func:`backtest_lab.data.validate_ohlcv`).
        signals: target direction per bar (values are clipped to -1/0/+1; NaN = 0).
        config: execution settings; defaults to :class:`BacktestConfig`.
        periods_per_year: bars per year, used to annualise metrics.

    Returns:
        A :class:`BacktestResult` with equity curve, positions and trade log.
    """
    cfg = config or BacktestConfig()
    sig = _prepare_signals(signals, data.index, cfg.allow_short)
    o = data["open"].to_numpy(dtype=float)
    h = data["high"].to_numpy(dtype=float)
    lo = data["low"].to_numpy(dtype=float)
    c = data["close"].to_numpy(dtype=float)
    idx = data.index
    n = len(data)

    cash = cfg.initial_cash
    qty = 0.0  # signed
    side = 0
    entry_price = 0.0
    entry_bar = -1
    entry_cost = 0.0  # commission paid on entry
    blocked_signal: float | None = None  # signal value suppressed after SL/TP exit
    equity = np.empty(n)
    position = np.empty(n)
    trades: list[dict[str, object]] = []

    def fill_price(raw: float, direction: int) -> float:
        """Apply slippage: buying (direction>0) pays more, selling receives less."""
        return raw * (1 + cfg.slippage * direction)

    def close_position(bar: int, raw_price: float, reason: str) -> None:
        nonlocal cash, qty, side, entry_price, entry_bar, entry_cost
        px = fill_price(raw_price, -side)
        notional = abs(qty) * px
        fee = notional * cfg.commission
        cash += qty * px - fee
        pnl = qty * (px - entry_price) - fee - entry_cost
        trades.append(
            {
                "entry_time": idx[entry_bar],
                "exit_time": idx[bar],
                "side": "long" if side > 0 else "short",
                "qty": abs(qty),
                "entry_price": entry_price,
                "exit_price": px,
                "pnl": pnl,
                "return_pct": pnl / (abs(qty) * entry_price),
                "bars_held": bar - entry_bar,
                "exit_reason": reason,
            }
        )
        qty, side, entry_price, entry_bar, entry_cost = 0.0, 0, 0.0, -1, 0.0

    def open_position(bar: int, direction: int) -> None:
        nonlocal cash, qty, side, entry_price, entry_bar, entry_cost
        px = fill_price(o[bar], direction)
        equity_now = cash  # flat at this point
        notional = equity_now * cfg.fraction / (1 + cfg.commission)
        q = notional / px
        fee = q * px * cfg.commission
        cash -= direction * q * px + fee
        qty, side, entry_price, entry_bar, entry_cost = direction * q, direction, px, bar, fee

    for i in range(n):
        # 1) act at the open on the previous bar's signal
        if i > 0:
            target = int(sig[i - 1])
            if blocked_signal is not None and sig[i - 1] != blocked_signal:
                blocked_signal = None
            if blocked_signal is not None:
                target = side  # stay as we are until the signal changes
            if target != side:
                if side != 0:
                    close_position(i, o[i], "signal")
                if target != 0:
                    open_position(i, target)

        # 2) intrabar stop-loss / take-profit
        if side != 0 and (cfg.stop_loss or cfg.take_profit):
            exit_px: float | None = None
            reason = ""
            if side > 0:
                stop = entry_price * (1 - cfg.stop_loss) if cfg.stop_loss else None
                tp = entry_price * (1 + cfg.take_profit) if cfg.take_profit else None
                if stop is not None and lo[i] <= stop:
                    exit_px, reason = min(o[i], stop) if i > entry_bar else stop, "stop_loss"
                elif tp is not None and h[i] >= tp:
                    exit_px, reason = max(o[i], tp) if i > entry_bar else tp, "take_profit"
            else:
                stop = entry_price * (1 + cfg.stop_loss) if cfg.stop_loss else None
                tp = entry_price * (1 - cfg.take_profit) if cfg.take_profit else None
                if stop is not None and h[i] >= stop:
                    exit_px, reason = max(o[i], stop) if i > entry_bar else stop, "stop_loss"
                elif tp is not None and lo[i] <= tp:
                    exit_px, reason = min(o[i], tp) if i > entry_bar else tp, "take_profit"
            if exit_px is not None:
                close_position(i, exit_px, reason)
                blocked_signal = sig[i - 1] if i > 0 else None

        # 3) mark to market at the close
        equity[i] = cash + qty * c[i]
        position[i] = qty

    if side != 0:
        # Record the open trade at the final close for reporting. Equity already
        # reflects it marked-to-market; exit costs are included in the trade P&L only.
        saved_cash = cash
        close_position(n - 1, c[-1], "end_of_data")
        cash = saved_cash

    trades_df = pd.DataFrame(trades, columns=TRADE_COLUMNS)
    equity_s = pd.Series(equity, index=idx, name="equity")
    position_s = pd.Series(position, index=idx, name="position")
    return BacktestResult(
        equity=equity_s,
        position=position_s,
        trades=trades_df,
        signals=pd.Series(sig, index=idx, name="signal"),
        config=cfg,
        metrics=compute_metrics(equity_s, trades_df, position_s, periods_per_year=periods_per_year),
    )
