from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest_lab.engine import BacktestConfig, run_backtest

FREE = BacktestConfig(commission=0.0, slippage=0.0)


def _flat_bars(ohlcv_factory, opens: list[float]):
    """Bars whose open == close == the given price, with +-1 wicks."""
    return ohlcv_factory([(p, p + 1, p - 1, p) for p in opens])


def _sig(df: pd.DataFrame, values: list[float]) -> pd.Series:
    return pd.Series(values, index=df.index, dtype=float)


def test_signal_executes_on_next_bar_open(ohlcv_factory) -> None:
    df = ohlcv_factory(
        [
            (100, 101, 99, 100),
            (105, 106, 104, 105),
            (110, 111, 109, 110),
            (120, 121, 119, 120),
        ]
    )
    res = run_backtest(df, _sig(df, [1, 0, 0, 0]), FREE)
    t = res.trades.iloc[0]
    assert t["entry_time"] == df.index[1]
    assert t["entry_price"] == 105  # open of the NEXT bar, not close of the signal bar
    assert t["exit_time"] == df.index[2]
    assert t["exit_price"] == 110
    assert res.position.iloc[0] == 0


def test_no_look_ahead_truncation_invariance(random_ohlcv: pd.DataFrame) -> None:
    """Results up to bar t must not depend on any data after bar t."""
    sma = random_ohlcv["close"].rolling(10).mean()
    signals = np.sign(random_ohlcv["close"] - sma).fillna(0)
    cfg = BacktestConfig(stop_loss=0.03, take_profit=0.05)
    full = run_backtest(random_ohlcv, signals, cfg)
    cut = 237
    part_df = random_ohlcv.iloc[:cut]
    part = run_backtest(part_df, signals.iloc[:cut], cfg)
    pd.testing.assert_series_equal(full.equity.iloc[:cut], part.equity)
    pd.testing.assert_series_equal(full.position.iloc[:cut], part.position)


def test_future_data_change_does_not_affect_past(random_ohlcv: pd.DataFrame) -> None:
    """Scrambling future prices leaves the past equity curve untouched."""
    signals = np.sign(random_ohlcv["close"].diff()).fillna(0)
    base = run_backtest(random_ohlcv, signals, FREE)
    shocked = random_ohlcv.copy()
    shocked.iloc[300:, :4] *= 1.5
    shocked_signals = np.sign(shocked["close"].diff()).fillna(0)
    alt = run_backtest(shocked, shocked_signals, FREE)
    pd.testing.assert_series_equal(base.equity.iloc[:300], alt.equity.iloc[:300])


def test_signal_on_last_bar_is_never_executed(ohlcv_factory) -> None:
    df = _flat_bars(ohlcv_factory, [100, 101, 102])
    res = run_backtest(df, _sig(df, [0, 0, 1]), FREE)
    assert res.trades.empty
    assert (res.equity == FREE.initial_cash).all()


def test_commission_and_slippage_math(ohlcv_factory) -> None:
    df = _flat_bars(ohlcv_factory, [100, 100, 110, 110])
    cfg = BacktestConfig(initial_cash=1000, commission=0.001, slippage=0.002)
    res = run_backtest(df, _sig(df, [1, 0, 0, 0]), cfg)
    buy = 100 * 1.002
    qty = 1000 / (1 + 0.001) / buy
    sell = 110 * (1 - 0.002)
    expected = 1000 - qty * buy * 0.001 + qty * (sell - buy) - qty * sell * 0.001
    assert res.equity.iloc[-1] == pytest.approx(expected, rel=1e-12)
    assert res.trades.iloc[0]["pnl"] == pytest.approx(expected - 1000, rel=1e-9)


def test_short_profits_when_price_falls(ohlcv_factory) -> None:
    df = _flat_bars(ohlcv_factory, [100, 100, 90, 90])
    res = run_backtest(df, _sig(df, [-1, 0, 0, 0]), FREE)
    assert res.trades.iloc[0]["side"] == "short"
    assert res.equity.iloc[-1] == pytest.approx(11_000)


def test_allow_short_false_ignores_short_signals(ohlcv_factory) -> None:
    df = _flat_bars(ohlcv_factory, [100, 100, 90, 90])
    cfg = BacktestConfig(commission=0, slippage=0, allow_short=False)
    res = run_backtest(df, _sig(df, [-1, -1, -1, -1]), cfg)
    assert res.trades.empty


def test_position_sizing_fraction(ohlcv_factory) -> None:
    df = _flat_bars(ohlcv_factory, [100, 100, 100])
    cfg = BacktestConfig(commission=0, slippage=0, fraction=0.25)
    res = run_backtest(df, _sig(df, [1, 1, 1]), cfg)
    assert res.position.iloc[1] == pytest.approx(0.25 * 10_000 / 100)


def test_stop_loss_fills_at_stop_level(ohlcv_factory) -> None:
    df = ohlcv_factory(
        [(100, 101, 99, 100), (100, 101, 99, 100), (99, 99.5, 90, 92), (92, 93, 91, 92)]
    )
    cfg = BacktestConfig(commission=0, slippage=0, stop_loss=0.05)
    res = run_backtest(df, _sig(df, [1, 1, 1, 1]), cfg)
    t = res.trades.iloc[0]
    assert t["exit_reason"] == "stop_loss"
    assert t["exit_price"] == pytest.approx(95.0)
    # signal unchanged after the stop -> no re-entry
    assert len(res.trades) == 1
    assert res.position.iloc[-1] == 0


def test_stop_loss_gap_fills_at_open(ohlcv_factory) -> None:
    df = ohlcv_factory([(100, 101, 99, 100), (100, 101, 99, 100), (85, 86, 84, 85)])
    cfg = BacktestConfig(commission=0, slippage=0, stop_loss=0.05)
    res = run_backtest(df, _sig(df, [1, 1, 1]), cfg)
    assert res.trades.iloc[0]["exit_price"] == pytest.approx(85.0)


def test_take_profit_short(ohlcv_factory) -> None:
    df = ohlcv_factory([(100, 101, 99, 100), (100, 101, 99, 100), (99, 100, 88, 89)])
    cfg = BacktestConfig(commission=0, slippage=0, take_profit=0.1)
    res = run_backtest(df, _sig(df, [-1, -1, -1]), cfg)
    t = res.trades.iloc[0]
    assert t["exit_reason"] == "take_profit"
    assert t["exit_price"] == pytest.approx(90.0)
    assert t["pnl"] == pytest.approx(100 * 10)


def test_reentry_after_stop_requires_new_signal(ohlcv_factory) -> None:
    df = ohlcv_factory(
        [
            (100, 101, 99, 100),
            (100, 101, 90, 92),  # stop hit on entry bar
            (92, 93, 91, 92),
            (92, 93, 91, 92),
            (92, 93, 91, 92),
        ]
    )
    cfg = BacktestConfig(commission=0, slippage=0, stop_loss=0.05)
    res = run_backtest(df, _sig(df, [1, 1, 0, 1, 1]), cfg)
    assert list(res.trades["exit_reason"]) == ["stop_loss", "end_of_data"]
    assert res.trades.iloc[1]["entry_time"] == df.index[4]


def test_reversal_flips_position(ohlcv_factory) -> None:
    df = _flat_bars(ohlcv_factory, [100, 100, 100, 100])
    res = run_backtest(df, _sig(df, [1, -1, -1, -1]), FREE)
    assert res.position.iloc[1] > 0 > res.position.iloc[2]
    assert list(res.trades["side"]) == ["long", "short"]


def test_config_validation() -> None:
    with pytest.raises(ValueError):
        BacktestConfig(fraction=0)
    with pytest.raises(ValueError):
        BacktestConfig(stop_loss=-0.1)
    with pytest.raises(ValueError):
        BacktestConfig(commission=-1)


def test_signal_index_must_match(random_ohlcv: pd.DataFrame) -> None:
    bad = pd.Series(0.0, index=range(len(random_ohlcv)))
    with pytest.raises(ValueError, match="index"):
        run_backtest(random_ohlcv, bad)
