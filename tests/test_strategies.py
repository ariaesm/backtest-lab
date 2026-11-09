from __future__ import annotations

from typing import Any, ClassVar

import pandas as pd
import pytest

from backtest_lab.engine import run_backtest
from backtest_lab.strategies import (
    STRATEGIES,
    SmaCrossover,
    Strategy,
    get_strategy,
    hold_for,
    register_strategy,
)


def test_registry_contains_builtins() -> None:
    assert {"doji_reversal", "engulfing_trend", "sma_crossover"} <= set(STRATEGIES)


@pytest.mark.parametrize("name", ["doji_reversal", "engulfing_trend", "sma_crossover"])
def test_builtin_signals_are_valid_and_causal(name: str, random_ohlcv: pd.DataFrame) -> None:
    strat = get_strategy(name)
    sig = strat.generate_signals(random_ohlcv)
    assert sig.index.equals(random_ohlcv.index)
    assert set(sig.unique()) <= {-1.0, 0.0, 1.0}
    cut = 260
    partial = strat.generate_signals(random_ohlcv.iloc[:cut])
    pd.testing.assert_series_equal(sig.iloc[:cut], partial)
    run_backtest(random_ohlcv, sig)  # runs end to end


def test_sma_crossover_logic(ohlcv_factory) -> None:
    closes = [10, 10, 10, 11, 12, 13, 12, 11, 10, 9]
    df = ohlcv_factory([(c, c + 0.5, c - 0.5, c) for c in closes])
    sig = SmaCrossover(fast=2, slow=3, allow_short=True).generate_signals(df)
    fast = pd.Series(closes, dtype=float).rolling(2).mean()
    slow = pd.Series(closes, dtype=float).rolling(3).mean()
    for i in range(2, len(closes)):
        expected = 1.0 if fast[i] > slow[i] else (-1.0 if fast[i] < slow[i] else 0.0)
        assert sig.iloc[i] == expected
    assert sig.iloc[0] == sig.iloc[1] == 0.0


def test_long_only_flags(random_ohlcv: pd.DataFrame) -> None:
    for name in ("doji_reversal", "engulfing_trend", "sma_crossover"):
        sig = get_strategy(name, allow_short=False).generate_signals(random_ohlcv)
        assert (sig >= 0).all()


def test_param_validation() -> None:
    with pytest.raises(ValueError, match="unknown parameter"):
        get_strategy("sma_crossover", speed=3)
    with pytest.raises(ValueError, match="fast must be smaller"):
        SmaCrossover(fast=50, slow=20)
    with pytest.raises(ValueError, match="positive integer"):
        get_strategy("doji_reversal", hold_bars=0)
    with pytest.raises(KeyError, match="unknown strategy"):
        get_strategy("does_not_exist")


def test_hold_for() -> None:
    entries = pd.Series([0, 1, 0, 0, 0, -1, 0, 0])
    assert list(hold_for(entries, 3)) == [0, 1, 1, 1, 0, -1, -1, -1]
    assert list(hold_for(entries, 1)) == [0, 1, 0, 0, 0, -1, 0, 0]


def test_custom_strategy_registration(random_ohlcv: pd.DataFrame) -> None:
    @register_strategy
    class AlwaysLong(Strategy):
        name = "always_long_test"
        default_params: ClassVar[dict[str, Any]] = {}

        def generate_signals(self, data: pd.DataFrame) -> pd.Series:
            return pd.Series(1.0, index=data.index)

    try:
        res = run_backtest(
            random_ohlcv, get_strategy("always_long_test").generate_signals(random_ohlcv)
        )
        assert res.metrics["exposure"] == pytest.approx(1 - 1 / len(random_ohlcv))
    finally:
        STRATEGIES.pop("always_long_test")
