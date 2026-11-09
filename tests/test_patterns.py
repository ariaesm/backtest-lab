"""Pattern detector tests on hand-crafted candles.

Each frame starts with ``WARMUP`` ordinary candles (body 2.0) so the EMA of body
sizes is well defined before the candle under test.
"""

from __future__ import annotations

import pandas as pd
import pytest

from backtest_lab import patterns as P

WARMUP = [(100.0, 102.0, 99.0, 101.0)] * 20  # white candles, body 1.0


def _last(fn, ohlcv_factory, *candles) -> bool:
    df = ohlcv_factory([*WARMUP, *candles])
    return bool(fn(df).iloc[-1])


def test_doji_detected_with_equal_shadows(ohlcv_factory) -> None:
    assert _last(P.doji, ohlcv_factory, (100.0, 102.0, 98.0, 100.05))


def test_doji_rejected_when_body_too_large(ohlcv_factory) -> None:
    # body 0.5 on a range of 4 = 12.5% > 5%
    assert not _last(P.doji, ohlcv_factory, (100.0, 102.0, 98.0, 100.5))


def test_dragonfly_and_gravestone(ohlcv_factory) -> None:
    dragonfly = (100.0, 100.05, 96.0, 100.05)
    gravestone = (100.05, 104.0, 100.0, 100.0)
    assert _last(P.dragonfly_doji, ohlcv_factory, dragonfly)
    assert not _last(P.gravestone_doji, ohlcv_factory, dragonfly)
    assert _last(P.gravestone_doji, ohlcv_factory, gravestone)
    # regular doji excludes the dragonfly/gravestone variants (like the Pine script)
    assert not _last(P.doji, ohlcv_factory, dragonfly)


def test_doji_uses_body_ema(ohlcv_factory) -> None:
    # Warm-up of tiny bodies makes the EMA body tiny, so a 4%-of-range body is no
    # longer "small" relative to the average and must not count as a doji.
    tiny = [(100.0, 102.0, 98.0, 100.0)] * 20
    df = ohlcv_factory([*tiny, (100.0, 102.0, 98.0, 100.16)])
    assert not bool(P.doji(df).iloc[-1])


def test_hammer_and_shooting_star(ohlcv_factory) -> None:
    hammer = (100.0, 100.5, 98.0, 100.5)  # body .5, lower shadow 2, no upper
    star = (100.5, 102.5, 100.0, 100.0)  # body .5, upper shadow 2, no lower
    assert _last(P.hammer, ohlcv_factory, hammer)
    assert not _last(P.shooting_star, ohlcv_factory, hammer)
    assert _last(P.shooting_star, ohlcv_factory, star)
    assert not _last(P.hammer, ohlcv_factory, star)


def test_bullish_engulfing(ohlcv_factory) -> None:
    prev = (101.0, 101.5, 99.5, 100.0)  # black
    cur = (99.8, 102.5, 99.5, 101.5)  # white, engulfs
    assert _last(P.bullish_engulfing, ohlcv_factory, prev, cur)
    assert not _last(P.bearish_engulfing, ohlcv_factory, prev, cur)


def test_bearish_engulfing(ohlcv_factory) -> None:
    prev = (100.0, 101.5, 99.5, 101.0)  # white
    cur = (101.2, 101.5, 99.0, 99.5)  # black, engulfs
    assert _last(P.bearish_engulfing, ohlcv_factory, prev, cur)


def test_engulfing_requires_full_cover(ohlcv_factory) -> None:
    prev = (101.0, 101.5, 99.5, 100.0)
    cur = (100.2, 101.0, 100.0, 100.8)  # white but inside previous body
    assert not _last(P.bullish_engulfing, ohlcv_factory, prev, cur)


def test_morning_and_evening_star(ohlcv_factory) -> None:
    morning = [
        (104.0, 104.2, 99.8, 100.0),  # long black
        (99.5, 99.9, 99.0, 99.6),  # small body below midpoint (102)
        (99.8, 103.6, 99.7, 103.5),  # long white closing above 102
    ]
    evening = [
        (100.0, 104.2, 99.8, 104.0),  # long white
        (104.4, 105.0, 104.1, 104.5),  # small body above midpoint (102)
        (104.2, 104.3, 100.3, 100.5),  # long black closing below 102
    ]
    assert _last(P.morning_star, ohlcv_factory, *morning)
    assert not _last(P.evening_star, ohlcv_factory, *morning)
    assert _last(P.evening_star, ohlcv_factory, *evening)


def test_detectors_do_not_peek_forward(random_ohlcv: pd.DataFrame) -> None:
    full = P.detect_patterns(random_ohlcv)
    cut = 250
    partial = P.detect_patterns(random_ohlcv.iloc[:cut])
    pd.testing.assert_frame_equal(full.iloc[:cut], partial)


def test_detect_patterns_columns_and_unknown(random_ohlcv: pd.DataFrame) -> None:
    out = P.detect_patterns(random_ohlcv)
    assert list(out.columns) == list(P.PATTERNS)
    assert all(out[c].dtype == bool for c in out)
    with pytest.raises(KeyError, match="unknown pattern"):
        P.detect_patterns(random_ohlcv, ["not_a_pattern"])
