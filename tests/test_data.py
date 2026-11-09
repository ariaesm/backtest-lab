from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from backtest_lab.data import DataValidationError, load_ohlcv, validate_ohlcv

GOOD_CSV = """timestamp,open,high,low,close,volume
2024-01-02,10,11,9.5,10.5,100
2024-01-01,9,10.2,8.8,10,120
2024-01-03,10.5,10.9,10.1,10.2,90
"""


def _write(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "data.csv"
    p.write_text(text)
    return p


def test_load_valid_csv_sorts_and_indexes(tmp_path: Path) -> None:
    df = load_ohlcv(_write(tmp_path, GOOD_CSV))
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.is_monotonic_increasing
    assert df.iloc[0]["close"] == 10.0


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_ohlcv(tmp_path / "nope.csv")


def test_missing_column_message(tmp_path: Path) -> None:
    text = "timestamp,open,high,low,close\n2024-01-01,1,2,0.5,1.5\n"
    with pytest.raises(DataValidationError, match="volume"):
        load_ohlcv(_write(tmp_path, text))


def test_non_numeric_value(tmp_path: Path) -> None:
    text = GOOD_CSV.replace("10.5,100", "abc,100")
    with pytest.raises(DataValidationError, match="close"):
        load_ohlcv(_write(tmp_path, text))


def test_high_below_close_is_rejected(tmp_path: Path) -> None:
    text = GOOD_CSV.replace("10,11,9.5,10.5", "10,10.2,9.5,10.5")
    with pytest.raises(DataValidationError, match="violate"):
        load_ohlcv(_write(tmp_path, text))


def test_duplicate_timestamp_rejected(tmp_path: Path) -> None:
    text = GOOD_CSV.replace("2024-01-03", "2024-01-02")
    with pytest.raises(DataValidationError, match="duplicate"):
        load_ohlcv(_write(tmp_path, text))


def test_non_positive_price_rejected() -> None:
    df = pd.DataFrame(
        {
            "timestamp": ["2024-01-01"],
            "open": [0.0],
            "high": [1.0],
            "low": [0.0],
            "close": [0.5],
            "volume": [1],
        }
    )
    with pytest.raises(DataValidationError, match="positive"):
        validate_ohlcv(df)


def test_accepts_datetime_index(random_ohlcv: pd.DataFrame) -> None:
    out = validate_ohlcv(random_ohlcv)
    pd.testing.assert_index_equal(out.index, random_ohlcv.index)


def test_empty_file(tmp_path: Path) -> None:
    with pytest.raises(DataValidationError):
        load_ohlcv(_write(tmp_path, ""))
