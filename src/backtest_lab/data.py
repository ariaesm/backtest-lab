"""Loading and validation of OHLCV price data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS: tuple[str, ...] = ("timestamp", "open", "high", "low", "close", "volume")
PRICE_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close")


class DataValidationError(ValueError):
    """Raised when OHLCV input data is malformed."""


def validate_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalise an OHLCV frame.

    The input may either carry a ``timestamp`` column or already be indexed by a
    :class:`pandas.DatetimeIndex`. The returned frame is indexed by timestamp, sorted,
    has float price columns and contains exactly ``open, high, low, close, volume``.

    Raises:
        DataValidationError: if columns are missing, values are non-numeric, NaN,
            non-positive, internally inconsistent (e.g. ``high < low``) or timestamps
            are unparsable or duplicated.
    """
    if not isinstance(df, pd.DataFrame):
        raise DataValidationError(f"expected a pandas DataFrame, got {type(df).__name__}")

    frame = df.copy()
    frame.columns = [str(c).strip().lower() for c in frame.columns]

    if "timestamp" not in frame.columns:
        if isinstance(frame.index, pd.DatetimeIndex):
            frame = frame.rename_axis("timestamp").reset_index()
        else:
            raise DataValidationError("missing required column(s): timestamp")

    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise DataValidationError(f"missing required column(s): {', '.join(missing)}")

    if frame.empty:
        raise DataValidationError("data contains no rows")

    try:
        ts = pd.to_datetime(frame["timestamp"], utc=False)
    except (ValueError, TypeError) as exc:
        raise DataValidationError(f"could not parse 'timestamp' column: {exc}") from exc

    out = pd.DataFrame(index=pd.DatetimeIndex(ts, name="timestamp"))
    for col in (*PRICE_COLUMNS, "volume"):
        values = pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=float)
        bad = np.isnan(values)
        if bad.any():
            first = int(np.flatnonzero(bad)[0])
            raise DataValidationError(
                f"column '{col}' has {int(bad.sum())} missing or non-numeric value(s) "
                f"(first at row {first})"
            )
        out[col] = values

    if out.index.has_duplicates:
        dup = out.index[out.index.duplicated()][0]
        raise DataValidationError(f"duplicate timestamp: {dup}")

    out = out.sort_index()

    if (out[list(PRICE_COLUMNS)] <= 0).any().any():
        raise DataValidationError("prices must be strictly positive")
    if (out["volume"] < 0).any():
        raise DataValidationError("volume must be non-negative")

    body_hi = out[["open", "close"]].max(axis=1)
    body_lo = out[["open", "close"]].min(axis=1)
    inconsistent = (out["high"] < body_hi) | (out["low"] > body_lo) | (out["high"] < out["low"])
    if inconsistent.any():
        first_ts = out.index[inconsistent.to_numpy()][0]
        raise DataValidationError(
            f"{int(inconsistent.sum())} bar(s) violate low <= open/close <= high "
            f"(first at {first_ts})"
        )
    return out


def load_ohlcv(path: str | Path) -> pd.DataFrame:
    """Load an OHLCV CSV file and validate it.

    The CSV must contain the columns ``timestamp, open, high, low, close, volume``
    (case-insensitive). See :func:`validate_ohlcv` for the checks performed.

    Raises:
        FileNotFoundError: if ``path`` does not exist.
        DataValidationError: if the content is invalid.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"data file not found: {p}")
    try:
        raw = pd.read_csv(p)
    except pd.errors.EmptyDataError as exc:
        raise DataValidationError(f"{p} is empty") from exc
    return validate_ohlcv(raw)
