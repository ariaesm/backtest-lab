#!/usr/bin/env python3
"""Generate the SYNTHETIC sample dataset at data/sample_ohlcv.csv.

The data is produced by a seeded geometric Brownian motion and does not represent any
real instrument. Re-running this script with the same arguments yields identical bytes.

Usage:
    PYTHONPATH=src python3 scripts/generate_sample_data.py [--bars 1000] [--seed 42]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from backtest_lab.synthetic import generate_gbm_ohlcv

DEFAULT_OUT = Path(__file__).resolve().parents[1] / "data" / "sample_ohlcv.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bars", type=int, default=1000, help="number of daily bars")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output CSV path")
    args = parser.parse_args()

    df = generate_gbm_ohlcv(n_bars=args.bars, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.reset_index().assign(timestamp=lambda d: d["timestamp"].dt.strftime("%Y-%m-%d")).to_csv(
        args.out, index=False
    )
    print(f"wrote {len(df)} synthetic bars to {args.out}")


if __name__ == "__main__":
    main()
