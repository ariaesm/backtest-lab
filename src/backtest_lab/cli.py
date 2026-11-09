"""Command-line interface: ``backtest-lab {run,patterns,optimize,strategies}``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from backtest_lab import __version__
from backtest_lab.data import DataValidationError, load_ohlcv
from backtest_lab.engine import BacktestConfig, run_backtest
from backtest_lab.metrics import format_metrics
from backtest_lab.optimize import walk_forward
from backtest_lab.patterns import PATTERNS, detect_patterns
from backtest_lab.strategies import STRATEGIES, get_strategy


def parse_value(text: str) -> Any:
    """Parse a CLI parameter value into bool, int, float or str."""
    low = text.strip().lower()
    if low in {"true", "yes"}:
        return True
    if low in {"false", "no"}:
        return False
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return text


def parse_params(items: Sequence[str] | None) -> dict[str, Any]:
    """Parse ``key=value`` pairs."""
    out: dict[str, Any] = {}
    for item in items or []:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise ValueError(f"expected key=value, got '{item}'")
        out[key.strip()] = parse_value(value)
    return out


def parse_grid(items: Sequence[str] | None) -> dict[str, list[Any]]:
    """Parse ``key=v1,v2,...`` grid specifications."""
    out: dict[str, list[Any]] = {}
    for item in items or []:
        key, sep, values = item.partition("=")
        if not sep or not key or not values:
            raise ValueError(f"expected key=v1,v2,..., got '{item}'")
        out[key.strip()] = [parse_value(v) for v in values.split(",") if v.strip()]
    return out


def _config(args: argparse.Namespace) -> BacktestConfig:
    return BacktestConfig(
        initial_cash=args.cash,
        fraction=args.fraction,
        commission=args.commission,
        slippage=args.slippage,
        stop_loss=args.stop_loss,
        take_profit=args.take_profit,
        allow_short=not args.long_only,
    )


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--data", required=True, type=Path, help="OHLCV CSV file")


def _add_execution(p: argparse.ArgumentParser) -> None:
    p.add_argument("--strategy", required=True, choices=sorted(STRATEGIES))
    p.add_argument("--cash", type=float, default=10_000.0, help="initial cash")
    p.add_argument("--fraction", type=float, default=1.0, help="equity fraction per trade")
    p.add_argument("--commission", type=float, default=0.0005, help="per side, fraction")
    p.add_argument("--slippage", type=float, default=0.0005, help="per side, fraction")
    p.add_argument("--stop-loss", type=float, default=None, help="e.g. 0.05 for 5%%")
    p.add_argument("--take-profit", type=float, default=None, help="e.g. 0.10 for 10%%")
    p.add_argument("--long-only", action="store_true", help="ignore short signals")


def build_parser() -> argparse.ArgumentParser:
    """Create the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="backtest-lab", description="Candlestick-pattern backtesting engine."
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="backtest a strategy")
    _add_common(run)
    _add_execution(run)
    run.add_argument("--param", "-p", action="append", metavar="KEY=VALUE", help="strategy param")
    run.add_argument("--trades", type=Path, help="write the trade log to this CSV")

    pat = sub.add_parser("patterns", help="detect candlestick patterns")
    _add_common(pat)
    pat.add_argument("--only", nargs="+", choices=list(PATTERNS), help="patterns to detect")
    pat.add_argument("--last", type=int, default=10, help="show the N most recent hits")
    pat.add_argument("--csv", type=Path, help="write the boolean pattern table to CSV")

    opt = sub.add_parser("optimize", help="grid search with walk-forward validation")
    _add_common(opt)
    _add_execution(opt)
    opt.add_argument(
        "--grid", "-g", action="append", required=True, metavar="KEY=V1,V2", help="param grid"
    )
    opt.add_argument("--metric", default="sharpe", help="metric to maximise (default sharpe)")
    opt.add_argument("--train-frac", type=float, default=0.7, help="train fraction (0-1)")
    opt.add_argument("--top", type=int, default=5, help="rows of the train table to show")

    sub.add_parser("strategies", help="list available strategies")
    return parser


def cmd_run(args: argparse.Namespace) -> int:
    data = load_ohlcv(args.data)
    strat = get_strategy(args.strategy, **parse_params(args.param))
    result = run_backtest(data, strat.generate_signals(data), _config(args))
    print(f"strategy: {strat!r}")
    print(
        f"data:     {args.data} ({len(data)} bars, {data.index[0]:%Y-%m-%d} -> "
        f"{data.index[-1]:%Y-%m-%d})"
    )
    print(format_metrics(result.metrics))
    if args.trades:
        result.trades.to_csv(args.trades, index=False)
        print(f"trades written to {args.trades}")
    return 0


def cmd_patterns(args: argparse.Namespace) -> int:
    data = load_ohlcv(args.data)
    table = detect_patterns(data, args.only)
    counts = table.sum()
    width = max(len(n) for n in table.columns)
    print(f"{len(data)} bars scanned")
    for name, count in counts.items():
        print(f"{name:<{width}}  {int(count):>5}")
    hits = table.stack()
    hits = hits[hits].reset_index()
    hits.columns = ["timestamp", "pattern", "hit"]
    if args.last > 0 and not hits.empty:
        print(f"\nmost recent {min(args.last, len(hits))} hits:")
        recent = hits.tail(args.last)
        for _, row in recent.iterrows():
            close = data.at[row["timestamp"], "close"]
            print(f"  {row['timestamp']:%Y-%m-%d}  {row['pattern']:<{width}}  close={close:.2f}")
    if args.csv:
        table.astype(int).to_csv(args.csv)
        print(f"pattern table written to {args.csv}")
    return 0


def cmd_optimize(args: argparse.Namespace) -> int:
    data = load_ohlcv(args.data)
    grid = parse_grid(args.grid)
    wf = walk_forward(
        data, args.strategy, grid, _config(args), metric=args.metric, train_frac=args.train_frac
    )
    cols = [*grid, "total_return", "sharpe", "max_drawdown", "num_trades"]
    with pd.option_context("display.width", 120, "display.float_format", "{:.4f}".format):
        print(f"train window: {data.index[0]:%Y-%m-%d} -> before {wf.split_time:%Y-%m-%d}")
        print(f"top {args.top} by {args.metric} (in-sample):")
        print(wf.train_table[cols].head(args.top).to_string(index=False))
    print(f"\nbest params: {wf.best_params}")
    print("\n-- train (in-sample) --")
    print(format_metrics(wf.train_metrics))
    print(f"\n-- test (out-of-sample, from {wf.split_time:%Y-%m-%d}) --")
    print(format_metrics(wf.test_metrics))
    return 0


def cmd_strategies(_: argparse.Namespace) -> int:
    for name, cls in STRATEGIES.items():
        print(f"{name:<16} {cls.description}")
        print(f"{'':<16} defaults: {cls.default_params}")
    return 0


COMMANDS = {
    "run": cmd_run,
    "patterns": cmd_patterns,
    "optimize": cmd_optimize,
    "strategies": cmd_strategies,
}


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the ``backtest-lab`` console script."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return COMMANDS[args.command](args)
    except (DataValidationError, FileNotFoundError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
