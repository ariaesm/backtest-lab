from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from backtest_lab.cli import main, parse_grid, parse_params, parse_value
from backtest_lab.synthetic import generate_gbm_ohlcv


@pytest.fixture
def csv_path(tmp_path: Path) -> Path:
    p = tmp_path / "sample.csv"
    generate_gbm_ohlcv(400, seed=3).reset_index().to_csv(p, index=False)
    return p


def test_parse_helpers() -> None:
    assert parse_value("3") == 3
    assert parse_value("0.5") == 0.5
    assert parse_value("true") is True
    assert parse_value("abc") == "abc"
    assert parse_params(["fast=10", "allow_short=false"]) == {"fast": 10, "allow_short": False}
    assert parse_grid(["fast=5,10", "slow=50"]) == {"fast": [5, 10], "slow": [50]}
    with pytest.raises(ValueError):
        parse_params(["oops"])


def test_cli_run(csv_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    trades = tmp_path / "trades.csv"
    code = main(
        [
            "run",
            "--data",
            str(csv_path),
            "--strategy",
            "sma_crossover",
            "-p",
            "fast=10",
            "-p",
            "slow=30",
            "--trades",
            str(trades),
            "--plot",
            str(tmp_path / "eq.png"),
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "sharpe" in out and "max_drawdown" in out
    assert "entry_time" in pd.read_csv(trades).columns
    assert (tmp_path / "eq.png").stat().st_size > 0


def test_cli_patterns(csv_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    png = tmp_path / "p.png"
    assert main(["patterns", "--data", str(csv_path), "--last", "3", "--plot", str(png)]) == 0
    assert png.exists()
    out = capsys.readouterr().out
    assert "400 bars scanned" in out
    assert "bullish_engulfing" in out


def test_cli_optimize(csv_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(
        [
            "optimize",
            "--data",
            str(csv_path),
            "--strategy",
            "sma_crossover",
            "-g",
            "fast=5,10",
            "-g",
            "slow=30,60",
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "best params" in out and "out-of-sample" in out


def test_cli_reports_bad_input(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bad = tmp_path / "bad.csv"
    bad.write_text("timestamp,open,high,low,close\n2024-01-01,1,2,0.5,1\n")
    assert main(["run", "--data", str(bad), "--strategy", "sma_crossover"]) == 2
    assert "missing required column" in capsys.readouterr().err


def test_cli_strategies(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["strategies"]) == 0
    assert "doji_reversal" in capsys.readouterr().out
