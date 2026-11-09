# Contributing

Thanks for your interest in backtest-lab!

1. Fork the repo and create a branch from `main`.
2. Install in editable mode with dev extras: `make install`.
3. Make your change with type hints and docstrings, and add tests.
4. Run the checks locally before opening a PR:
   ```bash
   make format   # ruff format + autofix
   make lint     # ruff check
   make test     # pytest
   ```
5. Open a pull request describing *what* changed and *why*.

### Adding a strategy

Subclass `backtest_lab.strategies.Strategy`, set `name`, `description` and
`default_params`, implement `generate_signals` (return -1/0/+1 per bar using only
data up to that bar), and decorate it with `@register_strategy`. Add a test that the
signals are causal (computing them on a truncated frame gives the same prefix).

### Reporting bugs

Open an issue with a minimal CSV snippet and the exact command you ran.
