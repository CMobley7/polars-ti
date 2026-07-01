# Polars-TI

**Polars-TI** is a technical-analysis library for OHLCV data built on a pure
**Polars + Numba/NumPy** architecture. It provides **262 indicators and
candlestick-pattern groups** as native Polars expressions, with **zero runtime
pandas dependency**, a `df.ti` DataFrame namespace, study helpers, and optional
TA-Lib acceleration/parity.

```python
import polars as pl
import polars_ti as ti
from polars_ti.momentum import rsi
from polars_ti.overlap import sma

df = pl.read_csv("data/SPY_D.csv", try_parse_dates=True)

# Expression API
features = df.select(sma("close", length=20), rsi("close", length=14))

# DataFrame accessor (append=True keeps the original columns)
with_rsi = df.ti.rsi(length=14, append=True)

# A full study
all_features = df.ti.study(ti.AllStudy, talib=True)
```

## Features

- Native Polars `Expr` indicators across candles, cycles, momentum, overlap,
  performance, statistics, transform, trend, volatility, and volume.
- `df.ti` accessor and `Study` / `AllStudy` / `CommonStudy` helpers.
- Optional TA-Lib paths; fully functional **without** TA-Lib (native
  Polars/Numba), selectable per call via `talib=`.
- Numba kernels for stateful/recursive indicators.
- Parity-tested against the pandas baseline and TA-Lib, in both TA-Lib modes.

## Install

```bash
uv pip install "polars-ti[full]"     # everything
# or just the library:
uv pip install polars-ti
```

TA-Lib is optional (recommended for full candlestick coverage):

```bash
uv pip install TA-Lib
```

## Documentation

Full documentation lives in [`docs/`](docs/index.md):

- [Getting started](docs/getting-started.md) — install, quickstart, return types, `append=`, lazy eval.
- [Indicators](docs/indicators.md) — all 262 by category, with outputs and TA-Lib availability.
- [Studies](docs/studies.md) — `AllStudy` / `CommonStudy` / custom studies, `talib=` and `errors=`.
- [TA-Lib & native paths](docs/talib.md) — the `native = pandas-ta, talib = TA-Lib` design rule.
- [Migrating from pandas-ta](docs/migrating-from-pandas-ta.md) — porting guide.
- [Differences from pandas-ta](docs/differences-from-pandas-ta.md) — output differences (and why), new indicators and credits.
- [Development](docs/development.md) — quality gates, CI, and the parity oracle.

## Development

```bash
uv sync --extra test --dev
./scripts/check.sh --fast        # ruff, mypy, pytest+coverage, pandas-purge
```

## License & credits

Polars-TI is a Polars/Numba port of, and builds on,
[twopirllc/pandas-ta](https://github.com/twopirllc/pandas-ta). Additional
indicators were integrated from community forks with attribution — see
[Differences §6](docs/differences-from-pandas-ta.md#6-new-indicators--credits).
See [LICENSE](LICENSE).
