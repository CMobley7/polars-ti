# Polars TI

Polars TI is a Polars-native technical indicators library for OHLCV data. It provides expression-first indicators, a `df.ti` DataFrame namespace, study helpers, optional TA-Lib acceleration/parity paths, and focused Polars tests for the converted indicator surface.

The public indicator functions now use the indicator name directly:

```python
from polars_ti.momentum import rsi
from polars_ti.overlap import sma
from polars_ti.volatility import atr
```

Legacy `pl_*` function names were removed. Use `rsi(...)`, `sma(...)`, `atr(...)`, etc.

## Features

- Polars `Expr` indicators for candles, cycles, momentum, overlap, performance, statistics, transform, trend, volatility, and volume.
- `pl.DataFrame.ti` namespace for convenient indicator dispatch.
- `Study`, `AllStudy`, and `CommonStudy` helpers for grouped indicator runs.
- Optional TA-Lib support for indicators and candlestick patterns where TA-Lib has an equivalent.
- Numba kernels for indicators that need stateful numerical loops.
- Ruff, mypy, pytest, coverage, pandas-purge checks, and dependency audit wired into `scripts/check.sh` and CI.

## Installation

```bash
uv pip install "polars-ti[full]"
```

For local development:

```bash
uv sync --extra test --dev
```

TA-Lib is optional, but recommended for full candlestick coverage and independent parity checks:

```bash
uv pip install TA-Lib
```

## Quick Start

```python
import polars as pl
from polars_ti.momentum import rsi
from polars_ti.overlap import sma

df = pl.read_csv("data/SPY_D.csv", try_parse_dates=True)

result = df.select(
    sma("close", length=20),
    rsi("close", length=14),
)
```

The DataFrame namespace returns a result DataFrame by default:

```python
sma20 = df.ti.sma(length=20)
```

Pass `append=True` when you want the returned DataFrame to include the original columns plus the indicator columns:

```python
df_with_sma = df.ti.sma(length=20, append=True)
```

## Studies

Run a built-in study and keep the returned DataFrame:

```python
import polars as pl
import polars_ti as ti

df = pl.read_csv("data/SPY_D.csv", try_parse_dates=True)
all_features = df.ti.study(ti.AllStudy, cores=0, talib=True)
```

`talib=True` enables TA-Lib-backed implementations where available. `talib=False` forces native Polars/Numba paths for indicators that support both.

## Validation

Current validation status is captured in [VALIDATION_REPORT.md](VALIDATION_REPORT.md).

At the time of this update:

- Ruff check passes.
- The active Polars test suite passes: `1128 passed`.
- All current public imports use non-`pl_` indicator names.
- All-study outputs were compared against the pandas implementation in `tmp/polars-ti` on the same deterministic random OHLCV dataset, with TA-Lib used independently for matching indicators.

## Development Checks

Run the same local quality gates used by CI:

```bash
./scripts/check.sh --fast
```

Run the full suite:

```bash
uv run pytest tests/ -q --tb=short
```

Run Ruff:

```bash
uv run ruff check polars_ti tests scripts
uv run ruff format --check polars_ti tests scripts
```
