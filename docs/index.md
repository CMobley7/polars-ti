# Polars-TI documentation

Polars-TI is a technical-analysis library for OHLCV data built on a pure
**Polars + Numba/NumPy** architecture — every indicator is a native Polars
expression with **zero runtime pandas dependency**. It provides **267
indicators and candlestick-pattern groups**, a `df.ti` DataFrame namespace,
study helpers, and optional TA-Lib acceleration/parity.

## Contents

- **[Getting started](getting-started.md)** — install, quickstart, return types,
  the `df.ti` accessor, `append=`, lazy evaluation, multi-output indicators.
- **[Indicators](indicators.md)** — all 267 indicators by category, with output
  counts and TA-Lib availability.
- **[Studies](studies.md)** — `Study`, `AllStudy`, `CommonStudy`, category
  studies, the `talib=` flag and `errors=` modes.
- **[TA-Lib & native paths](talib.md)** — when TA-Lib is used, how to force the
  native path, and the `native = pandas-ta, talib = TA-Lib` design rule.
- **[Migrating from pandas-ta](migrating-from-pandas-ta.md)** — a code-porting
  guide for existing pandas-ta users.
- **[Differences from pandas-ta](differences-from-pandas-ta.md)** — the
  authoritative catalog of behavioral/output differences (and why), plus the new
  indicators and their credits.
- **[Development](development.md)** — quality gates, CI, and the parity oracle.

## Install

```bash
uv pip install "polars-ti[full]"
```

TA-Lib is optional (used for speed and parity checks — every indicator,
including all candlestick patterns, also has a native implementation):
`uv pip install TA-Lib`.

## 30-second example

```python
import polars as pl
import polars_ti as ti
from polars_ti.momentum import rsi
from polars_ti.overlap import sma

df = pl.read_csv("data/SPY_D.csv", try_parse_dates=True)

# Expression API
features = df.select(sma("close", length=20), rsi("close", length=14))

# DataFrame accessor
rsi14 = df.ti.rsi(length=14)                 # result-only
with_rsi = df.ti.rsi(length=14, append=True) # original + RSI

# A full study
all_features = df.ti.study(ti.AllStudy, talib=True)
```
