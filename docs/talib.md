# TA-Lib & native paths

Polars-TI runs fully **without TA-Lib** — every indicator has a native
Polars/Numba implementation. TA-Lib is optional and used for speed/parity where
it has an equivalent.

```bash
uv pip install TA-Lib   # optional
```

## The `talib=` flag

Most indicators accept a `talib=` keyword (see the **TA-Lib** column in the
[indicator list](indicators.md)):

```python
from polars_ti.momentum import rsi

df.select(rsi("close", length=14, talib=True))    # TA-Lib path (if installed)
df.select(rsi("close", length=14, talib=False))   # native Polars/Numba path
```

- `talib=True` (default) uses TA-Lib **when it is installed**; if TA-Lib is
  absent it transparently falls back to the native path.
- `talib=False` forces the native path **and propagates that choice to every
  sub-indicator** (ATR, EMA, RSI, CMO, …) the indicator uses internally.

At the study level the flag applies to all indicators at once:

```python
df.ti.study(ti.AllStudy, talib=False)   # native everywhere
```

## Design rule: native = pandas-ta, talib = TA-Lib

When an indicator can be computed two ways, Polars-TI follows a deliberate rule:

- **`talib=True`** reproduces **TA-Lib** output (e.g. TA-Lib's SMA-seeded EMA,
  population-std `STDDEV`).
- **`talib=False`** reproduces **pandas-ta's native** semantics (e.g.
  `ewm(adjust=False)`-style EMA seeded from the first valid value).

This is why some indicators differ slightly between the two modes — each mode is
faithful to a different reference. See
[Differences §4c](differences-from-pandas-ta.md#4c-native-vs-ta-lib-paths-talib-flag).

## What TA-Lib affects

- **Candlestick patterns:** ~60 of the `cdl_pattern` patterns require TA-Lib and
  are omitted from the native study (a few have native implementations).
- **Indicators with a TA-Lib equivalent:** RSI, ATR, EMA, SMA, MACD, ADX, MFI,
  STDDEV, CMO, TRIX, LINEARREG, etc. — used under `talib=True`.

## CI coverage

CI runs the full test-suite in a matrix of **{TA-Lib installed, TA-Lib absent}**
× **{oldest-supported, latest Polars}**, so both code paths are always exercised.
You can reproduce the no-TA-Lib path locally even with TA-Lib installed:

```bash
POLARS_TI_SIMULATE_NO_TALIB=1 uv run pytest tests/ -q
```
