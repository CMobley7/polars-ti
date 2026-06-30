# Getting started

## Install

```bash
uv pip install "polars-ti[full]"
```

Or for local development:

```bash
uv sync --extra test --dev
```

TA-Lib is optional but recommended for full candlestick coverage and
independent parity checks:

```bash
uv pip install TA-Lib
```

## Two ways to call indicators

### 1. Expression API

Import the indicator by name and use it inside any Polars `select` /
`with_columns`:

```python
import polars as pl
from polars_ti.momentum import rsi
from polars_ti.overlap import sma

df = pl.read_csv("data/SPY_D.csv", try_parse_dates=True)

out = df.select(
    sma("close", length=20),
    rsi("close", length=14),
)
```

Indicators accept column **names** (`"close"`) or any Polars expression
(`pl.col("close")`, `pl.col("close").log()`, …).

> Legacy `pl_*` names (e.g. `pl_rsi`) were removed — use the indicator name
> directly.

### 2. DataFrame accessor (`df.ti`)

Every indicator is also a method on the `df.ti` namespace:

```python
rsi14 = df.ti.rsi(length=14)        # returns a result-only DataFrame
```

Pass `append=True` to get the original DataFrame with the indicator columns
hstacked on:

```python
with_rsi = df.ti.rsi(length=14, append=True)
```

## Return types

| Indicator kind | Expression API returns | Accessor returns |
| :--- | :--- | :--- |
| Single-output (e.g. `rsi`, `sma`) | one `pl.Expr` | DataFrame with one column |
| Multi-output (e.g. `macd`, `bbands`, `kc`) | a `pl.Expr` **struct** (or a list of exprs) | DataFrame with one **struct** column |

### Multi-output indicators

Multi-output indicators emit a single Polars **struct** column. Unnest it to get
flat columns:

```python
from polars_ti.momentum import macd

macd_struct = df.select(macd("close"))          # 1 struct column "MACD"
macd_flat = macd_struct.unnest("MACD")          # MACD_12_26_9, MACDs_..., MACDh_...

# via the accessor
flat = df.ti.macd().unnest("MACD")
```

The struct field names are the familiar pandas-ta column names
(`MACD_12_26_9`, `MACDs_12_26_9`, `MACDh_12_26_9`). A few indicators use shorter
struct keys; see the name map in
[Differences §4d](differences-from-pandas-ta.md#4d-renames--struct-outputs).

## Lazy evaluation

Because indicators are plain Polars expressions, they work in lazy queries:

```python
out = (
    pl.scan_csv("data/SPY_D.csv", try_parse_dates=True)
    .select(sma("close", 20), rsi("close", 14))
    .collect()
)
```

## TA-Lib vs native

Many indicators accept `talib=` to choose the TA-Lib path (default, when
installed) or the native Polars/Numba path:

```python
df.select(rsi("close", length=14, talib=False))   # native
```

See [TA-Lib & native paths](talib.md).

## Next steps

- Browse the full [indicator list](indicators.md).
- Run grouped [studies](studies.md).
