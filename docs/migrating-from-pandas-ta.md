# Migrating from pandas-ta

A practical guide for porting [pandas-ta](https://github.com/twopirllc/pandas-ta)
code to Polars-TI. For the *why* behind output differences, see
[Differences from pandas-ta](differences-from-pandas-ta.md).

## 1. DataFrame engine

Polars-TI operates on **Polars** DataFrames, not pandas. Convert at the
boundary:

```python
import polars as pl
pl_df = pl.from_pandas(pd_df)          # pandas -> Polars
pd_df = pl_df.to_pandas()              # Polars -> pandas
```

`from_pandas` / `to_pandas` require `pandas` and `pyarrow` (e.g.
`uv pip install pandas pyarrow`); they are not needed for Polars-TI itself.

## 2. Accessor name: `.ta` → `.ti`

```python
# pandas-ta
pd_df.ta.rsi()
# Polars-TI
pl_df.ti.rsi()
```

## 3. Function imports

```python
# pandas-ta
import pandas_ta as ta
ta.rsi(df["close"])
# Polars-TI
from polars_ti.momentum import rsi
df.select(rsi("close"))
```

`pl_*`-prefixed names (a transitional naming) were removed — always use the
indicator name directly (`rsi`, `sma`, `atr`, …).

## 4. Appending results

```python
# pandas-ta
pd_df.ta.rsi(append=True)
# Polars-TI (equivalent)
pl_df.ti.rsi(append=True)
# or, expression style
pl_df.with_columns(rsi("close"))
```

By default the accessor returns a **result-only** DataFrame; pass `append=True`
to keep the original columns.

## 5. Multi-output indicators return a struct

pandas-ta returns several columns; Polars-TI returns one **struct** column.
Unnest it for the familiar flat columns:

```python
# pandas-ta -> columns MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9
pd_df.ta.macd()

# Polars-TI -> one "MACD" struct column
pl_df.ti.macd().unnest("MACD")          # -> the same flat columns
```

## 6. Column-name changes

Most column names are identical. A few multi-output indicators use restructured
(struct/dotted) names; the mapping is documented in
[Differences §4d](differences-from-pandas-ta.md#4d-renames--struct-outputs).
Notable ones:

| pandas-ta | Polars-TI |
| :--- | :--- |
| `AROOND_14` / `AROONU_14` / `AROONOSC_14` | fields of the `AROON_14` struct |
| `KCLe_20_2` / `KCBe_20_2` / `KCUe_20_2` | `KC_e_20_2` struct |
| `VWAP_D` | `VWAP_1D` |

## 7. Studies

```python
# pandas-ta
pd_df.ta.strategy("All")
# Polars-TI
pl_df.ti.study(ti.AllStudy)
pl_df.ti.study("momentum")              # a single category
```

See [Studies](studies.md) for custom study definitions.

## 8. Expect some values to differ — for the better

A small set of indicators produce different numbers than pandas-ta because
pandas-ta's native path was buggy (Polars-TI matches TA-Lib/canonical) or
because of a documented convention change. None are Polars-TI being wrong — the
full list with magnitudes and reasons is in
[Differences §4](differences-from-pandas-ta.md#4-output-differences-with-reasons).
