# Studies

A **study** runs a group of indicators and returns the original DataFrame with
every indicator's columns appended.

```python
import polars as pl
import polars_ti as ti

df = pl.read_csv("data/SPY_D.csv", try_parse_dates=True)
all_features = df.ti.study(ti.AllStudy, talib=True)
```

## Built-in studies

| Study | What it runs |
| :--- | :--- |
| `ti.AllStudy` | every registered indicator (all 262) |
| `ti.CommonStudy` | a small common set |
| `"momentum"`, `"overlap"`, … | one category (pass the category name as a string) |

```python
momentum = df.ti.study("momentum")          # category shorthand
common = df.ti.study(ti.CommonStudy)
```

## Custom studies

Define a `Study` with a list of indicator specs (each a dict with a `kind` and
optional parameters). Indicators can be chained — a later indicator can read a
column produced by an earlier one:

```python
study = ti.Study(
    name="My Features",
    ti=[
        {"kind": "rsi"},
        {"kind": "macd"},
        {"kind": "sma", "length": 50},
        {"kind": "bbands", "length": 20},
        {"kind": "log_return", "cumulative": True},
        {"kind": "ema", "close": "CUMLOGRET_1", "length": 5},
    ],
)
features = df.ti.study(study)
```

## Options

```python
df.ti.study(study, talib=True, errors="warn", cores=0)
```

- **`talib`** (default `False`): pass `talib=True` to use TA-Lib-backed paths
  where available; `talib=False` forces native Polars/Numba for every indicator
  and its sub-indicators. See [TA-Lib & native paths](talib.md).
- **`errors`** (default `"warn"`): how to handle an indicator that fails.
  - `"warn"` — collect failures and emit one summary `warning`, completing the
    rest of the study.
  - `"raise"` — re-raise the first failure immediately.
  - `"ignore"` — silently skip failures.

  (Note: an unrecognized `kind` — e.g. a typo — is silently skipped under every
  error mode; it is not treated as a failure.)
- **`cores`** (default `0`): reserved for future multiprocessing; currently all
  indicators run sequentially.

> A handful of *composition* indicators (`long_run`, `short_run`, `tsignals`,
> `xsignals`) operate on columns produced by other indicators, not raw OHLCV, so
> they are skipped by `AllStudy` unless those input columns are present. With
> `errors="warn"` (the default) they are reported and the study still completes.

## Output shape & completeness

The all-study output is a **superset** of the pandas-ta all-study output: every
pandas-ta column is reproduced (0 dropped) plus a couple of struct extras. The
exact column manifest is pinned by `tests/test_study_completeness.py`. See
[Differences §3](differences-from-pandas-ta.md#3-indicator--column-counts).
