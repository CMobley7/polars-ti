# Polars-TI: Phase 5 — Complete Pandas Removal

## Goal

Eliminate **every** remaining `pandas` dependency from the `polars-ti` library and its test suite, producing a codebase whose only runtime dependencies are `polars`, `numpy`, `numba`, and (optionally) `talib`. Once complete, `pandas` should appear nowhere in `pip install polars-ti`'s dependency tree.

---

## Context & Current State

Phase 4 migrated all **indicator** modules (`pl_sma`, `pl_rsi`, etc.) to pure Polars + Numba. However, pandas vestiges remain in five areas:

| Area | Files | Severity |
|------|-------|----------|
| Test fixtures | 113 test files | High — blocks pandas removal from deps |
| Utility modules | 7 files in `polars_ti/utils/` | High — imported at runtime |
| Type aliases | `polars_ti/_typing.py` | Medium — `from pandas import DataFrame, Series` |
| Root conftest | `tests/conftest.py` | Medium — `from pandas import read_csv` |
| Dead parity test bodies | 48 test files | Low — dead code behind `pytest.skip` |

**Test suite baseline** (must remain green throughout): `1097 passed, 128 skipped, 0 failures`.

---

## Phase 5A: Test Suite Pandas Purge

### Objective

Remove `import pandas as pd` from all 113 `tests/*/test_*_polars.py` files. After this phase, zero test files should import pandas.

### Step 1 — Delete Dead Parity Test Bodies (48 files)

**What:** In 48 test files, `pd.` appears **only** inside the body of `test_numerical_parity` methods that are already guarded by `pytest.skip(...)`. The code after the skip is dead.

**Action:** For each of these 48 files:
1. Find the `test_numerical_parity` method
2. Delete every line **after** the `pytest.skip(...)` call, up to the next `def`, end of class, or end of file
3. Remove the `import pandas as pd` line
4. **Run the test file** in isolation to confirm it still passes

**How to identify these files:** Run the following (already validated):
```bash
# Files where pd. appears ONLY inside test_numerical_parity bodies (after pytest.skip)
python3 -c "
import re
from pathlib import Path

for fp in sorted(Path('tests').rglob('test_*_polars.py')):
    content = fp.read_text()
    if 'pd.' not in content: continue
    lines = content.split('\n')
    pd_in_fixture = False
    in_parity = False
    for line in lines:
        s = line.strip()
        if s.startswith('def test_numerical_parity'): in_parity = True
        elif s.startswith('def test_') and 'numerical_parity' not in s: in_parity = False
        elif s.startswith('@pytest.fixture'): in_parity = False
        if 'pd.' in line and not in_parity: pd_in_fixture = True
    if not pd_in_fixture:
        print(fp)
"
```

**Example before:**
```python
import pandas as pd  # Restored for fixtures

def test_numerical_parity(self, sample_data):
    """Numerical parity with Pandas implementation."""
    pytest.skip("Pandas implementation removed in Phase 4 purge")
    pdf = pd.DataFrame({'high': high, 'low': low, 'close': close})
    pd_result = cci(pdf['high'], pdf['low'], pdf['close'], length=14)
    # ... more dead pandas code
```

**Example after:**
```python
# (no pandas import)

def test_numerical_parity(self, sample_data):
    """Numerical parity with Pandas implementation."""
    pytest.skip("Pandas implementation removed in Phase 4 purge")
```

**Verification:** After each file, run:
```bash
.venv/bin/python -m pytest <file> --tb=short -q
```

---

### Step 2 — Rewrite Fixture Data Prep (62 "fixtures-only" files)

**What:** In 62 test files, `pd.Series(...)` or `pd.DataFrame(...)` appears inside `@pytest.fixture` methods to construct sample data. These need to be replaced with raw NumPy arrays or `pl.DataFrame`/`pl.Series`.

**Common patterns and their replacements:**

| Pandas Pattern | Polars/NumPy Replacement |
|----------------|--------------------------|
| `pd.Series(close)` | `close` (it's already a numpy array) |
| `pd.DataFrame({'close': close})` | `pl.DataFrame({'close': close})` |
| `return pl.DataFrame({...}), pd.Series(close)` | `return pl.DataFrame({...}), close` (just return the numpy array) |
| `pd.Series(close).ewm(span=N).mean().values` | Use helper: `_np_ema(close, span=N)` (see below) |

**NumPy EMA helper** (copy into any test file that needs it):
```python
def _np_ema(arr: np.ndarray, span: int) -> np.ndarray:
    """Pure NumPy EWM equivalent for test fixtures."""
    alpha = 2.0 / (span + 1)
    result = np.empty_like(arr)
    result[0] = arr[0]
    for i in range(1, len(arr)):
        result[i] = alpha * arr[i] + (1 - alpha) * result[i - 1]
    return result
```

**Action for each file:**
1. Open the file
2. Find every `pd.Series(...)` and `pd.DataFrame(...)` call **outside** of the skipped parity test
3. Replace with the appropriate NumPy/Polars equivalent from the table above
4. Remove the `import pandas as pd` line
5. If the file has a parity test with dead code after `pytest.skip`, also clean that up (as in Step 1)
6. Run the test file to confirm pass

> [!IMPORTANT]
> Some fixtures return a tuple like `(pl_df, pd_series)` and downstream test methods receive and index into that tuple. When you replace `pd.Series(close)` with just `close` (numpy array), check that all callers handle the numpy array correctly. Typically the callers pass it to `.to_numpy()` which is a no-op on numpy arrays, or index into it directly — both are safe.

**Verification:** After each file, run:
```bash
.venv/bin/python -m pytest <file> --tb=short -q
```

---

### Step 3 — Rewrite 3 "Both" Files

**What:** Three volatility test files use `pd.Series(...)` in **both** fixtures and parity tests:
- `tests/volatility/test_accbands_polars.py`
- `tests/volatility/test_atr_polars.py`
- `tests/volatility/test_bbands_polars.py`

**Action:** Apply both Step 1 (clean dead parity code) and Step 2 (rewrite fixtures) to each.

---

### Step 4 — Rewrite `tests/conftest.py`

**What:** The root conftest imports `from pandas import read_csv` and constructs a pandas DataFrame fixture `df` from `data/SPY_D.csv`. This fixture is used by the legacy test files (`test_studies.py`, `test_indicator_*.py`) which are already excluded from the active test suite.

**Action:**
1. Replace `from pandas import read_csv` with `import polars as pl`
2. Replace the `testdf` fixture:
   ```python
   @pytest.fixture(name="df", scope="function")
   def testdf():
       """Yields a truncated df from TEST_CSV file"""
       df = pl.read_csv(TEST_CSV, try_parse_dates=True)
       df = df.drop(["dividends", "stock splits"])
       yield df.head(TEST_ROWS)
   ```
3. Note: The `all_study`, `custom_study_*` fixtures reference `ti.Study` which is a legacy pandas class. These fixtures are only used by `test_studies.py` (which is already excluded from the test runner). Leave them as-is or wrap them in a try/except. They are dead code.

**Verification:**
```bash
.venv/bin/python -m pytest tests/cycles tests/candles tests/performance \
  tests/statistics tests/transform tests/trend tests/volatility \
  tests/volume tests/overlap tests/momentum --tb=no -q
```

---

## Phase 5B: Library Source Pandas Purge

### Objective

Remove `pandas` from the runtime dependency tree in `polars_ti/`.

### Step 5 — Clean `polars_ti/_typing.py`

**What:** Line 14 has `from pandas import DataFrame, Series`. Several type aliases on lines 58–63 and 82–83 reference these types.

**Action:**
1. Remove `from pandas import DataFrame, Series` (line 14)
2. Remove or replace the aliases that depend on them:

   | Old Alias | Replacement |
   |-----------|-------------|
   | `SeriesFrame = Series \| DataFrame` | Delete — not used by Polars indicators |
   | `MaybeSeries = T \| Series` | Delete |
   | `MaybeSeriesFrame = T \| Series \| DataFrame` | Delete |
   | `AnyArray = Array \| Series \| DataFrame` | `AnyArray = Array` |
   | `AnyArray1d = Array1d \| Series` | `AnyArray1d = Array1d` |
   | `AnyArray2d = Array2d \| DataFrame` | `AnyArray2d = Array2d` |
   | `DualFrame = DataFrame \| pl.DataFrame \| pl.LazyFrame` | `DualFrame = pl.DataFrame \| pl.LazyFrame` |
   | `DualSeries = Series \| pl.Series` | `DualSeries = pl.Series` |

3. After removing these, grep for any remaining callers and update them:
   ```bash
   grep -rn "SeriesFrame\|MaybeSeries\b\|MaybeSeriesFrame\|DualFrame\|DualSeries" \
     polars_ti/ --include="*.py" | grep -v __pycache__ | grep -v _typing.py
   ```
   Expected callers: `_validate.py` (lines 12, 14, 33, 68) and `_time.py` (line 7, 11). Update these to use `pl.DataFrame | pl.Series` or similar.

**Verification:**
```bash
.venv/bin/python -c "import polars_ti; print('import OK')"
.venv/bin/python -m pytest tests/cycles --tb=no -q  # Quick smoke test
```

---

### Step 6 — Clean `polars_ti/utils/` Modules (7 files)

These files are the deepest pandas entanglement. Not every function in these files is actively called by the Polars-native indicator modules. The strategy is:

1. **Identify which functions are live** (called by `pl_*` indicator functions or `core.py`)
2. **Rewrite live functions** to pure NumPy/Polars
3. **Mark dead functions** with `# DEPRECATED: legacy pandas` or delete them

**File-by-file plan:**

#### `polars_ti/utils/_core.py` (345 lines, 4 pandas refs)
- **Live functions:** `nb_non_zero_range` (Numba — already pandas-free), `PolarsTI`/`PolarsTILazy` classes
- **Dead functions:** `rma_pandas` (replaced by `_nb_rma` in `kdj.py`)
- **Action:**
  1. Remove `from pandas import DataFrame, Series`
  2. Delete `rma_pandas()` function entirely
  3. Update any remaining type hints from `DataFrame` → `pl.DataFrame`
  4. Remove `rma_pandas` from `polars_ti/utils/__init__.py` if exported

#### `polars_ti/utils/_math.py` (389 lines, 2 pandas refs)
- **Live functions:** `fibonacci`, `pascals_triangle`, `symmetric_triangle`, `pl_non_zero_range` — all already pandas-free
- **Note:** The only `pd.Series` reference is inside a **docstring example** in `hpoly()` (line 117: `coeffs_2 = pd.Series(coeffs_0).values`). This is documentation, not executable code.
- **Action:**
  1. Remove `from pandas import DataFrame, Series` (line 28)
  2. Update the `hpoly` docstring to use `np.array` instead of `pd.Series`
  3. Check if any internal functions use `Series` type hints and update

#### `polars_ti/utils/_validate.py` (223 lines, 2 pandas refs)
- **Live functions:** `v_expr` (Polars-native), `v_dataframe` (accepts `MaybeSeriesFrame` from `_typing.py`)
- **Action:** After Step 5 updates `_typing.py`, update imports here. Change `v_dataframe` to accept `pl.DataFrame` only.

#### `polars_ti/utils/_candles.py` (60 lines, 1 pandas ref)
- **Live functions:** `pl_high_low_range`, `pl_real_body` — already Polars-native
- **Action:** Remove `from pandas import Series`. Remove any `Series` type hints.

#### `polars_ti/utils/_metrics.py` (357 lines, 13 pandas refs)

> [!IMPORTANT]
> All 11 pandas-based metric functions are **DEAD CODE — zero callers in the codebase**. This was verified by grep. The `pl_*` functions at the bottom (lines 308+) are already Polars-native and are the only live code.

**Dead functions** (all accept `pd.Series`, all have zero callers):
- `cagr`, `calmar_ratio`, `downside_deviation`, `jensens_alpha`
- `log_max_drawdown`, `max_drawdown`, `optimal_leverage`
- `pure_profit_score`, `sharpe_ratio`, `sortino_ratio`, `volatility`

**Live functions** (already pandas-free):
- `pl_log_return`, `pl_percent_return`, `pl_cumulative_return`
- `pl_rolling_volatility`, `pl_drawdown`, `pl_max_drawdown`

**Action:**
1. Delete all 11 dead functions listed above
2. Remove `from pandas import Series, Timedelta`
3. Keep all `pl_*` functions
4. Remove the dead function names from `polars_ti/utils/__init__.py`

#### `polars_ti/utils/_signals.py` (198 lines, 1 pandas ref)
- **Live functions:** `pl_signal` — already returns `pl.Expr`. Legacy `signal_indicators` is dead.
- **Action:** Remove `from pandas import DataFrame, Series`. Delete dead functions.

#### `polars_ti/utils/_time.py` (197 lines, 2 pandas refs)
- **Live functions:** `df_dates` — accepts DataFrame. Zero callers from Polars indicators.
- **Action:** Remove `from pandas import DataFrame, Series, Timestamp, to_datetime`. Replace type hints with Polars equivalents or mark as deprecated.

#### Step 6 Final: Update `polars_ti/utils/__init__.py`

After cleaning the modules above, update the re-exports in `__init__.py` to remove deleted functions (especially the 11 dead metric functions and `rma_pandas`).

> [!WARNING]
> Before deleting ANY function, verify it has zero callers with:
> ```bash
> grep -rn "function_name" polars_ti/ tests/ --include="*.py" | grep -v __pycache__ | grep -v "def function_name"
> ```

**Verification after each file:**
```bash
.venv/bin/python -c "import polars_ti; print('import OK')"
.venv/bin/python -m pytest tests/cycles tests/momentum --tb=no -q  # Spot check
```

---

### Step 7 — Suppress `core.py` UserWarning

**What:** `@pl.api.register_dataframe_namespace("ti")` at line 225 of `core.py` emits:
```
UserWarning: Overriding existing custom namespace 'ti' (on 'DataFrame')
```
This happens because the module is imported multiple times (via `conftest.py` and test files).

**Action:** Wrap the decorator in a warning filter:
```python
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)

    @pl.api.register_dataframe_namespace("ti")
    class PolarsTIAccessor:
        ...
```

Or gate the registration:
```python
if not hasattr(pl.DataFrame, "ti"):
    @pl.api.register_dataframe_namespace("ti")
    class PolarsTIAccessor:
        ...
```

**Verification:**
```bash
.venv/bin/python -m pytest tests/cycles --tb=no -q 2>&1 | grep -c "UserWarning"
# Expected: 0
```

---

### Step 8 — Remove `pandas` From `pyproject.toml` Dependencies

**What:** After all code changes, `pandas` should no longer be a runtime dependency.

**Action:**
1. Open `pyproject.toml`
2. Move `pandas` from `[project.dependencies]` to `[project.optional-dependencies.dev]` (or remove entirely if it's already only in dev)
3. Run `uv lock` to regenerate the lockfile

**Verification:**
```bash
grep -i pandas pyproject.toml  # Should only appear in [dev] or test deps, not main deps
.venv/bin/python -c "import polars_ti; print('import OK')"  # Must work without pandas
```

> [!CAUTION]
> Do **NOT** uninstall pandas from the venv yet. The test suite still uses it in `conftest.py` fixtures until Step 4 is done. Remove it from `[project.dependencies]` only after ALL Phase 5A and 5B steps pass.

---

## Phase 5C: Final Verification

### Step 9 — Full Grep Audit

```bash
# Zero pandas imports in library source
grep -rn "import pandas\|from pandas" polars_ti/ --include="*.py" | grep -v __pycache__
# Expected: 0 results

# Zero pandas imports in test files
grep -rn "import pandas\|from pandas" tests/ --include="*.py" | grep -v __pycache__
# Expected: 0 results (or only in conftest.py if Study fixtures are preserved)

# Zero pd. references in library source
grep -rn "\bpd\." polars_ti/ --include="*.py" | grep -v __pycache__ | grep -v "docstrings"
# Expected: 0 results
```

### Step 10 — Full Test Suite

```bash
.venv/bin/python -m pytest \
  tests/cycles tests/candles tests/performance tests/statistics tests/transform \
  tests/trend tests/volatility tests/volume tests/overlap tests/momentum \
  --tb=short -q
```

**Target:** ≥1097 passed, 0 failures. The skip count should decrease (dead parity code removed).

---

## Execution Order Summary

| Step | Scope | Difficulty | Description |
|------|-------|-----------|-------------|
| 1 | Tests | Easy | Delete dead code in 48 parity tests |
| 2 | Tests | Medium | Rewrite fixtures in 62 test files |
| 3 | Tests | Medium | Fix 3 "both" test files |
| 4 | Tests | Easy | Rewrite `conftest.py` |
| 5 | Library | Easy | Clean `_typing.py` |
| 6 | Library | Hard | Rewrite 7 utils modules |
| 7 | Library | Easy | Suppress `core.py` warning |
| 8 | Config | Easy | Move pandas from deps to dev-deps |
| 9 | Verification | Easy | Full grep audit |
| 10 | Verification | Easy | Full test suite |

> [!IMPORTANT]
> Execute steps in order. Each step depends on the previous. Run the test suite after every step to catch regressions immediately. Never batch more than one step without verifying.

---

## Environment

- **Python**: 3.12.9
- **Polars**: 1.x (check version with `.venv/bin/python -c "import polars; print(polars.__version__)"`)
- **Numba**: installed in `.venv/`
- **Venv**: `.venv/` at project root
- **Test runner**: `.venv/bin/python -m pytest`
- **Numba cache**: If you get a segfault, clear stale cache with `find polars_ti -name "*.nbi" -o -name "*.nbc" -delete`
