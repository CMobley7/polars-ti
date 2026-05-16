# Verification Log

Date: 2026-05-16

## Conversion Audit

- Runtime library audit: `rg -n "import pandas|from pandas|\bpd\." polars_ti --glob '*.py'`
  - Result: no matches.
- Import smoke test: `.venv/bin/python -c "import polars_ti; import polars as pl; ..."`
  - Result: import OK, `pl.DataFrame(...).ti` resolves to `TechnicalIndicators`.
  - Difference from prior state: the duplicate `df.ti` namespace warning is gone.
- Dependency check: `pyproject.toml`
  - Result: pandas is not in `[project].dependencies`; it remains only in the optional `test` extra for legacy/reference tests.

## Test Results

- Active/default suite: `.venv/bin/python -m pytest --tb=short -q`
  - Result: `1128 passed, 1 warning in 4.03s`.
- Numba helper shard: `.venv/bin/python -m pytest tests/test_numba.py --tb=short -q`
  - Result: `7 passed in 1.62s`.
- Earlier active Polars category suite:
  - Command: `.venv/bin/python -m pytest tests/cycles tests/candles tests/performance tests/statistics tests/transform tests/trend tests/volatility tests/volume tests/overlap tests/momentum tests/utils --tb=short -q`
  - Result: `1121 passed, 1 warning in 2.50s`.

## Known Difference

- `tests/volatility/test_bbands_polars.py::TestPlBbands::test_with_zeros` emits:
  - `RuntimeWarning: divide by zero encountered in divide`
  - Source: `polars_ti/volatility/bbands.py`
  - Behavior: test still passes; the zero-width band case returns the expected null/NaN-safe output.

## Legacy Test Collection

The root pandas-era tests are intentionally excluded in `tests/conftest.py` so default pytest collection exercises the converted Polars suite. This includes old pandas parity tests such as `test_indicator_*.py`, `test_metrics.py`, `test_studies.py`, `test_supertrend_verification.py`, and `test_utils.py`.
