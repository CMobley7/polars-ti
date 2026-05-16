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
  - Result: `1128 passed in 3.77s`.
- Numba helper shard: `.venv/bin/python -m pytest tests/test_numba.py --tb=short -q`
  - Result: `7 passed in 1.62s`.
- Earlier active Polars category suite:
  - Command: `.venv/bin/python -m pytest tests/cycles tests/candles tests/performance tests/statistics tests/transform tests/trend tests/volatility tests/volume tests/overlap tests/momentum tests/utils --tb=short -q`
  - Result: `1121 passed, 1 warning in 2.50s`.

## Known Difference

- Previous `tests/volatility/test_bbands_polars.py::TestPlBbands::test_with_zeros`
  divide-by-zero warning was fixed by using `np.divide(..., where=...)` in
  `polars_ti/volatility/bbands.py`.
- `scipy` is no longer a core runtime dependency. It remains available as an
  optional/transitive extra dependency and is used opportunistically by
  `polars_ti.utils.inv_norm` when installed.

## Audit Follow-up

- High-confidence Ruff audit: `uvx ruff check polars_ti tests scripts --select E9,F63,F7,F82,RUF100,RUF103,RUF104`
  - Result: `All checks passed!`
- Syntax audit: `.venv/bin/python -m compileall -q polars_ti tests scripts`
  - Result: passed.
- Security-pattern audit:
  - `rg -n "eval\(|exec\(|pickle\.|yaml\.load\(|shell=True|os\.system\(" polars_ti scripts tests --glob '*.py'`
  - Result: no matches.
- Dependency audit: `uvx pip-audit --desc off --progress-spinner off`
  - Result: no known vulnerabilities found.
- CI added at `.github/workflows/ci.yml` for install, Ruff high-confidence checks,
  compileall, runtime pandas purge, pytest, and dependency audit on Python 3.11
  and 3.12.

## Legacy Test Collection

The root pandas-era tests are intentionally excluded in `tests/conftest.py` so default pytest collection exercises the converted Polars suite. This includes old pandas parity tests such as `test_indicator_*.py`, `test_metrics.py`, `test_studies.py`, `test_supertrend_verification.py`, and `test_utils.py`.
