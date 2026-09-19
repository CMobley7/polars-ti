# Compatibility and documentation follow-up

This follow-up to `cf108754f40b49010723cfce85e1c6c7dbd29abd` and
`ea15bcb4eda83ff4e0a5f81ce76dc0d6e937af02` restores supported versions that were
unnecessarily narrowed during the latest-runtime upgrade. See the current
[compatibility policy](../compatibility.md).

## Changes and rationale

- Restore Python 3.12, 3.13 and 3.14; keep 3.14.7 as the development default.
  NumPy 2.5.3 and SciPy 1.18.1 establish the Python 3.12 minimum.
- Allow Polars >=1.41.1,<2 while retaining 1.44.2 in the lock. Polars 1.41.0 is
  yanked; the first non-yanked 1.41 patch is the tested floor.
- Expand CI to all 12 Python/Polars/TA-Lib combinations and all three Python
  versions with full optional dependencies. Assert actual environment versions;
  use `--no-sync` so testing cannot undo the selected floor or TA-Lib absence.
- Keep the latest scientific dependency versions; regenerate the universal lock
  for the expanded Python range. All 123 resolved package versions are unchanged.
- Target Python 3.12 in Ruff and mypy. Python 3.12 initially failed to parse two
  exception handlers reformatted for Python 3.14. Parenthesized exception tuples
  restore compatibility without changing exception handling.
- Audit both prior commits for removed explanatory comments. Restore the fixed
  CRSI denominator, regression coordinates, MAVP period/warmup rules, entropy
  definition, moment conventions, weight orientation and allocation guards.
  Explain the rolling-weight identity, order-statistic causality, anchored MAD,
  conditional moment rebuilding, FIR accuracy fallback and causal candle moments.
- Correct stale docs about RMA seeding, candle-z-score `ddof`, native versus TA-Lib
  gap recovery, Boolean doji warmup, ALMA's FIR implementation and SciPy's current
  direct dependencies. Distinguish historical benchmark environments from current
  support. Installation examples now use the source checkout; this project has
  no published PyPI distribution at the time of this audit.

## Validation

Before correction, Python 3.12 rejected the >=3.14 metadata and later exposed
`SyntaxError` in the two exception handlers. Floor installation also rejected the
previous exact Polars pin. The compatibility matrix is now a permanent CI gate.

Each row below ran the complete suite with all optional dependencies, then ran
it again after uninstalling TA-Lib and verifying it could not be imported.

| Python | Polars | TA-Lib installed: passed / skipped | TA-Lib absent: passed / skipped |
| :--- | :--- | ---: | ---: |
| 3.12.12 | 1.41.1 | 3,278 / 56 | 2,899 / 433 |
| 3.12.12 | 1.44.2 | 3,278 / 56 | 2,899 / 433 |
| 3.13.12 | 1.41.1 | 3,278 / 56 | 2,899 / 433 |
| 3.13.12 | 1.44.2 | 3,278 / 56 | 2,899 / 433 |
| 3.14.7 | 1.41.1 | 3,278 / 56 | 2,899 / 433 |
| 3.14.7 | 1.44.2 | 3,278 / 56 | 2,899 / 433 |

The standard quality script passed: lock check, Ruff lint/format, configured mypy,
syntax compilation, runtime pandas purge, tests with **91.28% coverage**, and
dependency audit. A separate audit of the full environment with Polars 1.41.1
also found no known vulnerabilities. The unpublished local package itself cannot
be checked against PyPI's vulnerability database.

Executable AST comparison against `ea15bcb` confirms no runtime algorithm change
in the 21 edited production modules after stripping docstrings. Numerical test
tolerances and golden fixtures are unchanged. This follow-up makes no new speed
claim; measured kernel improvements and accuracy evidence remain in the
[rolling-kernel report](../../scripts/rolling_kernel_audit/REPORT.md).

## Independent review

Astra reviewed the full diff, lock, CI and restored explanations. Its remaining
findings concerned the obsolete SMA/BLAS dependency rationale and an overly broad
moment-centering docstring; both were corrected and Astra independently rechecked
them. Terra then completed its own review of metadata, lock, workflows, test
evidence and documentation. Both final reviews found no significant unresolved
issues. Development CI must pass before integration into main.

## Files changed

- `.github/workflows/ci.yml`
- `.github/workflows/test.yml`
- `README.md`
- `docs/compatibility.md`
- `docs/development.md`
- `docs/differences-from-pandas-ta.md`
- `docs/getting-started.md`
- `docs/index.md`
- `docs/migrating-from-pandas-ta.md`
- `docs/talib.md`
- `docs/upstream/2026-09-compatibility-review.md`
- `docs/upstream/2026-09-release-review.md`
- `polars_ti/candles/cdl_z.py`
- `polars_ti/core.py`
- `polars_ti/momentum/cg.py`
- `polars_ti/momentum/crsi.py`
- `polars_ti/overlap/alma.py`
- `polars_ti/overlap/fwma.py`
- `polars_ti/overlap/linreg.py`
- `polars_ti/overlap/mavp.py`
- `polars_ti/overlap/pwma.py`
- `polars_ti/overlap/rma.py`
- `polars_ti/overlap/sinwma.py`
- `polars_ti/overlap/swma.py`
- `polars_ti/statistics/entropy.py`
- `polars_ti/statistics/kurtosis.py`
- `polars_ti/statistics/skew.py`
- `polars_ti/statistics/zscore.py`
- `polars_ti/trend/adx.py`
- `polars_ti/utils/_fir.py`
- `polars_ti/utils/_order_stats.py`
- `polars_ti/utils/_prefix.py`
- `polars_ti/utils/_rolling.py`
- `pyproject.toml`
- `scripts/rolling_kernel_audit/README.md`
- `scripts/rolling_kernel_audit/REPORT.md`
- `tests/test_numba_blas.py`
- `uv.lock`
