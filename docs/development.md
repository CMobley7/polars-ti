# Development

## Setup

```bash
uv sync --extra test --dev
```

## Quality gates

`scripts/check.sh` runs the same gates as CI:

```bash
./scripts/check.sh          # full (includes pip-audit)
./scripts/check.sh --fast   # skip the dependency audit
./scripts/check.sh --fix    # apply safe ruff fixes first
```

Gates: `uv lock` check, Ruff lint, Ruff format, mypy `--strict`, a syntax
compile, a runtime **pandas-purge** check (no `pandas`/`pd.` in the runtime
package), pytest with coverage (`--cov-fail-under=90`), and a dependency audit.

Run the suite directly:

```bash
uv run pytest tests/ -q --tb=short
```

## CI matrix

CI runs the test job across **{Python 3.11, 3.12} × {oldest-supported, latest
Polars} × {TA-Lib installed, TA-Lib absent}**. The no-TA-Lib leg is a hard gate
that exercises the native (Numba) code paths. Reproduce it locally with:

```bash
POLARS_TI_SIMULATE_NO_TALIB=1 uv run pytest tests/ -q
```

This forces the native paths even when TA-Lib is installed (and blocks
`import talib`), so parity tests that grade against the TA-Lib golden skip just
as they would in a real TA-Lib-absent environment.

## The parity oracle

Correctness is enforced against committed **golden fixtures** generated from the
pandas baseline and from TA-Lib, over a deterministic slice of `data/SPY_D.csv`:

| File | Purpose |
| :--- | :--- |
| `tests/fixtures/old_talib.parquet`, `old_notalib.parquet` | pandas baseline all-study output, both TA-Lib modes |
| `tests/fixtures/talib_reference.parquet` | direct TA-Lib reference for 1:1 functions |
| `tests/fixtures/expected_columns.json` | the per-mode column manifest |
| `tests/_parity.py` | comparison engine + the old↔new `RENAME_MAP` |
| `tests/parity_exceptions.py` | per-column verdicts (`match` / `match_talib` / `intentional`) |

Key test modules:

- `tests/test_parity_smoke.py` — all-study parity vs the pandas golden (TA-Lib mode).
- `tests/test_native_parity.py` — native-mode parity vs the native golden, with
  documented `NATIVE_DIVERGENCE` (columns where pandas-ta's native golden was
  TA-Lib-contaminated).
- `tests/test_study_completeness.py` — column manifest + no all-NaN columns, both modes.
- `tests/test_indicators_parametrized.py` — every indicator runs, is non-empty,
  and is deterministic.

When an indicator's expected output legitimately changes, update its verdict in
`tests/parity_exceptions.py` (and, if a column was renamed, `RENAME_MAP` in
`tests/_parity.py`). See [Differences from pandas-ta](differences-from-pandas-ta.md)
for the rationale behind the current exceptions.

## Project layout

```
polars_ti/<category>/<indicator>.py   # one module per indicator
polars_ti/core.py                     # the df.ti accessor + study()
polars_ti/maps.py                     # Category registry (drives studies)
polars_ti/ma.py                       # moving-average dispatcher
tests/                                # parity oracle + per-indicator tests
docs/                                 # this documentation
```
