# Development

## Setup

```bash
uv sync --locked --extra test --dev
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

The required test matrix has **12 environments**: Python **3.12 / 3.13 / 3.14**,
Polars **1.41.1 / the exact version in uv.lock**, and **TA-Lib installed / absent**.
The separate full-extras workflow tests all optional dependencies on each of the
three Python versions with locked Polars. Both workflows run on development and
main. Development must pass before merging into main.

`UV_PYTHON` overrides the developer default in `.python-version`. The native jobs
install locked test dependencies with TA-Lib excluded, then assert it is absent.
The floor jobs replace locked Polars with exactly 1.41.1. Tests use
`uv run --no-sync` so uv does not silently restore the locked version or TA-Lib.
Each job verifies its actual Python, Polars and TA-Lib selection before testing.

To test the compatibility floor locally in a separate environment:

```bash
export UV_PROJECT_ENVIRONMENT=.venv-312
export UV_PYTHON=3.12
uv sync --locked --extra test --dev
uv pip install --python "$UV_PROJECT_ENVIRONMENT/bin/python" "polars==1.41.1"
uv run --no-sync pytest tests/ -q --tb=short
# Also exercise a genuinely TA-Lib-free environment:
uv pip uninstall --python "$UV_PROJECT_ENVIRONMENT/bin/python" TA-Lib
uv run --no-sync pytest tests/ -q --tb=short
unset UV_PROJECT_ENVIRONMENT UV_PYTHON
```

For a quick native-path check in your usual environment:

```bash
POLARS_TI_SIMULATE_NO_TALIB=1 uv run pytest tests/ -q
```

This forces native paths and blocks `import talib`; parity tests requiring the
TA-Lib oracle skip. CI additionally verifies actual package absence.
See [compatibility](compatibility.md) for the support policy. Ruff targets
Python 3.12 syntax so formatting cannot introduce Python 3.14-only syntax.
The existing mypy configuration invokes strict mode but suppresses errors in
the runtime and audit packages; a passing gate does not establish full type safety.

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

- `tests/test_talib_parity.py` — the full per-column TA-Lib-mode gate: every
  shared all-study column matches the pandas golden within tolerance, except the
  documented `match_talib`/`intentional`/`TALIB_DIVERGENCE` exceptions.
- `tests/test_native_parity.py` — native-mode parity vs the native golden, with
  documented `NATIVE_DIVERGENCE` (columns where pandas-ta's native golden was
  TA-Lib-contaminated).
- `tests/test_study_completeness.py` — column manifest + no all-NaN columns, both modes.
- `tests/test_parity_smoke.py` — a fast oracle sanity subset (not the full gate).
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
