# Remediation Plan — Bring New Polars-TI to Parity with Old + Latest Polars + Comprehensive Tests

Date: 2026-06-20
Companion to: `COMPARISON_REPORT.md` (evidence for every item below).

## Objective

Make the new Polars+Numba library produce output **at parity with the old pandas library for every
indicator and for the overall study**, *except* where the old library was wrong or the new library is
already better (those are preserved and pinned by tests). Run on the **latest Polars**, and add
**comprehensive per-indicator and library-wide tests** that would have caught every defect found.

## Guiding principle: a Parity-Exceptions registry

Most differences should be driven to zero, but a known set must **not** regress to the old behavior.
Create `tests/parity_exceptions.py` (a single source of truth, consumed by the parity tests) encoding,
per column: `{mode: match_old | match_talib | match_canonical | renamed | intentional, tol, note}`.

**Preserve (new is correct/better — pin to TA-Lib/canonical, NOT to old):**
- `WCP` (old native off by 1337), `MFI_14`, `ADX_14`, `RSI_14`, `TRIX_30_9` → assert == direct TA-Lib.
- `CDL_HIGHWAVE`, `CDL_RICKSHAWMAN`, `CDL_SPINNINGTOP` → assert == direct TA-Lib (old was wrong by 100).
- `*_Z_30_1`, `ZS_30` → `ddof=0` (TA-Lib style). `PVI*` → `initial=100` (StockCharts). `MASSI_9_25`,
  `OBVe_*` → keep new (more canonical). Document each as intentional divergence from old.

**Renames/struct output (cosmetic — keep new's struct design + compat layer; Decision 1):**
- The 38 new-only dotted/struct names (`AROON_14.AROOND`↔`AROOND_14`, `KC_e_20_2.kcb`↔`KCBe_20_2`,
  `VWAP_1D`↔`VWAP_D`, `TOS_STDEVALL.L_1`↔`TOS_STDEVALL_L_1`, etc.). **Keep the struct/dotted output as
  canonical** (a deliberate design improvement), record the authoritative old↔new name map, and ship an
  optional **`flatten=True` / pandas-ta-compat accessor** that returns the old flat column names for
  drop-in use (vectorbt, legacy code). Parity tests compare values through the map.

**Fix to match old/canonical (everything else).**

---

## Workstream 0 — Test oracle & harness foundation (do first; everything depends on it)

The current 1128-test suite passes while ~28 indicators are broken — because it never exercises the
native (no-TA-Lib) path, never checks the all-study for completeness, and has no parity oracle. Fix the
oracle before fixing indicators so each fix is verifiable.

1. **Golden reference fixtures** (committed, deterministic): generate once from the OLD library and from
   TA-Lib over the full `data/SPY_D.csv`:
   - `tests/fixtures/old_talib.parquet`, `old_notalib.parquet` (old library, both modes).
   - `tests/fixtures/talib_reference.parquet` (direct `talib.*` for every 1:1 function).
   - Generation script `tests/fixtures/_generate.py` runs in the **old venv** (it can't co-import with new
     — run as a subprocess / documented one-off). Reuse the verified logic in `tmp/compare/dump_old.py`.
2. **Parity engine** `tests/_parity.py`: load new study output + golden, align rows by date, fold struct
   names via the name-map, and per column compute `max_abs/max_rel/null_mismatch` with the
   classification from `tmp/compare/compare.py` (match/warmup/material/no-overlap). Expose a
   `assert_parity(col, mode)` helper that reads `parity_exceptions.py` for the expected verdict + tol.
3. **Default tolerances**: floats `max_abs ≤ 1e-6 or max_rel ≤ 1e-6` after warmup; signal/flag/candle
   columns must be **exact**; warmup-null mismatch ≤ documented per indicator.
4. **Study error visibility (Decision 3)**: add `df.ti.study(..., errors="warn"|"raise"|"ignore")`,
   **default `"warn"`** — dropped indicators are collected and surfaced (logged) but the study still
   completes; non-breaking. Tests run with `errors="raise"`. Today the bare `except Exception` in
   `core.py:_run` is what hides every regression — make failures surfaceable.

Verify: `_generate.py` reproduces the parquet byte-for-byte on re-run; parity engine reruns deterministically.

---

## Workstream 1 — Upgrade to latest Polars + fix packaging (low risk, do early)

The library **already runs on Polars 1.41.2** (full suite green). The only true blocker is scipy.

1. `pyproject.toml`: bump `polars>=1.41` (and `polars-runtime-32`), refresh `uv.lock`.
2. Move **`scipy`** from optional to a **hard runtime dependency** (Numba's `np.convolve`/BLAS in
   `nb_sma` and friends abort the interpreter without it). Add a smoke test asserting `nb_sma` returns
   finite values (guards against this regressing silently).
3. CI matrix: run the suite on `{min-supported, latest}` Polars **and** on `{TA-Lib installed, TA-Lib
   absent}` — the missing no-TA-Lib job is why every native bug shipped.

Verify: `uv run pytest` green on latest Polars; a no-TA-Lib job is green (after WS3/WS4).

---

## Workstream 2 — Restore broken / missing indicators (parity with old)

These raise or emit nothing and are silently dropped from the study (`COMPARISON_REPORT.md` §3). Fix each
to produce correct output in both modes; pin with parity tests vs old/canonical.

| Bucket | Indicators | Likely fix |
|---|---|---|
| `Expr`-API misuse (`'Expr' has no attribute select`, `truth value of Expr ambiguous`, `Expr` as int) | `ha`, `ichimoku`, `mama`/`fama`, `pivots`, `psar`, `hilo`, `rvi`, `vfi`, `vhm` | Port the pandas logic to expression/`map_batches` correctly; stop passing `Expr` where a column name/int/bool is expected |
| Numba typing failures (`non-precise type pyobject`) | `pmax`, `alphatrend`, `zigzag` | Fix kernel signatures/dtypes so `@njit` compiles |
| `talib` kwarg `TypeError` → study retry yields nothing | `macd`, `ppo` | Add/accept-and-ignore `talib` param (or strip unsupported kwargs in the accessor dispatch); confirm output columns |
| No output | `ebsw`, `reflex` | Return the computed column(s); fix the expression that currently yields none |
| Present but all-NaN both modes | `atrts`, `chop`, `cksp`, `halftrend`, `inertia`, `pgo`, `rwi` | Debug each native path (same NaN-seed/Expr family). `cksp` (Decision 2): re-add the `tvmode` toggle and restore the old default (book mode → `10_3_20`); apply the same "restore dropped param / old default" rule to any other indicator that silently changed defaults |

Approach: fix in dependency order (primitives first), one indicator per commit, each gated by its parity
test. `trixh` (errors) is the signal/hist column of `trix` — fix alongside `trix`.

---

## Workstream 3 — Native (no-TA-Lib) correctness: the `presma` seeding bug

Single root cause behind the native all-NaN cluster **and** the KDJ/TSI/TMO/EFI drift
(`COMPARISON_REPORT.md` §5–§7).

1. Fix the Numba EMA/RMA seed so a **leading null/NaN in the input does not poison the recursion** —
   match pandas `ewm`: skip leading NaNs, seed the SMA on the first fully-valid window. Apply in
   `overlap/ema.py` (`_ema_numba` presma branch) and `overlap/rma.py`. This repairs `atr`, `natr`,
   `dema`, `zlma`, and `aberration`'s ATR bands in native mode.
2. **Decision 4 — native = pandas-ta, talib = TA-Lib.** Make the seed convention selectable; the
   **native (talib=False) path seeds from the first valid value** (`ewm(adjust=False)` semantics) so
   `KDJ`, `TSI`, `TMO`, `EFI` (and the rest) match the old golden exactly, while the **talib path keeps
   the TA-Lib SMA seed**. This gives native parity with old and TA-Lib fidelity with TA-Lib.
3. `EFI` native currently routes through TA-Lib EMA because `ma()` defaults `talib=True` — ensure the
   native branch truly uses the native EMA.

Verify: native-mode parity tests for the above pass; no all-NaN columns in `study(talib=False)`.

---

## Workstream 4 — Discrete value bugs

| Indicator | Fix | Pin test against |
|---|---|---|
| `INVFISHER` (`transform/ifisher.py`) | Restore the conditional remap of out-of-`[-1,1]` input to `[-1,1]` (fixed full-series min/max) before `exp(amp·x)` | old/canonical |
| `VIDYA` (`overlap/vidya.py`, `momentum/cmo.py`) | Re-add the `drift` param to `cmo()` (or drop it from the `vidya` call); clamp `α·|cmo|` so the recurrence can't diverge to inf | old (finite) |
| `AMAT`/`AOBV` LR/SR (`trend/increasing.py`, `decreasing.py`) | Guard against Polars `NaN > 0 == True`: `fill_nan(None)`/`is_not_nan()` before the `>0`/`<0` comparison so warmup yields `0`, not spurious `1` | old |
| `AVSL` (`volatility/avsl.py`) | Add OLD's `sma_fast == 0` zero-guard before division | old |
| `OBV` seed (`volume/obv.py`, `utils/_core.py`) | Confirm canonical `OBV[0]=0`; note OLD's `signed_series` ignores its `initial` arg (don't replicate that bug) | canonical |

**Audit for the same classes everywhere** (cheap, high value): grep all indicators for (a) `> 0`/`< 0`
on possibly-NaN expressions (NaN-comparison bug), and (b) `map_batches`/`njit` recurrences without a
divergence/zero guard.

---

## Workstream 5 — Comprehensive test suite

Goal: every defect in `COMPARISON_REPORT.md` would fail a test. Build on WS0.

**Per-indicator (parametrized over all 176 in `maps.Category`):**
1. **Runs clean** in `talib=True` and `talib=False` (no exception, non-empty output).
2. **No all-NaN / no all-null** output columns.
3. **TA-Lib parity** where a 1:1 function exists (assert == `talib_reference.parquet`, tol 1e-8).
4. **Golden parity** vs old (through `parity_exceptions.py`): match, or the documented exception.
5. **Warmup nulls** correct (first N null, rest finite).
6. **Determinism** (same input → same output) and **append semantics** (accessor `append=True`).

**Library-wide / study:**
7. **Completeness**: `study(AllStudy, errors="raise")` runs every indicator with **no silent drops**;
   assert the produced column set == expected set (old parity set − documented renames + struct names).
8. **No all-NaN columns** anywhere in the All study, both modes.
9. **Column count / shape** locked to a golden manifest (`tests/fixtures/expected_columns.json`).
10. **Both talib modes** of the All study compared to the old golden via the parity engine.
11. **Category studies** (`study("momentum")`, …) column counts.

**Infra:**
12. Re-enable/rewrite the `collect_ignore`d legacy oracles (`test_studies.py`, `test_metrics.py`,
    `test_indicator_*`) as the new parity suite (delete the pandas-era ones once superseded).
13. Coverage gate ≥ existing 96%; **add a no-TA-Lib CI job** so native paths are always tested.
14. Pre-warm Numba kernels in `conftest` (already present) — extend to all categories.

Verify: full suite green on latest Polars, **with and without TA-Lib**; intentionally reverting any WS2–4
fix turns a test red.

---

## Sequencing

1. **WS0** (oracle + fixtures + strict study) → 2. **WS1** (Polars/scipy, fast win) → 3. **WS3**
   (presma fix — unblocks native cluster + several value diffs) → 4. **WS2** (restore broken indicators)
   → 5. **WS4** (discrete bugs) → 6. **WS5** (flesh out comprehensive tests as each fix lands; finalize
   CI). WS5 scaffolding starts in WS0 and grows with every fix (TDD: write the failing parity test, then
   fix).

## Resolved decisions (confirmed 2026-06-20)

1. **Column naming → Struct + compat layer.** Keep the struct/dotted output as canonical; add the
   documented old↔new name map and an optional `flatten=True` / pandas-ta-compat accessor returning the
   old flat names. (Drives the Parity-Exceptions "renames" handling and WS5 compat-accessor tests.)
2. **Changed defaults → Restore old defaults + params.** Wherever new dropped a parameter or changed a
   default, restore the old/pandas-ta behavior (CKSP: re-add `tvmode`, default to book `10_3_20`); users
   can still opt into the new variant. (WS2.)
3. **Study errors → Default `"warn"`.** `study(errors="warn"|"raise"|"ignore")`, default `"warn"`:
   dropped indicators surfaced but the study completes; tests use `"raise"`. (WS0 #4.)
4. **EMA/RMA seeding → native = pandas-ta, talib = TA-Lib.** Native path seeds from the first valid
   value (matches old); talib path keeps the TA-Lib SMA seed. (WS3 #2.)

## Definition of done

- Latest Polars; `scipy` a hard dep; CI green on `{latest Polars} × {TA-Lib on/off}`.
- All-study (both modes) matches the old golden for every column except the documented Parity-Exceptions,
  which are pinned to TA-Lib/canonical.
- Zero silently-dropped indicators; zero all-NaN study columns.
- Every indicator has the 6 per-indicator checks; the library has the 5 study-wide checks; coverage ≥ 96%.
