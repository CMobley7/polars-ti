# Polars-TI: Old (pandas) vs New (Polars+Numba) — All-Study Comparison

Date: 2026-06-20

## 1. Scope & method

Goal: verify the new Polars+Numba rewrite produces the **same or better** output than the
previous all-pandas library, for every indicator and overall, **both with TA-Lib and without**.

- **Old** = `tmp/polars-ti/` (pandas 2.3.3, TA-Lib 0.6.8, Python 3.12).
- **New** = repo root (polars 1.40.1, numba 0.63.1, TA-Lib 0.6.8, scipy 1.17.1, Python 3.11).
- **Dataset**: `data/SPY_D.csv` — full 7192 daily SPY bars; byte-identical in both trees.
- **Runs (4)**: `df.ti.study(ti.AllStudy, cores=0, talib=True)` and `talib=False`, for each library,
  dumped to Parquet.
- **Ground truth**: direct TA-Lib 0.6.8 calls where a 1:1 function exists — this decides "better".
- **Harness (throwaway)**: `tmp/compare/{dump_old,dump_new,compare,ground_truth,diag_new}.py`,
  outputs under `tmp/compare/out/`.
- New emits several multi-output indicators as Polars structs; these were flattened to dotted
  `key.field` names and folded onto the old pandas names via suffix-matching before diffing.

One environment fix is required to run the new library: **scipy must be installed**, or every Numba
kernel that uses `np.convolve`/BLAS (e.g. `nb_sma`) aborts the interpreter with *"Specified LAPACK
function could not be found"* — even though scipy is only an optional dependency. (The library itself
runs fine on the latest **polars 1.41.2** — the full 1128-test suite passes; an earlier `map_batches
return_dtype` error was a misleading secondary symptom of the scipy/LAPACK fatal crash, not a real
Polars-version incompatibility.)

> Note on NaN: in Polars, `NaN` is **not** null. Several new indicators return columns that are
> "non-null" but entirely `NaN`; all counts below treat `NaN` as missing.

## 2. Overall study comparison (the "and overall" question)

| Run | Total cols | Indicator cols |
|---|---:|---:|
| OLD, talib=True | 394 | 387 |
| OLD, talib=False | 394 | 387 |
| NEW, talib=True | 341 | 240 |
| NEW, talib=False | 281 | 180 |

| Mode | Common | match | warmup/null | **material diff** | **all-NaN (no overlap)** | old-only | new-only |
|---|---:|---:|---:|---:|---:|---:|---:|
| talib=True | 295 | 254 | 7 | 29 | 5 | 92 | 38 |
| talib=False | 235 | 169 | 11 | 43 | 12 | 152 | 38 |

Reading this:

- **The old library produces the same 387 columns whether or not TA-Lib is used.** The new library
  produces **240 with TA-Lib and only 180 without** — i.e. it is far more dependent on TA-Lib, and
  loses 60 columns when TA-Lib is absent.
- Of the 295 columns common with TA-Lib on, **254 match to floating-point noise (≤1e-6)** — for the
  indicators that work, the rewrite is numerically faithful.
- **new-only (38)**: all are cosmetic struct/field renames of existing old columns
  (`AROON_14.AROOND`↔`AROOND_14`, `KC_e_20_2.kcb`↔`KCBe_20_2`, `TOS_STDEVALL.L_1`↔`TOS_STDEVALL_L_1`,
  `VWAP_1D`↔`VWAP_D`, …). The only non-cosmetic one is **CKSP**, whose default params changed
  (`10_3_20` → `10_1.0_9`). No genuinely-new indicator output.
- **old-only**: a mix of (a) the cosmetic counterparts of the new-only renames, and (b) **genuinely
  missing indicators** — these are the real reason the studies differ in shape (§3).

## 3. Why the studies differ in shape: broken / missing indicators in NEW

This is the dominant cause of the column-count gap. The all-study loop **silently swallows any
indicator that raises or returns nothing**, so broken indicators just vanish from the output. A
standalone health-check of all 176 indicators (`diag_new.json`) plus the column-set diff identifies
**~28 indicators that fail or degrade in the new library but worked in the old one**:

| Failure mode (NEW) | Indicators | Notes |
|---|---|---|
| **Raises in both modes** (absent from study) | `ha` (Heikin-Ashi), `ichimoku`, `mama`/`fama`, `pivots`, `pmax`, `psar`, `hilo`, `alphatrend`, `zigzag`, `rvi`, `vfi`, `vhm`, `trixh` | Errors like `'Expr' object has no attribute 'select'`, `truth value of an Expr is ambiguous`, Numba typing failures |
| **`talib` kwarg crashes → study retry yields nothing** | `macd`, `ppo` | Underlying `macd()`/`ppo()` don't accept `talib`; the study catches the `TypeError`, retries without it, and gets **no columns**. MACD/PPO are simply **missing** from the new study. |
| **Returns no output** | `ebsw`, `reflex` | Produce no columns at all |
| **Present but entirely NaN, both modes** | `atrts`, `chop`, `cksp`, `halftrend`, `inertia`, `pgo`, `rwi` | Column emitted, all values NaN |
| **OK with TA-Lib, all-NaN without** | `atr`, `natr`, `dema`, `zlma` (and `aberration`'s ATR bands) | Native fallback bug — see §5 |

The old library produced all of these correctly. **These are real regressions**, and several are
core indicators (MACD, PPO, PSAR, Heikin-Ashi, Ichimoku, PMAX, QQE). They are *not* visible from the
test suite because the study path hides the exceptions.

## 4. Where NEW is genuinely BETTER (validated against TA-Lib)

For indicators with a 1:1 TA-Lib function, deviation from direct TA-Lib output decides correctness.
In **native (talib=False)** mode the new implementations are *more* correct than the old ones:

| Indicator | OLD native vs TA-Lib (max abs) | NEW native vs TA-Lib (max abs) | Verdict |
|---|---:|---:|---|
| `WCP` (weighted close) | **1337** (old native is wrong) | ~0 | **NEW correct** |
| `MFI_14` | 8.94 | ~3e-13 | **NEW correct** |
| `ADX_14` | 6.96 | 0.19 | **NEW much closer** |
| `RSI_14` | 18.0 | 8.3 | NEW closer (both warmup-diverge early) |
| `TRIX_30_9` | 1.8e-3 | ~3e-13 | **NEW essentially exact** |

Plus 3 candlestick patterns that differ even with TA-Lib on — **new matches TA-Lib exactly, old was
wrong by 100**: `CDL_HIGHWAVE`, `CDL_RICKSHAWMAN`, `CDL_SPINNINGTOP` (the other ~60 CDL patterns
match exactly in both).

(`ADXR_14_2`, `DMP_14`/`DMN_14`, `KAMA_10_2_30` show large deviations from TA-Lib in **both** old and
new — these use pandas-ta's own definitions, not TA-Lib's `ADXR`/`PLUS_DI`/`KAMA`, so TA-Lib is not a
valid reference for them. Old and new agree with each other there.)

## 5. Root cause of the native all-NaN cluster (single bug)

`atr`, `natr`, `dema`, `zlma`, and `aberration`'s ATR bands return **all-NaN** in native mode while
the `RMA`/`EMA` primitives they build on work fine. Verified cause:

- The native MA path passes **`presma=True`**, which seeds the recursive RMA/EMA with an **SMA of the
  first `length` values**.
- When the *input* series has any leading null/NaN, that seed SMA is NaN, and the recursion
  propagates NaN through the entire column.
- `atr` feeds `rma(true_range, presma=True)` and `true_range` has a leading null → all-NaN. `dema`
  feeds `_ema_numba(ema1, presma=True)` where `ema1` already carries `length-1` leading NaN from its
  own presma warmup → all-NaN. Setting `presma=False` makes both produce correct values.

This is one defect with several symptoms; old handled the leading-NaN seed (pandas `ewm` skips leading
NaNs and seeds on the first valid value) and so worked. **NEW is worse here** until fixed.

This same seeding choice — NEW's Numba EMAs default to **`presma=True`** (an SMA-of-first-`length`
seed, TA-Lib style) while OLD seeds recursively from the first valid value (pandas `ewm(adjust=False)`,
the pandas-ta canonical convention) — is also the root of the small KDJ/TSI/TMO/EFI differences in §6.
It is the single highest-leverage area to align.

## 6. Genuine NEW bugs found in value-producing indicators

Beyond the all-NaN cluster (§5), three more indicators that *do* emit numbers are **wrong**, confirmed
by reading both source trees:

| Indicator | Δ | Code-level root cause | Verdict |
|---|---:|---|---|
| `INVFISHER_1.0` (`transform/ifisher.py:45-46`) | 1.46 | NEW **dropped the input rescaling**. OLD remaps the input to [-1,1] (via fixed full-series min/max) when it falls outside [-1,1], as the Inverse-Fisher transform requires; NEW feeds raw prices into `exp(amp·x)`, which **saturates to ≈1.0 for every bar**. | **NEW broken**; OLD canonical |
| `AMATe_LR/SR`, `AOBV_LR/SR` (`trend/increasing.py:56`, `decreasing.py:55`) | 1 (several early bars) | **Polars `NaN > 0` evaluates `True`** (NaN sorts greatest), opposite of numpy. The long/short-run signals compare `diff(length) > 0` on the MA warmup region, which is NaN → NEW emits **spurious `1` signals across the warmup span**. OLD (pandas, `nan>0 == False`) emits `0`. | **NEW wrong** (warmup signals) |
| `VIDYA_14` (`overlap/vidya.py:85,18`) | **inf** (native) | Two bugs: NEW `cmo()` dropped the `drift` parameter that `vidya` still passes → `TypeError`; and the recurrence `v[i]=α·|cmo|·c + v[i-1]·(1−α·|cmo|)` has **no clamp**, so any CMO-scaling slip makes `(1−α·|cmo|)` diverge to inf. | **NEW broken**; OLD finite |

## 7. Remaining value differences — convention only (no 1:1 TA-Lib reference)

These differ but neither is objectively wrong; the cause is a smoothing/warmup/normalization
convention. Magnitudes are max-abs-diff over 7192 bars.

| Indicator(s) | Δ | Cause | Verdict |
|---|---:|---|---|
| `close/open/high/low_Z_30_1`, `ZS_30` | ~0.07 | **NEW `ddof=0`** (population std, TA-Lib style) vs OLD `ddof=1` (sample) | NEW aligns with TA-Lib |
| `PVI`, `PVIe_255` | ~105 | **NEW seeds `initial=100`** (canonical StockCharts) vs OLD seeds at first close (~25.8) | NEW more canonical |
| `K/D/J_9_3` (KDJ) | 6–14 | RMA seed: NEW SMA-seed/`presma`, OLD pandas-ewm first-value seed; plus div-by-zero epsilon | Convention diff |
| `TSI/TSIs`, `TMO/TMOs`, `EFI_13` | ~0.7–614 | Same EMA-seed convention (`presma=True` vs `ewm(adjust=False)`); EFI native also silently routes through TA-Lib EMA (`ma()` defaults `talib=True`) | Convention diff; fix = `presma=False` for pandas-ta parity |
| `CRSI_3_2_100` | 0.017 | 1-bar window alignment in the PercentRank component | Convention diff (immaterial) |
| `MASSI_9_25` | 0.08 | NEW's NaN-skipping cascaded EMA vs OLD's nested TA-Lib EMA (which degenerates) | NEW arguably more correct |
| `OBVe_4/OBVe_12` | rel .04–.14 | 1-bar OBV seed: NEW `OBV[0]=0` (canonical) vs OLD `signed_series` ignores its `initial` arg | NEW marginally better |
| `AVSL_12_26` | ~38 (rel .08) | identical formula; differs only in division-guard / rolling-NaN edge handling (NEW lacks OLD's `sma_fast==0` guard) | Convention diff (NEW slightly riskier) |
| `SMC*` (7 cols) | up to 430 | Smart-Money-Concept normalization genuinely diverges (NEW `SMCbp` ranges to 415.7 vs OLD 56.7); non-canonical indicator, no reference | Real difference; neither verifiably correct |
| `KURT_30` | 3e-4 | rolling-window float noise | Equivalent |

## 8. Bottom line

- **With TA-Lib installed**, the new library is **as good or better** on the indicators it actually
  emits: 254/295 columns match to float noise, and where it differs against a TA-Lib reference it is
  *more* correct (WCP, MFI, ADX, RSI, TRIX, 3 candle patterns). Intentional convention changes
  (z-score `ddof=0`, PVI `initial=100`) move it closer to TA-Lib/StockCharts conventions.
- **It is not yet at parity overall.** The new library drops/breaks **~28 indicators** that the old
  one computed — including core ones (MACD, PPO, PSAR, Heikin-Ashi, Ichimoku, PMAX, QQE, MAMA/FAMA,
  PIVOTS, ZigZag) — and these failures are hidden because the study swallows exceptions.
- **Its native (no-TA-Lib) path is materially weaker**: it loses 60 columns versus its own TA-Lib run
  and a cluster of indicators go all-NaN from one `presma` seeding bug (ATR/NATR/DEMA/ZLMA/ABER), plus
  three more value-producing indicators are outright wrong (`INVFISHER` saturates to ≈1, `VIDYA`
  overflows to inf, `AMAT`/`AOBV` emit spurious warmup signals from Polars' `NaN > 0` semantics).
- **Recurring themes**: most defects trace to two things — EMA/RMA **seed convention** (`presma` SMA-seed
  vs pandas first-value seed) interacting with leading nulls, and **Polars NaN semantics** (`NaN` is not
  null; `NaN > 0` is `True`). These two ideas explain the native all-NaN cluster, the AMAT/AOBV signal
  bug, and the KDJ/TSI/TMO/EFI convention drift.

> Verification note: a sub-agent initially reported SMC as bit-identical between the two libraries; that
> was a false positive (its "old" import silently resolved to the new package). Re-checked directly —
> SMC genuinely differs (`SMCbp` ranges to 415.7 in new vs 56.7 in old). All other verdicts here are
> source- or TA-Lib-grounded.

### Recommended fixes (priority order)
1. Make `macd`/`ppo` accept/ignore `talib`; restore `ha`, `psar`, `ichimoku`, `mama`, `pmax`,
   `pivots`, `hilo`, `zigzag`, `rvi`, `vfi`, `vhm`, `alphatrend`, `trixh` (porting/`Expr` API bugs).
2. Fix the `presma`-seed-over-leading-NaN bug (ATR/NATR/DEMA/ZLMA/ABER native mode); consider defaulting
   `presma=False` so Numba EMAs match the pandas-ta convention (also fixes KDJ/TSI/TMO/EFI drift).
3. Fix the three value bugs: `INVFISHER` (restore the [-1,1] input remap), `VIDYA` (`cmo` `drift` kwarg +
   clamp the recurrence), and `AMAT`/`AOBV` (guard `increasing`/`decreasing` against NaN before `>0`/`<0`).
4. Fix all-NaN-both-modes: `atrts`, `chop`, `cksp`, `halftrend`, `inertia`, `pgo`, `rwi`; and no-output
   `ebsw`, `reflex`.
5. Stop the study from silently swallowing indicator exceptions (so regressions surface in tests).
6. Packaging: make `scipy` a **hard** runtime dependency (Numba BLAS); the library already runs on the
   latest `polars` (1.41.2 verified, full suite green) so just bump the floor and keep current.
