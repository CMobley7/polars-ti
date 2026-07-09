# Differences from pandas-ta

This document catalogs how **Polars-TI** differs from
[twopirllc/pandas-ta](https://github.com/twopirllc/pandas-ta) — both the
deliberate architectural/API changes and every known difference in indicator
**output values**, with the reason for each. It also lists the indicators added
on top of upstream and credits their sources.

If you are porting existing pandas-ta code, start with the
[migration guide](migrating-from-pandas-ta.md); this page is the *what changed
and why* reference.

---

## 1. Summary

- **Engine:** pure **Polars + Numba/NumPy**. There is **zero runtime pandas
  dependency** — indicators are Polars expressions, not pandas `Series`
  operations.
- **Indicators:** **267** total — **176** from the original pandas-ta
  (development) port (including **17 added** from community forks, see
  [§6](#6-new-indicators--credits)), plus **86** added for feature parity with the
  community fork `xgboosted/pandas-ta-classic` (see
  [§8](#8-feature-parity-with-pandas-ta-classic)), plus a further **3** TA-Lib
  indicators (`adxr`, `tsf`, `mavp`) added to complete TA-Lib coverage (see
  [§8h](#8h-additional-ta-lib-parity-indicators-3)). The classic additions include a
  full native candlestick-pattern suite, so the **candles** category alone holds
  **65** functions.
- **Columns:** the Polars all-study output is a **superset** of the pandas
  all-study output — every pandas column is reproduced; none are dropped
  (see [§3](#3-indicator--column-counts)).
- **Numerical differences:** the overwhelming majority of columns match the
  pandas output within floating-point tolerance. Every remaining difference is
  documented and falls into one of: Polars-TI matching TA-Lib/canonical where
  pandas-ta was buggy, a deliberate convention change, or offering a TA-Lib path
  pandas-ta lacked. None are Polars-TI being wrong, and **every shared column is
  parity-tested per-column** (see [§4](#4-output-differences-with-reasons)).

---

## 2. Architecture & API changes

| Area | pandas-ta | Polars-TI |
| :--- | :--- | :--- |
| Data engine | pandas `Series`/`DataFrame` | Polars `Expr` / `DataFrame`; Numba for stateful loops |
| Runtime pandas | required | **none** (enforced by a CI "pandas purge" check) |
| Call style | `df.ta.rsi()` | `df.ti.rsi()` **and** `from polars_ti.momentum import rsi` → `df.select(rsi("close"))` |
| Legacy `pl_*` names | n/a | removed — use the indicator name directly (`rsi`, `sma`, `atr`, …) |
| Multi-output indicators | multiple columns in a DataFrame | a single Polars **struct** column (accessor) or a **list of expressions**; unnest for flat columns |
| TA-Lib | `talib=` per call | `talib=` per call **and** a study-wide `talib=` flag; native paths fully implemented so TA-Lib is optional |
| TA-Lib param honoring | `talib=True` silently **ignored** non-default extra params the TA-Lib C function can't take (`scalar`, `c`, `mamode`, `ddof`, `drift`, `presma`, `min_periods`, weights, …) | those params are **honored** — linear ones (`scalar`, `c`) are rescaled, the rest fall through to the native path; defaults are byte-identical (see [§4e](#4e-parameter-honoring-on-the-ta-lib-path)) |
| Invalid period params | a negative / non-integer `length`/`fast`/`slow`/… could reach a Numba kernel and **crash the process** (heap OOB / SIGSEGV) | no indicator crashes the process on a bad period any more; the ~40 crash-prone kernels are guarded and raise a clear `ValueError` (see [§4e](#4e-parameter-honoring-on-the-ta-lib-path)) |
| Study errors | exceptions bubble up | `df.ti.study(..., errors="warn"|"raise"|"ignore")`, default `"warn"`; an unknown indicator `kind` in a custom `Study` honors this policy (raises under `"raise"`), and a `study` argument that is neither a category string nor a `Study` raises `TypeError` |

See [Getting started](getting-started.md) for the return-type details and how to
unnest struct columns.

---

## 3. Indicator & column counts

This section compares Polars-TI against the **original pandas-ta (development)
baseline** it was ported from. The numbers below therefore reflect the
**176-indicator** original port; the **86** additional indicators from
`pandas-ta-classic` plus **3** further TA-Lib-parity indicators are covered
separately in [§8](#8-feature-parity-with-pandas-ta-classic) and bring the current
total to **267** (see [§1](#1-summary)).

Measured on a deterministic 1,500-row slice of `data/SPY_D.csv`, all-study,
with TA-Lib installed:

| | pandas baseline | Polars-TI (original port) |
| :--- | :---: | :---: |
| Indicators (Category registry) | 176 | 176 |
| All-study indicator columns (TA-Lib mode) | 387 | **389** |
| Columns dropped vs pandas | — | **0** |
| Extra columns exposed | — | **+2** (`DMP_14`/`DMN_14` are also surfaced inside the `ADX` struct) |
| All-study indicator columns (native, no TA-Lib) | — | 329 |

The native (no-TA-Lib) figure above (**329**) reflects the **original
176-indicator port**, when ~60 TA-Lib candlestick patterns had no native
implementation (pandas-ta behaves the same way when TA-Lib is absent). The
`pandas-ta-classic` port has since added a full **native** candlestick suite
(see [§8e](#8e-full-native-candlestick-suite-60-patterns)), so the **current**
all-study produces the **same column set in both modes** — native mode no longer
drops the candlestick patterns.

Some indicators changed their **column names** (struct/dotted layout). Those are
not drops; they are folded by an authoritative old↔new name map
(see [§4d](#4d-renames--struct-outputs)).

---

## 4. Output differences (with reasons)

Of the 387 shared columns (measured in TA-Lib mode vs the pandas golden):
**363 match within tolerance** — 356 to floating-point noise
(`max_abs`/`max_rel` ≤ 1e-6) and 7 that agree post-warmup with only a warm-up
null-mask difference. The remaining 24 are accounted for below: **8** pinned to
TA-Lib (§4a), **14** intentional convention divergences (§4b), and **2**
documented special cases — KAMA's TA-Lib path (§4c) and the `HT_direction`
string column. Every shared column is enforced per-column by
`tests/test_talib_parity.py` (TA-Lib mode) and `tests/test_native_parity.py`
(native mode), using the verdicts in `tests/parity_exceptions.py`.

### 4a. Polars-TI is more correct (old value was a bug)

These were wrong in pandas-ta's **native** (no-TA-Lib) path. Polars-TI matches
TA-Lib / the canonical definition and is pinned to TA-Lib in the test-suite.

| Column(s) | pandas-ta native error | Polars-TI |
| :--- | :--- | :--- |
| `WCP` (Weighted Close) | off by ~1337 | matches TA-Lib `WCLPRICE` |
| `MFI_14` | off by ~8.94 | matches TA-Lib (~3e-13) |
| `ADX_14` | off by ~6.96 | much closer to TA-Lib |
| `RSI_14` | off by ~18 | matches TA-Lib |
| `TRIX_30_9` | off by ~1.8e-3 | matches TA-Lib (~3e-13) |
| `CDL_HIGHWAVE`, `CDL_RICKSHAWMAN`, `CDL_SPINNINGTOP` | off by 100 (sign/scale) | match TA-Lib exactly |

### 4b. Intentional convention divergences (Polars-TI is canonical/better)

Deliberate, documented changes — Polars-TI does **not** reproduce the old value
here on purpose.

| Column(s) | Change & reason |
| :--- | :--- |
| `*_Z_30_1`, `ZS_30` (z-scores) | use population std (`ddof=0`, TA-Lib style) instead of sample std (`ddof=1`) |
| `PVI`, `PVIe_255` | seed at `initial=100` (StockCharts canonical) instead of the first close |
| `MASSI_9_25` | NaN-skipping cascaded EMA (more canonical than the old nested TA-Lib EMA) |
| `OBVe_4`, `OBVe_12` | canonical `OBV[0]=0` seed (old `signed_series` ignored its `initial` arg) |
| `VHM_610` | proper **rolling** standard deviation. The old code called `pstdev(volume, slength)`, but `statistics.pstdev`'s 2nd argument is the assumed *mean*, not a window — so the old denominator was a single ~3.9M constant for the whole series. |
| `ABER_ATR/SG/XG_5_15` | native ATR bands. The old "native" golden actually held TA-Lib ATR values (a talib non-propagation bug); Polars-TI's native path is true pandas-ta semantics. |

### 4c. Native vs TA-Lib paths (`talib=` flag)

Polars-TI honors `talib=False` consistently: with the flag off it uses the
native Polars/Numba path for an indicator **and all of its sub-indicators**
(ATR, EMA, RSI, CMO, …). pandas-ta frequently failed to propagate the flag, so
its "native" output for ~45 indicators silently used TA-Lib internally. Because
of that, Polars-TI's *native-mode* output legitimately diverges from pandas-ta's
*native* golden for those indicators — Polars-TI is the correct one
(see `tests/test_native_parity.py`, `NATIVE_DIVERGENCE`). TA-Lib-mode output
matches. This is a consequence of the design rule **native = pandas-ta
semantics, talib = TA-Lib fidelity**.

The same rule runs the other way for **`KAMA`**: pandas-ta had *only* a native
KAMA, so the golden is native KAMA in both modes. Polars-TI's `talib=False` path
reproduces it exactly, while `talib=True` returns TA-Lib's KAMA — which uses
different internal fast/slow constants and so differs from the golden (~0.15).
This is the one TA-Lib-mode column that diverges from the pandas golden and is
not a bug; it is exempted in `tests/test_talib_parity.py` (`TALIB_DIVERGENCE`).

### 4d. Renames / struct outputs

Multi-output indicators emit a single **struct** column instead of several flat
columns. Unnesting yields the familiar names. A few use shorter/dotted struct
field keys; the parity suite folds them via an authoritative map. Representative
mappings (old flat → new):

| pandas-ta column | Polars-TI |
| :--- | :--- |
| `AROOND_14`, `AROONU_14`, `AROONOSC_14` | fields of the `AROON_14` struct |
| `KCLe_20_2`, `KCBe_20_2`, `KCUe_20_2` | `KC_e_20_2` struct (`kcl`/`kcb`/`kcu`) |
| `THERMO*_20_2_0.5` | `THERMO_20_2_0.5` struct |
| `QQE*_14_5_4.236` | `QQE_14_5_4.236` struct |
| `TOS_STDEVALL_*` | `TOS_STDEVALL` struct |
| `VWAP_D` | `VWAP_1D` |

The full map lives in `tests/_parity.py` (`RENAME_MAP`). To get flat columns with
the familiar names, unnest the struct column (e.g. `.unnest("AROON_14")`).

### 4e. Parameter honoring on the TA-Lib path

pandas-ta's `talib=True` branch calls the TA-Lib C function directly, which
**silently drops** any extra parameter TA-Lib cannot represent — so
`rsi(scalar=50, talib=True)` returned the same values as the default, ignoring
`scalar`. Polars-TI **honors** these parameters even on the TA-Lib path, while
keeping the default output **byte-identical** to TA-Lib:

- **Linear parameters are rescaled** on the TA-Lib result: `scalar` (bop, rsi,
  adx, adxr, dx, roc, cmo, ppo, trix) and `c` (cci). Example:
  `bop(scalar=2.0, talib=True)` is exactly `2 ×` the default.
- **Non-linear parameters fall through to the native path**, which already
  applies them: `mamode` (atr, natr, accbands, dx), `ddof` (stdev, variance,
  bbands), `drift` (cmo, dm, vidya, dx), `presma` (ema, t3, tema),
  `min_periods` (sma), the `fast_w`/`medium_w`/`slow_w` weights and `drift`
  (uo), and `c` (accbands). For `mama`/`ht_trendline` the `prenan` leading-NaN
  count is applied as a mask on both paths (routing to native would change the
  underlying algorithm, so it is not used).

At each parameter's TA-Lib default this is a no-op, so no golden column moves;
every case is verified in `tests/test_round4_param_honoring.py`. A few
capability gaps remain **deferred** (documented, not silently wrong): `dm`'s
`mamode` (neither path varies it — both are Wilder sum-smoothing), `adx`
`mamode`/`tvmode`, `vwap` anchoring without a datetime column, and
`variance`/`tos_stdevall` `min_periods`.

**Input validation.** Period/window parameters (`length`, `fast`, `slow`,
`signal`, `drift`, …) flow into Numba kernels that index and allocate with them.
In ~40 indicators a negative or non-integer value used to cause out-of-bounds
access — a hard **process crash** (SIGSEGV/SIGABRT). Those kernels are now guarded
by a shared `v_pos_int` validator that raises a clear `ValueError`
(`length must be an integer >= 1`) before the kernel runs; a fork-isolated sweep
of all indicators confirms none crash the process on a bad period any more. (The
remaining period-taking indicators never crashed — a bad period there raises a
different exception, e.g. `OverflowError`/`TypeError`, or yields a degenerate
result; only the crash-prone kernels needed the guard.) Separately, the
weight-based moving averages (`alma`, `swma`, `sinwma`, `fwma`, `pwma`) plus `cg`
and `msw` build/allocate their weights lazily, so a period larger than the data
returns all-null instead of hanging on an O(n) allocation.

---

## 5. Migration-era bug fixes

While migrating to Polars these bug classes were found and fixed. They can make
Polars-TI values differ from **earlier polars-ti snapshots** (not from correct
pandas-ta):

- **Accessor argument order** — several `df.ti.<name>()` wrappers passed columns
  into the wrong positional parameters (e.g. `close` landing in `length`, or all
  OHLC shifted by one in `smc`). Fixed across `rvi`, `vfi`, `vhm`, `avsl`,
  `alphatrend`, `zigzag`, `trixh`, `smc`.
- **NaN comparisons** — in Polars `NaN > 0` and `x < NaN` are `True` (NaN sorts
  highest), which produced spurious warm-up flags (`AMAT`, `AOBV`, `THERMO`
  long/short, Chandelier Exit direction). Guarded with `is_not_nan()`.
- **EMA/RMA presma seeding** — a leading null/NaN no longer poisons the native
  EMA/RMA recursion (repairs `atr`, `natr`, `dema`, `zlma`, native ATR bands).
- **TA-Lib propagation** — indicators now thread `talib=` into every
  sub-indicator (see [§4c](#4c-native-vs-ta-lib-paths-talib-flag)).
- **Restored/repaired indicators** — `ha`, `ichimoku`, `mama`, `pivots`, `hilo`,
  `psar`, `pmax`, `macd`, `ppo`, `ebsw`, `reflex`, `cksp`, `chop`, `atrts`,
  `halftrend`, `inertia`, `pgo`, `rwi`, `vidya`, `ifisher` (INVFISHER) were
  broken or emitted nothing in some path and now produce correct output in both
  modes.

---

## 6. New indicators & credits

17 indicators were added on top of the pandas baseline during the original port,
harvested from community forks. Credit to the original authors. (A further **86**
indicators were later added for `pandas-ta-classic` feature parity; those are
listed in [§8](#8-feature-parity-with-pandas-ta-classic).)

### From [xgboosted/pandas-ta-classic](https://github.com/xgboosted/pandas-ta-classic) (PR #7)

| Indicator | Category | Description |
| :--- | :--- | :--- |
| `dsp` | Cycles | Detrended Synthetic Price |
| `lrsi` | Momentum | Laguerre RSI (4-stage filter) |
| `po` | Momentum | Projection Oscillator |
| `trixh` | Momentum | TRIX Histogram (TRIX + Signal + Histogram) |
| `vwmacd` | Momentum | Volume-Weighted MACD |
| `mmar` | Overlap | Madrid Moving Average Ribbon |
| `rainbow` | Overlap | Rainbow Charts (10-level SMA stack) |
| `vfi` | Volume | Volume Flow Indicator |
| `pmax` | Trend | Price Max (ATR-based adaptive trailing stop) |

### From [rvuz1013/pandas-ta](https://github.com/rvuz1013/pandas-ta) (PR #8)

| Indicator | Category | Description |
| :--- | :--- | :--- |
| `rmi` | Momentum | Relative Momentum Index |
| `trama` | Trend | Trend Regulated Adaptive Moving Average |
| `avsl` | Volatility | Anti-Volume Stop Loss (VPCI methodology) |
| `halftrend` | Volatility | ATR-based trend indicator with channels |

### From a 133-fork community audit (PR #9)

| Indicator | Category | Source fork | Description |
| :--- | :--- | :--- | :--- |
| `fvg` | Volatility | aligheshlaghi97 | Fair Value Gap (price imbalances) |
| `imi` | Momentum | delfoxav | Intraday Momentum Index |
| `avwap` | Volume | locupleto | Anchored VWAP |
| `ott` | Overlap | bartua | Optimized Trend Tracker (Anıl Özekşi) |

The same audit contributed fixes credited to **bdelpey** (VIDYA TA-Lib CMO
scale), **hypersousage** (CCI operator precedence), **GSLabIt** (Supertrend
TradingView band preservation), **Rossco8** (FWMA `np.convolve` optimization),
**80sVectorz** (ZigZag lookahead mode), and **allyssonmacedo** (PVR `**kwargs`).

---

## 7. Verifying these claims

Every difference above is encoded in the parity test-suite, graded against
committed golden fixtures generated from the pandas baseline and from TA-Lib:

- `tests/_parity.py` — comparison engine + the rename map.
- `tests/parity_exceptions.py` — per-column verdicts (`match`, `match_talib`,
  `intentional`).
- `tests/test_talib_parity.py` — the full per-column TA-Lib-mode gate (every
  shared column matches the pandas golden except the documented exceptions).
- `tests/test_native_parity.py` — the native-mode gate (with `NATIVE_DIVERGENCE`
  for pandas-ta's TA-Lib-contaminated native columns).
- `tests/test_study_completeness.py` — the column manifest + no-all-NaN check in
  both modes.
- `tests/test_parity_smoke.py` — a fast oracle sanity subset (not the full
  per-column gate; that is `test_talib_parity.py`).

Run them with `./scripts/check.sh --fast` (see [Development](development.md)).

---

## 8. Feature parity with pandas-ta-classic

Beyond the original [twopirllc/pandas-ta](https://github.com/twopirllc/pandas-ta)
(development) port, Polars-TI also tracks the community fork
[xgboosted/pandas-ta-classic](https://github.com/xgboosted/pandas-ta-classic) —
a fork off pandas-ta's **main** branch that carries extra indicators and a set of
correctness fixes. Adopting them brings the library to **262 indicators** (176
original + **86** added here); three further TA-Lib-parity indicators
([§8h](#8h-additional-ta-lib-parity-indicators-3)) bring the current total to
**267**. This section documents what was ported and how accurately.

### 8a. Correctness fixes adopted from the classic fork

These were validated against TA-Lib (or the classic reference) and re-pinned in
the parity suite:

| Fix | What changed |
| :--- | :--- |
| `psar` | Guarded reversal logic + a 2-bar guard, matching TA-Lib. |
| `vidya` | SMA-seeded recurrence (correct warm-up seed). |
| `cdl_doji` | Shifted HL-range average + `<=` comparison. |
| `natr` | Default `mamode="rma"`. |
| `trima` | Asymmetric ceil/floor sub-windows. |
| `linreg` (TSF) | Native one-step-ahead forecast `m*(L+1)+b`, equal to `talib.TSF`; this in turn corrected `fosc` (which calls `linreg(..., tsf=True)`). |
| `po` | Honors its `talib=` argument instead of hard-coding `talib=True` on the underlying `linreg` call. (`po` uses the standard `m*L+b` fit, not TSF, so it was unaffected by the TSF fix above.) |

### 8b. New indicators (7)

`rocp`, `rocr`, `rocr100`, `dx` (all exact vs TA-Lib), `beta`, `correl` (exact vs
`talib.BETA` / `talib.CORREL`), and `smc_sweep` (validated vs the classic fork).

### 8c. Hilbert Transform suite (5)

`ht_dcperiod`, `ht_dcphase`, `ht_phasor`, `ht_sine`, `ht_trendmode`.

### 8d. Tulip-Indicators parity (14)

`msw`, `cvi`, `hvol`, `marketfi`, `vosc`, `wad`, `emv`, `fosc`, `avgprice`,
`medprice`, `typprice`, `stderr`, `md`, `avolume`.

### 8e. Full native candlestick suite (~60 patterns)

All ~60 TA-Lib candlestick patterns (`cdl_*`) were ported **natively**, so the
library now has complete candlestick coverage **without** requiring TA-Lib. This
is a functionality feature, not a performance one — see [§8g](#8g-performance-note).

### 8f. Accuracy of the ports

The ports are held to the strongest oracle available for each indicator:

- **Exact against TA-Lib (native path).** Every ported indicator that has a
  TA-Lib equivalent matches `talib.*` **exactly** on the native path, with **0
  mismatches**: all ~60 candlestick patterns (`native == talib.CDL*`),
  `rocp` / `rocr` / `rocr100` / `dx` / `beta` / `correl`,
  `avgprice` / `medprice` / `typprice`, and `ht_dcperiod` / `ht_phasor`.
- **Exact against the classic reference (no TA-Lib equivalent).** The 14 Tulip
  indicators and `smc_sweep` have no TA-Lib counterpart, so they are validated
  exactly against the `pandas-ta-classic` reference via committed golden
  fixtures.
- **HT `ht_dcphase` / `ht_sine` / `ht_trendmode` native-path caveat.** With
  `talib=True` (the default when TA-Lib is installed) these three are exact.
  Their **pure-native** (no-TA-Lib) path cannot be reproduced bit-for-bit during
  TA-Lib's warm-up because they carry path-dependent state accumulators
  (`prevDCPhase` / `daysInTrend`). TA-Lib's *nominal* unstable period is **63
  bars**, but full re-sync takes longer in practice. Measured on daily SPY, the
  native path converges to TA-Lib to **~0.001° for `ht_dcphase` and ~2e-5 for
  `ht_sine` by ~bar 200, with 0 `ht_trendmode` mismatches after ~bar 200**
  (essentially exact by bar 500). The 64–200 region still shows a handful of
  transient mismatches, at phase-wrap / trend-flip points. The same limitation
  exists in the classic fork's own native versions. **Bottom line:** after ~200
  bars of warm-up the native path matches TA-Lib for practical purposes; only the
  early window differs.

### 8g. Performance note

The native candlestick suite is a **no-TA-Lib functionality** feature. As already
benchmarked, TA-Lib's C implementation is roughly **19× faster** for candlestick
logic, so `talib=True` remains the fast path — the native patterns exist so the
library is fully functional when TA-Lib is not installed, not to beat it on
speed.

### 8h. Additional TA-Lib-parity indicators (3)

Three TA-Lib indicators that neither the original port nor the classic fork
exposed were added to complete TA-Lib coverage, each with a native path plus a
TA-Lib fast path:

| Indicator | Category | Description | Accuracy |
| :--- | :--- | :--- | :--- |
| `adxr` | Trend | Average Directional Movement Index Rating — `(ADX + ADX.shift(length-1)) / 2` | exact vs `talib.ADXR` on the TA-Lib path |
| `tsf` | Overlap | Time Series Forecast — the one-step-ahead linear-regression forecast | native `m*(L+1)+b` is exact vs `talib.TSF`; TA-Lib path routes to `talib.TSF` |
| `mavp` | Overlap | Moving Average with Variable Period — a per-bar variable-period SMA | matches `talib.MAVP` for `matype=0` (SMA); other `matype` values route to TA-Lib |

This brings the total to **267** indicators.
