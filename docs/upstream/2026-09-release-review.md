# pandas-ta-classic release review

This follow-up starts after ticket 1, commit
`cf108754f40b49010723cfce85e1c6c7dbd29abd`, was reviewed, committed and pushed.
Its [remote test run](https://github.com/CMobley7/polars-ti/actions/runs/35427378415)
passed. The first ticket's accuracy and performance results remain in
[`REPORT.md`](../../scripts/rolling_kernel_audit/REPORT.md); those measurements
belong to that checkpoint, not to this subsequent correctness changeset.

The supplied strings (`0.8.3` and `0.6.522`) did not identify published tags. The
release list identifies [0.8.32](https://github.com/xgboosted/pandas-ta-classic/releases/tag/0.8.32)
(commit `b905eaa`) and [0.6.52](https://github.com/xgboosted/pandas-ta-classic/releases/tag/0.6.52)
(commit `cd17af4`) as the corresponding releases. This interpretation was
communicated before implementation. Review uses those tags and their
predecessors 0.6.52 / 0.6.20; no post-tag main changes from that repository are included. The separately
requested Gudzenkov commit is reviewed below. The 0.8.32
snapshot still labels its release changes “Unreleased” in CHANGELOG.md.

## Changes adopted and why

| Upstream item | Polars-TI disposition and evidence |
| --- | --- |
| `tsignals` drift corrupts entries/exits | Always compare consecutive states. Explicit `drift` warns and is ignored; `xsignals` defaults avoid emitting an unsolicited warning. Transition regression covers drift 0/2/3. |
| TA-Lib-independent `tal_ma` | Map the nine supported names without importing TA-Lib; fail clearly on unknown/non-string names. Tested with TA-Lib disabled. |
| Fourteen recursive indicators fail on chained input | Fixed native ADOSC (through AD), Fisher, HWC, HWMA, JMA, KAMA, LRSI, MAMA, SSF, TOS_STDEVALL and VIDYA. Local MACD and MCGD already passed. Local MACD has no separate MACDFIX API. Tests compare a padded input against its trimmed valid suffix, including offsets ±2, all-null/all-NaN input, and 0/1/3 rows. |
| Candle running averages poisoned by a missing prefix | Shared native runner starts at the first jointly finite OHLC row and restores prefix zeros. Tests cover every locally exposed TA-Lib-style candle pattern; native Doji additionally guards Polars NaN comparisons. Finite-input pattern logic stays intact. |
| Hilbert prefix, NaN crash, undefined trendmode, lookbacks | Shared adapter processes the first contiguous finite segment. All six public indicators retain row count, restart their lookback after the prefix, and leave an interior gap plus its suffix undefined. Trendmode preserves nullable Int32 with null for undefined state and historical zero lookback on valid input. Both dispatch paths are tested. |
| Wilder ADX/ADXR/DX/DM initial bars | Native ADX now seeds Wilder sums from the first `length - 1` movements and starts DX at `length`, making ADX start at `length + lensig - 1`. DX reuses that seed with `lensig=1`; standalone ADXR composes corrected ADX. DM handles prefixes and its one-period boundary. TA-Lib oracle checks include full warmup masks and flat prices. |
| RMA/true-range warmup | `presma=True` RMA requires a complete seed after the prefix. Native true range is undefined without finite high/low/previous close. Existing native RSI's explicitly documented non-presma convention is retained. |
| `cdl_z(full=True)` future-data leak | Replaced whole-sample normalization with expanding, anchored moments. Honors ddof and skips missing observations. Exact prefix invariance, independent `math.fsum` reference and Decimal high-offset tests cover causality and precision. |
| TOS_STDEVALL / VP ignore causal request | Added explicit `lookahead=False` rejection (`ValueError`), appropriate for Polars expression construction. Default full-sample summaries remain available. Unlike pandas' warning/None path, callers receive a specific error. |
| Short RVI/SWMA/TOS/PVT inputs | SWMA length 1 is identity; RVI length 1/empty is undefined without invalid TA-Lib calls; TOS with fewer than two valid observations is undefined. PVT already handles drift longer than input; regression retained. |
| `candle_color` missing prices | Use nullable integer output; missing open/close cannot be classified bullish or bearish. Native Doji also cannot signal on NaN comparisons. |
| `linear_regression` changes meaning with sklearn | Use installed SciPy `linregress` consistently. Fix the local floor-division slope and zero-sum-x defect; preserve result keys, with `r` always Pearson r. Reject invalid inputs explicitly. |
| `combination` standard-library reuse | Delegate to exact `math.comb`, avoiding float conversion of large integers; preserve `repetition` and `multichoose` aliases and historical absolute-value convention. |
| Missing-data / causality / signal-input documentation | Added user-facing explanations to getting-started, indicators and differences pages, including precomputed fast/slow inputs, comparison series and threshold scales. |

## Already present, deliberately retained, or not applicable

| Release item(s) | Assessment |
| --- | --- |
| 0.6.52 SMC liquidity sweep | `momentum/smc_sweep.py` already exists with a Polars API and tests. No duplicate indicator. |
| 0.6.52 / 0.8.32 Ichimoku DataFrame, `as_dataframe`, `append_span`, tuple deprecation | Local output already uses row-aligned Polars structs, `forward` columns and `lookahead=False`; pandas future-dated index rows cannot be copied into that contract. |
| CPR numeric classifications, `open_` accessor repair, virgin-CPR causal switch | No local CPR API. These repair an upstream feature absent from this fork; adding an options-oriented indicator is separate feature work, not a necessary compatibility patch. |
| MACDEXT KAMA/MAMA double fallback | No local MACDEXT API. Existing MACD dispatch is tested; no corresponding fallback to repair. |
| Accessor `__call__` swallowed exceptions | Local Polars accessor does not implement the upstream pandas `except BaseException` dispatch path. No port. |
| Beta/correl required benchmark and long_run/short_run/xsignals docs | Local signatures already require comparison inputs. Clarified the local expression/threshold contract in indicator documentation. |
| Required MAVP periods | Already required locally and tested; never generated from batch length. |
| Pointwise non-zero-range epsilon | Public Polars and Numba helpers already apply epsilon only to zero-range rows. The legacy shadowed helper is not the exported expression implementation. |
| Stoch/T3 output truncation; TD_SEQ index replacement | Stoch/T3 outputs already retain row count and have no pandas index to replace; TD_SEQ is not a local API. Existing expression/indicator suites exercise these contracts. |
| EMA seed beyond input; linreg/weighted-MA short windows; DM/CDL_PATTERN short contracts | Existing local guards and Polars row-preserving outputs differ from pandas' `None` convention. Added regressions around changed recursive paths without adopting pandas' row dropping / None behavior. |
| Native-loop speedups: HiLo, JMA, HA, STC, WAD, Wilder/DM, PMAX, EBSW, KAMA, MAMA | These local recurrences are already compiled or expressed in Polars; ticket 1 additionally addressed window-dependent kernels. Upstream's pandas `.iloc` timing improvements are not transferable measurements. |
| Aroon / maxindex / minindex / minmaxindex window speedups | Ticket 1 already uses monotonic extrema for Aroon. The three index functions are not local APIs; no sliding-window matrix allocation is needed. |
| Candlestick Numba rewrite | Shared prefix correctness ported. Current default dispatch already uses TA-Lib; native fixed-lookback loops are linear. A rewrite of the entire native candle framework is not necessary for correctness or the requested superlinear-kernel fixes; no upstream speed claim is asserted locally. |
| `npround` vectorization, TD_SEQ compilation and standalone math-operator imports | No local `npround`, TD_SEQ, or corresponding standalone operator API. No runtime pandas scalar-loop bottleneck to port. |
| 13 TA-Lib passthroughs | Local explicit dispatch already covers the applicable math/overlap/Hilbert APIs. Defaults and caller overrides stay intact. |
| `optimal_leverage` float return | No corresponding local public utility. |
| Lazy pandas import / namespace / `ta.<category>` / `ta.cdl` / cross-package import fixes | Local expression registration and exports differ. Avoid replacing working Polars registration with pandas lazy accessor machinery. |
| pandas 3 accessor state, `adjusted`, `last_run`, attrs mutation, property errors, worker payloads | Pandas DataFrame/accessor/multiprocessing internals do not apply to Polars expressions and local study execution. |
| `indicators()` stdout and installation snippet | Local documentation/API do not use the broken pandas `len(df.ta.indicators())` snippet. |
| Type stubs, alias/dead-code/namespace lint cleanup | Existing local typing/lint gates remain in force. Do not copy pandas stubs or remove unrelated supported local exports. The candle counterattack no-op loop is already absent. |
| Deprecations: data fetching, constants/time/exchange helpers, inactive drift, CDL_PATTERN_NAMES, crossany, wrappers, ytd | Upstream API cleanup is not a local compatibility requirement. Preserve local exports rather than import upstream deprecation churn. Signal drift is the exception because it caused an observed numerical defect. |
| Dependency pruning, TA-Lib 0.8 tests, frozen tulipy oracle | Ticket 1 upgraded and audited the runtime, and the full local suite runs against TA-Lib 0.8.0. Existing frozen/independent local oracles remain; no golden regeneration or tolerance relaxation. |
| Snapshot/exact-reference/tolerance/index/short-input/lookahead suites | Retain local frozen sources and independent oracles. Added focused causality, missing-prefix, warmup and short-input tests for adopted changes. Upstream pandas fixture values and indicator registry do not substitute for Polars tests. |
| Backtesting.py / backtrader / vectorbt / manifoldbt integration tutorials | Pandas-specific integration examples and new optional engines are outside this indicator-correctness task. No new runtime dependency or external backtest execution. |
| GitHub action bumps / pages / changelog union merge driver | Project-specific maintenance. First ticket's updated CI already passes with the new runtime; no unrelated workflow/merge policy imported. |

## Validation and review

Implementation followed failing regressions before fixes. Initial prefix/mapping/
signal run: 48 failed / 7 passed. The expanded candle/Hilbert/Wilder/causality/
utility run reproduced another 58 failures; subsequent boundary probes exposed
additional short/missing cases. Final focused and complete gate counts are
recorded below after verification.

The historical rolling-expression tests now use the frozen finite suffix for
Fisher/KAMA missing-prefix cases instead of requiring the defective poisoned
state; their `2e-12` tolerances are unchanged. No golden fixture was regenerated.

After review fixes: **3,274 passed / 56 skipped**, **91.28% coverage**, with
all quality gates and dependency audit passing. Native-only: **2,895 passed /
433 skipped**; all optional dependencies: **3,274 passed / 56 skipped**. Both
reviewers independently passed **467 focused regressions** and reported no
remaining significant findings. Four additional KVO checks were then added for
the final user amendment. Final verification: **3,278 passed / 56 skipped**,
**91.28% coverage**; native-only **2,899 passed / 433 skipped**; all optional
dependencies **3,278 passed / 56 skipped**. All quality gates and the dependency
audit passed. The unpublished local package itself is skipped by pip-audit. PROMPT-2/3 and the consumer repository remain untouched.

### Accuracy evidence

On the 240-row deterministic OHLCV corpus in `prices()` (seed 1832), all 19
recursive/Hilbert native outputs are **bit-identical** on finite input to an
isolated source snapshot of `cf108754f40b49010723cfce85e1c6c7dbd29abd` in the same
Python/dependency environment: ADOSC, Fisher, HWC, HWMA, JMA, KAMA, LRSI, MACD,
MAMA, MCGD, SSF, TOS_STDEVALL, VIDYA and all six HT indicators. This checks that
the prefix adapters do not alter ordinary values. It does not claim equivalence
for deliberately corrected missing inputs, ADX-family seeding, or full candle
normalization.

Measured maximum absolute errors on that corpus on the final implementation:
ADX **1.4211e-14**, ADXR and DX **1.0659e-14** against TA-Lib; full candle close
normalization **2.7157e-13** against a two-pass `math.fsum` expanding reference.
Warmup masks match exactly. Permanent tests also cover multiple windows, missing
prefixes, flat prices and Decimal references for tightly spaced values near 1e12.
The `2e-12` test budgets were not relaxed. The later negative-scalar fix computes
an unscaled directional ratio before applying the scalar once; oracle tests
cover negative, zero and positive scalars.

### Review history

- Astra round 1 reviewed every changed/new file and reproduced two findings:
  native Doji warmed before the first jointly finite OHLC row when only open or
  close had a missing prefix; native DX lost the sign of negative scalar through
  shared ADX scaling. Both received failing regression tests before fixes.
- Root boundary checks additionally reproduced undefined Pearson correlation
  reporting an infinite t-statistic for constant y, and negative HT trendline
  `prenan` masking rows instead of adding no mask. Both were fixed through TDD.
- Astra round 2 and independent Terra review both reported no remaining
  significant findings and independently passed 467 focused tests. Terra is explicitly
  `gpt-5.6-terra` via local Codex CLI because the desktop agent count blocked its
  spawn; Astra is the model-specific desktop subagent `gpt-6-astra`.

## Additional Gudzenkov KVO commit

The user subsequently requested
[`8afd02d3775b41c232b4157616cb2af6de0324c8`](https://github.com/gudzenkov/pandas-ta/commit/8afd02d3775b41c232b4157616cb2af6de0324c8).
It replaces an incorrect cumulative-measurement epsilon adjustment with a
pointwise substitution for zero denominators in that fork's volume-force KVO.

Local `volume/kvo.py` implements the inherited signed-volume variant:
`MA(volume * sign(diff(HLC3)), fast) - MA(..., slow)`, then signal smoothing.
It has no cumulative-measurement division, so the supplied denominator patch
has no applicable line. Flat price changes already map to zero signed volume.
Replacing this with the other fork's volume-force formula would change the
indicator definition and all ordinary values, rather than apply this guard.

Four permanent tests verify all-flat candles with both dispatch settings,
zero-range and tiny-range candles against an independent `math.fsum` SMA
reference, and prefix causality. All four passed without a production change.
Astra and Terra independently reviewed this disposition; Astra also reran all
four added tests successfully. After the final feature-branch
push, the user explicitly authorized merging the session changes into
`development`, pushing, then merging `development` into `main` and pushing.


## Integration CI

The existing workflow is `.github/workflows/ci.yml`. Ticket 1 already upgraded
its Python matrix to **3.14.7**. This follow-up enables its push and pull-request
triggers on **development** as well as **main**, so the quality, TA-Lib-on,
TA-Lib-off and dependency-audit jobs run before integration reaches main. The
user requires a successful full development CI run before main is pushed;
`test.yml` additionally exercises all optional dependencies on both branches.


## Files changed in this follow-up

The first ticket's complete file inventory is in its rolling-kernel report.
This separate follow-up changes these files:

- `.github/workflows/ci.yml`
- `docs/differences-from-pandas-ta.md`
- `docs/getting-started.md`
- `docs/indicators.md`
- `docs/upstream/2026-09-release-review.md`
- `polars_ti/candles/_cdl_math.py`
- `polars_ti/candles/cdl_doji.py`
- `polars_ti/candles/cdl_z.py`
- `polars_ti/cycles/_ht_utils.py`
- `polars_ti/cycles/ht_dcperiod.py`
- `polars_ti/cycles/ht_dcphase.py`
- `polars_ti/cycles/ht_phasor.py`
- `polars_ti/cycles/ht_sine.py`
- `polars_ti/cycles/ht_trendmode.py`
- `polars_ti/momentum/dm.py`
- `polars_ti/momentum/fisher.py`
- `polars_ti/momentum/lrsi.py`
- `polars_ti/overlap/hwma.py`
- `polars_ti/overlap/jma.py`
- `polars_ti/overlap/kama.py`
- `polars_ti/overlap/mama.py`
- `polars_ti/overlap/rma.py`
- `polars_ti/overlap/ssf.py`
- `polars_ti/overlap/vidya.py`
- `polars_ti/statistics/tos_stdevall.py`
- `polars_ti/trend/adx.py`
- `polars_ti/trend/dx.py`
- `polars_ti/trend/ht_trendline.py`
- `polars_ti/trend/tsignals.py`
- `polars_ti/trend/xsignals.py`
- `polars_ti/utils/_candles.py`
- `polars_ti/utils/_core.py`
- `polars_ti/utils/_math.py`
- `polars_ti/utils/_prefix.py`
- `polars_ti/volatility/hwc.py`
- `polars_ti/volatility/rvi.py`
- `polars_ti/volatility/true_range.py`
- `polars_ti/volume/ad.py`
- `polars_ti/volume/vp.py`
- `tests/test_rolling_expressions.py`
- `tests/test_upstream_release_fixes.py`
