# PROMPT-1 rolling-kernel results

Branch: `rolling-kernels`. Baseline: `dc8fb75b761df43db7bc1460da7bb2083efba8ac`.
This report belongs to the final cohesive commit; use `git log -1 --format=%H`
for its SHA. Independent signoffs are recorded in `REVIEWS.md`.

## Scope and five defects

1. **Rolling NaN scans:** TRAMA, Aroon, STOCH/STOCHF and STC now share monotonic
   deque extrema. Any NaN in the required window invalidates the extremum;
   Aroon ties select the newest index and never dereference the invalid `-1`.
   STC retains its explicit downstream carry rules when an extremum is undefined.
   Finite-input comparison kernels are bit-identical in the measured corpus.
   This fixes the prompt's inconsistent semantics rather than preserving that bug.
2. **MSW:** missing windows now produce NaN and recover. A sliding DFT is rebuilt
   every period, with direct recalculation where roundoff could cross a phase
   branch. Flat/near-degenerate phase decisions retain the direct evaluation.
3. **Variance/stdev:** native paths use compensated shifted moments, periodic and
   condition-triggered rebuilding, and compensated summed means. Extreme scales
   use power-of-two normalized moments to avoid intermediate overflow/underflow. Variance,
   standard deviation and higher-moment references use independent `math.fsum`
   two-pass definitions. Explicit TA-Lib dispatch remains available and unchanged.
4. **Quantile:** bubble sort replaced by block-local Fenwick order statistics,
   O(n log w) work and O(w) auxiliary space. Finite interpolation is unchanged.
   Any NaN invalidates its full window, replacing order-dependent old output.
5. **Trendflex:** algebraic sum reduction and locally centered rolling sums replace
   per-bar window scans; the filter and normalization recurrence are unchanged.

Beyond the five: optimized Reflex, entropy, zscore, MAD/MD, skew, kurtosis,
linreg, KAMA, MFI, MASSI, AVSL, StochRSI, CRSI, TMO, SMA, WMA, CG, MAVP, RVI,
Fisher, KDJ, Zigzag pivots, AVWAP pivots, strict increasing/decreasing, five
weighted-filter callbacks, and Pascal-row construction. CRSI retains a bounded
1024-item vectorized scan below its measured order-statistic crossover.

## Runtime and gates

Python 3.14.7; NumPy 2.5.3; Polars 1.44.2; Numba 0.67.0; SciPy 1.18.1;
TA-Lib 0.8.0. The five requested dependencies are pinned exactly; the old lock
was deleted and regenerated. Python requirement, CI and development docs now
match 3.14. The existing Ruff 0.15.14 is pinned to preserve the repository's
lint rule set rather than adopting unrelated new default lint rules.

Pristine original environment: **2291 passed / 2 skipped**. New tests initially
reproduced **20 failures / 1 pass** for the five defect regressions. Later stress
checks reproduced cancellation failures before re-anchoring fixes; no existing
test tolerances or golden fixtures were relaxed. Final gate counts follow below.

Dependency audit: **no known vulnerabilities** in the resolved requirements;
the local project itself has no published PyPI package to audit. This is not a
claim that a dependency audit proves absence of all security defects.

## Accuracy

Every benchmark row below includes maximum absolute, relative and relative
error above 1% of the median reference magnitude in the JSON artifact. Pure
comparison/order-statistic changes have zero finite-output deviation. Arithmetic
changes differ within floating-point reordering budgets on the benchmark corpus;
NaN behavior changes are explicitly separate in `nan_changes.json`.

The independent oracle includes 256 sampled windows per case, short windows,
large offsets with tiny spread, abrupt scale changes and 120,000 rows with
window 23,040. The table gives the largest absolute error across that corpus.
Absolute errors at scale ~1e24 must be read with their per-case relative errors
in `accuracy.json`; they are not comparable to dimensionless indicator errors.

| Kernel | Original max absolute error | New max absolute error |
|---|---:|---:|
| kurtosis | 0.0783667 | 1.03029e-13 |
| linreg | 0.00335693 | 0.000488281 |
| linreg_slope | 5.34058e-05 | 7.62939e-06 |
| mad | 0.00286865 | 0.00012207 |
| skew | 0.069219 | 1.9007e-13 |
| variance | 2.34881e+08 | 2.01327e+08 |
| zscore | 0.0231442 | 2.13163e-14 |

The new maximum errors improve across these combined corpora. This does **not**
mean every individual result is closer: some well-conditioned results change a
last ULP, and some original sums happen to round exactly to the oracle. For
example, mixed-scale MAD and variance have individual configurations where the
new absolute error increases but remains at floating-point rounding scale.
No universal exact-rounding or accuracy proof is claimed. Tests assert the
fixed numerical budgets without loosening tolerances.

## Kernel benchmark

8192 fixed synthetic price rows, windows 40/480/960, five alternating repetitions,
medians in milliseconds. Both versions run under the same final dependencies;
JIT compilation is excluded. `%` means **100 × (old − new) / old**, so negative
values show a slowdown. `k` fits `time ~ window^k` over the three windows. These
measurements are not a research-grid benchmark or an exclusive-machine timing.

| Kernel | w40 old→new ms | w480 old→new ms | w960 old→new ms | w960 improvement | k old→new | max abs deviation (all windows) |
|---|---:|---:|---:|---:|---:|---:|
| statistics.quantile.nb_quantile.(0.37,) | 10.7718→1.2566 | 621.3198→1.7821 | 2094.6730→1.7784 | 99.92% | 1.651→0.118 | 0 |
| trend.trendflex.nb_trendflex.(20, 0.04, 3.14159, 1.414) | 0.1512→0.2000 | 1.4709→0.1866 | 2.7278→0.1775 | 93.49% | 0.912→-0.035 | 6.55e-15 |
| cycles.reflex.np_reflex.(20, 0.04, 3.14159, 1.414) | 0.1955→0.1989 | 1.8604→0.1908 | 3.4563→0.1846 | 94.66% | 0.905→-0.022 | 1.38e-14 |
| cycles.msw.nb_msw | 0.4316→0.4453 | 2.0427→0.4239 | 3.6400→0.4195 | 88.48% | 0.659→-0.019 | 3.15e-12 |
| trend.trama._nb_rolling_max | 0.0663→0.0750 | 2.4048→0.0775 | 5.0090→0.0728 | 98.55% | 1.383→-0.003 | 0 |
| trend.trama._nb_rolling_min | 0.0661→0.0839 | 2.4296→0.0821 | 5.0931→0.0792 | 98.45% | 1.389→-0.016 | 0 |
| trend.aroon._nb_aroon | 0.5535→0.1858 | 6.9146→0.1943 | 12.7915→0.1828 | 98.57% | 0.996→0.001 | 0 |
| momentum.stoch._stoch_rawk | 0.2027→0.1807 | 2.8572→0.2135 | 5.5640→0.1892 | 96.60% | 1.048→0.029 | 0 |
| momentum.stoch._stoch_core.(3, 3) | 0.2295→0.2056 | 2.8305→0.1891 | 5.5729→0.1977 | 96.45% | 1.006→-0.018 | 0 |
| momentum.stochf._stochf_core.(3,) | 0.2244→0.1883 | 2.8966→0.1970 | 5.4411→0.2056 | 96.22% | 1.010→0.025 | 0 |
| momentum.stochrsi._stochrsi_raw_core.(5,) | 0.3799→0.1808 | 5.0165→0.2249 | 9.3226→0.1893 | 97.97% | 1.015→0.034 | 0 |
| momentum.stc.nb_schaff_tc.(0.5,) | 0.3415→0.3782 | 5.6464→0.3782 | 11.4069→0.3752 | 96.71% | 1.111→-0.002 | 0 |
| statistics.entropy.nb_entropy.(2.0,) | 0.2323→0.1368 | 2.5623→0.1512 | 5.0021→0.1481 | 97.04% | 0.966→0.029 | 2.66e-14 |
| statistics.zscore.nb_zscore.(1.0,) | 0.2143→0.4133 | 2.8235→0.4170 | 5.3850→0.4037 | 92.50% | 1.021→-0.004 | 3.47e-13 |
| statistics.mad.nb_mad | 0.2094→3.4114 | 2.8519→4.8302 | 5.5281→4.9527 | 10.41% | 1.036→0.123 | 3.25e-11 |
| statistics.skew.nb_skew | 0.3388→0.5015 | 3.0257→0.4790 | 5.6475→0.4925 | 91.28% | 0.884→-0.009 | 1.08e-12 |
| statistics.kurtosis.nb_kurtosis | 0.2812→0.4729 | 2.9881→0.4626 | 5.6025→0.4638 | 91.72% | 0.944→-0.007 | 1.68e-12 |
| volume.mfi._nb_mfi | 0.8605→0.1173 | 3.6326→0.1254 | 6.1069→0.1268 | 97.92% | 0.607→0.025 | 7.11e-14 |
| volatility.massi.nb_massi_from_ema1 | 0.1719→0.0659 | 1.6986→0.0639 | 3.0895→0.0775 | 97.49% | 0.912→0.034 | 2.16e-12 |
| volatility.avsl.nb_avsl_core_logic | 0.2496→0.1224 | 2.8543→0.1377 | 5.4057→0.1353 | 97.50% | 0.971→0.036 | 1.38e-10 |
| momentum.crsi.nb_percent_rank | 0.1118→0.1119 | 0.9383→0.9412 | 1.7632→1.7659 | -0.15% | 0.865→0.865 | 0 |
| momentum.tmo._signed_rolling_deltas_numba.(True,) | 0.3880→1.2985 | 2.1666→2.0390 | 3.6709→2.0408 | 44.41% | 0.703→0.153 | 0 |
| momentum.tmo._signed_rolling_deltas_numba.(False,) | 0.3713→1.2493 | 2.1317→1.9705 | 3.7255→2.0451 | 45.11% | 0.720→0.163 | 0 |
| overlap.sma.nb_sma | 0.6827→0.2283 | 2.0219→0.2284 | 3.1622→0.2176 | 93.12% | 0.470→-0.011 | 2.91e-11 |
| overlap.wma.nb_wma.(True, True) | 0.2591→0.6567 | 2.8500→0.6317 | 5.1419→0.6365 | 87.62% | 0.947→-0.011 | 1.38e-10 |
| overlap.wma.nb_wma.(False, True) | 0.2699→0.7067 | 2.8637→0.7071 | 5.1009→0.7412 | 85.47% | 0.932→0.011 | 2.18e-10 |
| momentum.cg.nb_cg | 0.2454→0.4793 | 3.1266→0.4593 | 5.9330→0.4633 | 92.19% | 1.008→-0.012 | 2.27e-12 |
| overlap.mavp._nb_mavp | 0.0781→0.3254 | 0.6659→0.5639 | 1.2757→0.6260 | 50.93% | 0.874→0.210 | 9.46e-11 |
| volatility.rvi._rolling_std.(1,) | 44.1279→0.5325 | 44.2869→0.5234 | 45.9772→0.5016 | 98.91% | 0.010→-0.016 | 1.59e-12 |
| trend.zigzag.nb_rolling_hl | 0.4917→0.1907 | 2.3523→0.1713 | 4.3573→0.1886 | 95.67% | 0.671→-0.014 | 0 |
| overlap.linreg.nb_linreg.(False, False, False, False, False, False) | 0.1662→0.8434 | 1.8021→0.8376 | 3.3543→0.8357 | 75.09% | 0.949→-0.003 | 6.77e-10 |
| overlap.linreg.nb_linreg.(False, False, False, False, True, False) | 0.1241→0.8549 | 1.4678→0.8333 | 2.7710→0.8241 | 70.26% | 0.982→-0.011 | 8.65e-12 |
| overlap.linreg.nb_linreg.(False, False, True, False, False, False) | 0.1640→0.8598 | 1.8233→0.8456 | 3.3646→0.8145 | 75.79% | 0.956→-0.014 | 9.82e-10 |
| overlap.linreg.nb_linreg.(False, False, False, True, False, False) | 0.1757→0.8812 | 1.8309→0.8543 | 3.4136→0.8479 | 75.16% | 0.936→-0.012 | 7.11e-11 |
| overlap.linreg.nb_linreg.(True, True, False, False, False, False) | 0.2573→0.9022 | 1.8996→0.8571 | 3.4465→0.8354 | 75.76% | 0.813→-0.023 | 2.99e-10 |
| overlap.linreg.nb_linreg.(False, False, False, False, False, True) | 0.1637→0.8640 | 1.8072→0.8223 | 3.3488→0.8023 | 76.04% | 0.954→-0.022 | 6.77e-10 |
| alma | 101.1901→4.7701 | 101.9700→3.3606 | 158.8950→2.9171 | 98.16% | 0.105→-0.151 | 1.46e-11 |
| fwma | 39.4334→4.2054 | 103.8824→3.4660 | 166.8786→3.0810 | 98.15% | 0.437→-0.093 | 1.46e-11 |
| pwma | 43.6244→4.6424 | 103.9882→3.4977 | 163.2221→3.3683 | 97.94% | 0.398→-0.104 | 1.46e-11 |
| sinwma | 39.4873→4.5752 | 103.4560→3.3163 | 162.4499→2.9873 | 98.16% | 0.430→-0.133 | 1.46e-11 |
| swma | 38.7497→4.2982 | 102.2741→3.3258 | 163.3745→2.9161 | 98.22% | 0.436→-0.117 | 1.46e-11 |
| kama | 2.1493→4.3429 | 4.1553→4.6850 | 5.9085→4.6708 | 20.95% | 0.304→0.025 | 4.37e-11 |
| fisher | 94.5263→3.3514 | 89.9695→2.7845 | 177.0265→2.6810 | 98.49% | 0.140→-0.071 | 0 |
| kdj | 341.2556→45.3796 | 324.6591→44.3583 | 677.1711→51.4639 | 92.40% | 0.153→0.027 | 0 |
| increasing | 7.4253→1.2429 | 280.4877→0.7103 | 924.0149→0.7634 | 99.92% | 1.503→-0.172 | 0 |
| decreasing | 3.5876→0.5237 | 275.2565→1.4060 | 848.1966→0.6455 | 99.92% | 1.727→0.154 | 0 |
| variance | 0.1488→0.8138 | 0.1313→0.6874 | 0.1313→0.6768 | -415.66% | -0.042→-0.061 | 6.29e-08 |
| stdev | 0.1312→0.7158 | 0.1324→0.6771 | 0.1718→0.8247 | -380.12% | 0.063→0.027 | 2.96e-11 |
| sma | 0.1285→0.5017 | 0.1194→0.4454 | 0.0783→0.4382 | -459.74% | -0.122→-0.044 | 1.46e-11 |

CRSI retains the original bounded scan at these window sizes because the rank
tree was slower at w960; its near-zero timing differences are measurement noise. Kernel
and expression timing categories include different overhead and must not be
summed into an inferred production speedup.

Native variance, stdev and SMA already used linear Polars kernels. Their new
Python-callback/compensated paths prioritize accuracy and can be slower; their
negative percentages are retained above. Several order-statistics paths are
also slower at small windows. These are measured tradeoffs, not universal
per-call speedups. Precision fallback work remains explicitly qualified below.

## Mechanical audit: flags and negatives

Original sweep: **523 Python files**, including 297 production files. Current sweep: **542 Python files**, including 300 production files. Full inventories include every clear function.

Originally flagged production functions (deduplicated by file/function):

- `polars_ti/candles/cdl_concealbabyswall.py:21` — `_detect`
- `polars_ti/candles/cdl_kicking.py:21` — `_detect`
- `polars_ti/candles/cdl_kickingbylength.py:21` — `_detect`
- `polars_ti/candles/cdl_mathold.py:21` — `_detect`
- `polars_ti/candles/cdl_piercing.py:21` — `_detect`
- `polars_ti/candles/cdl_risefall3methods.py:21` — `_detect`
- `polars_ti/core.py:489` — `study`
- `polars_ti/custom.py:83` — `import_dir`
- `polars_ti/cycles/_ht_pipeline.py:35` — `nb_ht_pipeline`
- `polars_ti/cycles/msw.py:14` — `nb_msw`
- `polars_ti/cycles/reflex.py:7` — `np_reflex`
- `polars_ti/momentum/cg.py:15` — `nb_cg`
- `polars_ti/momentum/crsi.py:30` — `nb_percent_rank`
- `polars_ti/momentum/fisher.py:40` — `fisher`
- `polars_ti/momentum/fisher.py:67` — `compute_fisher`
- `polars_ti/momentum/kdj.py:45` — `kdj`
- `polars_ti/momentum/kdj.py:80` — `compute`
- `polars_ti/momentum/stc.py:15` — `nb_schaff_tc`
- `polars_ti/momentum/stoch.py:48` — `_stoch_rawk`
- `polars_ti/momentum/stoch.py:80` — `_stoch_core`
- `polars_ti/momentum/stochf.py:45` — `_stochf_core`
- `polars_ti/momentum/stochrsi.py:67` — `_sma_numba`
- `polars_ti/momentum/stochrsi.py:111` — `_stochrsi_raw_core`
- `polars_ti/momentum/stochrsi.py:186` — `stochrsi`
- `polars_ti/momentum/stochrsi.py:238` — `compute_stochrsi_talib`
- `polars_ti/momentum/tmo.py:14` — `_signed_rolling_deltas_numba`
- `polars_ti/overlap/alma.py:12` — `alma`
- `polars_ti/overlap/fwma.py:12` — `fwma`
- `polars_ti/overlap/jma.py:14` — `nb_jma`
- `polars_ti/overlap/kama.py:14` — `kama`
- `polars_ti/overlap/kama.py:59` — `nb_kama`
- `polars_ti/overlap/linreg.py:14` — `nb_linreg`
- `polars_ti/overlap/mavp.py:14` — `_nb_mavp`
- `polars_ti/overlap/pwma.py:12` — `pwma`
- `polars_ti/overlap/sinwma.py:11` — `sinwma`
- `polars_ti/overlap/sma.py:8` — `nb_sma`
- `polars_ti/overlap/swma.py:12` — `swma`
- `polars_ti/overlap/wma.py:7` — `nb_wma`
- `polars_ti/statistics/entropy.py:14` — `nb_entropy`
- `polars_ti/statistics/kurtosis.py:14` — `nb_kurtosis`
- `polars_ti/statistics/mad.py:15` — `nb_mad`
- `polars_ti/statistics/quantile.py:15` — `nb_quantile`
- `polars_ti/statistics/skew.py:14` — `nb_skew`
- `polars_ti/statistics/zscore.py:14` — `nb_zscore`
- `polars_ti/trend/aroon.py:14` — `_nb_aroon`
- `polars_ti/trend/decreasing.py:11` — `decreasing`
- `polars_ti/trend/ht_trendline.py:7` — `nb_ht_trendline`
- `polars_ti/trend/increasing.py:11` — `increasing`
- `polars_ti/trend/trama.py:40` — `_nb_rolling_max`
- `polars_ti/trend/trama.py:54` — `_nb_rolling_min`
- `polars_ti/trend/trendflex.py:7` — `nb_trendflex`
- `polars_ti/trend/zigzag.py:7` — `nb_rolling_hl`
- `polars_ti/utils/_numba.py:75` — `nb_rolling`
- `polars_ti/volatility/avsl.py:14` — `nb_avsl_core_logic`
- `polars_ti/volatility/halftrend.py:7` — `nb_halftrend`
- `polars_ti/volatility/massi.py:16` — `nb_massi_from_ema1`
- `polars_ti/volatility/rvi.py:12` — `_rolling_std`
- `polars_ti/volume/avwap.py:14` — `_nb_find_pivots`
- `polars_ti/volume/mfi.py:14` — `_nb_mfi`

Current flagged production functions and classification:

- `polars_ti/candles/cdl_concealbabyswall.py:21` `_detect`: fixed 2/3-item candle scan.
- `polars_ti/candles/cdl_kicking.py:21` `_detect`: fixed 2/3-item candle scan.
- `polars_ti/candles/cdl_kickingbylength.py:21` `_detect`: fixed 2/3-item candle scan.
- `polars_ti/candles/cdl_mathold.py:21` `_detect`: fixed 2/3-item candle scan.
- `polars_ti/candles/cdl_piercing.py:21` `_detect`: fixed 2/3-item candle scan.
- `polars_ti/candles/cdl_risefall3methods.py:21` `_detect`: fixed 2/3-item candle scan.
- `polars_ti/core.py:488` `study`: scales with requested output count, not an indicator window.
- `polars_ti/custom.py:83` `import_dir`: filesystem/module enumeration, not a rolling numeric kernel.
- `polars_ti/cycles/_ht_pipeline.py:35` `nb_ht_pipeline`: bounded Hilbert loops, period clamped to 50.
- `polars_ti/cycles/msw.py:15` `nb_msw`: amortized O(n) periodic rebuild, with O(n·w) worst-case precision fallback.
- `polars_ti/momentum/crsi.py:32` `nb_percent_rank`: bounded scan up to 1024; O(n log w) rank structure above that.
- `polars_ti/momentum/stochrsi.py:69` `_sma_numba`: disjoint finite runs; amortized O(n).
- `polars_ti/overlap/jma.py:14` `nb_jma`: bounded 66-bar average.
- `polars_ti/overlap/linreg.py:23` `nb_linreg`: O(n) normal path, O(w) normalized fallback per extreme-scale window.
- `polars_ti/overlap/sma.py:10` `nb_sma`: O(n) normal path, normalized O(w) fallback when a raw sum overflows.
- `polars_ti/trend/ht_trendline.py:7` `nb_ht_trendline`: bounded Hilbert loop, period clamped to 50.
- `polars_ti/utils/_fir.py:10` `rolling_fir`: O(n log w) extended-precision convolution; direct bounded/precision/platform fallback.
- `polars_ti/utils/_math.py:153` `pascals_triangle`: one-time O(w) recurrence over exact integers; bit cost grows with coefficient size.
- `polars_ti/utils/_numba.py:75` `nb_rolling`: arbitrary user-supplied callback; no general sub-window shortcut exists.
- `polars_ti/utils/_order_stats.py:59` `variable_mean`: O(n log w) range tree; normalized O(w) fallback after intermediate overflow.
- `polars_ti/utils/_order_stats.py:120` `rolling_mad`: block-local O(n log w) order statistics; direct ill-conditioned-window fallback.
- `polars_ti/utils/_rolling.py:25` `rolling_sum`: O(n) including one O(w) rebuild per w bars.
- `polars_ti/utils/_rolling.py:67` `rolling_extreme`: amortized O(n); each index enters/leaves the deque once.
- `polars_ti/utils/_rolling.py:104` `rolling_moments`: O(n) periodic rebuild and fixed moment order; extra condition-triggered rebuilds can be O(n·w).
- `polars_ti/utils/_rolling.py:237` `rolling_quantile`: O(n log w): O(w log w) sort per w outputs, logarithmic updates/selection.
- `polars_ti/utils/_rolling.py:276` `rolling_ranks`: O(n log w): O(w log w) sort per w outputs, logarithmic updates/queries.
- `polars_ti/utils/_rolling.py:311` `_rolling_linear`: O(n) periodic rebuild; additional conditioning rebuilds can be O(n·w).
- `polars_ti/utils/_rolling.py:426` `scaled_window_moments`: O(w) per extreme-scale window; deliberate precision fallback.
- `polars_ti/utils/_rolling.py:466` `rolling_difference`: centered O(n) normal recurrence, with conditional reanchoring for accuracy.
- `polars_ti/volatility/halftrend.py:7` `nb_halftrend`: false positive from an aliased scalar; single-pass recurrence.

Manually identified conditional paths also include `nb_wma`/`nb_cg` calling
`scaled_window_linear` on non-finite arithmetic, and SMA/MAVP binary-scaled
finite-window fallbacks. Each fallback scans O(w) values and can be O(n·w)
over a fully extreme series. These calls can be syntactically clear in the
AST inventory because their loop bounds live in a separate helper.

Explicitly cleared: EBSW is a pure recurrence; AVWAP accumulation was already O(n).
The Hilbert pipeline, JMA, candle scans and StochRSI finite-run SMA are bounded
or amortized linear despite conservative syntax flags. All rewritten direct
window reductions are clear or have the classified shared helpers above.
No additional unbounded whole-history quadratic kernel was found by this sweep
and independent read-only audit. This remains a checkable audit, not a proof
about arbitrary dynamically supplied callbacks.

## Corrections to the old diagnosis (consumer kept read-only)

- The original repository has 523 Python files, of which 297 are production
  modules; the old suite counts were not reproduced in this environment.
- Polars 1.41.2 does **not** implement rolling variance as naive E[x²]−E[x]².
  Its `VarState` uses centered online insert/remove updates. The reproduced
  accuracy issue is real, but that proposed mechanism was wrong. See the
  [versioned Polars implementation](https://raw.githubusercontent.com/pola-rs/polars/py-1.41.2/crates/polars-compute/src/moment.rs)
  and [rolling adapter](https://raw.githubusercontent.com/pola-rs/polars/py-1.41.2/crates/polars-compute/src/rolling/moment.rs).
- Aroon's historical 640-unit error was attributed to a lost optimization's
  sentinel-index bug; it is not a reproduced baseline magnitude. Current Aroon
  bounds and sentinels are directly tested.
- The former 99.8% grid attribution, 3801× pipeline speedup, week/month schedule,
  37,772 MSW mismatches, and historical scaling exponents were not reproduced
  and are not used as current evidence. No PROMPT-2/3 work was started.

## Verification and reviews

- Fresh-cache full suite: **2,974 passed / 56 skipped**, **91.18% coverage**.
- TA-Lib disabled: **2,632 passed / 396 skipped**.
- Isolated environment with all `full` and `test` extras: **2,974 passed / 56 skipped**.
- Ruff lint/format, configured mypy, syntax, pandas-purge and lock checks pass.
- Runtime/test and all-optional-extras dependency audits: no known vulnerabilities.
- Astra and Terra independently report no significant remaining actionable findings;
  see `REVIEWS.md` for the complete fix-and-review history.
- The manually dispatched CI workflow now uses locked `uv sync` to create its
  environment before invoking pytest; the full-extra locked resolution was verified.

Primary defect entry points: `polars_ti/utils/_rolling.py:67` (extrema),
`polars_ti/cycles/msw.py:15`, `polars_ti/statistics/variance.py:14`,
`polars_ti/statistics/stdev.py:14`, `polars_ti/statistics/quantile.py:17`,
and `polars_ti/trend/trendflex.py:10`. Each benchmark row gives its measured
deviation; permanent `test_rolling_*` modules contain the independent checks.

## Larger-window crossover

32,768 rows, five paired warmed repetitions under the same runtime.

| Kernel | Window | Old ms | New ms | Runtime reduction | Max absolute deviation |
|---|---:|---:|---:|---:|---:|
| momentum.crsi.nb_percent_rank | 1024 | 8.1713 | 8.1667 | 0.06% | 0 |
| momentum.crsi.nb_percent_rank | 2048 | 15.7277 | 11.2732 | 28.32% | 0 |
| momentum.crsi.nb_percent_rank | 8192 | 50.6734 | 10.8643 | 78.56% | 0 |
| statistics.mad.nb_mad | 1024 | 25.7259 | 21.3242 | 17.11% | 4.37e-11 |
| statistics.mad.nb_mad | 2048 | 50.4877 | 22.9895 | 54.47% | 5.07e-11 |
| statistics.mad.nb_mad | 8192 | 163.0485 | 23.7537 | 85.43% | 6.87e-11 |

## FIR cancellation and outlier stress

20,000 rows; paired direct legacy SWMA and accelerated SWMA at window 2,048.
The full window sweep and raw samples are in `fir_stress.json`.

| Corpus | Old ms | New ms | Runtime reduction | Max absolute deviation |
|---|---:|---:|---:|---:|
| positive | 799.169 | 5.929 | 99.26% | 0 |
| cancellation | 866.732 | 5.796 | 99.33% | 3.13e-17 |
| outlier | 878.749 | 28.990 | 96.70% | 0 |

## Changed files

- `.github/workflows/ci.yml`
- `.github/workflows/test.yml`
- `.gitignore`
- `.python-version`
- `README.md`
- `docs/development.md`
- `polars_ti/core.py`
- `polars_ti/cycles/msw.py`
- `polars_ti/cycles/reflex.py`
- `polars_ti/momentum/cg.py`
- `polars_ti/momentum/crsi.py`
- `polars_ti/momentum/fisher.py`
- `polars_ti/momentum/kdj.py`
- `polars_ti/momentum/stc.py`
- `polars_ti/momentum/stoch.py`
- `polars_ti/momentum/stochf.py`
- `polars_ti/momentum/stochrsi.py`
- `polars_ti/momentum/tmo.py`
- `polars_ti/overlap/alma.py`
- `polars_ti/overlap/fwma.py`
- `polars_ti/overlap/kama.py`
- `polars_ti/overlap/linreg.py`
- `polars_ti/overlap/mavp.py`
- `polars_ti/overlap/pwma.py`
- `polars_ti/overlap/sinwma.py`
- `polars_ti/overlap/sma.py`
- `polars_ti/overlap/swma.py`
- `polars_ti/overlap/wma.py`
- `polars_ti/statistics/entropy.py`
- `polars_ti/statistics/kurtosis.py`
- `polars_ti/statistics/mad.py`
- `polars_ti/statistics/quantile.py`
- `polars_ti/statistics/skew.py`
- `polars_ti/statistics/stdev.py`
- `polars_ti/statistics/variance.py`
- `polars_ti/statistics/zscore.py`
- `polars_ti/trend/aroon.py`
- `polars_ti/trend/decreasing.py`
- `polars_ti/trend/increasing.py`
- `polars_ti/trend/trama.py`
- `polars_ti/trend/trendflex.py`
- `polars_ti/trend/zigzag.py`
- `polars_ti/utils/_fir.py`
- `polars_ti/utils/_math.py`
- `polars_ti/utils/_order_stats.py`
- `polars_ti/utils/_rolling.py`
- `polars_ti/volatility/avsl.py`
- `polars_ti/volatility/massi.py`
- `polars_ti/volatility/rvi.py`
- `polars_ti/volume/avwap.py`
- `polars_ti/volume/mfi.py`
- `pyproject.toml`
- `scripts/check.sh`
- `scripts/rolling_kernel_audit/.gitignore`
- `scripts/rolling_kernel_audit/README.md`
- `scripts/rolling_kernel_audit/REPORT.md`
- `scripts/rolling_kernel_audit/REVIEWS.md`
- `scripts/rolling_kernel_audit/accuracy.json`
- `scripts/rolling_kernel_audit/accuracy.py`
- `scripts/rolling_kernel_audit/after.json`
- `scripts/rolling_kernel_audit/audit.py`
- `scripts/rolling_kernel_audit/baseline.tar.gz`
- `scripts/rolling_kernel_audit/baseline_commit.sha`
- `scripts/rolling_kernel_audit/before.json`
- `scripts/rolling_kernel_audit/benchmark.json`
- `scripts/rolling_kernel_audit/benchmark.py`
- `scripts/rolling_kernel_audit/cases.py`
- `scripts/rolling_kernel_audit/crossover.json`
- `scripts/rolling_kernel_audit/crossover.py`
- `scripts/rolling_kernel_audit/expressions.json`
- `scripts/rolling_kernel_audit/expressions.py`
- `scripts/rolling_kernel_audit/fir_stress.json`
- `scripts/rolling_kernel_audit/fir_stress.py`
- `scripts/rolling_kernel_audit/nan_changes.json`
- `scripts/rolling_kernel_audit/reference.py`
- `tests/test_rolling_boundaries.py`
- `tests/test_rolling_equivalence.py`
- `tests/test_rolling_execution_modes.py`
- `tests/test_rolling_expressions.py`
- `tests/test_rolling_fir.py`
- `tests/test_rolling_fixes.py`
- `tests/test_rolling_kernel_audit.py`
- `tests/test_rolling_precision.py`
- `uv.lock`
