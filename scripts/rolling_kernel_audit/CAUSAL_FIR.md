# FIR prefix-consistency correction

The downstream ML consumer requires bit-for-bit prefix identity. Its SWMA(35)
feature differed by about 1.11e-14 at row 93 when evaluated on 150 rows versus
its complete input. The previous FFT filter used later values in a transformed
block; mathematical cancellation removed their contributions, but not necessarily
the last bits of floating-point roundoff. Future outliers could also change an
earlier window's accuracy-fallback decision.

## Algorithm and tradeoff

The corrected filter divides weight lags into [B,2B) chunks. Each chunk convolves
only completed B-row input blocks and contributes at least B rows after the block
starts. Thus every input contributing to an output has already arrived. Fixed
per-block FFT shapes, a fixed addition order, and causal error bounds preserve
both values and fallback choices when the input is extended. Batching independent
blocks along an untransformed axis avoids a Python call per small block.

The normal path costs O(n log² w), with O(n) working memory. Extreme-scale and
nonfinite windows retain accurate direct reevaluation, whose worst case is O(nw).
This is slower than the preceding O(n log w) implementation for some sizes, in
exchange for the strict downstream prefix contract. The earlier measurements in
REPORT.md remain historical; they do not describe this revised FFT schedule.

## Accuracy and validation

A permanent regression failed before the correction. Seven prefix tests cover
ordinary values, future outliers and missing/nonfinite data. Existing independent
accuracy and cancellation/fallback tests retain their original tolerances.
The engine quality script passed: **3,285 passed / 56 skipped**, **91.28% coverage**,
lint/format/type/syntax checks, and dependency audit. The targeted rolling suite
passed **630 / 54**. Astra independently passed all 29 FIR tests, 3,055 prefix
comparisons and 11,200 extended-precision oracle checks, finding no defect.

No ML prefix tolerance was relaxed. Its new tests cover all five weighted filters
at windows 35, 127 and 480, plus recursive warmup and Hilbert missing-state handling.

## Reproducible timing

Run `uv run python -m scripts.rolling_kernel_audit.causal_fir` in the engine
checkout. `causal_fir.json` contains every sample and numerical comparison to the
preserved pre-optimization baseline dc8fb75. Both implementations use the same
Python 3.14.7 / NumPy 2.5.3 / Polars 1.44.2 / SciPy 1.18.1 environment and
8,192 input rows. The table shows the 960-row window, five alternating samples,
and percent time reduction 100*(old-new)/old. These are shared-host synthetic
filter timings, not estimates of full research-run speed.

| Filter | Original ms | Corrected ms | Time reduction | Max absolute deviation |
| :--- | ---: | ---: | ---: | ---: |
| alma | 163.702 | 5.587 | 96.59% | 1.46e-11 |
| fwma | 167.437 | 5.756 | 96.56% | 1.46e-11 |
| pwma | 165.666 | 6.099 | 96.32% | 1.46e-11 |
| sinwma | 165.823 | 5.772 | 96.52% | 1.46e-11 |
| swma | 165.288 | 5.550 | 96.64% | 1.46e-11 |

All measured 960-window rows have matching NaN/infinity masks. Agreement with
old floating-point values is not the sole accuracy oracle: the independent
precision tests above evaluate the mathematical result separately.
