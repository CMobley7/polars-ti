"""Extended-precision convolution for arbitrary finite rolling weight vectors."""

import numpy as np
import polars as pl
from scipy.signal import oaconvolve

from polars_ti.utils._rolling import FloatArray, rolling_extreme, rolling_sum


def rolling_fir(values: FloatArray, weights: FloatArray) -> FloatArray:
    """Apply oldest-first weights with extended-precision overlap-add convolution.

    The normal path is O(n log w). Small windows and cancellation-dominated
    outputs use direct extended-precision products to protect accuracy. On
    platforms without extended long double, the direct path is required.
    """
    window = len(weights)
    if window < 1:
        raise ValueError("weights must be nonempty")
    result = np.full(len(values), np.nan)
    if len(values) < window:
        return result
    extended_weights = weights.astype(np.longdouble)
    finite = np.isfinite(values)
    clean = np.where(finite, values, 0.0).astype(np.longdouble)
    invalid = rolling_sum((~finite).astype(np.float64), window)
    extended_epsilon = np.finfo(np.longdouble).eps
    use_direct = window <= 32 or extended_epsilon >= np.finfo(np.float64).eps
    if use_direct:
        for i in range(window - 1, len(values)):
            result[i] = np.sum(values[i - window + 1 : i + 1].astype(np.longdouble) * extended_weights)
        return result
    local_scale = rolling_extreme(np.abs(values), window, True)[0]
    weight_norm = float(np.sum(np.abs(extended_weights)))
    rounding_budget = 4 * np.finfo(np.float64).eps * local_scale * weight_norm
    missing = rolling_sum(np.isnan(values).astype(np.float64), window) > 0
    block_size = max(256, window)
    for block in range(window - 1, len(values), block_size):
        end = min(len(values), block + block_size)
        start = block - window + 1
        segment = clean[start:end]
        computed = oaconvolve(segment, extended_weights[::-1], mode="valid")
        result[block:end] = computed.astype(np.float64)
        # A distant extreme cannot contaminate an independently transformed
        # block. Compare absolute error with the local direct-sum budget;
        # cancellation alone must never trigger full-window reevaluation.
        bound = float(64 * extended_epsilon * np.max(np.abs(segment)) * weight_norm * np.log2(len(segment) + window))
        fallback = ((invalid[block:end] > 0) | (bound > rounding_budget[block:end])) & ~missing[block:end]
        for i in np.flatnonzero(fallback) + block:
            direct = values[i - window + 1 : i + 1].astype(np.longdouble)
            result[i] = np.sum(direct * extended_weights)
    result[missing] = np.nan
    return result


def fir_series(series: pl.Series, weights: FloatArray) -> pl.Series:
    """Apply FIR weights while preserving Polars null-window semantics."""
    window = len(weights)
    result = pl.Series(rolling_fir(series.to_numpy().astype(np.float64), weights))
    missing = series.is_null().cast(pl.Int64).rolling_sum(window, min_samples=window)
    return result.set(missing.is_null() | (missing > 0), None)
