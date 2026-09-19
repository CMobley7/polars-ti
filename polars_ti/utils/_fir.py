"""Extended-precision convolution for arbitrary finite rolling weight vectors."""

import numpy as np
import polars as pl
from scipy.signal import fftconvolve

from polars_ti.utils._rolling import FloatArray, rolling_extreme, rolling_sum


def rolling_fir(values: FloatArray, weights: FloatArray) -> FloatArray:
    """Apply oldest-first weights with prefix-stable extended-precision convolution.

    The normal path is O(n log² w). Small windows, platforms without extended
    long double, and windows whose convolution error estimate exceeds the local
    direct-sum budget use direct products. Cancellation alone does not trigger
    fallback; exceptional inputs can require O(n*w) work to protect accuracy.
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
    # Lag chunks [B, 2B) use only completed B-row input blocks. Their first
    # contribution is at least B rows later, after every source value is known.
    # Appending rows can therefore add no operands to an earlier output, even
    # through FFT roundoff or an accuracy-fallback decision.
    kernel = extended_weights[::-1]
    accumulated = np.zeros(len(values), dtype=np.longdouble)
    error_bound = np.zeros(len(values), dtype=np.longdouble)
    head = 32
    for lag in range(head):
        accumulated[lag:] += clean[: len(values) - lag] * kernel[lag]
    block_size = head
    while block_size < window:
        block_count = len(values) // block_size
        if block_count == 0:
            break
        chunk = kernel[block_size : 2 * block_size]
        blocks = clean[: block_count * block_size].reshape(block_count, block_size)
        # Transform only the time axis: the fixed per-row FFT shape is independent
        # of how many complete blocks the caller supplies. Batching avoids one
        # Python/SciPy call per small block without changing the arithmetic.
        convolved = fftconvolve(blocks, chunk[None, :], mode="full", axes=-1)
        bounds = (
            64
            * extended_epsilon
            * np.max(np.abs(blocks), axis=1)
            * np.sum(np.abs(chunk))
            * np.log2(block_size + len(chunk))
        )
        # Each convolution overlaps at most two output blocks. Add all heads,
        # then all tails at every lag level, preserving that order in prefixes.
        for part in range(2):
            section = convolved[:, part * block_size : (part + 1) * block_size]
            width = section.shape[1]
            if width == 0:
                continue
            destination = (part + 1) * block_size
            count = min(block_count, (len(values) - destination + block_size - 1) // block_size)
            if count <= 0:
                continue
            # Pad the partial convolution tail so each source block keeps its
            # fixed output position instead of packing adjacent tails together.
            packed = np.zeros((count, block_size), dtype=np.longdouble)
            packed[:, :width] = section[:count]
            budget = np.zeros_like(packed)
            budget[:, :width] = bounds[:count, None]
            size = min(count * block_size, len(values) - destination)
            accumulated[destination : destination + size] += packed.ravel()[:size]
            error_bound[destination : destination + size] += budget.ravel()[:size]
        block_size *= 2
    result[window - 1 :] = accumulated[window - 1 :].astype(np.float64)
    # Both the bound and the local budget use only past observations. Direct
    # reevaluation protects extreme scales and infinities without admitting a
    # future outlier into the rounding decision for an earlier window.
    fallback = ((invalid > 0) | (error_bound > rounding_budget)) & ~missing
    fallback[: window - 1] = False
    for i in np.flatnonzero(fallback):
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
