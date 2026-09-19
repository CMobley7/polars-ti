"""Shared missing-input and warmup policy for Hilbert indicators."""

import numpy as np

from polars_ti.utils._prefix import ArrayKernel, FloatArray, first_finite_index


def run_hilbert(values: FloatArray, compute: ArrayKernel, lookback: int, outputs: int = 1) -> tuple[FloatArray, ...]:
    """Compute the first contiguous finite segment and preserve undefined rows.

    Recursive Hilbert state cannot recover after an interior gap. Warmup counts
    from the first finite price; the gap and all subsequent rows stay missing.
    """
    start = first_finite_index((values,))
    result = tuple(np.full(len(values), np.nan) for _ in range(outputs))
    if start == len(values):
        return result
    missing = np.flatnonzero(~np.isfinite(values[start:]))
    end = start + int(missing[0]) if len(missing) else len(values)
    calculated = compute((values[start:end],))
    if len(calculated) != outputs:
        raise ValueError("Hilbert kernel output count does not match its schema")
    for target, source in zip(result, calculated, strict=True):
        target[start:end] = source
        target[start : min(start + max(0, lookback), end)] = np.nan
    return result
