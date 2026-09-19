"""Stable range sums and mean absolute deviations over rolling windows."""

import numpy as np
from numba import njit

from polars_ti.utils._rolling import FloatArray, binary_scale, compensated_add, rolling_sum


@njit(cache=True)
def _range_sum_components(
    sums: FloatArray,
    corrections: FloatArray,
    start: int,
    end: int,
    capacity: int,
    total: float = 0.0,
    correction: float = 0.0,
) -> tuple[float, float]:
    """Accumulate tree nodes while retaining compensation across split ranges."""
    start += capacity
    end += capacity
    while start < end:
        if start & 1:
            total, correction = compensated_add(total, correction, sums[start])
            total, correction = compensated_add(total, correction, corrections[start])
            start += 1
        if end & 1:
            end -= 1
            total, correction = compensated_add(total, correction, sums[end])
            total, correction = compensated_add(total, correction, corrections[end])
        start //= 2
        end //= 2
    return total, correction


@njit(cache=True)
def _range_sum(sums: FloatArray, corrections: FloatArray, start: int, end: int, capacity: int) -> float:
    """Combine disjoint tree nodes without subtracting large prefix sums."""
    total, correction = _range_sum_components(sums, corrections, start, end, capacity)
    return total + correction


@njit(cache=True)
def _set_sum(sums: FloatArray, corrections: FloatArray, index: int, value: float, capacity: int) -> None:
    """Replace one tree leaf and rebuild its ancestors without removal drift."""
    index += capacity
    sums[index] = value
    corrections[index] = 0.0
    index //= 2
    while index:
        total, correction = compensated_add(sums[2 * index], corrections[2 * index], sums[2 * index + 1])
        total, correction = compensated_add(total, correction, corrections[2 * index + 1])
        sums[index] = total
        corrections[index] = correction
        index //= 2


@njit(cache=True)
def variable_mean(values: FloatArray, periods: FloatArray, minimum: int, maximum: int) -> FloatArray:
    """Compute variable-window means in O(n log w), preserving local NaN recovery."""
    size = len(values)
    result = np.full(size, np.nan)
    if minimum < 1 or maximum < minimum:
        raise ValueError("period bounds must satisfy 1 <= minimum <= maximum")
    if size < maximum:
        return result
    capacity = 1
    while capacity < maximum:
        capacity *= 2
    sums = np.zeros(2 * capacity)
    corrections = np.zeros(2 * capacity)
    invalid = np.zeros(size + 1, dtype=np.int64)
    positive_inf = np.zeros(size + 1, dtype=np.int64)
    negative_inf = np.zeros(size + 1, dtype=np.int64)
    for i in range(size):
        value = values[i]
        invalid[i + 1] = invalid[i] + np.isnan(value)
        positive_inf[i + 1] = positive_inf[i] + (value == np.inf)
        negative_inf[i + 1] = negative_inf[i] + (value == -np.inf)
        _set_sum(sums, corrections, i % maximum, value if np.isfinite(value) else 0.0, capacity)
        if i < maximum - 1:
            continue
        period = minimum if np.isnan(periods[i]) else min(max(int(periods[i]), minimum), maximum)
        first = i - period + 1
        if invalid[i + 1] != invalid[first]:
            continue
        has_positive = positive_inf[i + 1] != positive_inf[first]
        has_negative = negative_inf[i + 1] != negative_inf[first]
        if has_positive or has_negative:
            if not (has_positive and has_negative):
                result[i] = np.inf if has_positive else -np.inf
            continue
        start = first % maximum
        end = i % maximum + 1
        if start < end:
            total = _range_sum(sums, corrections, start, end, capacity)
        else:
            total, correction = _range_sum_components(sums, corrections, start, maximum, capacity)
            total, correction = _range_sum_components(sums, corrections, 0, end, capacity, total, correction)
            total += correction
        if np.isfinite(total):
            result[i] = total / period
        else:
            # Tree nodes may overflow before opposite signs cancel at the root.
            scale = 0.0
            for j in range(first, i + 1):
                scale = max(scale, abs(values[j]))
            total = correction = 0.0
            if scale == 0.0:
                result[i] = 0.0
                continue
            scale = binary_scale(scale)
            for j in range(first, i + 1):
                total, correction = compensated_add(total, correction, values[j] / scale)
            result[i] = ((total + correction) / period) * scale
    return result


@njit(cache=True)
def rolling_mad(values: FloatArray, window: int) -> FloatArray:
    """Compute MAD using ordered centered sums; ill-conditioned windows recheck directly."""
    if window < 1:
        raise ValueError("window must be positive")
    size = len(values)
    result = np.full(size, np.nan)
    means = rolling_sum(values, window) / window
    for block in range(window - 1, size, window):
        end = min(size, block + window)
        start = block - window + 1
        ordered = np.sort(values[start:end])
        finite_count = 0
        for value in ordered:
            finite_count += np.isfinite(value)
        if finite_count == 0:
            continue
        anchor = ordered[np.searchsorted(ordered, -np.inf, side="right") + finite_count // 2]
        capacity = 1
        while capacity < len(ordered):
            capacity *= 2
        sums = np.zeros(2 * capacity)
        corrections = np.zeros(2 * capacity)
        counts = np.zeros(2 * capacity)
        count_corrections = np.zeros(2 * capacity)
        frequencies = np.zeros(len(ordered), dtype=np.int64)
        invalid = 0
        extent = 0.0
        for i in range(start, end):
            value = values[i]
            if np.isfinite(value):
                rank = np.searchsorted(ordered, value)
                frequencies[rank] += 1
                extent = max(extent, abs(value - anchor))
                _set_sum(sums, corrections, rank, frequencies[rank] * (value - anchor), capacity)
                _set_sum(counts, count_corrections, rank, float(frequencies[rank]), capacity)
            else:
                invalid += 1
            if i < block:
                continue
            if invalid == 0:
                mean = means[i]
                split = np.searchsorted(ordered, mean)
                below = _range_sum(sums, corrections, 0, split, capacity)
                above = _range_sum(sums, corrections, split, len(ordered), capacity)
                below_count = _range_sum(counts, count_corrections, 0, split, capacity)
                centered_mean = mean - anchor
                deviation = (below_count * centered_mean - below) + (above - (window - below_count) * centered_mean)
                # An old extreme can make even centered prefix differences ill-conditioned.
                if deviation <= extent * window * 1e-3:
                    total = correction = 0.0
                    for j in range(i - window + 1, i + 1):
                        total, correction = compensated_add(total, correction, abs(values[j] - mean))
                    deviation = total + correction
                result[i] = deviation / window
            outgoing = values[i - window + 1]
            if np.isfinite(outgoing):
                rank = np.searchsorted(ordered, outgoing)
                frequencies[rank] -= 1
                _set_sum(sums, corrections, rank, frequencies[rank] * (outgoing - anchor), capacity)
                _set_sum(counts, count_corrections, rank, float(frequencies[rank]), capacity)
            else:
                invalid -= 1
    return result
