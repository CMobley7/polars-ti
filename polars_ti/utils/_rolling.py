"""Shared compensated rolling primitives with explicit missing-window semantics."""

from math import frexp, ldexp

import numpy as np
from numba import njit
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@njit(cache=True)
def compensated_add(total: float, correction: float, value: float) -> tuple[float, float]:
    """Add a finite value with Neumaier compensation."""
    updated = total + value
    if abs(total) >= abs(value):
        correction += (total - updated) + value
    else:
        correction += (value - updated) + total
    return updated, correction


@njit(cache=True)
def rolling_sum(values: FloatArray, window: int) -> FloatArray:
    """Return O(n) full-window sums, recovering after NaN or infinity leaves."""
    result = np.full(len(values), np.nan)
    if window < 1:
        raise ValueError("window must be positive")
    total = correction = 0.0
    invalid = positive_inf = negative_inf = 0
    for i in range(len(values)):
        if i % window == 0:
            total = correction = 0.0
            invalid = positive_inf = negative_inf = 0
            start = max(0, i - window + 1)
        else:
            start = i
            if i >= window:
                outgoing = values[i - window]
                if np.isnan(outgoing):
                    invalid -= 1
                elif outgoing == np.inf:
                    positive_inf -= 1
                elif outgoing == -np.inf:
                    negative_inf -= 1
                else:
                    total, correction = compensated_add(total, correction, -outgoing)
        for j in range(start, i + 1):
            incoming = values[j]
            if np.isnan(incoming):
                invalid += 1
            elif incoming == np.inf:
                positive_inf += 1
            elif incoming == -np.inf:
                negative_inf += 1
            else:
                total, correction = compensated_add(total, correction, incoming)
        if i >= window - 1 and invalid == 0:
            if positive_inf and negative_inf:
                continue
            result[i] = np.inf if positive_inf else -np.inf if negative_inf else total + correction
    return result


@njit(cache=True)
def rolling_extreme(
    values: FloatArray, window: int, maximum: bool, min_samples: int = 0
) -> tuple[FloatArray, IntArray]:
    """Return deque extrema and newest tied indices; invalid windows use -1."""
    if window < 1:
        raise ValueError("window must be positive")
    if min_samples == 0:
        min_samples = window
    size = len(values)
    result = np.full(size, np.nan)
    indices = np.full(size, -1, dtype=np.int64)
    queue = np.empty(size, dtype=np.int64)
    head = tail = invalid = 0
    for i in range(size):
        if i >= window and np.isnan(values[i - window]):
            invalid -= 1
        while head < tail and queue[head] <= i - window:
            head += 1
        value = values[i]
        if np.isnan(value):
            invalid += 1
        else:
            while head < tail:
                previous = values[queue[tail - 1]]
                if (maximum and previous <= value) or (not maximum and previous >= value):
                    tail -= 1
                else:
                    break
            queue[tail] = i
            tail += 1
        if i + 1 >= min_samples and invalid == 0 and head < tail:
            indices[i] = queue[head]
            result[i] = values[queue[head]]
    return result, indices


@njit(cache=True)
def rolling_moments(
    values: FloatArray, window: int, order: int = 2
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Compute mean and centered moments with compensated, periodically rebased sums.

    Values are shifted before powers are formed. Rebuilding once per window
    bounds drift while retaining linear total work; no large raw squares enter
    the variance calculation. The final center is the rounded float64 mean,
    matching a two-pass reference.
    """
    if window < 1:
        raise ValueError("window must be positive")
    size = len(values)
    mean = np.full(size, np.nan)
    second = np.full(size, np.nan)
    third = np.full(size, np.nan)
    fourth = np.full(size, np.nan)
    exact_means = rolling_sum(values, window) / window
    minimum = rolling_extreme(values, window, False)[0]
    maximum = rolling_extreme(values, window, True)[0]
    sums = np.zeros(4)
    corrections = np.zeros(4)
    anchor = 0.0
    max_delta = 0.0
    invalid = 0
    for i in range(size):
        reanchor = (
            i >= window - 1
            and np.isfinite(exact_means[i])
            and abs(anchor - exact_means[i]) > 2 * (maximum[i] - minimum[i])
        )
        if i >= window - 1 and np.isfinite(minimum[i]):
            extent = max(abs(minimum[i] - anchor), abs(maximum[i] - anchor))
            reanchor = reanchor or max_delta > 2 * extent
        if i % window == 0 or reanchor:
            start = max(0, i - window + 1)
            anchor = values[i] if np.isfinite(values[i]) else 0.0
            max_delta = 0.0
            sums[:] = 0.0
            corrections[:] = 0.0
            invalid = 0
        else:
            start = i
            if i >= window:
                outgoing = values[i - window]
                if not np.isfinite(outgoing):
                    invalid -= 1
                else:
                    delta = outgoing - anchor
                    power = delta
                    for exponent in range(order):
                        sums[exponent], corrections[exponent] = compensated_add(
                            sums[exponent], corrections[exponent], -power
                        )
                        power *= delta
        for j in range(start, i + 1):
            if not np.isfinite(values[j]):
                invalid += 1
                continue
            delta = values[j] - anchor
            max_delta = max(max_delta, abs(delta))
            power = delta
            for exponent in range(order):
                sums[exponent], corrections[exponent] = compensated_add(sums[exponent], corrections[exponent], power)
                power *= delta
        if i < window - 1 or invalid:
            continue
        # Summing and then dividing a large constant can round the mean away
        # from that constant (or overflow). Its exact centered moments are zero.
        if minimum[i] == maximum[i]:
            mean[i] = minimum[i]
            second[i] = third[i] = fourth[i] = 0.0
            continue
        magnitude = max(abs(minimum[i]), abs(maximum[i]))
        if needs_scaling(magnitude):
            mean[i], normalized2, normalized3, normalized4, scale, _ = scaled_window_moments(
                values, i - window + 1, i + 1
            )
            second[i] = (normalized2 * scale) * scale
            third[i] = ((normalized3 * scale) * scale) * scale
            fourth[i] = (((normalized4 * scale) * scale) * scale) * scale
            continue
        first = (sums[0] + corrections[0]) / window
        mean[i] = exact_means[i]
        shift = mean[i] - anchor
        raw2 = (sums[1] + corrections[1]) / window
        centered_second = raw2 - 2 * shift * first + shift * shift
        second[i] = np.nan if np.isnan(centered_second) else max(0.0, centered_second)
        if order >= 3:
            raw3 = (sums[2] + corrections[2]) / window
            third[i] = raw3 - 3 * shift * raw2 + 3 * shift * shift * first - shift**3
        if order >= 4:
            raw4 = (sums[3] + corrections[3]) / window
            fourth[i] = raw4 - 4 * shift * raw3 + 6 * shift**2 * raw2 - 4 * shift**3 * first + shift**4
    return mean, second, third, fourth


@njit(cache=True)
def _count_add(tree: IntArray, index: int, delta: int) -> None:
    """Update one Fenwick count."""
    index += 1
    while index < len(tree):
        tree[index] += delta
        index += index & -index


@njit(cache=True)
def _count_prefix(tree: IntArray, end: int) -> int:
    """Count ranks strictly below an exclusive endpoint."""
    count = 0
    while end > 0:
        count += tree[end]
        end -= end & -end
    return count


@njit(cache=True)
def _select(tree: IntArray, rank: int) -> int:
    """Find a zero-based order statistic in logarithmic time."""
    index = 0
    step = 1
    while step < len(tree):
        step *= 2
    while step:
        candidate = index + step
        if candidate < len(tree) and tree[candidate] <= rank:
            index = candidate
            rank -= tree[candidate]
        step //= 2
    return index


@njit(cache=True)
def rolling_quantile(values: FloatArray, window: int, quantile: float) -> FloatArray:
    """Compute O(n log w) quantiles with O(w) block-local order statistics."""
    size = len(values)
    result = np.full(size, np.nan)
    if window < 1 or not 0 <= quantile <= 1:
        raise ValueError("positive window and quantile in [0, 1] required")
    index = quantile * (window - 1)
    lower = int(index)
    fraction = index - lower
    for block in range(window - 1, size, window):
        end = min(size, block + window)
        start = block - window + 1
        ordered = np.sort(values[start:end])
        tree = np.zeros(len(ordered) + 1, dtype=np.int64)
        invalid = 0
        for i in range(start, end):
            value = values[i]
            if np.isnan(value):
                invalid += 1
            else:
                _count_add(tree, np.searchsorted(ordered, value), 1)
            if i < block:
                continue
            if invalid == 0:
                low = ordered[_select(tree, lower)]
                result[i] = (
                    low
                    if fraction == 0.0 or lower == window - 1
                    else low * (1 - fraction) + ordered[_select(tree, lower + 1)] * fraction
                )
            outgoing = values[i - window + 1]
            if np.isnan(outgoing):
                invalid -= 1
            else:
                _count_add(tree, np.searchsorted(ordered, outgoing), -1)
    return result


@njit(cache=True)
def rolling_ranks(history: FloatArray, queries: FloatArray, window: int) -> tuple[FloatArray, FloatArray]:
    """Count strict less/greater values in the preceding window, ignoring NaNs."""
    size = len(history)
    less = np.full(size, np.nan)
    greater = np.full(size, np.nan)
    if window == 0:
        return np.zeros(size), np.zeros(size)
    if window < 0:
        raise ValueError("window must be nonnegative")
    for block in range(window, size, window):
        end = min(size, block + window)
        start = block - window
        ordered = np.sort(history[start:end])
        tree = np.zeros(len(ordered) + 1, dtype=np.int64)
        count = 0
        for i in range(start, end):
            if i >= block:
                query = queries[i]
                if np.isnan(query):
                    less[i] = greater[i] = 0.0
                else:
                    less[i] = _count_prefix(tree, np.searchsorted(ordered, query, side="left"))
                    greater[i] = count - _count_prefix(tree, np.searchsorted(ordered, query, side="right"))
                outgoing = history[i - window]
                if not np.isnan(outgoing):
                    _count_add(tree, np.searchsorted(ordered, outgoing), -1)
                    count -= 1
            incoming = history[i]
            if not np.isnan(incoming):
                _count_add(tree, np.searchsorted(ordered, incoming), 1)
                count += 1
    return less, greater


@njit(cache=True)
def _rolling_linear(values: FloatArray, window: int, centered: bool) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return sums, ascending weighted sums, and centered linear covariances."""
    if window < 1:
        raise ValueError("window must be positive")
    size = len(values)
    totals = np.full(size, np.nan)
    weighted = np.full(size, np.nan)
    covariance = np.full(size, np.nan)
    minimum = rolling_extreme(values, window, False)[0]
    maximum = rolling_extreme(values, window, True)[0]
    total = correction = weight_sum = weight_correction = anchor = 0.0
    state_magnitude = 0.0
    invalid = 0
    weight_total = window * (window + 1) / 2.0
    for i in range(size):
        reanchor = (
            centered
            and i >= window - 1
            and np.isfinite(minimum[i])
            and (anchor < minimum[i] - (maximum[i] - minimum[i]) or anchor > maximum[i] + (maximum[i] - minimum[i]))
        )
        magnitude = max(abs(minimum[i]), abs(maximum[i]))
        scale_drop = np.isfinite(magnitude) and state_magnitude > magnitude * 1e6
        if i % window == 0 or reanchor or scale_drop:
            start = max(0, i - window + 1)
            anchor = values[i] if centered and np.isfinite(values[i]) else 0.0
            total = correction = weight_sum = weight_correction = 0.0
            state_magnitude = 0.0
            invalid = 0
        else:
            start = i
            weight_sum, weight_correction = compensated_add(weight_sum, weight_correction, -total)
            weight_sum, weight_correction = compensated_add(weight_sum, weight_correction, -correction)
            if i >= window:
                if not np.isfinite(values[i - window]):
                    invalid -= 1
                else:
                    total, correction = compensated_add(total, correction, -(values[i - window] - anchor))
        for j in range(start, i + 1):
            if not np.isfinite(values[j]):
                invalid += 1
                continue
            delta = values[j] - anchor
            state_magnitude = max(state_magnitude, abs(values[j]))
            weight = j - i + window
            total, correction = compensated_add(total, correction, delta)
            weight_sum, weight_correction = compensated_add(weight_sum, weight_correction, weight * delta)
        if i >= window - 1 and invalid == 0:
            centered_total = total + correction
            centered_weight = weight_sum + weight_correction
            # A vanished outlier can leave a small absolute residue that becomes
            # dominant when the current weighted sum nearly cancels.
            if state_magnitude > magnitude and abs(centered_weight + weight_total * anchor) < (
                state_magnitude * weight_total * 1e-3
            ):
                total = correction = weight_sum = weight_correction = 0.0
                state_magnitude = magnitude
                for j in range(i - window + 1, i + 1):
                    delta = values[j] - anchor
                    total, correction = compensated_add(total, correction, delta)
                    weight_sum, weight_correction = compensated_add(
                        weight_sum, weight_correction, (j - i + window) * delta
                    )
                centered_total = total + correction
                centered_weight = weight_sum + weight_correction
            totals[i] = centered_total + window * anchor
            weighted[i] = centered_weight + weight_total * anchor
            covariance[i] = centered_weight - ((window + 1) / 2.0) * centered_total
    return totals, weighted, covariance


@njit(cache=True)
def rolling_linear(values: FloatArray, window: int) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return compensated raw totals/weights and separately centered covariance."""
    total, weighted, _ = _rolling_linear(values, window, False)
    _, _, covariance = _rolling_linear(values, window, True)
    return total, weighted, covariance


@njit(cache=True)
def needs_scaling(magnitude: float) -> bool:
    """Identify finite scales where fourth powers overflow or underflow."""
    return np.isfinite(magnitude) and magnitude != 0.0 and (magnitude > 1e75 or magnitude < 1e-75)


@njit(cache=True)
def binary_scale(magnitude: float) -> float:
    """Return the power-of-two scale of a positive finite magnitude."""
    _, exponent = frexp(magnitude)
    return ldexp(1.0, exponent - 1)


@njit(cache=True)
def scaled_window_linear(
    values: FloatArray, start: int, end: int, ascending: bool = True
) -> tuple[float, float, float]:
    """Return normalized total, weighted total and scale for an extreme window."""
    magnitude = 0.0
    for i in range(start, end):
        if not np.isfinite(values[i]):
            return np.nan, np.nan, 1.0
        magnitude = max(magnitude, abs(values[i]))
    if magnitude == 0.0:
        return 0.0, 0.0, 1.0
    scale = binary_scale(magnitude)
    total = correction = weighted = weight_correction = 0.0
    for i in range(start, end):
        value = values[i] / scale
        weight = i - start + 1 if ascending else end - i
        total, correction = compensated_add(total, correction, value)
        weighted, weight_correction = compensated_add(weighted, weight_correction, value * weight)
    return total + correction, weighted + weight_correction, scale


@njit(cache=True)
def scaled_window_moments(values: FloatArray, start: int, end: int) -> tuple[float, float, float, float, float, float]:
    """Evaluate an extreme window using exact power-of-two scaling and centering.

    Returns its mean, normalized second/third/fourth moments, binary scale and
    normalized last centered value. Keeping the center as an offset avoids
    losing fractional ULPs when prices are large relative to their spread.
    """
    magnitude = 0.0
    for i in range(start, end):
        if not np.isfinite(values[i]):
            return np.nan, np.nan, np.nan, np.nan, 1.0, np.nan
        magnitude = max(magnitude, abs(values[i]))
    if magnitude == 0.0:
        return 0.0, 0.0, 0.0, 0.0, 1.0, 0.0
    scale = binary_scale(magnitude)
    anchor = values[start] / scale
    total = correction = 0.0
    for i in range(start, end):
        total, correction = compensated_add(total, correction, values[i] / scale - anchor)
    center = (total + correction) / (end - start)
    sums = np.zeros(3)
    corrections = np.zeros(3)
    for i in range(start, end):
        delta = (values[i] / scale - anchor) - center
        power = delta * delta
        for order in range(3):
            sums[order], corrections[order] = compensated_add(sums[order], corrections[order], power)
            power *= delta
    moments = (sums + corrections) / (end - start)
    return (
        (anchor + center) * scale,
        moments[0],
        moments[1],
        moments[2],
        scale,
        (values[end - 1] / scale - anchor) - center,
    )


@njit(cache=True)
def rolling_difference(values: FloatArray, window: int) -> FloatArray:
    """Sum differences from the last value using a locally centered recurrence."""
    if window < 1:
        raise ValueError("window must be positive")
    result = np.full(len(values), np.nan)
    minimum = rolling_extreme(values, window, False)[0]
    maximum = rolling_extreme(values, window, True)[0]
    anchor = total = correction = 0.0
    invalid = 0
    for i in range(len(values)):
        spread = maximum[i] - minimum[i]
        reanchor = np.isfinite(spread) and (anchor < minimum[i] - spread or anchor > maximum[i] + spread)
        if i % window == 0 or reanchor:
            anchor = values[i] if np.isfinite(values[i]) else 0.0
            total = correction = 0.0
            invalid = 0
            start = max(0, i - window + 1)
        else:
            start = i
            if i >= window:
                if np.isfinite(values[i - window]):
                    total, correction = compensated_add(total, correction, -(values[i - window] - anchor))
                else:
                    invalid -= 1
        for j in range(start, i + 1):
            if np.isfinite(values[j]):
                total, correction = compensated_add(total, correction, values[j] - anchor)
            else:
                invalid += 1
        if i >= window - 1 and invalid == 0:
            result[i] = (window * (values[i] - anchor) - total) - correction
    return result


@njit(cache=True)
def rolling_variance(values: FloatArray, window: int, ddof: int = 0) -> FloatArray:
    """Apply the sample adjustment before rounding extreme variances to float64."""
    if window <= ddof:
        return np.full(len(values), np.nan)
    adjustment = window / (window - ddof)
    result = rolling_moments(values, window)[1] * adjustment
    minimum = rolling_extreme(values, window, False)[0]
    maximum = rolling_extreme(values, window, True)[0]
    for i in range(window - 1, len(values)):
        magnitude = max(abs(minimum[i]), abs(maximum[i]))
        if minimum[i] != maximum[i] and needs_scaling(magnitude):
            _, second, _, _, scale, _ = scaled_window_moments(values, i - window + 1, i + 1)
            result[i] = ((second * adjustment) * scale) * scale
    return result


@njit(cache=True)
def rolling_standard_deviation(values: FloatArray, window: int, ddof: int = 0) -> FloatArray:
    """Compute standard deviation without first overflowing/underflowing variance."""
    result = np.full(len(values), np.nan)
    if window <= ddof:
        return result
    _, variance, _, _ = rolling_moments(values, window)
    minimum = rolling_extreme(values, window, False)[0]
    maximum = rolling_extreme(values, window, True)[0]
    adjustment = window / (window - ddof)
    for i in range(window - 1, len(values)):
        magnitude = max(abs(minimum[i]), abs(maximum[i]))
        if minimum[i] == maximum[i] and np.isfinite(minimum[i]):
            result[i] = 0.0
        elif needs_scaling(magnitude):
            _, second, _, _, scale, _ = scaled_window_moments(values, i - window + 1, i + 1)
            result[i] = np.sqrt(second * adjustment) * scale
        else:
            result[i] = np.sqrt(variance[i] * adjustment)
    return result
