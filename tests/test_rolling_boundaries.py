"""Boundary, recovery, and tie behavior of the shared rolling algorithms."""

import numpy as np
import pytest

from polars_ti.momentum.stc import nb_schaff_tc
from polars_ti.utils._order_stats import rolling_mad, variable_mean
from polars_ti.utils._rolling import rolling_extreme, rolling_quantile, rolling_ranks, rolling_sum


@pytest.mark.parametrize("window", [1, 2, 17, 1000])
@pytest.mark.parametrize("maximum", [True, False])
def test_extrema_indices_recovery_and_ties(window, maximum):
    values = np.array([np.nan] * 5 + [3.0, 2.0, 2.0, 4.0, 4.0, np.nan, 1.0, 5.0, np.inf, -np.inf] + [3.0] * 30)
    actual, indices = rolling_extreme(values, window, maximum)
    for i in range(len(values)):
        if i < window - 1 or np.isnan(values[max(0, i - window + 1) : i + 1]).any():
            assert np.isnan(actual[i])
            assert indices[i] == -1
        else:
            segment = values[i - window + 1 : i + 1]
            expected = np.max(segment) if maximum else np.min(segment)
            assert actual[i] == expected
            assert indices[i] == i - window + 1 + np.flatnonzero(segment == expected)[-1]


def test_empty_and_oversized_windows():
    for values in [np.array([], dtype=float), np.array([1.0, 2.0])]:
        for window in [3, 10**9]:
            for result in [
                rolling_sum(values, window),
                rolling_quantile(values, window, 0.5),
                rolling_mad(values, window),
            ]:
                assert len(result) == len(values)
                assert np.isnan(result).all()


@pytest.mark.parametrize("window", [1, 17, 128, 1536])
def test_rank_counts_are_exact_including_nan_and_infinity(window):
    values = np.random.default_rng(200).integers(-5, 6, size=3500).astype(float)
    values[::29] = np.nan
    values[::41] = np.inf
    values[::67] = -np.inf
    queries = values[::-1].copy()
    less, greater = rolling_ranks(values, queries, window)
    for i in range(window, len(values)):
        history = values[i - window : i]
        assert less[i] == np.count_nonzero(history < queries[i])
        assert greater[i] == np.count_nonzero(history > queries[i])


def test_variable_mean_local_missing_windows():
    values = np.array([1.0, np.nan, 3.0, 4.0, np.inf, 6.0, 7.0, 8.0])
    periods = np.array([np.nan, 0.0, 2.0, 9.0, 1.0, 2.0, 3.0, 2.0])
    result = variable_mean(values, periods, 1, 3)
    np.testing.assert_array_equal(result, [np.nan, np.nan, np.nan, np.nan, np.inf, np.inf, np.inf, 7.5])


@pytest.mark.parametrize("input_class", ["prefix", "interior", "all_nan"])
def test_stc_full_window_extrema_keep_explicit_carry_policy(input_class):
    values = 3 + np.random.default_rng(50).normal(size=150)
    if input_class == "prefix":
        values[:19] = np.nan
    elif input_class == "interior":
        values[70] = np.nan
    else:
        values[:] = np.nan
    window, factor = 7, 0.5
    first = np.zeros(len(values))
    smooth = np.zeros(len(values))
    second = np.zeros(len(values))
    result = np.zeros(len(values))
    for i in range(1, len(values)):
        segment = values[max(0, i - window + 1) : i + 1]
        low, high = np.min(segment), np.max(segment)
        span = high - low
        if span == 0:
            span = 1
        first[i] = 100 * (values[i] - low) / span if low > 0 else first[i - 1]
        smooth[i] = smooth[i - 1] + factor * (first[i] - smooth[i - 1])
        filtered = smooth[max(0, i - window + 1) : i + 1]
        low, high = np.min(filtered), np.max(filtered)
        span = high - low
        if span == 0:
            span = 1
        second[i] = 100 * (smooth[i] - low) / span if span > 0 else second[i - 1]
        result[i] = result[i - 1] + factor * (second[i] - result[i - 1])
    actual, actual_smooth = nb_schaff_tc(values, window, factor)
    np.testing.assert_array_equal(actual, result)
    np.testing.assert_array_equal(actual_smooth, smooth)
