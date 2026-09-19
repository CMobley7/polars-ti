"""Regression and independent-reference tests for the rolling kernel repairs."""

import math

import numpy as np
import polars as pl
import pytest

from polars_ti.cycles.msw import nb_msw
from polars_ti.momentum.stoch import _stoch_rawk
from polars_ti.statistics.quantile import nb_quantile
from polars_ti.statistics.stdev import stdev
from polars_ti.statistics.variance import variance
from polars_ti.trend.aroon import _nb_aroon
from polars_ti.trend.trama import _nb_rolling_max, _nb_rolling_min


@pytest.mark.parametrize("position", range(5))
def test_extrema_nan_is_independent_of_seed(position):
    values = np.arange(5, dtype=float)
    values[position] = np.nan
    assert np.isnan(_nb_rolling_max(values, 5)[-1])
    assert np.isnan(_nb_rolling_min(values, 5)[-1])
    assert np.isnan(_nb_aroon(values, values, 4, 100.0)[0][-1])
    assert np.isnan(_stoch_rawk(values + 2, values, values + 1, 5)[-1])


@pytest.mark.parametrize("window", [1, 2, 5, 17, 40])
def test_quantile_nan_windows_recover(window):
    values = np.random.default_rng(13).normal(size=150)
    values[:7] = np.nan
    values[80] = np.nan
    for quantile in [0.0, 0.1, 0.5, 0.9, 1.0]:
        expected = np.full(len(values), np.nan)
        for i in range(window - 1, len(values)):
            segment = values[i - window + 1 : i + 1]
            if not np.isnan(segment).any():
                ordered = np.sort(segment)
                index = quantile * (window - 1)
                lower = int(index)
                fraction = index - lower
                expected[i] = (
                    ordered[lower]
                    if lower == window - 1
                    else ordered[lower] * (1 - fraction) + ordered[lower + 1] * fraction
                )
        np.testing.assert_array_equal(nb_quantile(values, window, quantile), expected)


@pytest.mark.parametrize("period", [2, 5, 17])
def test_msw_nan_window_and_recovery(period):
    values = np.random.default_rng(11).normal(size=100)
    values[:3] = np.nan
    values[50] = np.nan
    sine, lead = nb_msw(values, period)
    for i in range(period, len(values)):
        invalid = np.isnan(values[i - period + 1 : i + 1]).any()
        assert np.isnan(sine[i]) == invalid
        assert np.isnan(lead[i]) == invalid


@pytest.mark.parametrize("window", [2, 5, 40, 480])
@pytest.mark.parametrize("ddof", [0, 1])
def test_variance_against_fsum(window, ddof):
    values = 60648 + np.random.default_rng(7).normal(scale=0.01, size=1100)
    expected = np.full(len(values), np.nan)
    for i in range(window - 1, len(values)):
        segment = values[i - window + 1 : i + 1]
        mean = math.fsum(segment) / window
        expected[i] = math.fsum((value - mean) ** 2 for value in segment) / (window - ddof)
    frame = pl.DataFrame({"x": values})
    actual = frame.select(variance("x", window, ddof, talib=False)).to_series().to_numpy()
    np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=1e-22)
    actual_std = frame.select(stdev("x", window, ddof, talib=False)).to_series().to_numpy()
    np.testing.assert_allclose(actual_std, np.sqrt(expected), rtol=2e-13, atol=1e-20)


@pytest.mark.parametrize("function, native", [(variance, "rolling_var"), (stdev, "rolling_std")])
@pytest.mark.parametrize("ddof", [0, 1, 2])
def test_variance_preserves_null_versus_nan(function, native, ddof):
    frame = pl.DataFrame({"x": [1.0, 2.0, np.nan, 4.0, None, 6.0, 7.0, 8.0, np.inf, 9.0, 10.0]})
    expected = frame.select(getattr(pl.col("x"), native)(window_size=2, min_samples=2, ddof=ddof)).to_series()
    actual = frame.select(function("x", length=2, ddof=ddof, talib=False)).to_series()
    assert actual.is_null().to_list() == expected.is_null().to_list()
    assert actual.is_nan().fill_null(False).to_list() == expected.is_nan().fill_null(False).to_list()


@pytest.mark.parametrize("value", [1e300, -1e300, 1e308, 1e-300])
def test_exact_constants_have_zero_variance_at_extreme_scale(value):
    from polars_ti.utils._rolling import rolling_moments

    values = np.full(20, value)
    mean, second, third, fourth = rolling_moments(values, 7, 4)
    np.testing.assert_array_equal(mean[6:], values[6:])
    for moments in [second, third, fourth]:
        np.testing.assert_array_equal(moments[6:], 0)
    frame = pl.DataFrame({"x": values})
    for function in [variance, stdev]:
        np.testing.assert_array_equal(frame.select(function("x", 7, talib=False)).to_series().to_numpy()[6:], 0)


@pytest.mark.parametrize("base, step", [(1e308, 2e292), (1e-200, 1e-200)])
def test_nonconstant_extreme_windows_keep_representable_outputs(base, step):
    from polars_ti.overlap.linreg import nb_linreg
    from polars_ti.statistics.kurtosis import nb_kurtosis
    from polars_ti.statistics.skew import nb_skew
    from polars_ti.statistics.zscore import nb_zscore

    normalized = np.array([0.0, 0.0, 1.0, 0.0, 1.0])
    values = base + step * normalized
    actual_step = values[2] - values[0]
    expected_std = actual_step * np.sqrt(0.4 * 0.6)
    actual = pl.DataFrame({"x": values}).select(stdev("x", 5, talib=False)).to_series()[-1]
    np.testing.assert_allclose(actual, expected_std, rtol=2e-14, atol=0)
    for function, extra in [(nb_zscore, (1.0,)), (nb_skew, ()), (nb_kurtosis, ())]:
        np.testing.assert_allclose(
            function(values, 5, *extra)[-1], function(normalized, 5, *extra)[-1], rtol=2e-14, atol=2e-14
        )
    flags = (False, False, False, True, False, False)
    np.testing.assert_allclose(
        nb_linreg(values, 5, *flags)[-1], nb_linreg(normalized, 5, *flags)[-1], rtol=2e-14, atol=0
    )
    endpoint = nb_linreg(values, 5, False, False, False, False, False, False)[-1]
    assert np.isfinite(endpoint)
