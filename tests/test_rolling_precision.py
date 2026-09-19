"""Independent numerical oracles, including cancellation and changing baselines."""

import math

import numpy as np
import pytest

from polars_ti.utils._rolling import rolling_linear, rolling_moments, rolling_sum


@pytest.mark.parametrize("window", [2, 5, 40, 97])
@pytest.mark.parametrize("kind", ["prices", "tiny_spread", "outlier", "alternating"])
def test_compensated_primitives(window, kind):
    values = 60648 + np.random.default_rng(144).normal(size=300)
    if kind == "tiny_spread":
        values = 1e9 + np.random.default_rng(144).normal(scale=1e-4, size=300)
    elif kind == "outlier":
        values[:120] = 1e12
    elif kind == "alternating":
        values[::2] *= -1
    total = rolling_sum(values, window)
    linear_total, weighted, covariance = rolling_linear(values, window)
    means, second, third, fourth = rolling_moments(values, window, 4)
    for i in range(window - 1, len(values)):
        segment = values[i - window + 1 : i + 1]
        expected_sum = math.fsum(segment)
        expected_mean = expected_sum / window
        expected_second = math.fsum((value - expected_mean) ** 2 for value in segment) / window
        expected_third = math.fsum((value - expected_mean) ** 3 for value in segment) / window
        expected_fourth = math.fsum((value - expected_mean) ** 4 for value in segment) / window
        np.testing.assert_allclose(total[i], expected_sum, rtol=2e-15, atol=1e-10)
        np.testing.assert_allclose(linear_total[i], expected_sum, rtol=2e-15, atol=1e-10)
        np.testing.assert_allclose(
            weighted[i], math.fsum((j + 1) * value for j, value in enumerate(segment)), rtol=2e-14, atol=1e-8
        )
        np.testing.assert_allclose(second[i], expected_second, rtol=2e-12, atol=1e-18)
        np.testing.assert_allclose(third[i], expected_third, rtol=2e-10, atol=max(1e-22, expected_second**1.5 * 2e-12))
        np.testing.assert_allclose(fourth[i], expected_fourth, rtol=2e-11, atol=1e-24)


@pytest.mark.parametrize("window", [2, 3])
@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_sma_extreme_means_and_recovery(window, sign):
    from polars_ti.overlap.sma import nb_sma

    values = np.array([sign * 1e308, sign * 1e308, 1.0, -1.0, 1.0, 1.0])
    result = nb_sma(values, window)
    for i in range(window - 1, len(values)):
        segment = values[i - window + 1 : i + 1]
        scale = max(abs(segment))
        expected = math.fsum(segment / scale) / window * scale
        np.testing.assert_allclose(result[i], expected, rtol=2e-15, atol=0)


@pytest.mark.parametrize("infinity", [np.inf, -np.inf])
@pytest.mark.parametrize("ascending", [True, False])
def test_wma_preserves_signed_infinity_and_recovery(infinity, ascending):
    from polars_ti.overlap.wma import nb_wma

    result = nb_wma(np.array([1.0, infinity, 3.0, 4.0, 5.0]), 2, ascending, True)
    assert np.isnan(result[0])
    np.testing.assert_array_equal(result[1:3], [infinity, infinity])
    np.testing.assert_allclose(result[3:], [11 / 3, 14 / 3] if ascending else [10 / 3, 13 / 3])


def test_descending_wma_preserves_cancellation_residual():
    from polars_ti.overlap.wma import nb_wma

    values = np.array([2.0**53, -(2.0**54) + 2])
    expected = math.fsum([2 * values[0], values[1]]) / 3
    assert nb_wma(values, 2, False, True)[-1] == expected


@pytest.mark.parametrize("module,function", [("trend.trendflex", "nb_trendflex"), ("cycles.reflex", "np_reflex")])
def test_nearly_flat_normalized_filters(module, function):
    from importlib import import_module

    from scripts.rolling_kernel_audit.reference import original

    values = 100 + np.random.default_rng(71).normal(scale=1e-12, size=3000)
    args = (values, 40, 20, 0.04, np.pi, np.sqrt(2))
    expected = getattr(original(module), function)(*args)
    actual = getattr(import_module("polars_ti." + module), function)(*args)
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)


def test_sample_variance_adjusts_before_subnormal_rounding():
    import polars as pl

    from polars_ti.statistics.variance import variance

    values = [0.0, 3e-162]
    expected = float((np.longdouble(values[1]) ** 2) / 2)
    actual = pl.DataFrame({"x": values}).select(variance("x", 2, ddof=1, talib=False)).item(-1, 0)
    assert expected > 0
    assert actual == expected


def test_exact_quantile_does_not_evaluate_unused_infinite_neighbor():
    from polars_ti.utils._rolling import rolling_quantile

    assert rolling_quantile(np.array([1.0, np.inf]), 2, 0.0)[1] == 1.0
    assert rolling_quantile(np.array([1.0, 2.0, np.inf]), 3, 0.5)[2] == 2.0


@pytest.mark.parametrize("outlier", [3, 4, 6, 8])
def test_linear_weights_recover_after_finite_outlier(outlier):
    from polars_ti.momentum.cg import nb_cg
    from polars_ti.overlap.wma import nb_wma

    values = np.arange(1.0, 21.0)
    values[outlier] = 1e100
    for ascending in [True, False]:
        weights = np.arange(1.0, 6.0) if ascending else np.arange(5.0, 0.0, -1.0)
        actual = nb_wma(values, 5, ascending, True)
        for i in range(4, len(values)):
            if i - 4 <= outlier <= i:
                continue
            expected = math.fsum(values[i - 4 : i + 1] * weights) / 15
            np.testing.assert_allclose(actual[i], expected, rtol=2e-15)
    actual = nb_cg(values, 5)
    for i in range(outlier + 5, len(values)):
        segment = values[i - 4 : i + 1]
        expected = -math.fsum(segment * np.arange(1.0, 6.0)) / math.fsum(segment)
        np.testing.assert_allclose(actual[i], expected, rtol=2e-15)


def test_msw_rebuilds_state_after_finite_outlier():
    from polars_ti.cycles.msw import nb_msw
    from scripts.rolling_kernel_audit.reference import original

    values = np.arange(1.0, 31.0)
    values[6] = 1e100
    expected = original("cycles.msw").nb_msw(values, 5)
    actual = nb_msw(values, 5)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_variable_mean_preserves_compensation_across_ring_boundary():
    from polars_ti.utils._order_stats import variable_mean

    values = np.array([0.0, 0.0, 1e16, -1e16, 1.0])
    actual = variable_mean(values, np.full(5, 3.0), 1, 3)
    assert actual[-1] == math.fsum(values[-3:]) / 3


@pytest.mark.parametrize("base", [1.0, 100.0, 1e6])
@pytest.mark.parametrize("ratio", [1e3, 1e5, 999999.0, 1e6, 1e6 + 1])
def test_msw_flat_windows_after_moderate_outlier(base, ratio):
    from polars_ti.cycles.msw import nb_msw
    from scripts.rolling_kernel_audit.reference import original

    values = np.full(30, base)
    values[6] = base * ratio
    actual = nb_msw(values, 5)
    expected = original("cycles.msw").nb_msw(values, 5)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_public_sma_preserves_extreme_mean_and_recovers():
    import polars as pl

    from polars_ti.overlap.sma import sma

    values = [1e308, 1e308, 1.0, 1.0, 1.0]
    result = pl.DataFrame({"x": values}).select(sma("x", 2, talib=False)).to_series()
    assert result.is_null().to_list() == [True, False, False, False, False]
    np.testing.assert_allclose(result.to_numpy(), [np.nan, 1e308, 5e307, 1.0, 1.0], rtol=2e-15)


def test_linear_weights_rebuild_when_outlier_error_dominates_cancellation():
    from polars_ti.momentum.cg import nb_cg
    from polars_ti.overlap.wma import nb_wma

    values = np.array([1.0, 2.0, 3.0, 100000.1, 1.0, -2.0, 1.0, 0.0, 1e-8])
    assert nb_cg(values, 5)[-1] == -5.0
    expected = math.fsum(values[-5:] * np.arange(1.0, 6.0)) / 15
    np.testing.assert_allclose(nb_wma(values, 5, True, True)[-1], expected, rtol=2e-15, atol=0)


def test_variable_mean_recovers_from_tree_intermediate_overflow():
    from polars_ti.utils._order_stats import variable_mean

    values = np.array([0.0, 1e308, -1e308, -1e308, 1e308])
    assert variable_mean(values, np.full(5, 4.0), 1, 4)[-1] == 0.0


def test_msw_preserves_small_trend_after_moderate_outlier():
    from polars_ti.cycles.msw import nb_msw
    from scripts.rolling_kernel_audit.reference import original

    values = 1.0 + np.arange(30) * 1e-12
    values[6] = 1e5
    np.testing.assert_allclose(nb_msw(values, 5), original("cycles.msw").nb_msw(values, 5), rtol=1e-12, atol=1e-12)


def test_overflow_fallback_scaling_preserves_exact_cancellation():
    from polars_ti.overlap.sma import nb_sma
    from polars_ti.utils._order_stats import variable_mean

    values = np.array([1.5e308, 5e307, -1e308, -1e308])
    assert nb_sma(values, 4)[-1] == 0.0
    wrapped = np.array([0.0, 1.5e308, -1e308, -1e308, 5e307])
    assert variable_mean(wrapped, np.full(5, 4.0), 1, 4)[-1] == 0.0


def test_weighted_indicators_avoid_unrepresentable_intermediates():
    import polars as pl

    from polars_ti.momentum.cg import nb_cg
    from polars_ti.overlap.wma import wma

    frame = pl.DataFrame({"x": [1e308, -1e308, 1e308]})
    for ascending, expected in [(True, 1e308 / 3), (False, -1e308 / 3)]:
        actual = frame.select(wma("x", 2, asc=ascending, talib=False)).item(-1, 0)
        np.testing.assert_allclose(actual, expected, rtol=2e-15)
    assert nb_cg(frame["x"].to_numpy(), 3)[-1] == -2.0
