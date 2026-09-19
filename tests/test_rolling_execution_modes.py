"""Check compiled/interpreted parity and obtain actual executed-line coverage."""

import numpy as np
import pytest

from polars_ti.utils import _order_stats, _rolling


@pytest.mark.parametrize("kind", ["finite", "nan", "infinity", "changing_scale"])
def test_numba_and_interpreter_agree(kind):
    values = 10 + np.random.default_rng(81).normal(size=150)
    if kind == "nan":
        values[:7] = np.nan
        values[50] = np.nan
    elif kind == "infinity":
        values[30], values[31] = np.inf, -np.inf
    elif kind == "changing_scale":
        values[:50] = 1e12
    periods = (np.arange(len(values)) % 17 + 1).astype(float)
    invocations = [
        (_rolling.rolling_sum, (values, 17)),
        (_rolling.rolling_difference, (values, 17)),
        (_rolling.rolling_variance, (values, 17, 1)),
        (_rolling.rolling_standard_deviation, (values, 17, 1)),
        (_rolling.rolling_extreme, (values, 17, True)),
        (_rolling.rolling_extreme, (values, 17, False, 1)),
        (_rolling.rolling_moments, (values, 17, 4)),
        (_rolling.rolling_quantile, (values, 17, 0.5)),
        (_rolling.rolling_ranks, (values, values, 17)),
        (_rolling._rolling_linear, (values, 17, True)),
        (_rolling._rolling_linear, (values, 17, False)),
        (_rolling.rolling_linear, (values, 17)),
        (_order_stats.variable_mean, (values, periods, 1, 17)),
        (_order_stats.rolling_mad, (values, 17)),
    ]
    for function, args in invocations:
        np.testing.assert_allclose(function.py_func(*args), function(*args), rtol=3e-13, atol=1e-12, equal_nan=True)


@pytest.mark.parametrize("scale", [0.0, 1e-200, 1.0, 1e200])
def test_scaled_moments_compiled_and_interpreted_agree(scale):
    values = np.array([1.0, 2.0, 1.0, 2.0, 1.0]) * scale
    np.testing.assert_allclose(
        _rolling.scaled_window_moments.py_func(values, 0, 5),
        _rolling.scaled_window_moments(values, 0, 5),
        rtol=2e-15,
        atol=0,
    )
    assert _rolling.needs_scaling.py_func(scale) == _rolling.needs_scaling(scale)
    for ascending in [True, False]:
        np.testing.assert_allclose(
            _rolling.scaled_window_linear.py_func(values, 0, 5, ascending),
            _rolling.scaled_window_linear(values, 0, 5, ascending),
            rtol=2e-15,
            atol=0,
        )
    if scale > 0:
        assert _rolling.binary_scale.py_func(scale) == _rolling.binary_scale(scale)


def test_python_tree_queries_and_updates():
    sums, corrections = np.zeros(16), np.zeros(16)
    for index, value in enumerate([1.0, -2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]):
        _order_stats._set_sum.py_func(sums, corrections, index, value, 8)
    assert _order_stats._range_sum.py_func(sums, corrections, 1, 7, 8) == 23
    tree = np.zeros(9, dtype=np.int64)
    for index in range(8):
        _rolling._count_add.py_func(tree, index, 1)
    assert _rolling._count_prefix.py_func(tree, 5) == 5
    assert _rolling._select.py_func(tree, 3) == 3
    assert sum(_rolling.compensated_add.py_func(1e16, 0.0, 1.0)) == 1e16
    total, correction = _rolling.compensated_add.py_func(1.0, 0.0, 1e16)
    assert correction == 1
