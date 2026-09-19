"""FIR optimization must beat direct float64 summation accuracy."""

import math

import numpy as np
import pytest

from polars_ti.utils._fir import rolling_fir


@pytest.mark.parametrize("window", [1, 5, 40, 480])
@pytest.mark.parametrize("kind", ["random", "prefix", "cancel", "outlier", "zero"])
def test_fir_independent_accuracy(window, kind):
    values = np.random.default_rng(21).normal(size=1200) + 60648
    if kind == "prefix":
        values[:17] = np.nan
    elif kind == "cancel":
        values[::2] *= -1
    elif kind == "outlier":
        values[700] = 1e16
    elif kind == "zero":
        values[:] = 0
    weights = np.exp(-0.5 * ((np.arange(window) - window / 3) / window * 6) ** 2)
    weights /= weights.sum()
    actual = rolling_fir(values, weights)
    expected = np.full(len(values), np.nan)
    for i in range(window - 1, len(values)):
        segment = values[i - window + 1 : i + 1]
        if np.isfinite(segment).all():
            expected[i] = math.fsum(float(value) * float(weight) for value, weight in zip(segment, weights))
    np.testing.assert_allclose(actual, expected, rtol=3e-15, atol=2e-11)


def test_cancellation_does_not_force_direct_windows(monkeypatch):
    from polars_ti.utils import _fir

    values = np.tile(np.array([-1.0, 1.0]), 5000)
    weights = np.full(128, 1 / 128)
    original_sum = _fir.np.sum
    window_sum_calls = []

    def tracked_sum(array, *args, **kwargs):
        if isinstance(array, np.ndarray) and array.dtype == np.longdouble and array.shape == weights.shape:
            window_sum_calls.append(1)
        return original_sum(array, *args, **kwargs)

    monkeypatch.setattr(_fir.np, "sum", tracked_sum)
    actual = _fir.rolling_fir(values, weights)
    np.testing.assert_allclose(actual[127:], 0, atol=2e-16, rtol=0)
    # The weight norm is summed once; cancellation alone must not scan windows.
    assert len(window_sum_calls) <= 1


def test_distant_outlier_does_not_force_all_later_windows(monkeypatch):
    from polars_ti.utils import _fir

    values = np.ones(20_000)
    values[0] = 1e12
    window = 128
    weights = np.full(window, 1 / window)
    original_sum = _fir.np.sum
    direct_calls = []

    def tracked_sum(array, *args, **kwargs):
        if isinstance(array, np.ndarray) and array.dtype == np.longdouble and array.shape == weights.shape:
            direct_calls.append(1)
        return original_sum(array, *args, **kwargs)

    monkeypatch.setattr(_fir.np, "sum", tracked_sum)
    actual = _fir.rolling_fir(values, weights)
    np.testing.assert_allclose(actual[window:], 1, atol=3e-15, rtol=0)
    assert len(direct_calls) <= 4 * window


@pytest.mark.parametrize("window", [35, 64, 127, 480])
def test_fir_prefix_is_bitwise_stable(window):
    """Appending observations must not change even the rounding of earlier rows."""
    values = 60648 + np.random.default_rng(71).normal(size=1600)
    weights = np.exp(-0.5 * ((np.arange(window) - window / 3) / window * 6) ** 2)
    weights /= weights.sum()
    full = rolling_fir(values, weights)
    for end in [window, window + 17, 601, 999]:
        prefix = rolling_fir(values[:end], weights)
        np.testing.assert_array_equal(full[:end].view(np.uint64), prefix.view(np.uint64))


@pytest.mark.parametrize("window", [35, 127, 480])
def test_fir_future_scale_and_missingness_cannot_change_prefix(window):
    """Error-budget decisions must also depend only on already observed values."""
    values = 60648 + np.random.default_rng(71).normal(size=1600)
    weights = np.full(window, 1 / window)
    before = rolling_fir(values[:601], weights)
    values[601:900] = 1e20
    values[900:1000] = np.nan
    values[1000] = np.inf
    full = rolling_fir(values, weights)
    np.testing.assert_array_equal(before.view(np.uint64), full[:601].view(np.uint64))
