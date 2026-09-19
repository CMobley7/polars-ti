"""Public-expression parity for the optimized callbacks."""

import numpy as np
import polars as pl
import pytest

from scripts.rolling_kernel_audit.expressions import NAMES, evaluate


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("window", [5, 40, 128])
@pytest.mark.parametrize("kind", ["finite", "nan_prefix", "interior_null"])
def test_expression_parity(name, window, kind):
    values = 100 + np.random.default_rng(88).normal(size=300)
    series = pl.Series(values)
    if kind == "nan_prefix":
        values[:17] = np.nan
        series = pl.Series(values)
    elif kind == "interior_null":
        series = series.set(pl.Series(np.arange(len(values)) == 140), None)
    frame = pl.DataFrame({"close": series, "high": series + 2, "low": series - 2})
    if kind == "nan_prefix" and name in {"kama", "fisher"}:
        # Upstream warmup fix: retain the frozen finite-input oracle, but seed
        # after the prefix rather than pinning the historical poisoned state.
        suffix = evaluate(name, frame.slice(17), window, True)
        expected = np.concatenate((np.full(17 * (2 if name == "fisher" else 1), np.nan), suffix))
    else:
        expected = evaluate(name, frame, window, True)
    actual = evaluate(name, frame, window, False)
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize("name", ["increasing", "decreasing"])
@pytest.mark.parametrize("drift", [-3, 0, 1, 2, 5])
@pytest.mark.parametrize("window", [1, 2, 3, 5, 17])
def test_strict_drift_preserved(name, drift, window):
    values = np.random.default_rng(99).normal(size=100)
    values[::17] = np.nan
    frame = pl.DataFrame({"close": values})
    np.testing.assert_array_equal(
        evaluate(name, frame, window, False, drift), evaluate(name, frame, window, True, drift)
    )
