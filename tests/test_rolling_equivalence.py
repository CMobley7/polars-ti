"""Permanent baseline equivalence across randomized windows and NaN warmups."""

import numpy as np
import pytest

from scripts.rolling_kernel_audit.cases import CASES


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
@pytest.mark.parametrize("window", [5, 17, 40])
@pytest.mark.parametrize("input_class", ["random", "prefix", "interior", "constant"])
def test_preserved_reference(case, window, input_class):
    if case.changed_nan and input_class in {"prefix", "interior"}:
        pytest.skip("Intentional NaN correction tested against explicit semantics")
    values = 100 + np.random.default_rng(42).normal(size=257)
    if input_class == "prefix":
        values[:11] = np.nan
    elif input_class == "interior":
        values[133] = np.nan
    elif input_class == "constant":
        values[:] = 100
    args = case.arguments(values, window)
    expected = case.kernel(baseline=True)(*args)
    actual = case.kernel()(*args)
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=2e-11, equal_nan=True)
