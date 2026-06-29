# -*- coding: utf-8 -*-
"""Guard the SciPy hard-dependency (WS1 #2).

``nb_sma`` (and other Numba kernels) call ``np.convolve`` / BLAS. When SciPy's
LAPACK is unavailable the Numba runtime aborts the interpreter with
"Specified LAPACK function could not be found" instead of raising a catchable
error. SciPy is therefore a *hard* runtime dependency (declared in
``pyproject.toml``). This test fails loudly if that dependency silently
regresses: it asserts ``nb_sma`` compiles and returns finite values past warmup.
"""

import numpy as np

from polars_ti.overlap.sma import nb_sma


def test_nb_sma_returns_finite_after_warmup():
    n = 5
    x = np.arange(50.0)
    out = nb_sma(x, n)

    assert out.shape == x.shape
    # First n-1 values are the warmup prepend (NaN); the rest must be finite.
    assert np.all(np.isnan(out[: n - 1]))
    tail = out[n - 1 :]
    assert np.all(np.isfinite(tail)), "nb_sma produced non-finite values (scipy/LAPACK missing?)"

    # SMA of consecutive integers over a window of n equals the window midpoint.
    expected = np.arange(50.0)[n - 1 :] - (n - 1) / 2.0
    assert np.allclose(tail, expected)


def test_scipy_importable():
    # The dependency must actually be present in the environment.
    import scipy  # noqa: F401
