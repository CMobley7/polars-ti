# -*- coding: utf-8 -*-
"""Retain the historical SMA regression and check SciPy availability.

The original SMA used Numba convolution, which could abort when SciPy's LAPACK
was missing. SMA now uses rolling sums; its regression still checks compilation,
warmup and values. SciPy remains a hard dependency because the FIR kernel and
shared regression utility import scipy.signal.oaconvolve and scipy.stats.linregress.
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
    assert np.all(np.isfinite(tail)), "nb_sma produced non-finite values after warmup"

    # SMA of consecutive integers over a window of n equals the window midpoint.
    expected = np.arange(50.0)[n - 1 :] - (n - 1) / 2.0
    assert np.allclose(tail, expected)


def test_scipy_importable():
    # The dependency must actually be present in the environment.
    import scipy  # noqa: F401
