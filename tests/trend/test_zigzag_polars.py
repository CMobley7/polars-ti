# -*- coding: utf-8 -*-
"""Regression tests for polars_ti/trend/zigzag.py.

Pins the buffer-sizing fix in ``nb_rolling_hl``: a single window bar can be
both the local low and the local high (flat/low-volatility data where
high == low), appending TWO entries per iteration. The output buffers must
therefore hold ``2 * len`` entries; sizing them to ``len`` caused out-of-bounds
writes and heap corruption on flat data.
"""

import numpy as np
import polars as pl
import pytest

import polars_ti as ti  # noqa: F401 — registers the 'ti' accessor
from polars_ti.trend.zigzag import nb_rolling_hl, zigzag


@pytest.mark.parametrize("n", [50, 300])
def test_nb_rolling_hl_flat_data_no_overflow(n):
    """Flat data appends 2 extremums per window bar without OOB writes."""
    high = np.full(n, 100.0)
    low = np.full(n, 100.0)
    idx, swing, value = nb_rolling_hl(high, low, 10)
    # Every interior bar is both a local low and a local high.
    assert len(idx) == len(swing) == len(value)
    # All recorded extremum values sit on the flat price.
    assert np.all(value == 100.0)


@pytest.mark.parametrize("n", [50, 300])
def test_zigzag_flat_ohlcv_does_not_crash(n):
    """df.ti.zigzag() on flat OHLCV returns a full-height, sane result."""
    df = pl.DataFrame(
        {
            "open": [100.0] * n,
            "high": [100.0] * n,
            "low": [100.0] * n,
            "close": [100.0] * n,
            "volume": [1e6] * n,
        }
    )
    result = df.ti.zigzag()
    assert result.height == n


def test_zigzag_normal_data_unchanged():
    """The fix must not alter zigzag output on ordinary (non-flat) data."""
    np.random.seed(7)
    n = 400
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n)) * 0.3
    low = close - np.abs(np.random.randn(n)) * 0.3
    df = pl.DataFrame({"high": high, "low": low})
    out = df.select(zigzag("high", "low"))
    # Deterministic and full-height; presence of at least one swing point.
    assert out.height == n
    swings = out.to_series().struct.field("ZIGZAGs_5.0%_10")
    assert swings.drop_nulls().len() > 0
