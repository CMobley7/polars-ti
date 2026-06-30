# -*- coding: utf-8 -*-
"""Tests for newly implemented momentum Polars indicators."""

import numpy as np
import polars as pl
import pytest


@pytest.fixture
def ohlcv_df() -> pl.DataFrame:
    """Generate reproducible OHLCV test data."""
    np.random.seed(42)
    n = 200
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n) * 0.3)
    low = close - np.abs(np.random.randn(n) * 0.3)
    open_ = close + np.random.randn(n) * 0.1
    volume = np.random.randint(1000, 10000, n).astype(float)
    return pl.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


class TestPlKst:
    def test_returns_expression(self, ohlcv_df):
        from polars_ti.momentum.kst import kst

        result = ohlcv_df.select(kst("close"))
        assert result.height == 200


class TestPlLrsi:
    def test_returns_expression(self, ohlcv_df):
        from polars_ti.momentum.lrsi import lrsi

        result = ohlcv_df.select(lrsi("close"))
        assert result.height == 200


class TestPlPgo:
    def test_returns_expression(self, ohlcv_df):
        from polars_ti.momentum.pgo import pgo

        result = ohlcv_df.select(pgo("high", "low", "close"))
        assert result.height == 200
