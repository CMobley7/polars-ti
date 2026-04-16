# -*- coding: utf-8 -*-
"""Tests for newly implemented momentum Polars indicators."""
import numpy as np
import pandas as pd  # REMOVED: pandas dependency  # Restored for fixtures
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
    return pl.DataFrame({
        "open": open_, "high": high, "low": low,
        "close": close, "volume": volume,
    })


class TestPlKst:
    def test_returns_expression(self, ohlcv_df):
        from polars_ti.momentum.kst import pl_kst
        result = ohlcv_df.select(pl_kst("close"))
        assert result.height == 200


class TestPlLrsi:
    def test_returns_expression(self, ohlcv_df):
        from polars_ti.momentum.lrsi import pl_lrsi
        result = ohlcv_df.select(pl_lrsi("close"))
        assert result.height == 200

    def test_parity(self, ohlcv_df):
        pytest.skip("Pandas implementation removed in Phase 4 purge")
        from polars_ti.momentum.lrsi import pl_lrsi
        pd_close = pd.Series(ohlcv_df["close"].to_numpy())
        pd_result = lrsi(pd_close, length=14, gamma=0.5)
        pl_result = ohlcv_df.select(pl_lrsi("close", length=14, gamma=0.5))
        pl_arr = pl_result[pl_result.columns[0]].to_numpy()
        pd_arr = pd_result.to_numpy()
        mask = ~np.isnan(pd_arr) & ~np.isnan(pl_arr)
        if mask.sum() > 0:
            assert np.allclose(pl_arr[mask], pd_arr[mask], atol=1e-6)


class TestPlPgo:
    def test_returns_expression(self, ohlcv_df):
        from polars_ti.momentum.pgo import pl_pgo
        result = ohlcv_df.select(pl_pgo("high", "low", "close"))
        assert result.height == 200
