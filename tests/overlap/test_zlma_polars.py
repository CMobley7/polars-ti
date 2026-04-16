# -*- coding: utf-8 -*-
"""Tests for pl_zlma."""
import numpy as np
import polars as pl
import pytest
from polars_ti.overlap.zlma import pl_zlma


class TestPlZlma:
    @pytest.fixture
    def sample_df(self) -> pl.DataFrame:
        np.random.seed(42)
        return pl.DataFrame({
            'close': 100 + np.cumsum(np.random.randn(100) * 0.5),
        })

    def test_returns_correct_column(self, sample_df):
        result = sample_df.select(pl_zlma("close", length=10, mamode="ema"))
        assert "ZL_EMA_10" in result.columns

    def test_different_lengths(self, sample_df):
        r10 = sample_df.select(pl_zlma("close", length=10))
        r20 = sample_df.select(pl_zlma("close", length=20))
        assert "ZL_EMA_10" in r10.columns
        assert "ZL_EMA_20" in r20.columns

    def test_different_mamodes(self, sample_df):
        r_ema = sample_df.select(pl_zlma("close", mamode="ema"))
        r_sma = sample_df.select(pl_zlma("close", mamode="sma"))
        assert "ZL_EMA_10" in r_ema.columns
        assert "ZL_SMA_10" in r_sma.columns

    def test_offset(self, sample_df):
        result = sample_df.select(pl_zlma("close", offset=5))
        arr = result["ZL_EMA_10"].to_numpy()
        assert np.isnan(arr[:5]).all()

    def test_has_valid_values(self, sample_df):
        result = sample_df.select(pl_zlma("close", length=10))
        arr = result["ZL_EMA_10"].to_numpy()
        mask = ~np.isnan(arr)
        assert mask.sum() > 50

    def test_with_null_values(self):
        """Handles null values gracefully."""
        df = pl.DataFrame({"close": [None] + [100.0] * 49})
        result = df.select(pl_zlma("close", length=10))
        assert result.height == 50

    def test_with_zeros(self):
        """Handles zero values."""
        df = pl.DataFrame({"close": [0.0] * 5 + [100.0] * 45})
        result = df.select(pl_zlma("close", length=10))
        assert result.height == 50

    def test_lazy_execution(self, sample_df):
        """Works with LazyFrame."""
        lazy_df = sample_df.lazy()
        result = lazy_df.select(pl_zlma("close", length=10)).collect()
        assert "ZL_EMA_10" in result.columns

