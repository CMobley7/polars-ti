# -*- coding: utf-8 -*-
"""Tests for pl_kama."""
import numpy as np
import pandas as pd  # REMOVED: pandas dependency  # Restored for fixtures
import polars as pl
import pytest
from polars_ti.overlap.kama import pl_kama


class TestPlKama:
    @pytest.fixture
    def sample_df(self) -> pl.DataFrame:
        np.random.seed(42)
        return pl.DataFrame({
            'close': 100 + np.cumsum(np.random.randn(100) * 0.5),
        })

    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        close = 100 + np.random.randn(100).cumsum()
        return {
            'pd_close': pd.Series(close),
            'pl_df': pl.DataFrame({'close': close}),
        }

    def test_returns_correct_column(self, sample_df):
        result = sample_df.select(pl_kama("close", length=10))
        assert "KAMA_10_2_30" in result.columns

    def test_pure_version(self, sample_df):
        result = sample_df.select(pl_kama("close", length=10, talib=False))
        assert "KAMA_10_2_30" in result.columns
        arr = result["KAMA_10_2_30"].to_numpy()
        assert np.isnan(arr[:9]).all()
        assert not np.isnan(arr[9])

    def test_talib_version(self, sample_df):
        result = sample_df.select(pl_kama("close", length=10, talib=True))
        assert "KAMA_10_2_30" in result.columns
        arr = result["KAMA_10_2_30"].to_numpy()
        mask = ~np.isnan(arr)
        assert mask.sum() > 50

    def test_custom_periods(self, sample_df):
        result = sample_df.select(pl_kama("close", length=5, fast=3, slow=20))
        assert "KAMA_5_3_20" in result.columns

    def test_offset(self, sample_df):
        result = sample_df.select(pl_kama("close", offset=5, talib=False))
        arr = result["KAMA_10_2_30"].to_numpy()
        assert np.isnan(arr[:14]).all()

    def test_numerical_parity(self, sample_data):
        """Numerical parity with Pandas implementation."""
        pytest.skip("Pandas implementation removed in Phase 4 purge")
        pd_result = kama(sample_data['pd_close'], length=10, fast=2, slow=30)
        pl_result = sample_data['pl_df'].select(pl_kama('close', talib=False)).to_series()
        
        warmup = 15
        pd_vals = pd_result.iloc[warmup:].values
        pl_vals = pl_result[warmup:].to_numpy()
        
        valid = ~np.isnan(pd_vals) & ~np.isnan(pl_vals)
        if valid.sum() > 0:
            diff = np.abs(pd_vals[valid] - pl_vals[valid])
            assert np.max(diff) < 1e-10, f"Max diff: {np.max(diff)}"

    def test_with_null_values(self):
        """Handles null values gracefully."""
        df = pl.DataFrame({"close": [None] + [100.0] * 39})
        result = df.select(pl_kama("close", talib=False))
        assert result.height == 40

    def test_with_zeros(self):
        """Handles zero values."""
        df = pl.DataFrame({"close": [0.0] * 5 + [100.0] * 35})
        result = df.select(pl_kama("close", talib=False))
        assert result.height == 40

    def test_lazy_execution(self, sample_df):
        """Works with LazyFrame."""
        lazy_df = sample_df.lazy()
        result = lazy_df.select(pl_kama("close", talib=False)).collect()
        assert "KAMA_10_2_30" in result.columns

