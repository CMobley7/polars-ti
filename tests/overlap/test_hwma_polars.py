# -*- coding: utf-8 -*-
"""Tests for pl_hwma."""
import numpy as np
import polars as pl
import pytest
from polars_ti.overlap.hwma import pl_hwma


class TestPlHwma:
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
            'pd_close': close,
            'pl_df': pl.DataFrame({'close': close}),
        }

    def test_returns_correct_column(self, sample_df):
        result = sample_df.select(pl_hwma("close"))
        assert "HWMA_0.2_0.1_0.1" in result.columns

    def test_custom_parameters(self, sample_df):
        result = sample_df.select(pl_hwma("close", na=0.3, nb=0.2, nc=0.15))
        assert "HWMA_0.3_0.2_0.15" in result.columns

    def test_different_params_give_different_results(self, sample_df):
        r1 = sample_df.select(pl_hwma("close", na=0.2)).get_column("HWMA_0.2_0.1_0.1").to_numpy()
        r2 = sample_df.select(pl_hwma("close", na=0.5)).get_column("HWMA_0.5_0.1_0.1").to_numpy()
        assert not np.allclose(r1, r2)

    def test_offset(self, sample_df):
        result = sample_df.select(pl_hwma("close", offset=5))
        arr = result.get_column("HWMA_0.2_0.1_0.1").to_numpy()
        assert np.isnan(arr[:5]).all()

    def test_no_nan_without_offset(self, sample_df):
        result = sample_df.select(pl_hwma("close"))
        arr = result.get_column("HWMA_0.2_0.1_0.1").to_numpy()
        assert not np.any(np.isnan(arr))

    def test_with_null_values(self):
        """Handles null values gracefully."""
        df = pl.DataFrame({"close": [None] + [100.0] * 29})
        result = df.select(pl_hwma("close"))
        assert result.height == 30

    def test_with_zeros(self):
        """Handles zero values."""
        df = pl.DataFrame({"close": [0.0] * 5 + [100.0] * 25})
        result = df.select(pl_hwma("close"))
        assert result.height == 30

    def test_lazy_execution(self, sample_df):
        """Works with LazyFrame."""
        lazy_df = sample_df.lazy()
        result = lazy_df.select(pl_hwma("close")).collect()
        assert "HWMA_0.2_0.1_0.1" in result.columns

