# -*- coding: utf-8 -*-
"""Tests for pl_supertrend - Numba @njit implementation."""
import numpy as np
import polars as pl
import pytest
from polars_ti.overlap.supertrend import pl_supertrend


class TestPlSupertrend:
    @pytest.fixture
    def sample_df(self) -> pl.DataFrame:
        np.random.seed(42)
        n = 100
        high = 100 + np.cumsum(np.random.randn(n) * 0.5) + np.abs(np.random.randn(n) * 0.3)
        low = high - np.abs(np.random.randn(n) * 0.5)
        close = (high + low) / 2 + np.random.randn(n) * 0.1
        return pl.DataFrame({'high': high, 'low': low, 'close': close})

    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        n = 100
        close = 100 + np.random.randn(n).cumsum()
        high = close + np.abs(np.random.randn(n))
        low = close - np.abs(np.random.randn(n))
        return {
            'pd_df': pl.DataFrame({'high': high, 'low': low, 'close': close}),
            'pl_df': pl.DataFrame({'high': high, 'low': low, 'close': close}),
        }

    def test_returns_expr(self):
        expr = pl_supertrend("high", "low", "close")
        assert isinstance(expr, pl.Expr)

    def test_returns_struct(self, sample_df):
        result = sample_df.select(pl_supertrend("high", "low", "close"))
        assert len(result.columns) == 1  # Returns a struct column

    def test_columns_present(self, sample_df):
        result = sample_df.select(pl_supertrend("high", "low", "close"))
        result = result.unnest(result.columns[0])
        assert "SUPERT_7_3.0" in result.columns
        assert "SUPERTd_7_3.0" in result.columns
        assert "SUPERTl_7_3.0" in result.columns
        assert "SUPERTs_7_3.0" in result.columns

    def test_custom_parameters(self, sample_df):
        result = sample_df.select(pl_supertrend("high", "low", "close", length=10, multiplier=2.0))
        result = result.unnest(result.columns[0])
        assert "SUPERT_10_2.0" in result.columns

    def test_has_valid_values(self, sample_df):
        result = sample_df.select(pl_supertrend("high", "low", "close"))
        result = result.unnest(result.columns[0])
        arr = result["SUPERT_7_3.0"].to_numpy()
        mask = ~np.isnan(arr)
        assert mask.sum() > 50

    def test_with_null_values(self):
        """Handles null values gracefully."""
        df = pl.DataFrame({
            "high": [None] + [102.0] * 49,
            "low": [None] + [98.0] * 49,
            "close": [None] + [100.0] * 49,
        })
        result = df.select(pl_supertrend("high", "low", "close"))
        assert result.height == 50

    def test_with_zeros(self):
        """Handles zero values."""
        df = pl.DataFrame({
            "high": [0.0] * 5 + [102.0] * 45,
            "low": [0.0] * 5 + [98.0] * 45,
            "close": [0.0] * 5 + [100.0] * 45,
        })
        result = df.select(pl_supertrend("high", "low", "close"))
        assert result.height == 50

    def test_lazy_execution(self, sample_df):
        """Works with LazyFrame."""
        lazy_df = sample_df.lazy()
        result = lazy_df.select(pl_supertrend("high", "low", "close")).collect()
        assert result.height == 100

