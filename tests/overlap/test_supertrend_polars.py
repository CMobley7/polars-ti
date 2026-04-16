# -*- coding: utf-8 -*-
"""Tests for pl_supertrend - Numba @njit implementation."""
import numpy as np
import pandas as pd  # REMOVED: pandas dependency  # Restored for fixtures
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
            'pd_df': pd.DataFrame({'high': high, 'low': low, 'close': close}),
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

    def test_numerical_parity(self, sample_data):
        """Numerical parity with Pandas implementation."""
        pytest.skip("Pandas implementation removed in Phase 4 purge")
        pd_df = sample_data['pd_df']
        pd_result = supertrend(pd_df['high'], pd_df['low'], pd_df['close'], length=7, multiplier=3.0)
        pl_result = sample_data['pl_df'].select(pl_supertrend('high', 'low', 'close')).unnest("SUPERT_7_3.0")
        
        warmup = 20
        for col in ["SUPERT_7_3.0", "SUPERTd_7_3.0"]:
            pd_vals = pd_result[col].iloc[warmup:].values
            pl_vals = pl_result[col][warmup:].to_numpy()
            valid = ~np.isnan(pd_vals) & ~np.isnan(pl_vals)
            if valid.sum() > 0:
                diff = np.abs(pd_vals[valid] - pl_vals[valid])
                assert np.max(diff) < 1e-10, f"{col} max diff: {np.max(diff)}"

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

