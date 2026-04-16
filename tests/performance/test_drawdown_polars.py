# -*- coding: utf-8 -*-
"""Tests for pl_drawdown."""
import numpy as np
import pandas as pd  # REMOVED: pandas dependency  # Restored for fixtures
import polars as pl
import pytest
from polars_ti.performance.drawdown import pl_drawdown


class TestPlDrawdown:
    @pytest.fixture
    def sample_df(self) -> pl.DataFrame:
        np.random.seed(42)
        return pl.DataFrame({'close': 100 + np.cumsum(np.random.randn(100) * 0.5)})

    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        close = 100 + np.random.randn(100).cumsum()
        return {
            'pd_close': pd.Series(close),
            'pl_df': pl.DataFrame({'close': close}),
        }

    def test_returns_struct(self, sample_df):
        result = sample_df.select(pl_drawdown("close"))
        assert "DRAWDOWN" in result.columns

    def test_columns_present(self, sample_df):
        result = sample_df.select(pl_drawdown("close")).unnest("DRAWDOWN")
        assert "DD" in result.columns
        assert "DD_PCT" in result.columns
        assert "DD_LOG" in result.columns

    def test_numerical_parity(self, sample_data):
        """Numerical parity with Pandas implementation."""
        pytest.skip("Pandas implementation removed in Phase 4 purge")
        pd_result = drawdown(sample_data['pd_close'])
        pl_result = sample_data['pl_df'].select(pl_drawdown('close')).unnest("DRAWDOWN")
        
        warmup = 5
        for col in ["DD", "DD_PCT", "DD_LOG"]:
            pd_vals = pd_result[col].iloc[warmup:].values
            pl_vals = pl_result[col][warmup:].to_numpy()
            valid = ~np.isnan(pd_vals) & ~np.isnan(pl_vals)
            if valid.sum() > 0:
                diff = np.abs(pd_vals[valid] - pl_vals[valid])
                assert np.max(diff) < 1e-10, f"{col} max diff: {np.max(diff)}"

    def test_with_null_values(self):
        """Handles null values gracefully."""
        df = pl.DataFrame({"close": [None] + [100.0] * 49})
        result = df.select(pl_drawdown("close"))
        assert result.height == 50

    def test_with_zeros(self):
        """Handles zero values."""
        df = pl.DataFrame({"close": [0.0] * 5 + [100.0] * 45})
        result = df.select(pl_drawdown("close"))
        assert result.height == 50

    def test_lazy_execution(self, sample_df):
        """Works with LazyFrame."""
        lazy_df = sample_df.lazy()
        result = lazy_df.select(pl_drawdown("close")).collect()
        assert "DRAWDOWN" in result.columns
