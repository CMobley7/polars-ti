# -*- coding: utf-8 -*-
"""Tests for pl_kdj - Polars implementation."""
import numpy as np
import polars as pl
import pandas as pd
import pytest
from polars_ti.momentum.kdj import pl_kdj


class TestPlKdj:
    @pytest.fixture
    def sample_df(self):
        np.random.seed(42)
        n = 200
        high = 100 + np.cumsum(np.random.randn(n) * 0.5) + np.random.rand(n)
        low = 100 + np.cumsum(np.random.randn(n) * 0.5) - np.random.rand(n)
        close = (high + low) / 2
        return pl.DataFrame({'high': high, 'low': low, 'close': close})

    def test_returns_list(self):
        exprs = pl_kdj("high", "low", "close")
        assert isinstance(exprs, list)
        assert len(exprs) == 3

    def test_has_kdj_columns(self, sample_df):
        result = sample_df.select(pl_kdj("high", "low", "close"))
        assert "K_9_3" in result.columns
        assert "D_9_3" in result.columns
        assert "J_9_3" in result.columns

    def test_has_valid_values(self, sample_df):
        result = sample_df.select(pl_kdj("high", "low", "close"))
        assert result["K_9_3"][20:].null_count() == 0

    def test_offset_parameter(self, sample_df):
        result_no_offset = sample_df.select(pl_kdj("high", "low", "close", offset=0))
        result_with_offset = sample_df.select(pl_kdj("high", "low", "close", offset=5))
        # Offset should add more nulls
        assert result_with_offset["K_9_3"].null_count() > result_no_offset["K_9_3"].null_count()

    def test_lazy_execution(self, sample_df):
        result = sample_df.lazy().select(pl_kdj("high", "low", "close")).collect()
        assert "K_9_3" in result.columns

    def test_custom_lengths(self, sample_df):
        result = sample_df.select(pl_kdj("high", "low", "close", length=14, signal=5))
        assert "K_14_5" in result.columns
        assert "D_14_5" in result.columns
        assert "J_14_5" in result.columns

    def test_numerical_parity(self, sample_df):
        """Verify numerical parity with Pandas implementation."""
        pytest.skip("Pandas implementation removed in Phase 4 purge")
        import pandas as pd
        # from polars_ti.momentum.kdj import kdj as pandas_kdj  # REMOVED: pandas func removed
        
        h = sample_df["high"].to_numpy()
        l = sample_df["low"].to_numpy()
        c = sample_df["close"].to_numpy()
        
        pandas_result = pandas_kdj(pd.Series(h), pd.Series(l), pd.Series(c), length=9, signal=3)
        polars_result = sample_df.select(pl_kdj("high", "low", "close", length=9, signal=3))
        
        for col in ['K_9_3', 'D_9_3', 'J_9_3']:
            pandas_arr = pandas_result[col].to_numpy()[20:]
            polars_arr = polars_result[col].to_numpy()[20:]
            
            valid_mask = ~np.isnan(pandas_arr) & ~np.isnan(polars_arr)
            max_diff = np.max(np.abs(pandas_arr[valid_mask] - polars_arr[valid_mask]))
            assert max_diff < 1e-6, f"{col} max diff {max_diff} exceeds tolerance"
