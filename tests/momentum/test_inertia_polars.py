# -*- coding: utf-8 -*-
"""Tests for pl_inertia - Polars implementation."""
import numpy as np
import polars as pl
import pandas as pd
import pytest
from polars_ti.momentum.inertia import pl_inertia


class TestPlInertia:
    @pytest.fixture
    def sample_df(self):
        np.random.seed(42)
        n = 200
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        return pl.DataFrame({'close': close})

    def test_returns_expr(self):
        expr = pl_inertia("close")
        assert isinstance(expr, pl.Expr)

    def test_has_inertia_column(self, sample_df):
        result = sample_df.select(pl_inertia("close"))
        assert "INERTIA_20_14" in result.columns

    def test_has_valid_values(self, sample_df):
        result = sample_df.select(pl_inertia("close"))
        assert result["INERTIA_20_14"][55:].null_count() == 0

    def test_offset_parameter(self, sample_df):
        result_no_offset = sample_df.select(pl_inertia("close", offset=0))
        result_with_offset = sample_df.select(pl_inertia("close", offset=5))
        # Offset should add more nulls
        assert result_with_offset["INERTIA_20_14"].null_count() > result_no_offset["INERTIA_20_14"].null_count()

    def test_lazy_execution(self, sample_df):
        result = sample_df.lazy().select(pl_inertia("close")).collect()
        assert "INERTIA_20_14" in result.columns

    def test_custom_lengths(self, sample_df):
        result = sample_df.select(pl_inertia("close", length=10, rvi_length=7))
        assert "INERTIA_10_7" in result.columns

    def test_numerical_parity(self, sample_df):
        """Verify numerical parity with Pandas implementation."""
        pytest.skip("Pandas implementation removed in Phase 4 purge")
        import pandas as pd
        # from polars_ti.momentum.inertia import inertia as pandas_inertia  # REMOVED: pandas func removed
        
        close = sample_df["close"].to_numpy()
        pdf = pd.DataFrame({'close': close})
        
        pandas_result = pandas_inertia(pdf['close'], length=20, rvi_length=14)
        polars_result = sample_df.select(pl_inertia("close", length=20, rvi_length=14))
        
        pandas_arr = pandas_result.to_numpy()[55:]
        polars_arr = polars_result["INERTIA_20_14"].to_numpy()[55:]
        
        valid_mask = ~np.isnan(pandas_arr) & ~np.isnan(polars_arr)
        max_diff = np.max(np.abs(pandas_arr[valid_mask] - polars_arr[valid_mask]))
        assert max_diff < 1e-6, f"Max diff {max_diff} exceeds tolerance"
