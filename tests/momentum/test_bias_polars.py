# -*- coding: utf-8 -*-
"""Tests for pl_bias - Pure Polars implementation."""
import numpy as np
import polars as pl
import pytest
from polars_ti.momentum.bias import pl_bias


class TestPlBias:
    @pytest.fixture
    def sample_df(self):
        np.random.seed(42)
        n = 100
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        return pl.DataFrame({'close': close})

    def test_returns_expr(self):
        expr = pl_bias("close")
        assert isinstance(expr, pl.Expr)

    def test_has_bias_column(self, sample_df):
        result = sample_df.select(pl_bias("close"))
        assert "BIAS_SMA_26" in result.columns

    def test_has_valid_values(self, sample_df):
        result = sample_df.select(pl_bias("close"))
        # After warmup (26), should have values
        assert result["BIAS_SMA_26"][30:].null_count() == 0

    def test_offset_parameter(self, sample_df):
        result_no_offset = sample_df.select(pl_bias("close", offset=0))
        result_with_offset = sample_df.select(pl_bias("close", offset=5))
        # First 5 values of offset result should be null
        assert result_with_offset["BIAS_SMA_26"][:5].null_count() == 5

    def test_lazy_execution(self, sample_df):
        result = sample_df.lazy().select(pl_bias("close")).collect()
        assert "BIAS_SMA_26" in result.columns

    def test_custom_length(self, sample_df):
        result = sample_df.select(pl_bias("close", length=10))
        assert "BIAS_SMA_10" in result.columns

    def test_ema_mamode(self, sample_df):
        result = sample_df.select(pl_bias("close", mamode="ema"))
        assert "BIAS_EMA_26" in result.columns

    def test_with_null_values(self):
        df = pl.DataFrame({'close': [100.0, None, 102.0, 103.0] + [100.0] * 50})
        result = df.select(pl_bias("close", length=5))
        # Should handle nulls gracefully
        assert result.height == 54
