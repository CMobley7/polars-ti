# -*- coding: utf-8 -*-
"""Tests for pl_bop - Pure Polars + TA-Lib implementation."""
import numpy as np
import polars as pl
import pytest
from polars_ti.momentum.bop import pl_bop


class TestPlBop:
    @pytest.fixture
    def sample_df(self):
        np.random.seed(42)
        n = 100
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n) * 0.5)
        low = close - np.abs(np.random.randn(n) * 0.5)
        open_ = close + np.random.randn(n) * 0.2
        return pl.DataFrame({'open': open_, 'high': high, 'low': low, 'close': close})

    def test_returns_expr(self):
        expr = pl_bop("open", "high", "low", "close")
        assert isinstance(expr, pl.Expr)

    def test_has_bop_column(self, sample_df):
        result = sample_df.select(pl_bop("open", "high", "low", "close"))
        assert "BOP" in result.columns

    def test_has_valid_values(self, sample_df):
        result = sample_df.select(pl_bop("open", "high", "low", "close"))
        # BOP should have values immediately (no warmup)
        assert result["BOP"].null_count() == 0

    def test_offset_parameter(self, sample_df):
        result_with_offset = sample_df.select(pl_bop("open", "high", "low", "close", offset=5))
        # First 5 values should be null
        assert result_with_offset["BOP"][:5].null_count() == 5

    def test_lazy_execution(self, sample_df):
        result = sample_df.lazy().select(pl_bop("open", "high", "low", "close")).collect()
        assert "BOP" in result.columns

    def test_talib_true(self, sample_df):
        result = sample_df.select(pl_bop("open", "high", "low", "close", talib=True))
        assert "BOP" in result.columns

    def test_talib_false(self, sample_df):
        result = sample_df.select(pl_bop("open", "high", "low", "close", talib=False))
        assert "BOP" in result.columns

    def test_scalar_parameter(self, sample_df):
        result = sample_df.select(pl_bop("open", "high", "low", "close", scalar=2.0, talib=False))
        # With scalar=2.0, values should be doubled
        assert "BOP" in result.columns

    def test_with_null_values(self):
        df = pl.DataFrame({
            'open': [100.0, None, 102.0, 103.0],
            'high': [101.0, 102.0, None, 104.0],
            'low': [99.0, 100.0, 101.0, None],
            'close': [100.5, 101.0, 102.5, 103.5]
        })
        result = df.select(pl_bop("open", "high", "low", "close", talib=False))
        assert result.height == 4
