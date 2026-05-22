# -*- coding: utf-8 -*-
"""Tests for pl_cci - Polars + Numba implementation with TA-Lib support."""

import numpy as np
import polars as pl
import pytest
from polars_ti.momentum.cci import pl_cci


class TestPlCci:
    @pytest.fixture
    def sample_df(self):
        np.random.seed(42)
        n = 100
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n) * 0.5)
        low = close - np.abs(np.random.randn(n) * 0.5)
        return pl.DataFrame({"high": high, "low": low, "close": close})

    def test_returns_expr(self):
        expr = pl_cci("high", "low", "close")
        assert isinstance(expr, pl.Expr)

    def test_has_cci_column(self, sample_df):
        result = sample_df.select(pl_cci("high", "low", "close"))
        assert "CCI_14_0.015" in result.columns

    def test_has_valid_values(self, sample_df):
        result = sample_df.select(pl_cci("high", "low", "close"))
        assert result["CCI_14_0.015"][20:].null_count() == 0

    def test_offset_parameter(self, sample_df):
        result = sample_df.select(pl_cci("high", "low", "close", offset=5))
        assert result["CCI_14_0.015"][:5].null_count() == 5

    def test_lazy_execution(self, sample_df):
        result = sample_df.lazy().select(pl_cci("high", "low", "close")).collect()
        assert "CCI_14_0.015" in result.columns

    def test_talib_true(self, sample_df):
        result = sample_df.select(pl_cci("high", "low", "close", talib=True))
        assert "CCI_14_0.015" in result.columns

    def test_talib_false(self, sample_df):
        result = sample_df.select(pl_cci("high", "low", "close", talib=False))
        assert "CCI_14_0.015" in result.columns

    def test_custom_length(self, sample_df):
        result = sample_df.select(pl_cci("high", "low", "close", length=20))
        assert "CCI_20_0.015" in result.columns

    def test_custom_c_constant(self, sample_df):
        result = sample_df.select(pl_cci("high", "low", "close", c=0.02))
        assert "CCI_14_0.02" in result.columns

    def test_with_null_values(self):
        df = pl.DataFrame(
            {
                "high": [101.0, None, 103.0] + [102.0] * 50,
                "low": [99.0, 100.0, None] + [98.0] * 50,
                "close": [100.0, 101.0, 102.0] + [100.0] * 50,
            }
        )
        result = df.select(pl_cci("high", "low", "close", talib=False))
        assert result.height == 53
