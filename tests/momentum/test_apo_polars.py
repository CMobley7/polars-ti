# -*- coding: utf-8 -*-
"""Tests for pl_apo - Pure Polars + TA-Lib implementation."""

import numpy as np
import polars as pl
import pytest
from polars_ti.momentum.apo import pl_apo


class TestPlApo:
    @pytest.fixture
    def sample_df(self):
        np.random.seed(42)
        n = 100
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        return pl.DataFrame({"close": close})

    def test_returns_expr(self):
        expr = pl_apo("close")
        assert isinstance(expr, pl.Expr)

    def test_has_apo_column(self, sample_df):
        result = sample_df.select(pl_apo("close"))
        assert "APO_12_26" in result.columns

    def test_has_valid_values(self, sample_df):
        result = sample_df.select(pl_apo("close"))
        # After warmup (26), should have values
        assert result["APO_12_26"][30:].null_count() == 0

    def test_custom_periods(self, sample_df):
        result = sample_df.select(pl_apo("close", fast=5, slow=10))
        assert "APO_5_10" in result.columns

    def test_offset_parameter(self, sample_df):
        result_with_offset = sample_df.select(pl_apo("close", offset=5))
        # First 5 values should be null
        assert result_with_offset["APO_12_26"][:5].null_count() == 5

    def test_lazy_execution(self, sample_df):
        result = sample_df.lazy().select(pl_apo("close")).collect()
        assert "APO_12_26" in result.columns

    def test_talib_true(self, sample_df):
        result = sample_df.select(pl_apo("close", talib=True))
        assert "APO_12_26" in result.columns

    def test_talib_false(self, sample_df):
        result = sample_df.select(pl_apo("close", talib=False))
        assert "APO_12_26" in result.columns

    def test_ema_mamode(self, sample_df):
        result = sample_df.select(pl_apo("close", mamode="ema", talib=False))
        assert "APO_12_26" in result.columns

    def test_sma_mamode(self, sample_df):
        result = sample_df.select(pl_apo("close", mamode="sma", talib=False))
        assert "APO_12_26" in result.columns

    def test_swap_fast_slow_if_needed(self, sample_df):
        # If slow < fast, they should be swapped internally
        result = sample_df.select(pl_apo("close", fast=26, slow=12))
        assert "APO_12_26" in result.columns

    def test_with_null_values(self):
        df = pl.DataFrame({"close": [100.0, None, 102.0, 103.0] + [100.0] * 50})
        result = df.select(pl_apo("close", talib=False))
        assert result.height == 54
