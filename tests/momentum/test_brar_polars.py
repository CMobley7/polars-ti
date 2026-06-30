# -*- coding: utf-8 -*-
"""Tests for pl_brar - Pure Polars implementation."""

import numpy as np
import polars as pl
import pytest
from polars_ti.momentum.brar import brar


class TestPlBrar:
    @pytest.fixture
    def sample_df(self):
        np.random.seed(42)
        n = 100
        open_ = 100 + np.cumsum(np.random.randn(n) * 0.35)
        high = open_ + np.abs(np.random.randn(n) * 0.3) + 0.1
        low = open_ - np.abs(np.random.randn(n) * 0.3) - 0.1
        close = (high + low) / 2
        return pl.DataFrame({"open": open_, "high": high, "low": low, "close": close})

    def test_returns_list_of_expr(self):
        exprs = brar("open", "high", "low", "close")
        assert isinstance(exprs, list)
        assert len(exprs) == 2
        for expr in exprs:
            assert isinstance(expr, pl.Expr)

    def test_has_ar_br_columns(self, sample_df):
        result = sample_df.select(brar("open", "high", "low", "close"))
        assert "AR_26" in result.columns
        assert "BR_26" in result.columns

    def test_has_valid_values(self, sample_df):
        result = sample_df.select(brar("open", "high", "low", "close"))
        # After warmup, should have values
        assert result["AR_26"][50:].null_count() == 0
        assert result["BR_26"][50:].null_count() == 0

    def test_offset_parameter(self, sample_df):
        result_no_offset = sample_df.select(brar("open", "high", "low", "close", offset=0))
        result_with_offset = sample_df.select(brar("open", "high", "low", "close", offset=5))
        # Values should be shifted
        assert result_with_offset["AR_26"][:5].null_count() == 5

    def test_lazy_execution(self, sample_df):
        result = sample_df.lazy().select(brar("open", "high", "low", "close")).collect()
        assert "AR_26" in result.columns
        assert "BR_26" in result.columns

    def test_custom_parameters(self, sample_df):
        result = sample_df.select(brar("open", "high", "low", "close", length=10, scalar=50.0, drift=2))
        assert "AR_10" in result.columns
        assert "BR_10" in result.columns
