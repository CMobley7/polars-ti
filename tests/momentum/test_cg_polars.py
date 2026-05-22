# -*- coding: utf-8 -*-
"""Tests for pl_cg - Polars + Numba implementation."""

import numpy as np
import polars as pl
import pytest
from polars_ti.momentum.cg import pl_cg


class TestPlCg:
    @pytest.fixture
    def sample_df(self):
        np.random.seed(42)
        n = 100
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        return pl.DataFrame({"close": close})

    def test_returns_expr(self):
        expr = pl_cg("close")
        assert isinstance(expr, pl.Expr)

    def test_has_cg_column(self, sample_df):
        result = sample_df.select(pl_cg("close"))
        assert "CG_10" in result.columns

    def test_has_valid_values(self, sample_df):
        result = sample_df.select(pl_cg("close"))
        assert result["CG_10"][15:].null_count() == 0

    def test_offset_parameter(self, sample_df):
        result_no_offset = sample_df.select(pl_cg("close", offset=0))
        result_with_offset = sample_df.select(pl_cg("close", offset=5))
        # With offset, values are shifted
        assert result_with_offset["CG_10"].is_nan().sum() > result_no_offset["CG_10"].is_nan().sum()

    def test_lazy_execution(self, sample_df):
        result = sample_df.lazy().select(pl_cg("close")).collect()
        assert "CG_10" in result.columns

    def test_custom_length(self, sample_df):
        result = sample_df.select(pl_cg("close", length=15))
        assert "CG_15" in result.columns

    def test_with_null_values(self):
        df = pl.DataFrame({"close": [100.0, None, 102.0] + [100.0] * 50})
        result = df.select(pl_cg("close"))
        assert result.height == 53

    def test_values_are_negative(self, sample_df):
        """CG values should be negative (formula has negative sign)."""
        result = sample_df.select(pl_cg("close"))
        # Most CG values should be negative
        valid_values = result["CG_10"].filter(~result["CG_10"].is_nan())
        assert valid_values.mean() < 0
