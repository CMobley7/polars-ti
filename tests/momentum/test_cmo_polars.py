# -*- coding: utf-8 -*-
"""Tests for pl_cmo - Pure Polars + TA-Lib implementation."""

import numpy as np
import polars as pl
import pytest
from polars_ti.momentum.cmo import pl_cmo


class TestPlCmo:
    @pytest.fixture
    def sample_df(self):
        np.random.seed(42)
        n = 100
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        return pl.DataFrame({"close": close})

    def test_returns_expr(self):
        expr = pl_cmo("close")
        assert isinstance(expr, pl.Expr)

    def test_has_cmo_column(self, sample_df):
        result = sample_df.select(pl_cmo("close"))
        assert "CMO_14" in result.columns

    def test_has_valid_values(self, sample_df):
        result = sample_df.select(pl_cmo("close"))
        assert result["CMO_14"][20:].null_count() == 0

    def test_offset_parameter(self, sample_df):
        result_no_offset = sample_df.select(pl_cmo("close", offset=0, talib=False))
        result_with_offset = sample_df.select(pl_cmo("close", offset=5, talib=False))
        # With offset 5, first 5 values are shifted so more nulls
        assert result_with_offset["CMO_14"].null_count() > result_no_offset["CMO_14"].null_count()

    def test_lazy_execution(self, sample_df):
        result = sample_df.lazy().select(pl_cmo("close")).collect()
        assert "CMO_14" in result.columns

    def test_talib_true(self, sample_df):
        result = sample_df.select(pl_cmo("close", talib=True))
        assert "CMO_14" in result.columns

    def test_talib_false(self, sample_df):
        result = sample_df.select(pl_cmo("close", talib=False))
        assert "CMO_14" in result.columns

    def test_custom_length(self, sample_df):
        result = sample_df.select(pl_cmo("close", length=20))
        assert "CMO_20" in result.columns

    def test_scalar_parameter(self, sample_df):
        result = sample_df.select(pl_cmo("close", scalar=50.0, talib=False))
        assert "CMO_14" in result.columns

    def test_values_in_range(self, sample_df):
        """CMO should be between -100 and 100."""
        result = sample_df.select(pl_cmo("close", talib=False))
        valid = result["CMO_14"].filter(~result["CMO_14"].is_nan())
        assert valid.min() >= -100
        assert valid.max() <= 100
