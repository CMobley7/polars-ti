# -*- coding: utf-8 -*-
"""Tests for pl_cti - Polars + Numba implementation."""

import numpy as np
import polars as pl
import pytest
from polars_ti.momentum.cti import pl_cti


class TestPlCti:
    @pytest.fixture
    def sample_df(self):
        np.random.seed(42)
        n = 100
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        return pl.DataFrame({"close": close})

    def test_returns_expr(self):
        expr = pl_cti("close")
        assert isinstance(expr, pl.Expr)

    def test_has_cti_column(self, sample_df):
        result = sample_df.select(pl_cti("close"))
        assert "CTI_12" in result.columns

    def test_has_valid_values(self, sample_df):
        result = sample_df.select(pl_cti("close"))
        assert result["CTI_12"][15:].is_nan().sum() == 0

    def test_offset_parameter(self, sample_df):
        result_no_offset = sample_df.select(pl_cti("close", offset=0))
        result_with_offset = sample_df.select(pl_cti("close", offset=5))
        # offset shifts all values by 5, adding 5 more nulls
        assert result_with_offset["CTI_12"].null_count() > result_no_offset["CTI_12"].null_count()

    def test_lazy_execution(self, sample_df):
        result = sample_df.lazy().select(pl_cti("close")).collect()
        assert "CTI_12" in result.columns

    def test_custom_length(self, sample_df):
        result = sample_df.select(pl_cti("close", length=20))
        assert "CTI_20" in result.columns

    def test_values_in_range(self, sample_df):
        """CTI should be between -1 and 1."""
        result = sample_df.select(pl_cti("close"))
        valid = result["CTI_12"].filter(~result["CTI_12"].is_nan())
        assert valid.min() >= -1.0
        assert valid.max() <= 1.0

    def test_with_null_values(self):
        df = pl.DataFrame({"close": [100.0, None, 102.0] + [100.0] * 50})
        result = df.select(pl_cti("close"))
        assert result.height == 53
