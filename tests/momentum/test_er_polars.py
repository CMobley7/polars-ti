# -*- coding: utf-8 -*-
"""Tests for pl_er - Pure Polars implementation."""

import numpy as np
import polars as pl
import pytest
from polars_ti.momentum.er import er


class TestPlEr:
    @pytest.fixture
    def sample_df(self):
        np.random.seed(42)
        n = 100
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        return pl.DataFrame({"close": close})

    def test_returns_expr(self):
        expr = er("close")
        assert isinstance(expr, pl.Expr)

    def test_has_er_column(self, sample_df):
        result = sample_df.select(er("close"))
        assert "ER_10" in result.columns

    def test_has_valid_values(self, sample_df):
        result = sample_df.select(er("close"))
        assert result["ER_10"][15:].null_count() == 0

    def test_offset_parameter(self, sample_df):
        result_no_offset = sample_df.select(er("close", offset=0))
        result_with_offset = sample_df.select(er("close", offset=5))
        assert result_with_offset["ER_10"].null_count() > result_no_offset["ER_10"].null_count()

    def test_lazy_execution(self, sample_df):
        result = sample_df.lazy().select(er("close")).collect()
        assert "ER_10" in result.columns

    def test_custom_length(self, sample_df):
        result = sample_df.select(er("close", length=20))
        assert "ER_20" in result.columns

    def test_values_in_range(self, sample_df):
        """ER should be between 0 and 1."""
        result = sample_df.select(er("close"))
        valid = result["ER_10"].filter(~result["ER_10"].is_null())
        assert valid.min() >= 0
        assert valid.max() <= 1.05  # slight tolerance for floating point
