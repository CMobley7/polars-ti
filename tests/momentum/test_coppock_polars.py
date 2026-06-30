# -*- coding: utf-8 -*-
"""Tests for pl_coppock - Pure Polars implementation using pl_roc + pl_wma."""

import numpy as np
import polars as pl
import pytest
from polars_ti.momentum.coppock import coppock


class TestPlCoppock:
    @pytest.fixture
    def sample_df(self):
        np.random.seed(42)
        n = 100
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        return pl.DataFrame({"close": close})

    def test_returns_expr(self):
        expr = coppock("close")
        assert isinstance(expr, pl.Expr)

    def test_has_coppock_column(self, sample_df):
        result = sample_df.select(coppock("close"))
        assert "COPC_11_14_10" in result.columns

    def test_has_valid_values(self, sample_df):
        result = sample_df.select(coppock("close"))
        # After warmup (14 + 10 = 24), should have values
        assert result["COPC_11_14_10"][30:].null_count() == 0

    def test_offset_parameter(self, sample_df):
        result_no_offset = sample_df.select(coppock("close", offset=0))
        result_with_offset = sample_df.select(coppock("close", offset=5))
        assert result_with_offset["COPC_11_14_10"].null_count() > result_no_offset["COPC_11_14_10"].null_count()

    def test_lazy_execution(self, sample_df):
        result = sample_df.lazy().select(coppock("close")).collect()
        assert "COPC_11_14_10" in result.columns

    def test_custom_lengths(self, sample_df):
        result = sample_df.select(coppock("close", length=5, fast=7, slow=10))
        assert "COPC_7_10_5" in result.columns

    def test_with_null_values(self):
        df = pl.DataFrame({"close": [100.0, None, 102.0] + [100.0] * 50})
        result = df.select(coppock("close"))
        assert result.height == 53

    def test_composition(self, sample_df):
        """Verify it uses pl_roc and pl_wma composition."""
        result = sample_df.select(coppock("close"))
        # Should produce valid non-zero values
        valid = result["COPC_11_14_10"].filter(~result["COPC_11_14_10"].is_nan())
        assert valid.std() > 0  # Should have variation
