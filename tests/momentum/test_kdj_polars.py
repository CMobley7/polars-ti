# -*- coding: utf-8 -*-
"""Tests for pl_kdj - Polars implementation."""

import numpy as np
import polars as pl
import pytest
from polars_ti.momentum.kdj import kdj


class TestPlKdj:
    @pytest.fixture
    def sample_df(self):
        np.random.seed(42)
        n = 200
        high = 100 + np.cumsum(np.random.randn(n) * 0.5) + np.random.rand(n)
        low = 100 + np.cumsum(np.random.randn(n) * 0.5) - np.random.rand(n)
        close = (high + low) / 2
        return pl.DataFrame({"high": high, "low": low, "close": close})

    def test_returns_list(self):
        exprs = kdj("high", "low", "close")
        assert isinstance(exprs, list)
        assert len(exprs) == 3

    def test_has_kdj_columns(self, sample_df):
        result = sample_df.select(kdj("high", "low", "close"))
        assert "K_9_3" in result.columns
        assert "D_9_3" in result.columns
        assert "J_9_3" in result.columns

    def test_has_valid_values(self, sample_df):
        result = sample_df.select(kdj("high", "low", "close"))
        assert result["K_9_3"][20:].null_count() == 0

    def test_offset_parameter(self, sample_df):
        result_no_offset = sample_df.select(kdj("high", "low", "close", offset=0))
        result_with_offset = sample_df.select(kdj("high", "low", "close", offset=5))
        # Offset should add more nulls
        assert result_with_offset["K_9_3"].null_count() > result_no_offset["K_9_3"].null_count()

    def test_lazy_execution(self, sample_df):
        result = sample_df.lazy().select(kdj("high", "low", "close")).collect()
        assert "K_9_3" in result.columns

    def test_custom_lengths(self, sample_df):
        result = sample_df.select(kdj("high", "low", "close", length=14, signal=5))
        assert "K_14_5" in result.columns
        assert "D_14_5" in result.columns
        assert "J_14_5" in result.columns
