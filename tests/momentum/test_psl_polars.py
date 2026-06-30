# -*- coding: utf-8 -*-
"""Tests for pl_psl (Psychological Line)."""

import numpy as np
import polars as pl
import pytest
from polars_ti.momentum.psl import psl


class TestPlPsl:
    @pytest.fixture
    def sample_df(self) -> pl.DataFrame:
        np.random.seed(42)
        n = 100
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        open_ = close + np.random.randn(n) * 0.2
        return pl.DataFrame({"close": close, "open": open_})

    def test_returns_expression(self, sample_df):
        """Test that pl_psl returns a valid expression."""
        result = sample_df.select(psl("close"))
        assert result.height == 100

    def test_output_has_correct_alias(self, sample_df):
        """Test that output column has correct name."""
        result = sample_df.select(psl("close", length=14))
        assert "PSL_14" in result.columns

    def test_offset_shifts_result(self, sample_df):
        """Test that offset parameter shifts results correctly."""
        result = sample_df.select(psl("close", offset=5))
        arr = result[result.columns[0]].to_numpy()
        # First 16 values should be null (11 from warmup + 5 offset)
        assert all(np.isnan(arr[:16]))  # length-1 + offset = 11 + 5 = 16

    def test_with_null_values(self):
        """Test handling of null values."""
        df = pl.DataFrame({"close": [None] + [100.0] * 49})
        result = df.select(psl("close", length=10))
        assert result.height == 50

    def test_lazy_execution(self, sample_df):
        """Test that pl_psl works with lazy frames."""
        lazy_df = sample_df.lazy()
        result = lazy_df.select(psl("close")).collect()
        assert "PSL_12" in result.columns

    def test_psl_bounds(self, sample_df):
        """Test that PSL values are bounded 0-100."""
        result = sample_df.select(psl("close", length=12))
        values = result.to_numpy().flatten()
        valid = values[~np.isnan(values)]
        assert np.all(valid >= 0), "PSL should not be negative"
        assert np.all(valid <= 100), "PSL should not exceed 100"
