# -*- coding: utf-8 -*-
"""Tests for pl_rmi (Relative Momentum Index)."""

import numpy as np
import polars as pl
import pytest
from polars_ti.momentum.rmi import pl_rmi


class TestPlRmi:
    @pytest.fixture
    def sample_df(self) -> pl.DataFrame:
        np.random.seed(42)
        n = 100
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        return pl.DataFrame({"close": close})

    def test_returns_expression(self, sample_df):
        """Test that pl_rmi returns a valid expression."""
        result = sample_df.select(pl_rmi("close"))
        assert result.height == 100

    def test_output_has_correct_alias(self, sample_df):
        """Test that output column has correct name."""
        result = sample_df.select(pl_rmi("close", length=10, momentum=3))
        assert "RMI_10_3" in result.columns

    def test_offset_shifts_result(self, sample_df):
        """Test that offset parameter shifts results correctly."""
        result = sample_df.select(pl_rmi("close", offset=5))
        arr = result[result.columns[0]].to_numpy()
        # First values should include offset NaNs (momentum(5) + offset(5) = 10)
        assert all(np.isnan(arr[:10]))

    def test_with_null_values(self):
        """Test handling of null values."""
        df = pl.DataFrame({"close": [None] + [100.0] * 49})
        result = df.select(pl_rmi("close", length=10, momentum=3))
        assert result.height == 50

    def test_lazy_execution(self, sample_df):
        """Test that pl_rmi works with lazy frames."""
        lazy_df = sample_df.lazy()
        result = lazy_df.select(pl_rmi("close")).collect()
        assert "RMI_14_5" in result.columns

    def test_rmi_bounds(self, sample_df):
        """Test that RMI values are bounded 0-100."""
        result = sample_df.select(pl_rmi("close"))
        values = result.to_numpy().flatten()
        valid = values[~np.isnan(values)]
        assert np.all(valid >= 0), "RMI should not be negative"
        assert np.all(valid <= 100), "RMI should not exceed 100"
