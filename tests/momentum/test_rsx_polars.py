# -*- coding: utf-8 -*-
"""Tests for Polars RSX (Relative Strength Xtra) implementation."""

import numpy as np
import polars as pl
import pytest

from polars_ti.momentum.rsx import pl_rsx


class TestPlRsx:
    """Test suite for pl_rsx Polars implementation."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample price data for testing."""
        np.random.seed(42)
        n = 200
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        return pl.DataFrame({"close": close}), close

    def test_returns_expression(self, sample_data):
        """pl_rsx should return a Polars expression."""
        result = pl_rsx("close")
        assert isinstance(result, pl.Expr)

    def test_output_has_correct_alias(self, sample_data):
        """Output column should have correct RSX_<length> alias."""
        pl_df, _ = sample_data
        result = pl_df.select(pl_rsx("close", length=14))
        assert "RSX_14" in result.columns

    def test_offset_shifts_result(self, sample_data):
        """Offset parameter should shift results."""
        pl_df, _ = sample_data
        offset = 5

        result_no_offset = pl_df.select(pl_rsx("close", offset=0))
        result_with_offset = pl_df.select(pl_rsx("close", offset=offset))

        # Values should be shifted
        no_offset_vals = result_no_offset[result_no_offset.columns[0]].to_numpy()
        with_offset_vals = result_with_offset[result_with_offset.columns[0]].to_numpy()

        # Check shift at a specific non-NaN position
        valid_idx = 30
        assert np.isclose(no_offset_vals[valid_idx], with_offset_vals[valid_idx + offset], atol=1e-10)

    def test_with_null_values(self, sample_data):
        """pl_rsx should handle null values gracefully."""
        pl_df, _ = sample_data

        # Insert nulls
        df_with_nulls = pl_df.with_columns(
            pl.when(pl.col("close").is_between(50, 60)).then(None).otherwise(pl.col("close")).alias("close")
        )

        result = df_with_nulls.select(pl_rsx("close"))
        assert result.height == pl_df.height

    def test_lazy_execution(self, sample_data):
        """pl_rsx should work with lazy DataFrames."""
        pl_df, _ = sample_data
        lazy_df = pl_df.lazy()

        result = lazy_df.select(pl_rsx("close", length=14)).collect()
        assert result.height == pl_df.height
        assert "RSX_14" in result.columns

    def test_rsx_bounds(self, sample_data):
        """RSX values should be bounded between 0 and 100."""
        pl_df, _ = sample_data
        result = pl_df.select(pl_rsx("close", length=14))
        values = result[result.columns[0]].to_numpy()

        valid_vals = values[~np.isnan(values)]
        assert np.all(valid_vals >= 0), "RSX values should be >= 0"
        assert np.all(valid_vals <= 100), "RSX values should be <= 100"

    def test_different_length_parameters(self, sample_data):
        """RSX should work with different length parameters."""
        pl_df, _ = sample_data

        for length in [7, 14, 21]:
            result = pl_df.select(pl_rsx("close", length=length))
            assert f"RSX_{length}" in result.columns
            assert result.height == pl_df.height
