# -*- coding: utf-8 -*-
"""Tests for Polars SLOPE implementation."""

import numpy as np
import polars as pl
import pytest

from polars_ti.momentum.slope import pl_slope


class TestPlSlope:
    """Test suite for pl_slope Polars implementation."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample price data for testing."""
        np.random.seed(42)
        n = 200
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        return pl.DataFrame({"close": close}), close

    def test_returns_expression(self, sample_data):
        """pl_slope should return a Polars expression."""
        result = pl_slope("close")
        assert isinstance(result, pl.Expr)

    def test_output_has_correct_alias_slope(self, sample_data):
        """Output column should have SLOPE_<length> alias in slope mode."""
        pl_df, _ = sample_data
        result = pl_df.select(pl_slope("close", length=10))
        assert "SLOPE_10" in result.columns

    def test_output_has_correct_alias_angle_radians(self, sample_data):
        """Output column should have ANGLEr_<length> alias in radians mode."""
        pl_df, _ = sample_data
        result = pl_df.select(pl_slope("close", length=10, as_angle=True, to_degrees=False))
        assert "ANGLEr_10" in result.columns

    def test_output_has_correct_alias_angle_degrees(self, sample_data):
        """Output column should have ANGLEd_<length> alias in degrees mode."""
        pl_df, _ = sample_data
        result = pl_df.select(pl_slope("close", length=10, as_angle=True, to_degrees=True))
        assert "ANGLEd_10" in result.columns

    def test_offset_shifts_result(self, sample_data):
        """Offset parameter should shift results."""
        pl_df, _ = sample_data
        offset = 5

        result_no_offset = pl_df.select(pl_slope("close", offset=0))
        result_with_offset = pl_df.select(pl_slope("close", offset=offset))

        no_offset_vals = result_no_offset[result_no_offset.columns[0]].to_numpy()
        with_offset_vals = result_with_offset[result_with_offset.columns[0]].to_numpy()

        valid_idx = 20
        assert np.isclose(no_offset_vals[valid_idx], with_offset_vals[valid_idx + offset], atol=1e-10)

    def test_lazy_execution(self, sample_data):
        """pl_slope should work with lazy DataFrames."""
        pl_df, _ = sample_data
        lazy_df = pl_df.lazy()

        result = lazy_df.select(pl_slope("close", length=10)).collect()
        assert result.height == pl_df.height

    def test_different_length_parameters(self, sample_data):
        """SLOPE should work with different length parameters."""
        pl_df, _ = sample_data

        for length in [1, 5, 10, 20]:
            result = pl_df.select(pl_slope("close", length=length))
            assert f"SLOPE_{length}" in result.columns
