# -*- coding: utf-8 -*-
"""Tests for Polars SLOPE implementation."""
import numpy as np
import pandas as pd  # REMOVED: pandas dependency  # Restored for fixtures
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
        return pl.DataFrame({"close": close}), pd.Series(close)

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

    def test_numerical_parity_slope_mode(self, sample_data):
        """Polars SLOPE should match Pandas within 1e-6 tolerance in slope mode."""
        pytest.skip("Pandas implementation removed in Phase 4 purge")
        pl_df, pd_close = sample_data
        length = 10

        pd_result = slope(pd_close, length=length)
        pd_arr = pd_result.to_numpy()

        pl_result = pl_df.select(pl_slope("close", length=length))
        pl_arr = pl_result[pl_result.columns[0]].to_numpy()

        warmup = length + 5
        mask = ~np.isnan(pd_arr[warmup:]) & ~np.isnan(pl_arr[warmup:])
        max_diff = np.max(np.abs(pl_arr[warmup:][mask] - pd_arr[warmup:][mask]))
        assert max_diff < 1e-6

    def test_numerical_parity_angle_mode(self, sample_data):
        """Polars SLOPE should match Pandas in angle mode."""
        pytest.skip("Pandas implementation removed in Phase 4 purge")
        pl_df, pd_close = sample_data
        length = 10

        pd_result = slope(pd_close, length=length, as_angle=True, to_degrees=True)
        pd_arr = pd_result.to_numpy()

        pl_result = pl_df.select(pl_slope("close", length=length, as_angle=True, to_degrees=True))
        pl_arr = pl_result[pl_result.columns[0]].to_numpy()

        warmup = length + 5
        mask = ~np.isnan(pd_arr[warmup:]) & ~np.isnan(pl_arr[warmup:])
        max_diff = np.max(np.abs(pl_arr[warmup:][mask] - pd_arr[warmup:][mask]))
        assert max_diff < 1e-6

    def test_offset_shifts_result(self, sample_data):
        """Offset parameter should shift results."""
        pl_df, _ = sample_data
        offset = 5

        result_no_offset = pl_df.select(pl_slope("close", offset=0))
        result_with_offset = pl_df.select(pl_slope("close", offset=offset))

        no_offset_vals = result_no_offset[result_no_offset.columns[0]].to_numpy()
        with_offset_vals = result_with_offset[result_with_offset.columns[0]].to_numpy()

        valid_idx = 20
        assert np.isclose(
            no_offset_vals[valid_idx],
            with_offset_vals[valid_idx + offset],
            atol=1e-10
        )

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
