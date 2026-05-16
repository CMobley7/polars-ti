# -*- coding: utf-8 -*-
"""Unit tests for polars_ti/overlap/dema.py Polars implementation."""
import numpy as np
import polars as pl
import pytest

from polars_ti.overlap.dema import pl_dema


class TestPlDema:
    """Tests for pl_dema - Double Exponential Moving Average."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        np.random.seed(42)
        close = 100 + np.random.randn(200).cumsum()
        return {
            'pd_series': close,
            'pl_df': pl.DataFrame({'close': close}),
        }

    def test_output_has_correct_alias(self, sample_data):
        """Test that output column has correct alias."""
        result = sample_data['pl_df'].select(pl_dema('close', length=10))
        assert result.columns[0] == 'DEMA_10'

    def test_offset_shifts_result(self, sample_data):
        """Test that offset parameter shifts the result."""
        no_offset = sample_data['pl_df'].select(pl_dema('close', length=10)).to_series()
        with_offset = sample_data['pl_df'].select(pl_dema('close', length=10, offset=5)).to_series()
        
        # Values at index i+5 in offset result should equal index i in non-offset
        for i in range(20, 50):
            if not np.isnan(no_offset[i]):
                assert no_offset[i] == with_offset[i + 5], f"Offset mismatch at {i}"

    def test_warmup_period_has_nan(self, sample_data):
        """Test that warmup period contains NaN values."""
        result = sample_data['pl_df'].select(pl_dema('close', length=10)).to_series()
        # First 2*length - 2 values should be NaN (EMA warmup for both stages)
        assert result[:17].is_nan().all()

    def test_talib_parameter_accepted(self, sample_data):
        """Test that talib parameter is accepted (for API compatibility)."""
        result = sample_data['pl_df'].select(pl_dema('close', length=10, talib=False))
        assert result is not None

    def test_different_lengths(self, sample_data):
        """Test that different lengths produce different results."""
        result_10 = sample_data['pl_df'].select(pl_dema('close', length=10)).to_series()
        result_20 = sample_data['pl_df'].select(pl_dema('close', length=20)).to_series()
        
        # After both warmups, values should be different
        warmup = 40
        valid_10 = ~result_10[warmup:].is_nan()
        valid_20 = ~result_20[warmup:].is_nan()
        
        if valid_10.sum() > 0 and valid_20.sum() > 0:
            # Values should be different (not exactly equal)
            diff = result_10[warmup:].to_numpy() - result_20[warmup:].to_numpy()
            valid = ~np.isnan(diff)
            assert np.any(np.abs(diff[valid]) > 1e-10)

    def test_with_null_values(self):
        """Handles null values gracefully."""
        df = pl.DataFrame({"close": [None] + [100.0] * 49})
        result = df.select(pl_dema("close", length=10))
        assert result.height == 50

    def test_with_zeros(self):
        """Handles zero values."""
        df = pl.DataFrame({"close": [0.0] * 5 + [100.0] * 45})
        result = df.select(pl_dema("close", length=10))
        assert result.height == 50

    def test_lazy_execution(self, sample_data):
        """Works with LazyFrame."""
        lazy_df = sample_data['pl_df'].lazy()
        result = lazy_df.select(pl_dema("close", length=10)).collect()
        assert "DEMA_10" in result.columns
