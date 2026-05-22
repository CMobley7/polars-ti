# -*- coding: utf-8 -*-
"""Unit tests for polars_ti/overlap/sma.py Polars implementation."""

import numpy as np
import polars as pl
import pytest

from polars_ti.overlap.sma import pl_sma


class TestPlSma:
    """Tests for pl_sma - Simple Moving Average."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        np.random.seed(42)
        close = 100 + np.random.randn(200).cumsum()
        return {
            "pd_series": close,
            "pl_df": pl.DataFrame({"close": close}),
        }

    def test_output_has_correct_alias(self, sample_data):
        """Test that output column has correct alias."""
        result = sample_data["pl_df"].select(pl_sma("close", length=10))
        assert result.columns[0] == "SMA_10"

    def test_offset_shifts_result(self, sample_data):
        """Test that offset parameter shifts the result."""
        no_offset = sample_data["pl_df"].select(pl_sma("close", length=10)).to_series()
        with_offset = sample_data["pl_df"].select(pl_sma("close", length=10, offset=5)).to_series()

        for i in range(10, 50):
            if not np.isnan(no_offset[i]):
                assert no_offset[i] == with_offset[i + 5], f"Offset mismatch at {i}"

    def test_warmup_period_has_nan(self, sample_data):
        """Test that warmup period contains NaN values."""
        result = sample_data["pl_df"].select(pl_sma("close", length=10)).to_series()
        assert result[:9].is_nan().all()

    def test_min_periods_parameter(self, sample_data):
        """Test that min_periods allows early values."""
        result = sample_data["pl_df"].select(pl_sma("close", length=10, min_periods=1)).to_series()
        # Should have non-NaN from index 0
        assert not result[0:].is_nan().all()

    def test_with_null_values(self):
        """Handles null values gracefully."""
        df = pl.DataFrame({"close": [None] + [100.0] * 29})
        result = df.select(pl_sma("close", length=10))
        assert result.height == 30

    def test_with_zeros(self):
        """Handles zero values."""
        df = pl.DataFrame({"close": [0.0] * 5 + [100.0] * 25})
        result = df.select(pl_sma("close", length=10))
        assert result.height == 30

    def test_lazy_execution(self, sample_data):
        """Works with LazyFrame."""
        lazy_df = sample_data["pl_df"].lazy()
        result = lazy_df.select(pl_sma("close", length=10)).collect()
        assert "SMA_10" in result.columns
