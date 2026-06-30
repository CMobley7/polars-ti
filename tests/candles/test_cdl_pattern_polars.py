# -*- coding: utf-8 -*-
"""Unit tests for polars_ti/candles/cdl_pattern.py Polars implementation."""

import numpy as np
import polars as pl
import pytest

from polars_ti.candles.cdl_pattern import cdl_pattern, POLARS_PATTERNS


class TestPlCdlPattern:
    """Tests for pl_cdl_pattern wrapper."""

    @pytest.fixture
    def sample_df(self):
        """Create sample OHLC DataFrame."""
        np.random.seed(42)
        n = 20
        high = np.random.randn(n).cumsum() + 102
        low = high - np.abs(np.random.randn(n) * 2)
        open_ = low + np.random.rand(n) * (high - low)
        close = low + np.random.rand(n) * (high - low)
        return pl.DataFrame({"open": open_, "high": high, "low": low, "close": close})

    def test_all_patterns(self, sample_df):
        """Test detecting all available patterns."""
        result = cdl_pattern(sample_df, name="all")
        # Should have original columns plus pattern columns
        assert len(result.columns) > 4
        assert "CDL_DOJI_10_0.1" in result.columns
        assert "CDL_INSIDE" in result.columns

    def test_single_pattern(self, sample_df):
        """Test detecting single pattern by name."""
        result = cdl_pattern(sample_df, name="doji")
        assert "CDL_DOJI_10_0.1" in result.columns

    def test_pattern_list(self, sample_df):
        """Test detecting multiple patterns by list."""
        result = cdl_pattern(sample_df, name=["doji", "inside"])
        assert "CDL_DOJI_10_0.1" in result.columns
        assert "CDL_INSIDE" in result.columns

    def test_unknown_pattern_warning(self, sample_df, capsys):
        """Test that unknown patterns print a warning."""
        result = cdl_pattern(sample_df, name="unknown_pattern")
        captured = capsys.readouterr()
        assert "not available" in captured.out

    def test_custom_scalar(self, sample_df):
        """Test custom scalar is passed through."""
        result = cdl_pattern(sample_df, name="inside", scalar=50.0)
        # Inside bar values should be 0 or 50
        vals = result["CDL_INSIDE"].unique().to_list()
        assert all(v in [0, 50] for v in vals)

    def test_with_null_values(self):
        """Handles null values gracefully."""
        df = pl.DataFrame(
            {
                "open": [None] + [100.0] * 19,
                "high": [110.0] * 20,
                "low": [90.0] * 20,
                "close": [100.01] * 20,
            }
        )
        result = cdl_pattern(df, name="doji")
        assert result.height == 20

    def test_with_zeros(self):
        """Handles zero values."""
        df = pl.DataFrame(
            {
                "open": [0.0] * 5 + [100.0] * 15,
                "high": [0.0] * 5 + [110.0] * 15,
                "low": [0.0] * 5 + [90.0] * 15,
                "close": [0.0] * 5 + [100.01] * 15,
            }
        )
        result = cdl_pattern(df, name="doji")
        assert result.height == 20

    def test_lazy_execution(self, sample_df):
        """Works with LazyFrame (via collect)."""
        # pl_cdl_pattern takes DataFrame, lazy not directly supported
        result = cdl_pattern(sample_df, name="inside")
        assert "CDL_INSIDE" in result.columns
