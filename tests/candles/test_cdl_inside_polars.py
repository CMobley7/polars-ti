# -*- coding: utf-8 -*-
"""Unit tests for polars_ti/candles/cdl_inside.py Polars implementation."""

import numpy as np
import polars as pl
import pytest

from polars_ti.candles.cdl_inside import cdl_inside


class TestPlCdlInside:
    """Tests for pl_cdl_inside."""

    def test_detects_inside_bar(self):
        """Test that inside bars are detected correctly."""
        df = pl.DataFrame(
            {
                "open": [100.0, 100.0, 98.0],
                "high": [110.0, 105.0, 100.0],  # Decreasing highs
                "low": [90.0, 92.0, 95.0],  # Increasing lows
                "close": [105.0, 102.0, 99.0],
            }
        )
        result = df.select(cdl_inside("open", "high", "low", "close"))["CDL_INSIDE"]
        # Index 1 and 2 should be inside bars
        assert result.to_list()[1] == 100  # First inside bar
        assert result.to_list()[2] == 100  # Second inside bar

    def test_no_inside_bar_higher_high(self):
        """Test no detection when high is higher than previous."""
        df = pl.DataFrame(
            {
                "open": [95.0, 100.0, 110.0],
                "high": [100.0, 110.0, 120.0],  # Increasing highs
                "low": [90.0, 92.0, 95.0],
                "close": [98.0, 105.0, 115.0],
            }
        )
        result = df.select(cdl_inside("open", "high", "low", "close"))["CDL_INSIDE"]
        # Should not detect inside bar when high increases
        assert result.to_list()[1] == 0
        assert result.to_list()[2] == 0

    def test_custom_scalar(self):
        """Test that custom scalar is applied."""
        df = pl.DataFrame(
            {
                "open": [100.0, 100.0],
                "high": [110.0, 105.0],
                "low": [90.0, 92.0],
                "close": [105.0, 102.0],
            }
        )
        result = df.select(cdl_inside("open", "high", "low", "close", scalar=50.0))["CDL_INSIDE"]
        assert result.to_list()[1] == 50

    def test_first_value_is_zero(self):
        """Test that first value is 0 (no previous bar to compare)."""
        df = pl.DataFrame(
            {
                "open": [100.0, 100.0, 98.0],
                "high": [110.0, 105.0, 100.0],
                "low": [90.0, 92.0, 95.0],
                "close": [105.0, 102.0, 99.0],
            }
        )
        result = df.select(cdl_inside("open", "high", "low", "close"))["CDL_INSIDE"]
        assert result.to_list()[0] == 0

    def test_asbool_returns_boolean(self):
        """Test that asbool=True returns boolean values."""
        df = pl.DataFrame(
            {
                "open": [100.0, 100.0],
                "high": [110.0, 105.0],
                "low": [90.0, 92.0],
                "close": [105.0, 102.0],
            }
        )
        result = df.select(cdl_inside("open", "high", "low", "close", asbool=True))["CDL_INSIDE"]
        assert result.dtype == pl.Boolean
        assert result.to_list()[1] is True

    def test_offset_shifts_result(self):
        """Test that offset shifts the result."""
        df = pl.DataFrame(
            {
                "open": [100.0, 100.0, 98.0],
                "high": [110.0, 105.0, 100.0],
                "low": [90.0, 92.0, 95.0],
                "close": [105.0, 102.0, 99.0],
            }
        )
        result = df.select(cdl_inside("open", "high", "low", "close", offset=1))["CDL_INSIDE"]
        # With offset=1, values should be shifted forward
        assert result.to_list()[0] is None  # First value becomes null
        assert result.to_list()[2] == 100  # Inside bar shifted from index 1 to 2

    def test_with_null_values(self):
        """Handles null values gracefully."""
        df = pl.DataFrame(
            {
                "open": [None] + [100.0] * 9,
                "high": [110.0] * 10,
                "low": [90.0] * 10,
                "close": [105.0] * 10,
            }
        )
        result = df.select(cdl_inside("open", "high", "low", "close"))
        assert result.height == 10

    def test_with_zeros(self):
        """Handles zero values."""
        df = pl.DataFrame(
            {
                "open": [0.0] * 5 + [100.0] * 5,
                "high": [0.0] * 4 + [110.0, 105.0, 100.0, 95.0, 90.0, 85.0],
                "low": [0.0] * 4 + [90.0, 92.0, 95.0, 97.0, 85.0, 80.0],
                "close": [0.0] * 5 + [105.0] * 5,
            }
        )
        result = df.select(cdl_inside("open", "high", "low", "close"))
        assert result.height == 10

    def test_lazy_execution(self):
        """Works with LazyFrame."""
        df = pl.DataFrame(
            {
                "open": [100.0, 100.0, 98.0],
                "high": [110.0, 105.0, 100.0],
                "low": [90.0, 92.0, 95.0],
                "close": [105.0, 102.0, 99.0],
            }
        )
        lazy_df = df.lazy()
        result = lazy_df.select(cdl_inside("open", "high", "low", "close")).collect()
        assert "CDL_INSIDE" in result.columns
