# -*- coding: utf-8 -*-
"""Unit tests for polars_ti/cycles/ebsw.py Polars implementation."""

import numpy as np
import polars as pl
import pytest

from polars_ti.cycles.ebsw import ebsw


class TestPlEbsw:
    """Tests for the native-Expr ebsw."""

    def test_ebsw_returns_expression(self):
        """Test that ebsw() returns a Polars expression."""
        assert isinstance(ebsw("close"), pl.Expr)

    def test_ebsw_adds_column(self):
        """Test that selecting ebsw produces the EBSW column."""
        np.random.seed(42)
        close = np.random.randn(60).cumsum() + 100
        df = pl.DataFrame({"close": close})
        result = df.select(ebsw("close", length=40, bars=10))
        assert "EBSW_40_10" in result.columns

    def test_ebsw_bounded(self):
        """Test that EBSW output is bounded between -1 and 1."""
        np.random.seed(42)
        close = np.random.randn(100).cumsum() + 100
        df = pl.DataFrame({"close": close})
        vals = df.select(ebsw("close", length=40, bars=10))["EBSW_40_10"].to_numpy()
        valid = vals[~np.isnan(vals)]
        assert np.abs(valid).max() <= 1.0 + 1e-10  # Small tolerance

    def test_warmup_period(self):
        """Test that first length-1 values are NaN (not computed)."""
        np.random.seed(42)
        close = np.random.randn(60).cumsum() + 100
        df = pl.DataFrame({"close": close})
        vals = df.select(ebsw("close", length=40, bars=10))["EBSW_40_10"].to_numpy()
        # First 39 should be NaN (length-1)
        assert np.isnan(vals[:39]).all()
        # Value at index 40 should exist
        assert not np.isnan(vals[40])

    def test_with_zeros(self):
        """Handles zero values."""
        df = pl.DataFrame({"close": [0.0] * 10 + [100.0] * 90})
        result = df.select(ebsw("close", length=40, bars=10))
        assert result.height == 100

    def test_preserves_original_columns(self):
        """Original columns are preserved with with_columns."""
        np.random.seed(42)
        close = np.random.randn(60).cumsum() + 100
        df = pl.DataFrame({"close": close, "other": range(60)})
        result = df.with_columns(ebsw("close", length=40, bars=10))
        assert "close" in result.columns
        assert "other" in result.columns
        assert "EBSW_40_10" in result.columns
