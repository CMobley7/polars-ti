# -*- coding: utf-8 -*-
"""Unit tests for polars_ti/cycles/ebsw.py Polars implementation."""
import numpy as np
import polars as pl
import pytest

from polars_ti.cycles.ebsw import pl_ebsw, pl_ebsw_apply


class TestPlEbsw:
    """Tests for pl_ebsw and pl_ebsw_apply."""

    def test_pl_ebsw_returns_function(self):
        """Test that pl_ebsw returns a callable."""
        compute_fn = pl_ebsw()
        assert callable(compute_fn)

    def test_pl_ebsw_apply_adds_column(self):
        """Test that pl_ebsw_apply adds EBSW column."""
        np.random.seed(42)
        close = np.random.randn(60).cumsum() + 100
        df = pl.DataFrame({"close": close})
        result = pl_ebsw_apply(df, close="close", length=40, bars=10)
        assert "EBSW_40_10" in result.columns

    def test_ebsw_bounded(self):
        """Test that EBSW output is bounded between -1 and 1."""
        np.random.seed(42)
        close = np.random.randn(100).cumsum() + 100
        df = pl.DataFrame({"close": close})
        result = pl_ebsw_apply(df, close="close", length=40, bars=10)
        vals = result["EBSW_40_10"].drop_nulls().to_numpy()
        # Filter out NaN values and check bounds
        valid = vals[~np.isnan(vals)]
        assert np.abs(valid).max() <= 1.0 + 1e-10  # Small tolerance

    def test_warmup_period(self):
        """Test that first length-1 values are NaN (not computed)."""
        np.random.seed(42)
        close = np.random.randn(60).cumsum() + 100
        df = pl.DataFrame({"close": close})
        result = pl_ebsw_apply(df, close="close", length=40, bars=10)
        vals = result["EBSW_40_10"].to_numpy()
        # First 39 should be NaN (length-1)
        assert np.isnan(vals[:39]).all()
        # Value at index 40 should exist
        assert not np.isnan(vals[40])

    def test_with_zeros(self):
        """Handles zero values."""
        df = pl.DataFrame({"close": [0.0] * 10 + [100.0] * 90})
        result = pl_ebsw_apply(df, length=40, bars=10)
        assert result.height == 100

    def test_preserves_original_columns(self):
        """Original columns are preserved."""
        np.random.seed(42)
        close = np.random.randn(60).cumsum() + 100
        df = pl.DataFrame({"close": close, "other": range(60)})
        result = pl_ebsw_apply(df, length=40, bars=10)
        assert "close" in result.columns
        assert "other" in result.columns
        assert "EBSW_40_10" in result.columns
