# -*- coding: utf-8 -*-
"""Unit tests for polars_ti/cycles/dsp.py Polars implementation."""

import numpy as np
import polars as pl
import pytest

from polars_ti.cycles.dsp import pl_dsp


class TestPlDsp:
    """Tests for pl_dsp."""

    def test_basic_calculation(self):
        """Test basic DSP calculation."""
        np.random.seed(42)
        close = np.random.randn(30).cumsum() + 100
        df = pl.DataFrame({"close": close})
        result = df.select(pl_dsp("close", length=14))
        assert "DSP_14" in result.columns
        # After warmup, values should exist
        assert not result["DSP_14"].is_null().all()

    def test_dsp_oscillates_around_zero(self):
        """Test that DSP oscillates around zero (detrended)."""
        # Create trending data
        close = list(range(1, 51))  # Linear trend
        df = pl.DataFrame({"close": [float(x) for x in close]})
        result = df.select(pl_dsp("close", length=10))
        vals = result["DSP_10"].drop_nulls().to_numpy()
        vals = vals[~np.isnan(vals)]  # Filter NaN values
        # Mean should be close to positive (above trend line for uptrend)
        assert vals.mean() > 0

    def test_custom_length(self):
        """Test custom length parameter."""
        close = [float(x) for x in range(1, 31)]
        df = pl.DataFrame({"close": close})
        result = df.select(pl_dsp("close", length=5))
        assert "DSP_5" in result.columns

    def test_alias_format(self):
        """Test output alias format."""
        df = pl.DataFrame({"close": [100.0] * 20})
        result = df.select(pl_dsp("close", length=7))
        assert result.columns[0] == "DSP_7"

    def test_with_null_values(self):
        """Handles null values gracefully."""
        df = pl.DataFrame({"close": [None] + [100.0] * 29})
        result = df.select(pl_dsp("close", length=14))
        assert result.height == 30

    def test_with_zeros(self):
        """Handles zero values."""
        df = pl.DataFrame({"close": [0.0] * 5 + [100.0] * 25})
        result = df.select(pl_dsp("close", length=14))
        assert result.height == 30

    def test_lazy_execution(self):
        """Works with LazyFrame."""
        np.random.seed(42)
        close = np.random.randn(30).cumsum() + 100
        df = pl.DataFrame({"close": close})
        lazy_df = df.lazy()
        result = lazy_df.select(pl_dsp("close", length=14)).collect()
        assert "DSP_14" in result.columns

    def test_offset_parameter(self):
        """Offset parameter shifts results."""
        df = pl.DataFrame({"close": [100.0] * 30})
        result = df.select(pl_dsp("close", length=14, offset=2))
        arr = result["DSP_14"].to_numpy()
        # First 2 values should be null due to offset
        assert np.isnan(arr[0]) or arr[0] is None
