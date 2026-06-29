# -*- coding: utf-8 -*-
"""Unit tests for polars_ti/cycles/reflex.py Polars implementation."""

import numpy as np
import polars as pl
import pytest

from polars_ti.cycles.reflex import reflex


class TestPlReflex:
    """Tests for the native-Expr reflex."""

    def test_reflex_returns_expression(self):
        """Test that reflex() returns a Polars expression."""
        assert isinstance(reflex("close"), pl.Expr)

    def test_reflex_adds_column(self):
        """Test that selecting reflex produces the REFLEX column."""
        np.random.seed(42)
        close = np.random.randn(50).cumsum() + 100
        df = pl.DataFrame({"close": close})
        result = df.select(reflex("close", length=20, smooth=20))
        assert "REFLEX_20_20_0.04" in result.columns

    def test_warmup_period(self):
        """Test that first length values are NaN."""
        np.random.seed(42)
        close = np.random.randn(50).cumsum() + 100
        df = pl.DataFrame({"close": close})
        vals = df.select(reflex("close", length=20, smooth=20))["REFLEX_20_20_0.04"].to_numpy()[:20]
        assert np.isnan(vals).all()

    def test_custom_parameters(self):
        """Test custom parameters."""
        np.random.seed(42)
        close = np.random.randn(50).cumsum() + 100
        df = pl.DataFrame({"close": close})
        result = df.select(reflex("close", length=15, smooth=15, alpha=0.1))
        assert "REFLEX_15_15_0.1" in result.columns

    def test_with_zeros(self):
        """Handles zero values."""
        df = pl.DataFrame({"close": [0.0] * 10 + [100.0] * 90})
        result = df.select(reflex("close", length=20, smooth=20))
        assert result.height == 100

    def test_preserves_original_columns(self):
        """Original columns are preserved with with_columns."""
        np.random.seed(42)
        close = np.random.randn(50).cumsum() + 100
        df = pl.DataFrame({"close": close, "other": range(50)})
        result = df.with_columns(reflex("close", length=20, smooth=20))
        assert "close" in result.columns
        assert "other" in result.columns
        assert "REFLEX_20_20_0.04" in result.columns

    def test_oscillator_bounded(self):
        """Reflex output should be bounded (normalized)."""
        np.random.seed(42)
        close = np.random.randn(100).cumsum() + 100
        df = pl.DataFrame({"close": close})
        vals = df.select(reflex("close", length=20, smooth=20))["REFLEX_20_20_0.04"].to_numpy()
        valid = vals[~np.isnan(vals)]
        # Should be roughly bounded like an oscillator
        assert np.abs(valid).max() <= 5  # Max normalized value is 5
