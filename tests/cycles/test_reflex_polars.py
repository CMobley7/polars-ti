# -*- coding: utf-8 -*-
"""Unit tests for polars_ti/cycles/reflex.py Polars implementation."""

import numpy as np
import polars as pl
import pytest

from polars_ti.cycles.reflex import reflex, reflex_apply


class TestPlReflex:
    """Tests for pl_reflex and pl_reflex_apply."""

    def test_pl_reflex_returns_function(self):
        """Test that pl_reflex returns a callable."""
        compute_fn = reflex()
        assert callable(compute_fn)

    def test_pl_reflex_apply_adds_column(self):
        """Test that pl_reflex_apply adds REFLEX column."""
        np.random.seed(42)
        close = np.random.randn(50).cumsum() + 100
        df = pl.DataFrame({"close": close})
        result = reflex_apply(df, close="close", length=20, smooth=20)
        assert "REFLEX_20_20_0.04" in result.columns

    def test_warmup_period(self):
        """Test that first length values are NaN."""
        np.random.seed(42)
        close = np.random.randn(50).cumsum() + 100
        df = pl.DataFrame({"close": close})
        result = reflex_apply(df, close="close", length=20, smooth=20)
        vals = result["REFLEX_20_20_0.04"].to_numpy()[:20]
        assert np.isnan(vals).all()

    def test_custom_parameters(self):
        """Test custom parameters."""
        np.random.seed(42)
        close = np.random.randn(50).cumsum() + 100
        df = pl.DataFrame({"close": close})
        result = reflex_apply(df, close="close", length=15, smooth=15, alpha=0.1)
        assert "REFLEX_15_15_0.1" in result.columns

    def test_with_zeros(self):
        """Handles zero values."""
        df = pl.DataFrame({"close": [0.0] * 10 + [100.0] * 90})
        result = reflex_apply(df, length=20, smooth=20)
        assert result.height == 100

    def test_preserves_original_columns(self):
        """Original columns are preserved."""
        np.random.seed(42)
        close = np.random.randn(50).cumsum() + 100
        df = pl.DataFrame({"close": close, "other": range(50)})
        result = reflex_apply(df, length=20, smooth=20)
        assert "close" in result.columns
        assert "other" in result.columns
        assert "REFLEX_20_20_0.04" in result.columns

    def test_oscillator_bounded(self):
        """Reflex output should be bounded (normalized)."""
        np.random.seed(42)
        close = np.random.randn(100).cumsum() + 100
        df = pl.DataFrame({"close": close})
        result = reflex_apply(df, length=20, smooth=20)
        vals = result["REFLEX_20_20_0.04"].drop_nulls().to_numpy()
        valid = vals[~np.isnan(vals)]
        # Should be roughly bounded like an oscillator
        assert np.abs(valid).max() <= 5  # Max normalized value is 5
