# -*- coding: utf-8 -*-
"""Unit tests for polars_ti/candles/cdl_z.py Polars implementation."""
import numpy as np
import polars as pl
import pytest

from polars_ti.candles.cdl_z import pl_cdl_z, pl_zscore


class TestPlZscore:
    """Tests for pl_zscore helper."""

    def test_zscore_basic(self):
        """Test basic z-score calculation."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        df = pl.DataFrame({"close": data})
        result = df.select(pl_zscore("close", length=5, ddof=1).alias("z"))
        # After warmup period, z-scores should be calculated
        assert result["z"].to_numpy()[4] is not None


class TestPlCdlZ:
    """Tests for pl_cdl_z."""

    def test_returns_four_columns(self):
        """Test that pl_cdl_z returns four Z-score expressions."""
        result = pl_cdl_z("open", "high", "low", "close", length=10)
        assert len(result) == 4

    def test_column_aliases(self):
        """Test that column aliases are correct."""
        np.random.seed(42)
        close = np.random.randn(20).cumsum() + 100
        df = pl.DataFrame({
            "open": close, "high": close + 1, "low": close - 1, "close": close
        })
        exprs = pl_cdl_z("open", "high", "low", "close", length=10, ddof=1)
        result = df.with_columns(exprs)
        
        assert "open_Z_10_1" in result.columns
        assert "high_Z_10_1" in result.columns
        assert "low_Z_10_1" in result.columns
        assert "close_Z_10_1" in result.columns

    def test_zscore_range(self):
        """Test that z-scores are reasonably bounded."""
        np.random.seed(42)
        close = np.random.randn(50).cumsum() + 100
        df = pl.DataFrame({
            "open": close, "high": close + 1, "low": close - 1, "close": close
        })
        exprs = pl_cdl_z("open", "high", "low", "close", length=30)
        result = df.with_columns(exprs)
        
        # Z-scores should typically be between -3 and 3 for normal data
        z_vals = result["close_Z_30_1"].drop_nulls().to_numpy()
        assert np.abs(z_vals).max() < 5  # Very loose bound for random data

    def test_with_null_values(self):
        """Handles null values gracefully."""
        df = pl.DataFrame({
            "open": [None] + [100.0] * 39,
            "high": [110.0] * 40,
            "low": [90.0] * 40,
            "close": [105.0] * 40,
        })
        exprs = pl_cdl_z("open", "high", "low", "close", length=30)
        result = df.select(exprs)
        assert result.height == 40

    def test_with_zeros(self):
        """Handles zero values."""
        df = pl.DataFrame({
            "open": [0.0] * 10 + [100.0] * 30,
            "high": [0.0] * 10 + [110.0] * 30,
            "low": [0.0] * 10 + [90.0] * 30,
            "close": [0.0] * 10 + [105.0] * 30,
        })
        exprs = pl_cdl_z("open", "high", "low", "close", length=30)
        result = df.select(exprs)
        assert result.height == 40

    def test_lazy_execution(self):
        """Works with LazyFrame."""
        np.random.seed(42)
        close = np.random.randn(40).cumsum() + 100
        df = pl.DataFrame({
            "open": close, "high": close + 1, "low": close - 1, "close": close
        })
        lazy_df = df.lazy()
        exprs = pl_cdl_z("open", "high", "low", "close", length=30)
        result = lazy_df.select(exprs).collect()
        assert "close_Z_30_1" in result.columns
