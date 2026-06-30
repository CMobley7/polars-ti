# -*- coding: utf-8 -*-
"""Tests for hilo (Gann HiLo Activator)."""

import numpy as np
import polars as pl
import pytest
from polars_ti.overlap.hilo import hilo


def _hilo(df, **kwargs):
    """Helper: run hilo and return the unnested struct as a flat DataFrame."""
    result = df.select(hilo("high", "low", "close", **kwargs))
    return result.unnest(result.columns[0])


class TestPlHilo:
    """Test suite for the hilo Polars implementation."""

    @pytest.fixture
    def sample_df(self) -> pl.DataFrame:
        """Create sample OHLC DataFrame."""
        np.random.seed(42)
        n = 100
        base = 100 + np.cumsum(np.random.randn(n) * 0.5)
        return pl.DataFrame(
            {
                "high": base + np.abs(np.random.randn(n)),
                "low": base - np.abs(np.random.randn(n)),
                "close": base,
            }
        )

    def test_returns_expression(self, sample_df):
        """hilo() returns a Polars expression."""
        assert isinstance(hilo("high", "low", "close"), pl.Expr)

    def test_returns_dataframe_with_correct_columns(self, sample_df):
        """Test that hilo returns a struct with the expected columns."""
        result = _hilo(sample_df, high_length=13, low_length=21)
        assert "HILO_13_21" in result.columns
        assert "HILOl_13_21" in result.columns
        assert "HILOs_13_21" in result.columns

    def test_preserves_original_columns(self, sample_df):
        """Original columns are preserved when added with with_columns."""
        result = sample_df.with_columns(hilo("high", "low", "close").alias("HILO"))
        assert "high" in result.columns
        assert "low" in result.columns
        assert "close" in result.columns

    def test_custom_lengths(self, sample_df):
        """Test with custom high and low lengths."""
        result = _hilo(sample_df, high_length=10, low_length=15)
        assert "HILO_10_15" in result.columns

    def test_different_mamode(self, sample_df):
        """Test with different MA modes."""
        result_sma = _hilo(sample_df, mamode="sma")
        result_ema = _hilo(sample_df, mamode="ema")

        hilo_sma = result_sma.get_column("HILO_13_21").to_numpy()
        hilo_ema = result_ema.get_column("HILO_13_21").to_numpy()

        mask = ~(np.isnan(hilo_sma) | np.isnan(hilo_ema))
        assert np.any(hilo_sma[mask] != hilo_ema[mask])

    def test_offset_shifts_results(self, sample_df):
        """Test that offset parameter shifts results."""
        result_offset = _hilo(sample_df, offset=5)

        arr = result_offset.get_column("HILO_13_21").to_numpy()
        assert np.isnan(arr[:5]).all()

    def test_hilo_values_are_numeric(self, sample_df):
        """Test that HILO values are numeric."""
        result = _hilo(sample_df)
        hilo_col = result.get_column("HILO_13_21").to_numpy()

        non_null = ~np.isnan(hilo_col)
        assert non_null.sum() > 50

    def test_with_null_values(self):
        """Handles null values gracefully."""
        df = pl.DataFrame(
            {
                "high": [None] + [110.0] * 39,
                "low": [None] + [90.0] * 39,
                "close": [None] + [100.0] * 39,
            }
        )
        result = _hilo(df)
        assert result.height == 40

    def test_with_zeros(self):
        """Handles zero values."""
        df = pl.DataFrame(
            {
                "high": [0.0] * 5 + [110.0] * 35,
                "low": [0.0] * 5 + [90.0] * 35,
                "close": [0.0] * 5 + [100.0] * 35,
            }
        )
        result = _hilo(df)
        assert result.height == 40

    def test_lazy_execution(self, sample_df):
        """Works with LazyFrame."""
        result = sample_df.lazy().select(hilo("high", "low", "close")).collect()
        unnested = result.unnest(result.columns[0])
        assert "HILO_13_21" in unnested.columns
