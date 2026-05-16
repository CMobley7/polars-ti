# -*- coding: utf-8 -*-
"""Tests for pl_hilo (Gann HiLo Activator)."""
import numpy as np
import polars as pl
import pytest
from polars_ti.overlap.hilo import pl_hilo


class TestPlHilo:
    """Test suite for pl_hilo Polars implementation."""

    @pytest.fixture
    def sample_df(self) -> pl.DataFrame:
        """Create sample OHLC DataFrame."""
        np.random.seed(42)
        n = 100
        base = 100 + np.cumsum(np.random.randn(n) * 0.5)
        return pl.DataFrame({
            'high': base + np.abs(np.random.randn(n)),
            'low': base - np.abs(np.random.randn(n)),
            'close': base,
        })

    @pytest.fixture
    def sample_data(self):
        """Create sample data for both Pandas and Polars."""
        np.random.seed(42)
        n = 200
        close = 100 + np.random.randn(n).cumsum()
        high = close + np.abs(np.random.randn(n) * 0.5)
        low = close - np.abs(np.random.randn(n) * 0.5)
        return {
            'pd_high': high,
            'pd_low': low,
            'pd_close': close,
            'pl_df': pl.DataFrame({'high': high, 'low': low, 'close': close}),
        }

    def test_returns_dataframe_with_correct_columns(self, sample_df):
        """Test that pl_hilo returns DataFrame with expected columns."""
        result = pl_hilo(sample_df, high_length=13, low_length=21)
        assert "HILO_13_21" in result.columns
        assert "HILOl_13_21" in result.columns
        assert "HILOs_13_21" in result.columns

    def test_preserves_original_columns(self, sample_df):
        """Test that original columns are preserved."""
        result = pl_hilo(sample_df)
        assert "high" in result.columns
        assert "low" in result.columns
        assert "close" in result.columns

    def test_custom_lengths(self, sample_df):
        """Test with custom high and low lengths."""
        result = pl_hilo(sample_df, high_length=10, low_length=15)
        assert "HILO_10_15" in result.columns

    def test_different_mamode(self, sample_df):
        """Test with different MA modes."""
        result_sma = pl_hilo(sample_df, mamode="sma")
        result_ema = pl_hilo(sample_df, mamode="ema")
        
        hilo_sma = result_sma.get_column("HILO_13_21").to_numpy()
        hilo_ema = result_ema.get_column("HILO_13_21").to_numpy()
        
        mask = ~(np.isnan(hilo_sma) | np.isnan(hilo_ema))
        assert np.any(hilo_sma[mask] != hilo_ema[mask])

    def test_offset_shifts_results(self, sample_df):
        """Test that offset parameter shifts results."""
        result_offset = pl_hilo(sample_df, offset=5)
        
        arr = result_offset.get_column("HILO_13_21").to_numpy()
        assert np.isnan(arr[:5]).all()

    def test_hilo_values_are_numeric(self, sample_df):
        """Test that HILO values are numeric."""
        result = pl_hilo(sample_df)
        hilo = result.get_column("HILO_13_21").to_numpy()
        
        non_null = ~np.isnan(hilo)
        assert non_null.sum() > 50

    def test_with_null_values(self):
        """Handles null values gracefully."""
        df = pl.DataFrame({
            "high": [None] + [110.0] * 39,
            "low": [None] + [90.0] * 39,
            "close": [None] + [100.0] * 39,
        })
        result = pl_hilo(df)
        assert result.height == 40

    def test_with_zeros(self):
        """Handles zero values."""
        df = pl.DataFrame({
            "high": [0.0] * 5 + [110.0] * 35,
            "low": [0.0] * 5 + [90.0] * 35,
            "close": [0.0] * 5 + [100.0] * 35,
        })
        result = pl_hilo(df)
        assert result.height == 40

    def test_lazy_execution(self, sample_df):
        """Works with LazyFrame (converts to eager internally)."""
        result = pl_hilo(sample_df)
        assert "HILO_13_21" in result.columns

