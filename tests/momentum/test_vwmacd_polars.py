# -*- coding: utf-8 -*-
"""Tests for Polars VWMACD (Volume Weighted MACD) implementation."""

import numpy as np
import polars as pl
import pytest

from polars_ti.momentum.vwmacd import pl_vwmacd


class TestPlVwmacd:
    """Test suite for pl_vwmacd function."""

    @pytest.fixture
    def sample_data(self) -> pl.DataFrame:
        """Create sample OHLCV data for testing."""
        np.random.seed(42)
        n = 500
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        volume = np.random.randint(1000, 10000, n).astype(float)
        return pl.DataFrame({"close": close, "volume": volume})

    def test_basic_output(self, sample_data: pl.DataFrame) -> None:
        """Test that pl_vwmacd returns correct number of expressions."""
        result = sample_data.select(pl_vwmacd("close", "volume"))
        
        assert result.shape[1] == 3, "Should return 3 columns"
        assert "VWMACD_12_26_9" in result.columns
        assert "VWMACDh_12_26_9" in result.columns
        assert "VWMACDs_12_26_9" in result.columns

    def test_custom_parameters(self, sample_data: pl.DataFrame) -> None:
        """Test pl_vwmacd with custom fast/slow/signal parameters."""
        result = sample_data.select(pl_vwmacd("close", "volume", fast=10, slow=20, signal=5))
        
        assert "VWMACD_10_20_5" in result.columns
        assert "VWMACDh_10_20_5" in result.columns
        assert "VWMACDs_10_20_5" in result.columns

    def test_fast_slow_swap(self, sample_data: pl.DataFrame) -> None:
        """Test that fast > slow causes them to swap."""
        result = sample_data.select(pl_vwmacd("close", "volume", fast=30, slow=20, signal=9))
        
        assert "VWMACD_20_30_9" in result.columns

    def test_histogram_is_difference(self, sample_data: pl.DataFrame) -> None:
        """Test that histogram = VWMACD - Signal."""
        result = sample_data.select(pl_vwmacd("close", "volume"))
        
        vwmacd = result["VWMACD_12_26_9"].to_numpy()
        signal = result["VWMACDs_12_26_9"].to_numpy()
        histogram = result["VWMACDh_12_26_9"].to_numpy()
        
        mask = ~np.isnan(vwmacd) & ~np.isnan(signal) & ~np.isnan(histogram)
        np.testing.assert_allclose(
            histogram[mask],
            vwmacd[mask] - signal[mask],
            rtol=1e-10
        )

    def test_offset_parameter(self, sample_data: pl.DataFrame) -> None:
        """Test offset shifts all outputs."""
        result_no_offset = sample_data.select(pl_vwmacd("close", "volume", offset=0))
        result_with_offset = sample_data.select(pl_vwmacd("close", "volume", offset=5))
        
        vwmacd_no = result_no_offset["VWMACD_12_26_9"].to_numpy()
        vwmacd_with = result_with_offset["VWMACD_12_26_9"].to_numpy()
        
        # Offset version should have more leading NaNs
        mask_no = ~np.isnan(vwmacd_no)
        mask_with = ~np.isnan(vwmacd_with)
        
        assert np.sum(mask_no) > np.sum(mask_with)

    def test_lazy_evaluation(self, sample_data: pl.DataFrame) -> None:
        """Test that pl_vwmacd works in lazy context."""
        lazy_df = sample_data.lazy()
        result = lazy_df.select(pl_vwmacd("close", "volume")).collect()
        
        assert result.shape[1] == 3
        assert not result["VWMACD_12_26_9"].is_empty()

    def test_null_handling(self) -> None:
        """Test pl_vwmacd handles data gracefully."""
        n = 200
        close = [100.0 + i * 0.1 for i in range(n)]
        volume = [1000.0 + i for i in range(n)]
        
        df = pl.DataFrame({
            "close": close,
            "volume": volume,
        })
        
        result = df.select(pl_vwmacd("close", "volume"))
        
        assert result.shape[0] == n
        # First value should be NaN due to warmup
        vwmacd = result["VWMACD_12_26_9"].to_numpy()
        assert np.isnan(vwmacd[0])

    def test_expr_input(self, sample_data: pl.DataFrame) -> None:
        """Test pl_vwmacd accepts pl.Expr as input."""
        result = sample_data.select(pl_vwmacd(pl.col("close"), pl.col("volume")))
        
        assert result.shape[1] == 3

    def test_column_order(self, sample_data: pl.DataFrame) -> None:
        """Test that column order matches Pandas (VWMACD, Histogram, Signal)."""
        result = sample_data.select(pl_vwmacd("close", "volume"))
        
        # Pandas order: VWMACD, VWMACDh, VWMACDs
        assert result.columns[0] == "VWMACD_12_26_9"
        assert result.columns[1] == "VWMACDh_12_26_9"
        assert result.columns[2] == "VWMACDs_12_26_9"
