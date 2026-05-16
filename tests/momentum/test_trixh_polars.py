# -*- coding: utf-8 -*-
"""Tests for Polars TRIXH (TRIX Histogram) implementation."""

import numpy as np
import polars as pl
import pytest

from polars_ti.momentum.trixh import pl_trixh


class TestPlTrixh:
    """Test suite for pl_trixh function."""

    @pytest.fixture
    def sample_data(self) -> pl.DataFrame:
        """Create sample OHLCV data for testing."""
        np.random.seed(42)
        n = 500
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        return pl.DataFrame({"close": close})

    def test_basic_output(self, sample_data: pl.DataFrame) -> None:
        """Test that pl_trixh returns correct number of expressions."""
        result = sample_data.select(pl_trixh("close"))
        
        assert result.shape[1] == 3, "Should return 3 columns"
        assert "TRIX_18_9" in result.columns
        assert "TRIXs_18_9" in result.columns
        assert "TRIXh_18_9" in result.columns

    def test_custom_parameters(self, sample_data: pl.DataFrame) -> None:
        """Test pl_trixh with custom length/signal parameters."""
        result = sample_data.select(pl_trixh("close", length=14, signal=7))
        
        assert "TRIX_14_7" in result.columns
        assert "TRIXs_14_7" in result.columns
        assert "TRIXh_14_7" in result.columns

    def test_length_signal_swap(self, sample_data: pl.DataFrame) -> None:
        """Test that length < signal causes them to swap (matching pandas)."""
        # length=5 < signal=10 should swap to length=10, signal=5
        result = sample_data.select(pl_trixh("close", length=5, signal=10))
        
        assert "TRIX_10_5" in result.columns
        assert "TRIXs_10_5" in result.columns
        assert "TRIXh_10_5" in result.columns

    def test_histogram_is_difference(self, sample_data: pl.DataFrame) -> None:
        """Test that histogram = TRIX - Signal."""
        result = sample_data.select(pl_trixh("close", talib=False))
        
        trix = result["TRIX_18_9"].to_numpy()
        signal = result["TRIXs_18_9"].to_numpy()
        histogram = result["TRIXh_18_9"].to_numpy()
        
        # Compare where not NaN
        mask = ~np.isnan(trix) & ~np.isnan(signal) & ~np.isnan(histogram)
        expected = trix[mask] - signal[mask]
        actual = histogram[mask]
        
        np.testing.assert_allclose(actual, expected, rtol=1e-10)

    def test_offset_parameter(self, sample_data: pl.DataFrame) -> None:
        """Test offset shifts all outputs."""
        result_no_offset = sample_data.select(pl_trixh("close", talib=False, offset=0))
        result_with_offset = sample_data.select(pl_trixh("close", talib=False, offset=5))
        
        trix_no = result_no_offset["TRIX_18_9"].to_numpy()
        trix_with = result_with_offset["TRIX_18_9"].to_numpy()
        
        # The offset version should have 5 more leading NaNs
        warmup = 3 * 18 + 1  # EMA warmup + drift
        mask_no = ~np.isnan(trix_no)
        mask_with = ~np.isnan(trix_with)
        
        # Offset version has more NaNs at the start
        assert np.sum(mask_no) > np.sum(mask_with)

    def test_scalar_parameter(self, sample_data: pl.DataFrame) -> None:
        """Test that scalar multiplier affects output magnitude."""
        result_100 = sample_data.select(pl_trixh("close", scalar=100.0, talib=False))
        result_1 = sample_data.select(pl_trixh("close", scalar=1.0, talib=False))
        
        trix_100 = result_100["TRIX_18_9"].to_numpy()
        trix_1 = result_1["TRIX_18_9"].to_numpy()
        
        # Remove NaN for comparison
        mask = ~np.isnan(trix_100) & ~np.isnan(trix_1)
        
        # Should be ~100x different
        np.testing.assert_allclose(
            trix_100[mask], 
            trix_1[mask] * 100, 
            rtol=1e-10
        )

    def test_lazy_evaluation(self, sample_data: pl.DataFrame) -> None:
        """Test that pl_trixh works in lazy context."""
        lazy_df = sample_data.lazy()
        result = lazy_df.select(pl_trixh("close", talib=False)).collect()
        
        assert result.shape[1] == 3
        assert not result["TRIX_18_9"].is_empty()

    def test_null_handling(self) -> None:
        """Test pl_trixh handles null values gracefully."""
        df = pl.DataFrame({
            "close": [100.0, None, 102.0, 103.0, None] + [100.0 + i * 0.1 for i in range(495)]
        })
        
        result = df.select(pl_trixh("close", talib=False))
        
        # Should not raise, and should have some valid values
        assert result.shape[0] == 500
        assert result["TRIX_18_9"].null_count() > 0  # Some nulls expected due to warmup

    def test_expr_input(self, sample_data: pl.DataFrame) -> None:
        """Test pl_trixh accepts pl.Expr as input."""
        result = sample_data.select(pl_trixh(pl.col("close"), talib=False))
        
        assert result.shape[1] == 3

    def test_talib_parameter(self, sample_data: pl.DataFrame) -> None:
        """Test that talib parameter controls TA-Lib usage."""
        # Both should work regardless of TA-Lib availability
        result_talib = sample_data.select(pl_trixh("close", talib=True))
        result_pure = sample_data.select(pl_trixh("close", talib=False))
        
        assert result_talib.shape[1] == 3
        assert result_pure.shape[1] == 3

    def test_drift_parameter(self, sample_data: pl.DataFrame) -> None:
        """Test that drift parameter affects TRIX calculation."""
        result_drift1 = sample_data.select(pl_trixh("close", drift=1, talib=False))
        result_drift2 = sample_data.select(pl_trixh("close", drift=2, talib=False))
        
        trix1 = result_drift1["TRIX_18_9"].to_numpy()
        trix2 = result_drift2["TRIX_18_9"].to_numpy()
        
        # Different drift should give different results
        mask = ~np.isnan(trix1) & ~np.isnan(trix2)
        # Values should be different (not exactly equal)
        assert not np.allclose(trix1[mask][:100], trix2[mask][:100], rtol=1e-6)
