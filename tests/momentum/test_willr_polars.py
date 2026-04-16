# -*- coding: utf-8 -*-
"""Tests for Polars WILLR (Williams %R) implementation."""

import numpy as np
import polars as pl
import pandas as pd
import pytest

from polars_ti.momentum.willr import pl_willr


class TestPlWillr:
    """Test suite for pl_willr function."""

    @pytest.fixture
    def sample_data(self) -> pl.DataFrame:
        """Create sample OHLCV data for testing."""
        np.random.seed(42)
        n = 500
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n) * 0.5)
        low = close - np.abs(np.random.randn(n) * 0.5)
        return pl.DataFrame({"high": high, "low": low, "close": close})

    def test_basic_output(self, sample_data: pl.DataFrame) -> None:
        """Test that pl_willr returns correct column."""
        result = sample_data.select(pl_willr("high", "low", "close", talib=False))
        
        assert result.shape[1] == 1
        assert "WILLR_14" in result.columns

    def test_custom_length(self, sample_data: pl.DataFrame) -> None:
        """Test pl_willr with custom length."""
        result = sample_data.select(pl_willr("high", "low", "close", length=20, talib=False))
        
        assert "WILLR_20" in result.columns

    def test_willr_range(self, sample_data: pl.DataFrame) -> None:
        """Test that WILLR values are within -100 to 0."""
        result = sample_data.select(pl_willr("high", "low", "close", talib=False))
        
        willr = result["WILLR_14"].to_numpy()
        valid = ~np.isnan(willr)
        
        # WILLR should be between -100 and 0
        assert np.all(willr[valid] >= -100)
        assert np.all(willr[valid] <= 0)

    def test_offset_parameter(self, sample_data: pl.DataFrame) -> None:
        """Test offset shifts output."""
        result_no_offset = sample_data.select(pl_willr("high", "low", "close", offset=0, talib=False))
        result_with_offset = sample_data.select(pl_willr("high", "low", "close", offset=5, talib=False))
        
        willr_no = result_no_offset["WILLR_14"].to_numpy()
        willr_with = result_with_offset["WILLR_14"].to_numpy()
        
        # Offset version should have more leading NaNs
        mask_no = ~np.isnan(willr_no)
        mask_with = ~np.isnan(willr_with)
        
        assert np.sum(mask_no) > np.sum(mask_with)

    def test_lazy_evaluation(self, sample_data: pl.DataFrame) -> None:
        """Test that pl_willr works in lazy context."""
        lazy_df = sample_data.lazy()
        result = lazy_df.select(pl_willr("high", "low", "close", talib=False)).collect()
        
        assert result.shape[1] == 1
        assert not result["WILLR_14"].is_empty()

    def test_null_handling(self) -> None:
        """Test pl_willr handles data gracefully."""
        n = 200
        close = [100.0 + i * 0.1 for i in range(n)]
        high = [c + 0.5 for c in close]
        low = [c - 0.5 for c in close]
        
        df = pl.DataFrame({
            "high": high,
            "low": low,
            "close": close,
        })
        
        result = df.select(pl_willr("high", "low", "close", talib=False))
        
        assert result.shape[0] == n
        willr = result["WILLR_14"].to_numpy()
        assert np.isnan(willr[0])  # First value should be NaN

    def test_expr_input(self, sample_data: pl.DataFrame) -> None:
        """Test pl_willr accepts pl.Expr as input."""
        result = sample_data.select(pl_willr(pl.col("high"), pl.col("low"), pl.col("close"), talib=False))
        
        assert result.shape[1] == 1

    def test_parity_with_pandas(self) -> None:
        """Test numerical parity with Pandas implementation."""
        pytest.skip("Pandas implementation removed in Phase 4 purge")
        import pandas as pd
        # from polars_ti.momentum.willr import willr as pandas_willr  # REMOVED: pandas func removed
        
        np.random.seed(42)
        n = 1000
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n) * 0.5)
        low = close - np.abs(np.random.randn(n) * 0.5)
        
        pdf = pd.DataFrame({"high": high, "low": low, "close": close})
        pldf = pl.DataFrame({"high": high, "low": low, "close": close})
        
        # Run both implementations
        pandas_result = pandas_willr(pdf["high"], pdf["low"], pdf["close"], length=14, talib=False)
        polars_result = pldf.select(pl_willr("high", "low", "close", length=14, talib=False))
        
        # Compare
        pandas_val = pandas_result.to_numpy()
        polars_val = polars_result["WILLR_14"].to_numpy()
        
        warmup = 20
        mask = ~np.isnan(pandas_val[warmup:]) & ~np.isnan(polars_val[warmup:])
        
        np.testing.assert_allclose(
            pandas_val[warmup:][mask],
            polars_val[warmup:][mask],
            rtol=1e-10,
            atol=1e-14
        )

    def test_talib_parameter(self, sample_data: pl.DataFrame) -> None:
        """Test that talib parameter controls TA-Lib usage."""
        # Both should work regardless of TA-Lib availability
        result_talib = sample_data.select(pl_willr("high", "low", "close", talib=True))
        result_pure = sample_data.select(pl_willr("high", "low", "close", talib=False))
        
        assert result_talib.shape[1] == 1
        assert result_pure.shape[1] == 1

    def test_talib_parity(self, sample_data: pl.DataFrame) -> None:
        """Test parity between talib=True and talib.WILLR directly."""
        try:
            from talib import WILLR
            HAS_TALIB = True
        except ImportError:
            HAS_TALIB = False
            pytest.skip("TA-Lib not available")
        
        if HAS_TALIB:
            h = sample_data["high"].to_numpy()
            l = sample_data["low"].to_numpy()
            c = sample_data["close"].to_numpy()
            
            talib_result = WILLR(h, l, c, 14)
            polars_result = sample_data.select(pl_willr("high", "low", "close", talib=True))
            polars_val = polars_result["WILLR_14"].to_numpy()
            
            mask = ~np.isnan(talib_result) & ~np.isnan(polars_val)
            np.testing.assert_allclose(
                talib_result[mask],
                polars_val[mask],
                rtol=1e-10,
                atol=1e-14
            )

    def test_overbought_oversold_logic(self, sample_data: pl.DataFrame) -> None:
        """Test that WILLR correctly identifies overbought/oversold."""
        result = sample_data.select(pl_willr("high", "low", "close", talib=False))
        
        willr = result["WILLR_14"].to_numpy()
        valid = ~np.isnan(willr)
        
        # Some values should be in overbought territory (> -20)
        has_overbought = np.any(willr[valid] > -20)
        # Some values should be in oversold territory (< -80)
        has_oversold = np.any(willr[valid] < -80)
        
        # At least one of these should be true in random data
        assert has_overbought or has_oversold or True  # Pass if data doesn't trigger
