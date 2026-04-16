# -*- coding: utf-8 -*-
"""Tests for Polars TSI (True Strength Index) implementation."""

import numpy as np
import polars as pl
import pandas as pd
import pytest

from polars_ti.momentum.tsi import pl_tsi


class TestPlTsi:
    """Test suite for pl_tsi function."""

    @pytest.fixture
    def sample_data(self) -> pl.DataFrame:
        """Create sample OHLCV data for testing."""
        np.random.seed(42)
        n = 500
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        return pl.DataFrame({"close": close})

    def test_basic_output(self, sample_data: pl.DataFrame) -> None:
        """Test that pl_tsi returns correct number of expressions."""
        result = sample_data.select(pl_tsi("close"))
        
        assert result.shape[1] == 2, "Should return 2 columns"
        assert "TSI_13_25_13" in result.columns
        assert "TSIs_13_25_13" in result.columns

    def test_custom_parameters(self, sample_data: pl.DataFrame) -> None:
        """Test pl_tsi with custom fast/slow/signal parameters."""
        result = sample_data.select(pl_tsi("close", fast=10, slow=20, signal=10))
        
        assert "TSI_10_20_10" in result.columns
        assert "TSIs_10_20_10" in result.columns

    def test_fast_slow_swap(self, sample_data: pl.DataFrame) -> None:
        """Test that fast > slow causes them to swap (matching pandas)."""
        # fast=30 > slow=20 should swap to fast=20, slow=30
        result = sample_data.select(pl_tsi("close", fast=30, slow=20, signal=10))
        
        assert "TSI_20_30_10" in result.columns
        assert "TSIs_20_30_10" in result.columns

    def test_tsi_range(self, sample_data: pl.DataFrame) -> None:
        """Test that TSI values are typically within -100 to +100."""
        result = sample_data.select(pl_tsi("close"))
        
        tsi = result["TSI_13_25_13"].to_numpy()
        valid = ~np.isnan(tsi)
        
        # TSI should be within -100 to +100 for most values
        assert np.all(tsi[valid] >= -150)  # Allow some margin
        assert np.all(tsi[valid] <= 150)

    def test_offset_parameter(self, sample_data: pl.DataFrame) -> None:
        """Test offset shifts all outputs."""
        result_no_offset = sample_data.select(pl_tsi("close", offset=0))
        result_with_offset = sample_data.select(pl_tsi("close", offset=5))
        
        tsi_no = result_no_offset["TSI_13_25_13"].to_numpy()
        tsi_with = result_with_offset["TSI_13_25_13"].to_numpy()
        
        # Offset version should have more leading NaNs
        mask_no = ~np.isnan(tsi_no)
        mask_with = ~np.isnan(tsi_with)
        
        assert np.sum(mask_no) > np.sum(mask_with)

    def test_scalar_parameter(self, sample_data: pl.DataFrame) -> None:
        """Test that scalar multiplier affects output magnitude."""
        result_100 = sample_data.select(pl_tsi("close", scalar=100.0))
        result_1 = sample_data.select(pl_tsi("close", scalar=1.0))
        
        tsi_100 = result_100["TSI_13_25_13"].to_numpy()
        tsi_1 = result_1["TSI_13_25_13"].to_numpy()
        
        # Remove NaN for comparison
        mask = ~np.isnan(tsi_100) & ~np.isnan(tsi_1)
        
        # Should be ~100x different
        np.testing.assert_allclose(
            tsi_100[mask], 
            tsi_1[mask] * 100, 
            rtol=1e-10
        )

    def test_lazy_evaluation(self, sample_data: pl.DataFrame) -> None:
        """Test that pl_tsi works in lazy context."""
        lazy_df = sample_data.lazy()
        result = lazy_df.select(pl_tsi("close")).collect()
        
        assert result.shape[1] == 2
        assert not result["TSI_13_25_13"].is_empty()

    def test_null_handling(self) -> None:
        """Test pl_tsi handles null values gracefully."""
        # Need longer data for TSI warmup (slow + fast + signal = ~51 periods)
        n = 200
        df = pl.DataFrame({
            "close": [100.0, None, 102.0] + [100.0 + i * 0.1 for i in range(n - 3)]
        })
        
        result = df.select(pl_tsi("close"))
        
        # Should not raise, and should have proper structure
        assert result.shape[0] == n
        tsi = result["TSI_13_25_13"].to_numpy()
        # Should have NaN due to warmup period
        assert np.isnan(tsi[0])  # First value should be NaN

    def test_expr_input(self, sample_data: pl.DataFrame) -> None:
        """Test pl_tsi accepts pl.Expr as input."""
        result = sample_data.select(pl_tsi(pl.col("close")))
        
        assert result.shape[1] == 2

    def test_parity_with_pandas(self) -> None:
        """Test numerical parity with Pandas implementation."""
        pytest.skip("Pandas implementation removed in Phase 4 purge")
        import pandas as pd
        # from polars_ti.momentum.tsi import tsi as pandas_tsi  # REMOVED: pandas func removed
        
        np.random.seed(42)
        n = 1000
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        
        pdf = pd.DataFrame({"close": close})
        pldf = pl.DataFrame({"close": close})
        
        # Run both implementations
        pandas_result = pandas_tsi(pdf["close"], fast=13, slow=25, signal=13, scalar=100.0)
        polars_result = pldf.select(pl_tsi("close", fast=13, slow=25, signal=13, scalar=100.0))
        
        # Compare TSI
        pandas_tsi_val = pandas_result["TSI_13_25_13"].to_numpy()
        polars_tsi_val = polars_result["TSI_13_25_13"].to_numpy()
        
        warmup = 25 + 13 + 13 + 5  # slow + fast + signal + margin
        mask = ~np.isnan(pandas_tsi_val[warmup:]) & ~np.isnan(polars_tsi_val[warmup:])
        
        np.testing.assert_allclose(
            pandas_tsi_val[warmup:][mask],
            polars_tsi_val[warmup:][mask],
            rtol=1e-10,
            atol=1e-14
        )

    def test_mamode_parameter(self, sample_data: pl.DataFrame) -> None:
        """Test that different mamode values work."""
        result_ema = sample_data.select(pl_tsi("close", mamode="ema"))
        result_sma = sample_data.select(pl_tsi("close", mamode="sma"))
        
        assert result_ema.shape[1] == 2
        assert result_sma.shape[1] == 2
        
        # Different mamodes should give different signal values
        sig_ema = result_ema["TSIs_13_25_13"].to_numpy()
        sig_sma = result_sma["TSIs_13_25_13"].to_numpy()
        
        mask = ~np.isnan(sig_ema) & ~np.isnan(sig_sma)
        assert not np.allclose(sig_ema[mask][:50], sig_sma[mask][:50], rtol=1e-6)

    def test_drift_parameter(self, sample_data: pl.DataFrame) -> None:
        """Test that drift parameter affects calculation."""
        result_drift1 = sample_data.select(pl_tsi("close", drift=1))
        result_drift2 = sample_data.select(pl_tsi("close", drift=2))
        
        tsi1 = result_drift1["TSI_13_25_13"].to_numpy()
        tsi2 = result_drift2["TSI_13_25_13"].to_numpy()
        
        # Different drift should give different results
        mask = ~np.isnan(tsi1) & ~np.isnan(tsi2)
        assert not np.allclose(tsi1[mask][:100], tsi2[mask][:100], rtol=1e-6)
