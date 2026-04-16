# -*- coding: utf-8 -*-
"""Tests for pl_ppo (Percentage Price Oscillator)."""
import numpy as np
import polars as pl
import pytest
import pandas as pd  # REMOVED: pandas dependency  # Restored for fixtures
from polars_ti.momentum.ppo import pl_ppo


class TestPlPpo:
    @pytest.fixture
    def sample_df(self) -> pl.DataFrame:
        np.random.seed(42)
        n = 100
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        return pl.DataFrame({'close': close})

    def test_returns_expression(self, sample_df):
        """Test that pl_ppo returns a valid expression."""
        result = sample_df.select(pl_ppo("close"))
        assert result.height == 100

    def test_output_struct_columns(self, sample_df):
        """Test that output struct has correct columns."""
        result = sample_df.select(pl_ppo("close", fast=12, slow=26, signal=9))
        unnested = result.unnest(result.columns[0])
        assert "PPO_12_26_9" in unnested.columns
        assert "PPOs_12_26_9" in unnested.columns
        assert "PPOh_12_26_9" in unnested.columns

    def test_numerical_parity_pandas(self, sample_df):
        """Numerical parity with Pandas implementation."""
        # Create pandas data directly
        pytest.skip("Pandas implementation removed in Phase 4 purge")
        np.random.seed(42)
        n = 100
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        sample_pdf = pd.DataFrame({'close': close})
        
        # Polars result (native)
        polars_result = sample_df.select(pl_ppo("close", talib=False))
        unnested = polars_result.unnest(polars_result.columns[0])
        polars_ppo = unnested["PPO_12_26_9"].to_numpy()
        
        # Pandas result
        pandas_result = ppo(sample_pdf['close'], talib=False)
        pandas_ppo = pandas_result.iloc[:, 0].to_numpy()
        
        # Compare after warmup
        warmup = 35
        valid_polars = polars_ppo[warmup:]
        valid_pandas = pandas_ppo[warmup:]
        
        mask = ~np.isnan(valid_polars) & ~np.isnan(valid_pandas)
        max_diff = np.max(np.abs(valid_polars[mask] - valid_pandas[mask]))
        
        assert max_diff < 1e-6, f"Max diff {max_diff} exceeds tolerance 1e-6"

    def test_talib_parameter_toggle(self, sample_df):
        """Test that both talib=True and talib=False work."""
        result_talib = sample_df.select(pl_ppo("close", talib=True))
        result_native = sample_df.select(pl_ppo("close", talib=False))
        
        # Both should produce valid results
        unnested_talib = result_talib.unnest(result_talib.columns[0])
        unnested_native = result_native.unnest(result_native.columns[0])
        
        ppo_talib = unnested_talib["PPO_12_26_9"].to_numpy()
        ppo_native = unnested_native["PPO_12_26_9"].to_numpy()
        
        # Both should have valid values
        assert (~np.isnan(ppo_talib)).sum() > 50
        assert (~np.isnan(ppo_native)).sum() > 50

    def test_with_null_values(self):
        """Test handling of null values."""
        df = pl.DataFrame({
            "close": [None] + [100.0] * 49
        })
        result = df.select(pl_ppo("close"))
        assert result.height == 50

    def test_with_zero_values(self):
        """Test handling of zero prices."""
        df = pl.DataFrame({
            "close": [0.0] * 20 + [100.0] * 30
        })
        result = df.select(pl_ppo("close", fast=5, slow=10, signal=3))
        # Should not raise and should handle zeros
        assert result.height == 50

    def test_lazy_execution(self, sample_df):
        """Test that pl_ppo works with lazy frames."""
        lazy_df = sample_df.lazy()
        result = lazy_df.select(pl_ppo("close")).collect()
        assert result.height == 100

    def test_different_ma_modes(self, sample_df):
        """Test different MA modes."""
        for mamode in ["sma", "ema"]:
            result = sample_df.select(pl_ppo("close", mamode=mamode))
            unnested = result.unnest(result.columns[0])
            ppo_vals = unnested["PPO_12_26_9"].to_numpy()
            assert (~np.isnan(ppo_vals)).sum() > 50
