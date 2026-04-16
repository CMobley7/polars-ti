# -*- coding: utf-8 -*-
"""Tests for pl_stoch - Polars + Numba Stochastic Oscillator with TA-Lib support."""
import numpy as np
import polars as pl
import pandas as pd
import pytest
from polars_ti.momentum.stoch import pl_stoch


class TestPlStoch:
    @pytest.fixture
    def sample_df(self):
        np.random.seed(42)
        n = 100
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n) * 0.5)
        low = close - np.abs(np.random.randn(n) * 0.5)
        return pl.DataFrame({'high': high, 'low': low, 'close': close})

    def test_returns_expr(self):
        expr = pl_stoch("high", "low", "close")
        assert isinstance(expr, pl.Expr)

    def test_has_stoch_column(self, sample_df):
        result = sample_df.select(pl_stoch("high", "low", "close"))
        assert "STOCH" in result.columns

    def test_struct_has_k_d_h_fields(self, sample_df):
        result = sample_df.select(pl_stoch("high", "low", "close"))
        stoch = result["STOCH"]
        k = 14
        d = 3
        smooth_k = 3
        assert f"STOCHk_{k}_{d}_{smooth_k}" in stoch.struct.fields
        assert f"STOCHd_{k}_{d}_{smooth_k}" in stoch.struct.fields
        assert f"STOCHh_{k}_{d}_{smooth_k}" in stoch.struct.fields

    def test_has_valid_values(self, sample_df):
        result = sample_df.select(pl_stoch("high", "low", "close"))
        stoch = result["STOCH"]
        k_vals = stoch.struct.field("STOCHk_14_3_3")
        # After warmup period, values should be valid
        assert k_vals[25:].null_count() == 0

    def test_offset_parameter(self, sample_df):
        result = sample_df.select(pl_stoch("high", "low", "close", offset=5))
        stoch = result["STOCH"]
        k_vals = stoch.struct.field("STOCHk_14_3_3")
        # First 5 should be null from offset
        assert k_vals[:5].null_count() == 5

    def test_lazy_execution(self, sample_df):
        result = sample_df.lazy().select(pl_stoch("high", "low", "close")).collect()
        assert "STOCH" in result.columns

    def test_talib_true(self, sample_df):
        result = sample_df.select(pl_stoch("high", "low", "close", talib=True))
        assert "STOCH" in result.columns

    def test_talib_false(self, sample_df):
        result = sample_df.select(pl_stoch("high", "low", "close", talib=False))
        assert "STOCH" in result.columns

    def test_custom_k_parameter(self, sample_df):
        result = sample_df.select(pl_stoch("high", "low", "close", k=20))
        stoch = result["STOCH"]
        assert "STOCHk_20_3_3" in stoch.struct.fields

    def test_custom_d_parameter(self, sample_df):
        result = sample_df.select(pl_stoch("high", "low", "close", d=5))
        stoch = result["STOCH"]
        assert "STOCHk_14_5_3" in stoch.struct.fields

    def test_custom_smooth_k_parameter(self, sample_df):
        result = sample_df.select(pl_stoch("high", "low", "close", smooth_k=5))
        stoch = result["STOCH"]
        assert "STOCHk_14_3_5" in stoch.struct.fields

    def test_smooth_k_1_raw_stoch(self, sample_df):
        """Test smooth_k=1 returns raw stochastic."""
        result = sample_df.select(pl_stoch("high", "low", "close", smooth_k=1, talib=False))
        stoch = result["STOCH"]
        assert "STOCHk_14_3_1" in stoch.struct.fields

    def test_with_null_values(self):
        df = pl.DataFrame({
            'high': [101.0, None, 103.0] + [102.0] * 50,
            'low': [99.0, 100.0, None] + [98.0] * 50,
            'close': [100.0, 101.0, 102.0] + [100.0] * 50
        })
        result = df.select(pl_stoch("high", "low", "close", talib=False))
        assert result.height == 53

    def test_with_zeros(self):
        df = pl.DataFrame({
            'high': [0.0] * 10 + [102.0] * 50,
            'low': [0.0] * 10 + [98.0] * 50,
            'close': [0.0] * 10 + [100.0] * 50
        })
        result = df.select(pl_stoch("high", "low", "close", talib=False))
        assert result.height == 60

    def test_numerical_parity(self):
        """Verify numerical parity with Pandas implementation."""
        pytest.skip("Pandas implementation removed in Phase 4 purge")
        import pandas as pd
        # from polars_ti.momentum.stoch import stoch as pandas_stoch  # REMOVED: pandas func removed
        
        np.random.seed(42)
        n = 500
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n) * 0.5)
        low = close - np.abs(np.random.randn(n) * 0.5)
        
        pdf = pd.DataFrame({'high': high, 'low': low, 'close': close})
        pldf = pl.DataFrame({'high': high, 'low': low, 'close': close})
        
        pandas_result = pandas_stoch(pdf['high'], pdf['low'], pdf['close'], 
                                     k=14, d=3, smooth_k=3, talib=False)
        polars_result = pldf.select(pl_stoch("high", "low", "close", 
                                             k=14, d=3, smooth_k=3, talib=False))
        
        stoch = polars_result["STOCH"]
        pandas_k = pandas_result["STOCHk_14_3_3"].to_numpy()[25:]
        polars_k = stoch.struct.field("STOCHk_14_3_3").to_numpy()[25:]
        
        valid_mask = ~np.isnan(pandas_k) & ~np.isnan(polars_k)
        max_diff = np.max(np.abs(pandas_k[valid_mask] - polars_k[valid_mask]))
        assert max_diff < 1e-6, f"Max diff {max_diff} exceeds tolerance"

    def test_talib_parity(self):
        """Verify TA-Lib path produces same results as direct TA-Lib call."""
        try:
            from talib import STOCH as TALIB_STOCH
        except ImportError:
            pytest.skip("TA-Lib not installed")
        
        np.random.seed(42)
        n = 500
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n) * 0.5)
        low = close - np.abs(np.random.randn(n) * 0.5)
        
        pldf = pl.DataFrame({'high': high, 'low': low, 'close': close})
        
        # Direct TA-Lib
        talib_k, talib_d = TALIB_STOCH(high, low, close, 14, 3, 0, 3, 0)
        
        # Polars with talib=True
        polars_result = pldf.select(pl_stoch("high", "low", "close", talib=True))
        stoch = polars_result["STOCH"]
        polars_k = stoch.struct.field("STOCHk_14_3_3").to_numpy()
        
        valid_mask = ~np.isnan(talib_k) & ~np.isnan(polars_k)
        max_diff = np.max(np.abs(talib_k[valid_mask] - polars_k[valid_mask]))
        assert max_diff < 1e-6, f"Max diff {max_diff} exceeds tolerance"
