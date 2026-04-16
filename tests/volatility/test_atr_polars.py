# -*- coding: utf-8 -*-
"""Tests for pl_atr."""
import numpy as np
import pandas as pd  # REMOVED: pandas dependency  # Restored for fixtures
import polars as pl
import pytest
from polars_ti.volatility.atr import pl_atr


class TestPlAtr:
    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        n = 100
        high = 100 + np.cumsum(np.random.randn(n) * 0.5) + np.abs(np.random.randn(n) * 0.3)
        low = high - np.abs(np.random.randn(n) * 0.5) - 0.2
        close = (high + low) / 2 + np.random.randn(n) * 0.1
        return {"high": high, "low": low, "close": close}

    def test_returns_expression(self, sample_data):
        result = pl_atr("high", "low", "close")
        assert isinstance(result, pl.Expr)

    def test_output_has_correct_alias(self, sample_data):
        df = pl.DataFrame(sample_data)
        result = df.select(pl_atr("high", "low", "close", length=14))
        assert "ATR" in result.columns[0]

    def test_numerical_parity_pandas(self, sample_data):
        """Test parity with Pandas ATR (mamode=rma, default)."""
        pytest.skip("Pandas implementation removed in Phase 4 purge")
        pd_high = pd.Series(sample_data["high"])
        pd_low = pd.Series(sample_data["low"])
        pd_close = pd.Series(sample_data["close"])
        pl_df = pl.DataFrame(sample_data)
        
        pd_result = atr(pd_high, pd_low, pd_close, length=14, mamode='rma', talib=False, presma=True)
        pl_result = pl_df.select(pl_atr("high", "low", "close", length=14, mamode='rma', talib=False))
        
        warmup = 20
        pd_vals = pd_result.to_numpy()[warmup:]
        pl_vals = pl_result[pl_result.columns[0]].to_numpy()[warmup:]
        mask = np.isfinite(pd_vals) & np.isfinite(pl_vals)
        if mask.sum() > 0:
            max_diff = np.abs(pd_vals[mask] - pl_vals[mask]).max()
            assert max_diff < 1e-6, f"RMA parity failed: {max_diff}"

    def test_talib_option(self, sample_data):
        """Test TA-Lib path if available."""
        pl_df = pl.DataFrame(sample_data)
        try:
            from talib import ATR as TALIB_ATR
            talib_result = TALIB_ATR(
                sample_data["high"], sample_data["low"], sample_data["close"], timeperiod=14
            )
            pl_result = pl_df.select(pl_atr("high", "low", "close", length=14, talib=True))
            warmup = 20
            talib_vals = talib_result[warmup:]
            pl_vals = pl_result[pl_result.columns[0]].to_numpy()[warmup:]
            mask = np.isfinite(talib_vals) & np.isfinite(pl_vals)
            if mask.sum() > 0:
                max_diff = np.abs(talib_vals[mask] - pl_vals[mask]).max()
                assert max_diff < 1e-6, f"TA-Lib parity failed: {max_diff}"
        except ImportError:
            pytest.skip("TA-Lib not installed")

    def test_mamode_sma(self, sample_data):
        """Test SMA mode for feature parity."""
        pytest.skip("Pandas implementation removed in Phase 4 purge")
        pd_high = pd.Series(sample_data["high"])
        pd_low = pd.Series(sample_data["low"])
        pd_close = pd.Series(sample_data["close"])
        pl_df = pl.DataFrame(sample_data)
        
        pd_result = atr(pd_high, pd_low, pd_close, length=14, mamode='sma', talib=False, presma=True)
        pl_result = pl_df.select(pl_atr("high", "low", "close", length=14, mamode='sma', talib=False))
        
        warmup = 30
        pd_vals = pd_result.to_numpy()[warmup:]
        pl_vals = pl_result[pl_result.columns[0]].to_numpy()[warmup:]
        mask = np.isfinite(pd_vals) & np.isfinite(pl_vals)
        if mask.sum() > 0:
            max_diff = np.abs(pd_vals[mask] - pl_vals[mask]).max()
            assert max_diff < 1e-6, f"SMA parity failed: {max_diff}"

    def test_mamode_ema(self, sample_data):
        """Test EMA mode for feature parity."""
        pytest.skip("Pandas implementation removed in Phase 4 purge")
        pd_high = pd.Series(sample_data["high"])
        pd_low = pd.Series(sample_data["low"])
        pd_close = pd.Series(sample_data["close"])
        pl_df = pl.DataFrame(sample_data)
        
        pd_result = atr(pd_high, pd_low, pd_close, length=14, mamode='ema', talib=False, presma=True)
        pl_result = pl_df.select(pl_atr("high", "low", "close", length=14, mamode='ema', talib=False))
        
        warmup = 20
        pd_vals = pd_result.to_numpy()[warmup:]
        pl_vals = pl_result[pl_result.columns[0]].to_numpy()[warmup:]
        mask = np.isfinite(pd_vals) & np.isfinite(pl_vals)
        if mask.sum() > 0:
            max_diff = np.abs(pd_vals[mask] - pl_vals[mask]).max()
            assert max_diff < 1e-6, f"EMA parity failed: {max_diff}"

    def test_with_null_values(self, sample_data):
        data = sample_data.copy()
        data["close"] = data["close"].copy()
        data["close"][10:15] = np.nan
        df = pl.DataFrame(data)
        result = df.select(pl_atr("high", "low", "close"))
        assert result.height == 100

    def test_with_zeros(self, sample_data):
        data = sample_data.copy()
        data["low"] = data["low"].copy()
        data["low"][50:55] = 50.0
        df = pl.DataFrame(data)
        result = df.select(pl_atr("high", "low", "close"))
        assert result.height == 100

    def test_lazy_execution(self, sample_data):
        df = pl.DataFrame(sample_data)
        result = df.lazy().select(pl_atr("high", "low", "close")).collect()
        assert result.height == 100
