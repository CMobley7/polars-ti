# -*- coding: utf-8 -*-
"""Tests for pl_accbands."""
import numpy as np
import pandas as pd  # REMOVED: pandas dependency  # Restored for fixtures
import polars as pl
import pytest
from polars_ti.volatility.accbands import pl_accbands


class TestPlAccbands:
    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        n = 100
        high = 100 + np.cumsum(np.random.randn(n) * 0.5) + np.abs(np.random.randn(n) * 0.3)
        low = high - np.abs(np.random.randn(n) * 0.5) - 0.2
        close = (high + low) / 2 + np.random.randn(n) * 0.1
        return {"high": high, "low": low, "close": close}

    def test_returns_expression(self, sample_data):
        result = pl_accbands("high", "low", "close")
        assert isinstance(result, pl.Expr)

    def test_output_has_correct_alias(self, sample_data):
        df = pl.DataFrame(sample_data)
        result = df.select(pl_accbands("high", "low", "close", length=20))
        assert "ACCBANDS" in result.columns[0]

    def test_numerical_parity_pandas(self, sample_data):
        pytest.skip("Pandas implementation removed in Phase 4 purge")
        pd_high = pd.Series(sample_data["high"])
        pd_low = pd.Series(sample_data["low"])
        pd_close = pd.Series(sample_data["close"])
        pl_df = pl.DataFrame(sample_data)
        
        pd_result = accbands(pd_high, pd_low, pd_close, length=20, c=4, mamode='sma')
        pl_result = pl_df.select(pl_accbands("high", "low", "close", length=20, c=4, mamode='sma', talib=False))
        pl_unnest = pl_result.unnest(pl_result.columns[0])
        
        warmup = 25
        for col in ["ACCBL_20", "ACCBM_20", "ACCBU_20"]:
            pd_vals = pd_result[col].to_numpy()[warmup:]
            pl_vals = pl_unnest[col].to_numpy()[warmup:]
            mask = np.isfinite(pd_vals) & np.isfinite(pl_vals)
            if mask.sum() > 0:
                max_diff = np.abs(pd_vals[mask] - pl_vals[mask]).max()
                assert max_diff < 1e-6, f"{col} parity failed: {max_diff}"

    def test_talib_option(self, sample_data):
        """Test TA-Lib path if available."""
        pl_df = pl.DataFrame(sample_data)
        try:
            from talib import ACCBANDS
            talib_upper, talib_mid, talib_lower = ACCBANDS(
                sample_data["high"], sample_data["low"], sample_data["close"], timeperiod=20
            )
            pl_result = pl_df.select(pl_accbands("high", "low", "close", length=20, talib=True))
            pl_unnest = pl_result.unnest(pl_result.columns[0])
            warmup = 25
            for name, talib_arr in [("ACCBL_20", talib_lower), ("ACCBM_20", talib_mid), ("ACCBU_20", talib_upper)]:
                pl_vals = pl_unnest[name].to_numpy()[warmup:]
                talib_vals = talib_arr[warmup:]
                mask = np.isfinite(talib_vals) & np.isfinite(pl_vals)
                if mask.sum() > 0:
                    max_diff = np.abs(talib_vals[mask] - pl_vals[mask]).max()
                    assert max_diff < 1e-6, f"{name} talib parity failed: {max_diff}"
        except ImportError:
            pytest.skip("TA-Lib not installed")

    def test_mamode_ema(self, sample_data):
        """Test EMA mode for feature parity."""
        pytest.skip("Pandas implementation removed in Phase 4 purge")
        pd_high = pd.Series(sample_data["high"])
        pd_low = pd.Series(sample_data["low"])
        pd_close = pd.Series(sample_data["close"])
        pl_df = pl.DataFrame(sample_data)
        
        pd_result = accbands(pd_high, pd_low, pd_close, length=20, c=4, mamode='ema')
        pl_result = pl_df.select(pl_accbands("high", "low", "close", length=20, c=4, mamode='ema', talib=False))
        pl_unnest = pl_result.unnest(pl_result.columns[0])
        
        warmup = 25
        for col in ["ACCBL_20", "ACCBM_20", "ACCBU_20"]:
            pd_vals = pd_result[col].to_numpy()[warmup:]
            pl_vals = pl_unnest[col].to_numpy()[warmup:]
            mask = np.isfinite(pd_vals) & np.isfinite(pl_vals)
            if mask.sum() > 0:
                max_diff = np.abs(pd_vals[mask] - pl_vals[mask]).max()
                assert max_diff < 1e-6, f"{col} EMA parity failed: {max_diff}"

    def test_with_null_values(self, sample_data):
        data = sample_data.copy()
        data["close"] = data["close"].copy()
        data["close"][10:15] = np.nan
        df = pl.DataFrame(data)
        result = df.select(pl_accbands("high", "low", "close"))
        assert result.height == 100

    def test_with_zeros(self, sample_data):
        data = sample_data.copy()
        data["low"] = data["low"].copy()
        data["low"][50:55] = 50.0
        df = pl.DataFrame(data)
        result = df.select(pl_accbands("high", "low", "close"))
        assert result.height == 100

    def test_lazy_execution(self, sample_data):
        df = pl.DataFrame(sample_data)
        result = df.lazy().select(pl_accbands("high", "low", "close")).collect()
        assert result.height == 100
