# -*- coding: utf-8 -*-
"""Tests for pl_atr."""

import numpy as np
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

    def test_talib_option(self, sample_data):
        """Test TA-Lib path if available."""
        pl_df = pl.DataFrame(sample_data)
        try:
            from talib import ATR as TALIB_ATR

            talib_result = TALIB_ATR(
                sample_data["high"],
                sample_data["low"],
                sample_data["close"],
                timeperiod=14,
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
