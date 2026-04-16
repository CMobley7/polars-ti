# -*- coding: utf-8 -*-
"""Tests for pl_chandelier_exit."""
import numpy as np
import pandas as pd  # REMOVED: pandas dependency  # Restored for fixtures
import polars as pl
import pytest
from polars_ti.volatility.chandelier_exit import pl_chandelier_exit


class TestPlChandelierExit:
    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        n = 100
        high = 100 + np.cumsum(np.random.randn(n) * 0.5) + np.abs(np.random.randn(n) * 0.3)
        low = high - np.abs(np.random.randn(n) * 0.5) - 0.2
        close = (high + low) / 2 + np.random.randn(n) * 0.1
        return {"high": high, "low": low, "close": close}

    def test_returns_expression(self, sample_data):
        result = pl_chandelier_exit("high", "low", "close")
        assert isinstance(result, pl.Expr)

    def test_output_has_correct_alias(self, sample_data):
        df = pl.DataFrame(sample_data)
        result = df.select(pl_chandelier_exit("high", "low", "close"))
        assert "CHDLREXT" in result.columns[0]

    def test_numerical_parity(self, sample_data):
        pytest.skip("Pandas implementation removed in Phase 4 purge")
        pd_high = pd.Series(sample_data["high"])
        pd_low = pd.Series(sample_data["low"])
        pd_close = pd.Series(sample_data["close"])
        pl_df = pl.DataFrame(sample_data)
        pd_result = chandelier_exit(pd_high, pd_low, pd_close, high_length=22, low_length=22, atr_length=14, multiplier=2.0, talib=False)
        pl_result = pl_df.select(pl_chandelier_exit("high", "low", "close", high_length=22, low_length=22, atr_length=14, multiplier=2.0)).unnest(pl_df.select(pl_chandelier_exit("high", "low", "close")).columns[0])
        warmup = 20
        pd_long = pd_result.iloc[:, 0].to_numpy()[warmup:]
        pl_long = pl_result["long"].to_numpy()[warmup:]
        mask = np.isfinite(pd_long) & np.isfinite(pl_long)
        max_diff = np.abs(pd_long[mask] - pl_long[mask]).max()
        assert max_diff < 1e-6

    def test_with_null_values(self, sample_data):
        data = sample_data.copy()
        data["close"] = data["close"].copy()
        data["close"][10:15] = np.nan
        df = pl.DataFrame(data)
        result = df.select(pl_chandelier_exit("high", "low", "close"))
        assert result.height == 100

    def test_with_zeros(self, sample_data):
        data = sample_data.copy()
        data["close"] = data["close"].copy()
        data["close"][50:55] = 50.0
        df = pl.DataFrame(data)
        result = df.select(pl_chandelier_exit("high", "low", "close"))
        assert result.height == 100

    def test_lazy_execution(self, sample_data):
        df = pl.DataFrame(sample_data)
        result = df.lazy().select(pl_chandelier_exit("high", "low", "close")).collect()
        assert result.height == 100
