# -*- coding: utf-8 -*-
"""Tests for pl_natr."""
import numpy as np
import pandas as pd  # REMOVED: pandas dependency  # Restored for fixtures
import polars as pl
import pytest
from polars_ti.volatility.natr import pl_natr


class TestPlNatr:
    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        n = 100
        high = 100 + np.cumsum(np.random.randn(n) * 0.5) + np.abs(np.random.randn(n) * 0.3)
        low = high - np.abs(np.random.randn(n) * 0.5) - 0.2
        close = (high + low) / 2 + np.random.randn(n) * 0.1
        return {"high": high, "low": low, "close": close}

    def test_returns_expression(self, sample_data):
        result = pl_natr("high", "low", "close")
        assert isinstance(result, pl.Expr)

    def test_output_has_correct_alias(self, sample_data):
        df = pl.DataFrame(sample_data)
        result = df.select(pl_natr("high", "low", "close"))
        assert "NATR" in result.columns[0]

    def test_numerical_parity(self, sample_data):
        pytest.skip("Pandas implementation removed in Phase 4 purge")
        pd_high = pd.Series(sample_data["high"])
        pd_low = pd.Series(sample_data["low"])
        pd_close = pd.Series(sample_data["close"])
        pl_df = pl.DataFrame(sample_data)
        pd_result = natr(pd_high, pd_low, pd_close, length=14, scalar=100, talib=False)
        pl_result = pl_df.select(pl_natr("high", "low", "close", length=14, scalar=100.0))
        warmup = 20
        pd_vals = pd_result.to_numpy()[warmup:]
        pl_vals = pl_result[pl_result.columns[0]].to_numpy()[warmup:]
        mask = np.isfinite(pd_vals) & np.isfinite(pl_vals)
        max_diff = np.abs(pd_vals[mask] - pl_vals[mask]).max()
        assert max_diff < 1e-6

    def test_with_null_values(self, sample_data):
        data = sample_data.copy()
        data["close"] = data["close"].copy()
        data["close"][10:15] = np.nan
        df = pl.DataFrame(data)
        result = df.select(pl_natr("high", "low", "close"))
        assert result.height == 100

    def test_with_zeros(self, sample_data):
        data = sample_data.copy()
        data["low"] = data["low"].copy()
        data["low"][50:55] = 50.0
        df = pl.DataFrame(data)
        result = df.select(pl_natr("high", "low", "close"))
        assert result.height == 100

    def test_lazy_execution(self, sample_data):
        df = pl.DataFrame(sample_data)
        result = df.lazy().select(pl_natr("high", "low", "close")).collect()
        assert result.height == 100
