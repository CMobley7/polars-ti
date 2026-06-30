# -*- coding: utf-8 -*-
"""Tests for pl_natr."""

import numpy as np
import polars as pl
import pytest
from polars_ti.volatility.natr import natr


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
        result = natr("high", "low", "close")
        assert isinstance(result, pl.Expr)

    def test_output_has_correct_alias(self, sample_data):
        df = pl.DataFrame(sample_data)
        result = df.select(natr("high", "low", "close"))
        assert "NATR" in result.columns[0]

    def test_with_null_values(self, sample_data):
        data = sample_data.copy()
        data["close"] = data["close"].copy()
        data["close"][10:15] = np.nan
        df = pl.DataFrame(data)
        result = df.select(natr("high", "low", "close"))
        assert result.height == 100

    def test_with_zeros(self, sample_data):
        data = sample_data.copy()
        data["low"] = data["low"].copy()
        data["low"][50:55] = 50.0
        df = pl.DataFrame(data)
        result = df.select(natr("high", "low", "close"))
        assert result.height == 100

    def test_lazy_execution(self, sample_data):
        df = pl.DataFrame(sample_data)
        result = df.lazy().select(natr("high", "low", "close")).collect()
        assert result.height == 100

    def test_default_mamode_is_rma(self):
        """classic b914429: NATR default mamode is rma (Wilder), matching ATR
        and TA-Lib. The talib-mode default must equal talib.NATR exactly."""
        talib = pytest.importorskip("talib")
        df = pl.read_csv("data/SPY_D.csv", try_parse_dates=True).head(1500)
        h = df["high"].to_numpy().astype(float)
        low_ = df["low"].to_numpy().astype(float)
        c = df["close"].to_numpy().astype(float)
        ref = talib.NATR(h, low_, c, timeperiod=14)
        # default talib=True path uses rma and must match talib.NATR exactly.
        got = df.select(natr("high", "low", "close", length=14, talib=True)).to_series().to_numpy()
        m = ~np.isnan(ref) & ~np.isnan(got)
        assert m.sum() > 100
        assert np.max(np.abs(ref[m] - got[m])) < 1e-9
