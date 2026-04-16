# -*- coding: utf-8 -*-
"""Tests for pl_hl2, pl_hlc3, pl_ohlc4."""
import numpy as np
import polars as pl
import pytest
from polars_ti.overlap.hl2 import pl_hl2
from polars_ti.overlap.hlc3 import pl_hlc3
from polars_ti.overlap.ohlc4 import pl_ohlc4


class TestAverageIndicators:
    @pytest.fixture
    def df(self) -> pl.DataFrame:
        np.random.seed(42)
        return pl.DataFrame({
            'open': 100 + np.random.randn(100),
            'high': 102 + np.random.randn(100),
            'low': 98 + np.random.randn(100),
            'close': 101 + np.random.randn(100),
        })

    def test_pl_hl2(self, df):
        result = df.select(pl_hl2("high", "low"))
        assert "HL2" in result.columns
        assert result.height == 100
        
    def test_pl_hlc3(self, df):
        result = df.select(pl_hlc3("high", "low", "close"))
        assert "HLC3" in result.columns
        assert result.height == 100

    def test_pl_ohlc4(self, df):
        result = df.select(pl_ohlc4("open", "high", "low", "close"))
        assert "OHLC4" in result.columns
        assert result.height == 100
        
    def test_hl2_formula(self, df):
        result = df.select(pl_hl2("high", "low"))
        expected = (df["high"] + df["low"]) / 2
        np.testing.assert_array_almost_equal(
            result["HL2"].to_numpy(), expected.to_numpy()
        )

    def test_hlc3_formula(self, df):
        result = df.select(pl_hlc3("high", "low", "close"))
        expected = (df["high"] + df["low"] + df["close"]) / 3
        np.testing.assert_array_almost_equal(
            result["HLC3"].to_numpy(), expected.to_numpy()
        )

    def test_ohlc4_formula(self, df):
        result = df.select(pl_ohlc4("open", "high", "low", "close"))
        expected = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
        np.testing.assert_array_almost_equal(
            result["OHLC4"].to_numpy(), expected.to_numpy()
        )
