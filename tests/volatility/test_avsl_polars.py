# -*- coding: utf-8 -*-
"""Tests for pl_avsl."""

import numpy as np
import polars as pl
import pytest
from polars_ti.volatility.avsl import pl_avsl


class TestPlAvsl:
    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        n = 150
        high = 100 + np.cumsum(np.random.randn(n) * 0.5) + np.abs(np.random.randn(n) * 0.3)
        low = high - np.abs(np.random.randn(n) * 0.5) - 0.2
        close = (high + low) / 2 + np.random.randn(n) * 0.1
        volume = np.abs(np.random.randn(n) * 1000000) + 100000
        return {"close": close, "low": low, "volume": volume}

    def test_returns_expression(self, sample_data):
        result = pl_avsl("close", "low", "volume")
        assert isinstance(result, pl.Expr)

    def test_output_has_correct_alias(self, sample_data):
        df = pl.DataFrame(sample_data)
        result = df.select(pl_avsl("close", "low", "volume", fast_period=12, slow_period=26))
        assert "AVSL" in result.columns[0]

    def test_with_null_values(self, sample_data):
        data = sample_data.copy()
        data["close"] = data["close"].copy()
        data["close"][10:15] = np.nan
        df = pl.DataFrame(data)
        result = df.select(pl_avsl("close", "low", "volume"))
        assert result.height == 150

    def test_with_zeros(self, sample_data):
        data = sample_data.copy()
        data["low"] = data["low"].copy()
        data["low"][50:55] = 50.0
        df = pl.DataFrame(data)
        result = df.select(pl_avsl("close", "low", "volume"))
        assert result.height == 150

    def test_lazy_execution(self, sample_data):
        df = pl.DataFrame(sample_data)
        result = df.lazy().select(pl_avsl("close", "low", "volume")).collect()
        assert result.height == 150
