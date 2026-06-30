# -*- coding: utf-8 -*-
"""Tests for pl_vwma."""

import numpy as np
import polars as pl
import pytest
from polars_ti.volume.vwma import vwma


class TestPlVwma:
    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        n = 100
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        volume = np.random.randint(1000, 10000, n).astype(float)
        return {"close": close, "volume": volume}

    def test_returns_expression(self, sample_data):
        result = vwma("close", "volume")
        assert isinstance(result, pl.Expr)

    def test_output_has_correct_alias(self, sample_data):
        df = pl.DataFrame(sample_data)
        result = df.select(vwma("close", "volume", length=10))
        assert "VWMA_10" in result.columns[0]

    def test_with_null_values(self, sample_data):
        data = sample_data.copy()
        data["close"] = data["close"].copy()
        data["close"][10:15] = np.nan
        df = pl.DataFrame(data)
        result = df.select(vwma("close", "volume"))
        assert result.height == 100

    def test_with_zeros(self, sample_data):
        data = sample_data.copy()
        data["volume"] = data["volume"].copy()
        data["volume"][50:55] = 1.0  # Low volume
        df = pl.DataFrame(data)
        result = df.select(vwma("close", "volume"))
        assert result.height == 100

    def test_lazy_execution(self, sample_data):
        df = pl.DataFrame(sample_data)
        result = df.lazy().select(vwma("close", "volume")).collect()
        assert result.height == 100
