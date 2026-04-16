# -*- coding: utf-8 -*-
"""Tests for pl_vwma."""
import numpy as np
import pandas as pd  # REMOVED: pandas dependency  # Restored for fixtures
import polars as pl
import pytest
from polars_ti.volume.vwma import pl_vwma


class TestPlVwma:
    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        n = 100
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        volume = np.random.randint(1000, 10000, n).astype(float)
        return {"close": close, "volume": volume}

    def test_returns_expression(self, sample_data):
        result = pl_vwma("close", "volume")
        assert isinstance(result, pl.Expr)

    def test_output_has_correct_alias(self, sample_data):
        df = pl.DataFrame(sample_data)
        result = df.select(pl_vwma("close", "volume", length=10))
        assert "VWMA_10" in result.columns[0]

    def test_numerical_parity(self, sample_data):
        pytest.skip("Pandas implementation removed in Phase 4 purge")
        pd_close = pd.Series(sample_data["close"])
        pd_volume = pd.Series(sample_data["volume"])
        pl_df = pl.DataFrame(sample_data)
        pd_result = vwma(pd_close, pd_volume, length=10)
        pl_result = pl_df.select(pl_vwma("close", "volume", length=10))
        warmup = 15
        pd_vals = pd_result.to_numpy()[warmup:]
        pl_vals = pl_result[pl_result.columns[0]].to_numpy()[warmup:]
        mask = ~np.isnan(pd_vals) & ~np.isnan(pl_vals)
        max_diff = np.abs(pd_vals[mask] - pl_vals[mask]).max()
        assert max_diff < 1e-6

    def test_with_null_values(self, sample_data):
        data = sample_data.copy()
        data["close"] = data["close"].copy()
        data["close"][10:15] = np.nan
        df = pl.DataFrame(data)
        result = df.select(pl_vwma("close", "volume"))
        assert result.height == 100

    def test_with_zeros(self, sample_data):
        data = sample_data.copy()
        data["volume"] = data["volume"].copy()
        data["volume"][50:55] = 1.0  # Low volume
        df = pl.DataFrame(data)
        result = df.select(pl_vwma("close", "volume"))
        assert result.height == 100

    def test_lazy_execution(self, sample_data):
        df = pl.DataFrame(sample_data)
        result = df.lazy().select(pl_vwma("close", "volume")).collect()
        assert result.height == 100
