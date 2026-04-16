# -*- coding: utf-8 -*-
"""Tests for pl_pdist."""
import numpy as np
import pandas as pd  # REMOVED: pandas dependency  # Restored for fixtures
import polars as pl
import pytest
from polars_ti.volatility.pdist import pl_pdist


class TestPlPdist:
    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        n = 100
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n) * 0.3)
        low = close - np.abs(np.random.randn(n) * 0.3)
        open_ = close + np.random.randn(n) * 0.2
        return {"open": open_, "high": high, "low": low, "close": close}

    def test_returns_expression(self, sample_data):
        result = pl_pdist("open", "high", "low", "close")
        assert isinstance(result, pl.Expr)

    def test_output_has_correct_alias(self, sample_data):
        df = pl.DataFrame(sample_data)
        result = df.select(pl_pdist("open", "high", "low", "close"))
        assert "PDIST" in result.columns[0]

    def test_numerical_parity(self, sample_data):
        pytest.skip("Pandas implementation removed in Phase 4 purge")
        pd_open = pd.Series(sample_data["open"])
        pd_high = pd.Series(sample_data["high"])
        pd_low = pd.Series(sample_data["low"])
        pd_close = pd.Series(sample_data["close"])
        pl_df = pl.DataFrame(sample_data)
        pd_result = pdist(pd_open, pd_high, pd_low, pd_close, drift=1)
        pl_result = pl_df.select(pl_pdist("open", "high", "low", "close", drift=1))
        warmup = 5
        pd_vals = pd_result.to_numpy()[warmup:]
        pl_vals = pl_result[pl_result.columns[0]].to_numpy()[warmup:]
        mask = np.isfinite(pd_vals) & np.isfinite(pl_vals)
        if mask.sum() > 0:
            max_diff = np.abs(pd_vals[mask] - pl_vals[mask]).max()
            assert max_diff < 1e-6

    def test_with_null_values(self, sample_data):
        data = sample_data.copy()
        data["high"] = np.array(data["high"])
        data["high"][10:15] = np.nan
        df = pl.DataFrame(data)
        result = df.select(pl_pdist("open", "high", "low", "close"))
        assert result.height == 100

    def test_with_zeros(self, sample_data):
        data = sample_data.copy()
        data["low"] = np.array(data["low"])
        data["low"][50:55] = data["high"][50:55]  # Make range zero
        df = pl.DataFrame(data)
        result = df.select(pl_pdist("open", "high", "low", "close"))
        assert result.height == 100

    def test_lazy_execution(self, sample_data):
        df = pl.DataFrame(sample_data)
        result = df.lazy().select(pl_pdist("open", "high", "low", "close")).collect()
        assert result.height == 100
