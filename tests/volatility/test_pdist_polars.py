# -*- coding: utf-8 -*-
"""Tests for pl_pdist."""
import numpy as np
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
