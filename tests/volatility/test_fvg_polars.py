# -*- coding: utf-8 -*-
"""Tests for pl_fvg."""

import numpy as np
import polars as pl
import pytest
from polars_ti.volatility.fvg import pl_fvg


class TestPlFvg:
    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        n = 100
        open_ = 100 + np.cumsum(np.random.randn(n) * 0.5)
        close = open_ + np.random.randn(n) * 0.8
        high = np.maximum(open_, close) + np.abs(np.random.randn(n) * 0.3)
        low = np.minimum(open_, close) - np.abs(np.random.randn(n) * 0.3)
        return {"open": open_, "high": high, "low": low, "close": close}

    def test_returns_expression(self, sample_data):
        result = pl_fvg("open", "high", "low", "close")
        assert isinstance(result, pl.Expr)

    def test_output_has_correct_alias(self, sample_data):
        df = pl.DataFrame(sample_data)
        result = df.select(pl_fvg("open", "high", "low", "close"))
        assert "FVG" in result.columns[0]

    def test_with_null_values(self, sample_data):
        data = sample_data.copy()
        data["close"] = data["close"].copy()
        data["close"][10:15] = np.nan
        df = pl.DataFrame(data)
        result = df.select(pl_fvg("open", "high", "low", "close"))
        assert result.height == 100

    def test_with_zeros(self, sample_data):
        data = sample_data.copy()
        data["close"] = data["close"].copy()
        data["close"][50:55] = 50.0
        df = pl.DataFrame(data)
        result = df.select(pl_fvg("open", "high", "low", "close"))
        assert result.height == 100

    def test_lazy_execution(self, sample_data):
        df = pl.DataFrame(sample_data)
        result = df.lazy().select(pl_fvg("open", "high", "low", "close")).collect()
        assert result.height == 100
