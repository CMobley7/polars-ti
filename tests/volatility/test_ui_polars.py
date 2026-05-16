# -*- coding: utf-8 -*-
"""Tests for pl_ui."""
import numpy as np
import polars as pl
import pytest
from polars_ti.volatility.ui import pl_ui


class TestPlUi:
    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        n = 100
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        return {"close": close}

    def test_returns_expression(self, sample_data):
        result = pl_ui("close")
        assert isinstance(result, pl.Expr)

    def test_output_has_correct_alias(self, sample_data):
        df = pl.DataFrame(sample_data)
        result = df.select(pl_ui("close"))
        assert "UI" in result.columns[0]

    def test_with_null_values(self, sample_data):
        data = sample_data.copy()
        data["close"] = np.array(data["close"])
        data["close"][10:15] = np.nan
        df = pl.DataFrame(data)
        result = df.select(pl_ui("close"))
        assert result.height == 100

    def test_with_zeros(self, sample_data):
        data = sample_data.copy()
        data["close"] = np.array(data["close"])
        data["close"][50:55] = 0.001  # Near-zero to avoid div/0
        df = pl.DataFrame(data)
        result = df.select(pl_ui("close"))
        assert result.height == 100

    def test_lazy_execution(self, sample_data):
        df = pl.DataFrame(sample_data)
        result = df.lazy().select(pl_ui("close")).collect()
        assert result.height == 100
