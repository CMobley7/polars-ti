# -*- coding: utf-8 -*-
"""Tests for pl_massi."""

import numpy as np
import polars as pl
import pytest
from polars_ti.volatility.massi import massi


class TestPlMassi:
    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        n = 100
        high = 100 + np.cumsum(np.random.randn(n) * 0.5) + np.abs(np.random.randn(n) * 0.3)
        low = high - np.abs(np.random.randn(n) * 0.5) - 0.2
        return {"high": high, "low": low}

    def test_returns_expression(self, sample_data):
        result = massi("high", "low")
        assert isinstance(result, pl.Expr)

    def test_output_has_correct_alias(self, sample_data):
        df = pl.DataFrame(sample_data)
        result = df.select(massi("high", "low"))
        assert "MASSI" in result.columns[0]

    def test_with_null_values(self, sample_data):
        data = sample_data.copy()
        data["high"] = data["high"].copy()
        data["high"][10:15] = np.nan
        df = pl.DataFrame(data)
        result = df.select(massi("high", "low"))
        assert result.height == 100

    def test_with_zeros(self, sample_data):
        data = sample_data.copy()
        data["low"] = data["low"].copy()
        data["low"][50:55] = 50.0
        df = pl.DataFrame(data)
        result = df.select(massi("high", "low"))
        assert result.height == 100

    def test_lazy_execution(self, sample_data):
        df = pl.DataFrame(sample_data)
        result = df.lazy().select(massi("high", "low")).collect()
        assert result.height == 100
