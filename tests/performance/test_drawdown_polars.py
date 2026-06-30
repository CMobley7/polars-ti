# -*- coding: utf-8 -*-
"""Tests for pl_drawdown."""

import numpy as np
import polars as pl
import pytest
from polars_ti.performance.drawdown import drawdown


class TestPlDrawdown:
    @pytest.fixture
    def sample_df(self) -> pl.DataFrame:
        np.random.seed(42)
        return pl.DataFrame({"close": 100 + np.cumsum(np.random.randn(100) * 0.5)})

    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        close = 100 + np.random.randn(100).cumsum()
        return {
            "pd_close": close,
            "pl_df": pl.DataFrame({"close": close}),
        }

    def test_returns_struct(self, sample_df):
        result = sample_df.select(drawdown("close"))
        assert "DRAWDOWN" in result.columns

    def test_columns_present(self, sample_df):
        result = sample_df.select(drawdown("close")).unnest("DRAWDOWN")
        assert "DD" in result.columns
        assert "DD_PCT" in result.columns
        assert "DD_LOG" in result.columns

    def test_with_null_values(self):
        """Handles null values gracefully."""
        df = pl.DataFrame({"close": [None] + [100.0] * 49})
        result = df.select(drawdown("close"))
        assert result.height == 50

    def test_with_zeros(self):
        """Handles zero values."""
        df = pl.DataFrame({"close": [0.0] * 5 + [100.0] * 45})
        result = df.select(drawdown("close"))
        assert result.height == 50

    def test_lazy_execution(self, sample_df):
        """Works with LazyFrame."""
        lazy_df = sample_df.lazy()
        result = lazy_df.select(drawdown("close")).collect()
        assert "DRAWDOWN" in result.columns
