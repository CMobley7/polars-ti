# -*- coding: utf-8 -*-
"""Tests for pl_mmar - Polars + Numba implementation."""

import numpy as np
import polars as pl
import pytest
from polars_ti.overlap.mmar import pl_mmar


class TestPlMmar:
    @pytest.fixture
    def sample_df(self):
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

    def test_returns_expr(self):
        expr = pl_mmar("close")
        assert isinstance(expr, pl.Expr)

    def test_returns_dataframe(self, sample_df):
        result = sample_df.select(pl_mmar("close"))
        assert isinstance(result, pl.DataFrame)

    def test_has_ribbon_columns(self, sample_df):
        # MMAR returns a struct - unnest to get individual columns
        result = sample_df.select(pl_mmar("close"))
        result = result.unnest(result.columns[0])
        assert "MMAR_10" in result.columns
        assert "MMAR_35" in result.columns

    def test_custom_parameters(self, sample_df):
        result = sample_df.select(pl_mmar("close", length=5, step=3, num_ribbons=4))
        result = result.unnest(result.columns[0])
        assert "MMAR_5" in result.columns
        assert "MMAR_14" in result.columns  # 5 + 3*3 = 14

    def test_with_null_values(self):
        """Handles null values gracefully."""
        df = pl.DataFrame({"close": [None] + [100.0] * 49})
        result = df.select(pl_mmar("close"))
        assert result.height == 50

    def test_with_zeros(self):
        """Handles zero values."""
        df = pl.DataFrame({"close": [0.0] * 5 + [100.0] * 45})
        result = df.select(pl_mmar("close"))
        assert result.height == 50

    def test_lazy_execution(self, sample_df):
        """Works with LazyFrame."""
        lazy_df = sample_df.lazy()
        result = lazy_df.select(pl_mmar("close")).collect()
        assert result.height == 100
