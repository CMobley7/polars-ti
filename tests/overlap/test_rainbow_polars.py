# -*- coding: utf-8 -*-
"""Tests for pl_rainbow - Native Polars pl.Expr API."""

import numpy as np
import polars as pl
import pytest
from polars_ti.overlap.rainbow import pl_rainbow


class TestPlRainbow:
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

    def test_returns_expr(self):
        expr = pl_rainbow("close")
        assert isinstance(expr, pl.Expr)

    def test_returns_struct(self, sample_df):
        result = sample_df.select(pl_rainbow("close"))
        assert len(result.columns) == 1

    def test_default_10_ribbons(self, sample_df):
        result = sample_df.select(pl_rainbow("close"))
        unnested = result.unnest(result.columns[0])
        assert len(unnested.columns) == 10

    def test_custom_ribbons(self, sample_df):
        result = sample_df.select(pl_rainbow("close", num_ribbons=5))
        unnested = result.unnest(result.columns[0])
        assert len(unnested.columns) == 5

    def test_has_valid_values(self, sample_df):
        result = sample_df.select(pl_rainbow("close"))
        unnested = result.unnest(result.columns[0])
        assert unnested["RAINBOW_1"].drop_nulls().len() > 0

    def test_with_null_values(self):
        """Handles null values gracefully."""
        df = pl.DataFrame({"close": [None] + [100.0] * 49})
        result = df.select(pl_rainbow("close"))
        assert result.height == 50

    def test_with_zeros(self):
        """Handles zero values."""
        df = pl.DataFrame({"close": [0.0] * 5 + [100.0] * 45})
        result = df.select(pl_rainbow("close"))
        assert result.height == 50

    def test_lazy_execution(self, sample_df):
        """Works with LazyFrame."""
        lazy_df = sample_df.lazy()
        result = lazy_df.select(pl_rainbow("close")).collect()
        assert result.height == 100
