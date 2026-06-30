# -*- coding: utf-8 -*-
"""Tests for pl_ao - Native Polars pl.Expr API."""

import numpy as np
import polars as pl
import pytest
from polars_ti.momentum.ao import ao


class TestPlAo:
    @pytest.fixture
    def sample_df(self) -> pl.DataFrame:
        np.random.seed(42)
        n = 100
        return pl.DataFrame(
            {
                "high": 102 + np.cumsum(np.random.randn(n) * 0.3),
                "low": 98 + np.cumsum(np.random.randn(n) * 0.3),
            }
        )

    def test_returns_expr(self):
        expr = ao("high", "low")
        assert isinstance(expr, pl.Expr)

    def test_has_ao_column(self, sample_df):
        result = sample_df.select(ao("high", "low"))
        assert "AO_5_34" in result.columns

    def test_custom_periods(self, sample_df):
        result = sample_df.select(ao("high", "low", fast=7, slow=21))
        assert "AO_7_21" in result.columns

    def test_has_valid_values(self, sample_df):
        result = sample_df.select(ao("high", "low"))
        arr = result["AO_5_34"].to_numpy()
        mask = ~np.isnan(arr)
        assert mask.sum() > 50

    def test_with_null_values(self, sample_df):
        """Handles null values gracefully."""
        df_with_nulls = sample_df.with_columns(
            pl.when(pl.col("high").is_first_distinct()).then(None).otherwise(pl.col("high")).alias("high")
        )
        result = df_with_nulls.select(ao("high", "low"))
        assert result.height == sample_df.height

    def test_with_zeros(self):
        """Handles zero values."""
        df = pl.DataFrame(
            {
                "high": [0.0] * 10 + [100.0] * 90,
                "low": [0.0] * 10 + [99.0] * 90,
            }
        )
        result = df.select(ao("high", "low", fast=5, slow=10))
        assert result.height == 100

    def test_lazy_execution(self, sample_df):
        """Works with LazyFrame."""
        lazy_df = sample_df.lazy()
        result = lazy_df.select(ao("high", "low")).collect()
        assert "AO_5_34" in result.columns
        assert result.height == sample_df.height

    def test_offset_parameter(self, sample_df):
        """Offset parameter shifts results."""
        result = sample_df.select(ao("high", "low", offset=2))
        arr = result["AO_5_34"].to_numpy()
        # First 2 values should be NaN due to offset
        assert np.isnan(arr[0])
        assert np.isnan(arr[1])
