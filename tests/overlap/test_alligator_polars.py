# -*- coding: utf-8 -*-
"""Tests for pl_alligator - Native Polars pl.Expr API."""
import numpy as np
import polars as pl
import pytest
from polars_ti.overlap.alligator import pl_alligator


class TestPlAlligator:
    @pytest.fixture
    def sample_df(self) -> pl.DataFrame:
        np.random.seed(42)
        return pl.DataFrame({'close': 100 + np.cumsum(np.random.randn(100) * 0.5)})

    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        close = 100 + np.random.randn(200).cumsum()
        return {
            'pd_series': close,
            'pl_df': pl.DataFrame({'close': close}),
        }

    def test_returns_expr(self):
        expr = pl_alligator("close")
        assert isinstance(expr, pl.Expr)

    def test_returns_struct(self, sample_df):
        result = sample_df.select(pl_alligator("close"))
        assert len(result.columns) == 1

    def test_has_three_fields(self, sample_df):
        result = sample_df.select(pl_alligator("close"))
        unnested = result.unnest(result.columns[0])
        assert len(unnested.columns) == 3

    def test_column_names(self, sample_df):
        result = sample_df.select(pl_alligator("close"))
        unnested = result.unnest(result.columns[0])
        assert any("AGj" in c for c in unnested.columns)
        assert any("AGt" in c for c in unnested.columns)
        assert any("AGl" in c for c in unnested.columns)

    def test_custom_parameters(self, sample_df):
        result = sample_df.select(pl_alligator("close", jaw=10, teeth=6, lips=4))
        assert "AG_10_6_4" in result.columns

    def test_with_null_values(self):
        """Handles null values gracefully."""
        df = pl.DataFrame({"close": [None] + [100.0] * 39})
        result = df.select(pl_alligator("close"))
        assert result.height == 40

    def test_with_zeros(self):
        """Handles zero values."""
        df = pl.DataFrame({"close": [0.0] * 5 + [100.0] * 35})
        result = df.select(pl_alligator("close"))
        assert result.height == 40

    def test_lazy_execution(self, sample_df):
        """Works with LazyFrame."""
        lazy_df = sample_df.lazy()
        result = lazy_df.select(pl_alligator("close")).collect()
        assert "AG_13_8_5" in result.columns
