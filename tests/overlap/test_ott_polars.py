# -*- coding: utf-8 -*-
"""Tests for pl_ott - Polars + Numba implementation."""
import numpy as np
import polars as pl
import pytest
from polars_ti.overlap.ott import pl_ott


class TestPlOtt:
    @pytest.fixture
    def sample_df(self) -> pl.DataFrame:
        np.random.seed(42)
        return pl.DataFrame({'close': 100 + np.cumsum(np.random.randn(100) * 0.5)})

    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        close = 100 + np.random.randn(100).cumsum()
        return {
            'pd_close': close,
            'pl_df': pl.DataFrame({'close': close}),
        }

    def test_returns_expr(self):
        expr = pl_ott("close")
        assert isinstance(expr, pl.Expr)

    def test_returns_dataframe(self, sample_df):
        result = sample_df.select(pl_ott("close"))
        assert isinstance(result, pl.DataFrame)

    def test_columns_present(self, sample_df):
        # OTT returns a struct - unnest to get individual columns
        result = sample_df.select(pl_ott("close"))
        result = result.unnest(result.columns[0])
        assert "OTT_5_2.4" in result.columns
        assert "OTTSL_5_2.4" in result.columns
        assert "OTTd_5_2.4" in result.columns

    def test_custom_parameters(self, sample_df):
        result = sample_df.select(pl_ott("close", length=10, multiplier=3.0))
        result = result.unnest(result.columns[0])
        assert "OTT_10_3.0" in result.columns

    def test_has_valid_values(self, sample_df):
        result = sample_df.select(pl_ott("close"))
        result = result.unnest(result.columns[0])
        # OTT should have mostly valid values after warmup
        arr = result["OTT_5_2.4"].to_numpy()
        mask = ~np.isnan(arr)
        assert mask.sum() > 50

    def test_with_null_values(self):
        """Handles null values gracefully."""
        df = pl.DataFrame({"close": [None] + [100.0] * 49})
        result = df.select(pl_ott("close"))
        assert result.height == 50

    def test_with_zeros(self):
        """Handles zero values."""
        df = pl.DataFrame({"close": [0.0] * 5 + [100.0] * 45})
        result = df.select(pl_ott("close"))
        assert result.height == 50

    def test_lazy_execution(self, sample_df):
        """Works with LazyFrame."""
        lazy_df = sample_df.lazy()
        result = lazy_df.select(pl_ott("close")).collect()
        assert result.height == 100

