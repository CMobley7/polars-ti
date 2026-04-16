# -*- coding: utf-8 -*-
"""Tests for pl_rainbow - Native Polars pl.Expr API."""
import numpy as np
import pandas as pd  # REMOVED: pandas dependency  # Restored for fixtures
import polars as pl
import pytest
from polars_ti.overlap.rainbow import pl_rainbow


class TestPlRainbow:
    @pytest.fixture
    def sample_df(self) -> pl.DataFrame:
        np.random.seed(42)
        return pl.DataFrame({'close': 100 + np.cumsum(np.random.randn(100) * 0.5)})

    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        close = 100 + np.random.randn(100).cumsum()
        return {
            'pd_close': pd.Series(close),
            'pl_df': pl.DataFrame({'close': close}),
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

    def test_numerical_parity(self, sample_data):
        """Numerical parity with Pandas implementation."""
        pytest.skip("Pandas implementation removed in Phase 4 purge")
        pd_result = rainbow(sample_data['pd_close'], length=2, num_ribbons=10)
        pl_result = sample_data['pl_df'].select(pl_rainbow('close')).unnest("RAINBOW_2_10")
        
        warmup = 30
        for col in ["RAINBOW_1", "RAINBOW_5"]:
            pd_vals = pd_result[col].iloc[warmup:].values
            pl_vals = pl_result[col][warmup:].to_numpy()
            valid = ~np.isnan(pd_vals) & ~np.isnan(pl_vals)
            if valid.sum() > 0:
                diff = np.abs(pd_vals[valid] - pl_vals[valid])
                assert np.max(diff) < 1e-10, f"{col} max diff: {np.max(diff)}"

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

