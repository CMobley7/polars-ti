# -*- coding: utf-8 -*-
"""Tests for pl_alligator - Native Polars pl.Expr API."""
import numpy as np
import pandas as pd  # REMOVED: pandas dependency  # Restored for fixtures
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
            'pd_series': pd.Series(close, name='close'),
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

    def test_numerical_parity(self, sample_data):
        """Numerical parity with Pandas implementation."""
        pytest.skip("Pandas implementation removed in Phase 4 purge")
        pd_result = alligator(sample_data['pd_series'], talib=False)
        pl_result = sample_data['pl_df'].select(pl_alligator('close')).unnest("AG_13_8_5")
        
        warmup = 20
        for col in ["AGj_13_8_5", "AGt_13_8_5", "AGl_13_8_5"]:
            pd_vals = pd_result[col].iloc[warmup:].values
            pl_vals = pl_result[col][warmup:].to_numpy()
            
            valid = ~np.isnan(pd_vals) & ~np.isnan(pl_vals)
            if valid.sum() > 0:
                diff = np.abs(pd_vals[valid] - pl_vals[valid])
                assert np.max(diff) < 1e-10, f"Max diff for {col}: {np.max(diff)}"

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
