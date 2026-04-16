# -*- coding: utf-8 -*-
"""Tests for pl_ott - Polars + Numba implementation."""
import numpy as np
import pandas as pd  # REMOVED: pandas dependency  # Restored for fixtures
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
            'pd_close': pd.Series(close),
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

    def test_numerical_parity(self, sample_data):
        """Numerical parity with Pandas implementation."""
        pytest.skip("Pandas implementation removed in Phase 4 purge")
        pd_result = ott(sample_data['pd_close'], length=5, multiplier=2.4)
        pl_result = sample_data['pl_df'].select(pl_ott('close')).unnest("OTT_5_2.4")
        
        warmup = 20
        for col in ["OTT_5_2.4", "OTTSL_5_2.4"]:
            pd_vals = pd_result[col].iloc[warmup:].values
            pl_vals = pl_result[col][warmup:].to_numpy()
            valid = ~np.isnan(pd_vals) & ~np.isnan(pl_vals)
            if valid.sum() > 0:
                diff = np.abs(pd_vals[valid] - pl_vals[valid])
                assert np.max(diff) < 1e-10, f"{col} max diff: {np.max(diff)}"

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

