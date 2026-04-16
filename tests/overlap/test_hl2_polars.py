# -*- coding: utf-8 -*-
"""Tests for pl_hl2."""
import numpy as np
import pandas as pd  # REMOVED: pandas dependency  # Restored for fixtures
import polars as pl
import pytest
from polars_ti.overlap.hl2 import pl_hl2


class TestPlHl2:
    """Test suite for pl_hl2 Polars implementation."""

    @pytest.fixture
    def sample_df(self) -> pl.DataFrame:
        np.random.seed(42)
        return pl.DataFrame({
            'high': 102 + np.random.randn(100),
            'low': 98 + np.random.randn(100),
        })

    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        high = 102 + np.random.randn(100)
        low = 98 + np.random.randn(100)
        return {
            'pd_high': pd.Series(high),
            'pd_low': pd.Series(low),
            'pl_df': pl.DataFrame({'high': high, 'low': low}),
        }

    def test_returns_correct_column(self, sample_df):
        result = sample_df.select(pl_hl2("high", "low"))
        assert "HL2" in result.columns

    def test_formula_correct(self, sample_df):
        result = sample_df.select(pl_hl2("high", "low"))
        expected = (sample_df["high"] + sample_df["low"]) / 2
        np.testing.assert_array_almost_equal(
            result["HL2"].to_numpy(), expected.to_numpy()
        )

    def test_with_expressions(self, sample_df):
        result = sample_df.select(pl_hl2(pl.col("high"), pl.col("low")))
        assert "HL2" in result.columns

    def test_numerical_parity(self, sample_data):
        """Numerical parity with Pandas implementation."""
        pytest.skip("Pandas implementation removed in Phase 4 purge")
        pd_result = hl2(sample_data['pd_high'], sample_data['pd_low'])
        pl_result = sample_data['pl_df'].select(pl_hl2('high', 'low')).to_series()
        
        pd_vals = pd_result.values
        pl_vals = pl_result.to_numpy()
        
        valid = ~np.isnan(pd_vals) & ~np.isnan(pl_vals)
        if valid.sum() > 0:
            diff = np.abs(pd_vals[valid] - pl_vals[valid])
            assert np.max(diff) < 1e-10, f"Max diff: {np.max(diff)}"

    def test_with_null_values(self):
        """Handles null values gracefully."""
        df = pl.DataFrame({
            "high": [None] + [102.0] * 29,
            "low": [None] + [98.0] * 29,
        })
        result = df.select(pl_hl2("high", "low"))
        assert result.height == 30

    def test_with_zeros(self):
        """Handles zero values."""
        df = pl.DataFrame({
            "high": [0.0] * 5 + [102.0] * 25,
            "low": [0.0] * 5 + [98.0] * 25,
        })
        result = df.select(pl_hl2("high", "low"))
        assert result.height == 30

    def test_lazy_execution(self, sample_df):
        """Works with LazyFrame."""
        lazy_df = sample_df.lazy()
        result = lazy_df.select(pl_hl2("high", "low")).collect()
        assert "HL2" in result.columns

