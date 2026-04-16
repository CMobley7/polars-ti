# -*- coding: utf-8 -*-
"""Unit tests for polars_ti/overlap/midpoint.py Polars implementation."""
import numpy as np
import pandas as pd  # REMOVED: pandas dependency  # Restored for fixtures
import polars as pl
import pytest

from polars_ti.overlap.midpoint import pl_midpoint


class TestPlMidpoint:
    """Tests for pl_midpoint - Midpoint."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        np.random.seed(42)
        close = 100 + np.random.randn(200).cumsum()
        return {
            'pd_series': pd.Series(close, name='close'),
            'pl_df': pl.DataFrame({'close': close}),
        }

    def test_returns_expression(self):
        """Returns a Polars expression."""
        result = pl_midpoint("close", length=14)
        assert isinstance(result, pl.Expr)

    def test_output_has_correct_alias(self, sample_data):
        """Output column has correct alias."""
        result = sample_data['pl_df'].select(pl_midpoint('close', length=14))
        assert result.columns[0] == 'MIDPOINT_14'

    def test_numerical_parity(self, sample_data):
        """Numerical parity with Pandas implementation."""
        pytest.skip("Pandas implementation removed in Phase 4 purge")
        pd_result = midpoint(sample_data['pd_series'], length=14, talib=False)
        pl_result = sample_data['pl_df'].select(pl_midpoint('close', length=14, talib=False)).to_series()
        
        warmup = 20
        pd_vals = pd_result.iloc[warmup:].values
        pl_vals = pl_result[warmup:].to_numpy()
        
        valid = ~np.isnan(pd_vals) & ~np.isnan(pl_vals)
        if valid.sum() > 0:
            diff = np.abs(pd_vals[valid] - pl_vals[valid])
            assert np.max(diff) < 1e-10, f"Max diff: {np.max(diff)}"

    def test_with_null_values(self):
        """Handles null values gracefully."""
        df = pl.DataFrame({"close": [None] + [100.0] * 29})
        result = df.select(pl_midpoint("close", length=14))
        assert result.height == 30

    def test_with_zeros(self):
        """Handles zero values."""
        df = pl.DataFrame({"close": [0.0] * 5 + [100.0] * 25})
        result = df.select(pl_midpoint("close", length=14))
        assert result.height == 30

    def test_lazy_execution(self, sample_data):
        """Works with LazyFrame."""
        lazy_df = sample_data['pl_df'].lazy()
        result = lazy_df.select(pl_midpoint("close", length=14)).collect()
        assert "MIDPOINT_14" in result.columns
