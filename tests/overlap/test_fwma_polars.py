# -*- coding: utf-8 -*-
"""Unit tests for polars_ti/overlap/fwma.py Polars implementation."""
import numpy as np
import polars as pl
import pytest

from polars_ti.overlap.fwma import pl_fwma


class TestPlFwma:
    """Tests for pl_fwma - Fibonacci Weighted Moving Average."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        np.random.seed(42)
        close = 100 + np.random.randn(200).cumsum()
        return {
            'pd_series': close,
            'pl_df': pl.DataFrame({'close': close}),
        }

    def test_returns_expression(self):
        """Returns a Polars expression."""
        result = pl_fwma("close", length=10)
        assert isinstance(result, pl.Expr)

    def test_output_has_correct_alias(self, sample_data):
        """Output column has correct alias."""
        result = sample_data['pl_df'].select(pl_fwma('close', length=10))
        assert result.columns[0] == 'FWMA_10'

    def test_with_null_values(self):
        """Handles null values gracefully."""
        df = pl.DataFrame({"close": [None] + [100.0] * 29})
        result = df.select(pl_fwma("close", length=10))
        assert result.height == 30

    def test_with_zeros(self):
        """Handles zero values."""
        df = pl.DataFrame({"close": [0.0] * 5 + [100.0] * 25})
        result = df.select(pl_fwma("close", length=10))
        assert result.height == 30

    def test_lazy_execution(self, sample_data):
        """Works with LazyFrame."""
        lazy_df = sample_data['pl_df'].lazy()
        result = lazy_df.select(pl_fwma("close", length=10)).collect()
        assert "FWMA_10" in result.columns
