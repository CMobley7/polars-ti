# -*- coding: utf-8 -*-
"""Unit tests for polars_ti/overlap/linreg.py Polars implementation."""

import numpy as np
import polars as pl
import pytest

from polars_ti.overlap.linreg import pl_linreg


class TestPlLinreg:
    """Tests for pl_linreg - Linear Regression Moving Average."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        np.random.seed(42)
        close = 100 + np.random.randn(200).cumsum()
        return {
            "pd_series": close,
            "pl_df": pl.DataFrame({"close": close}),
        }

    def test_returns_expression(self):
        """Returns a Polars expression."""
        result = pl_linreg("close", length=14)
        assert isinstance(result, pl.Expr)

    def test_output_has_correct_alias(self, sample_data):
        """Output column has correct alias."""
        result = sample_data["pl_df"].select(pl_linreg("close", length=14))
        assert result.columns[0] == "LINREG_14"

    def test_slope_mode(self, sample_data):
        """Slope mode works correctly."""
        result = sample_data["pl_df"].select(pl_linreg("close", length=14, slope=True))
        assert "LINREGm_14" in result.columns[0]

    def test_intercept_mode(self, sample_data):
        """Intercept mode works correctly."""
        result = sample_data["pl_df"].select(pl_linreg("close", length=14, intercept=True))
        assert "LINREGb_14" in result.columns[0]

    def test_with_null_values(self):
        """Handles null values gracefully."""
        df = pl.DataFrame({"close": [None] + [100.0] * 29})
        result = df.select(pl_linreg("close", length=14))
        assert result.height == 30

    def test_with_zeros(self):
        """Handles zero values."""
        df = pl.DataFrame({"close": [0.0] * 5 + [100.0] * 25})
        result = df.select(pl_linreg("close", length=14))
        assert result.height == 30

    def test_lazy_execution(self, sample_data):
        """Works with LazyFrame."""
        lazy_df = sample_data["pl_df"].lazy()
        result = lazy_df.select(pl_linreg("close", length=14)).collect()
        assert "LINREG_14" in result.columns
