# -*- coding: utf-8 -*-
"""Unit tests for polars_ti/overlap/t3.py Polars implementation."""

import numpy as np
import polars as pl
import pytest

from polars_ti.maps import Imports

from polars_ti.overlap.t3 import t3


class TestPlT3:
    """Tests for pl_t3 - Tim Tillson's T3 Moving Average."""

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
        result = t3("close", length=10)
        assert isinstance(result, pl.Expr)

    def test_output_has_correct_alias(self, sample_data):
        """Output column has correct alias."""
        result = sample_data["pl_df"].select(t3("close", length=10))
        assert result.columns[0] == "T3_10_0.7"

    def test_custom_a_parameter(self, sample_data):
        """Custom 'a' parameter works."""
        result = sample_data["pl_df"].select(t3("close", length=10, a=0.8))
        assert result.columns[0] == "T3_10_0.8"

    def test_offset_shifts_result(self, sample_data):
        """Offset parameter shifts the result."""
        no_offset = sample_data["pl_df"].select(t3("close", length=10)).to_series()
        with_offset = sample_data["pl_df"].select(t3("close", length=10, offset=5)).to_series()

        for i in range(60, 80):
            if not np.isnan(no_offset[i]):
                assert no_offset[i] == with_offset[i + 5], f"Offset mismatch at {i}"

    @pytest.mark.skipif(not Imports["talib"], reason="warmup structure assumes the TA-Lib multi-stage warmup")
    def test_warmup_period_has_nan(self, sample_data):
        """Warmup period contains NaN values."""
        result = sample_data["pl_df"].select(t3("close", length=10)).to_series()
        # T3 needs 6*length-5 warmup
        assert result[:53].is_nan().all()

    def test_with_null_values(self):
        """Handles null values gracefully."""
        df = pl.DataFrame({"close": [None] + [100.0] * 99})
        result = df.select(t3("close", length=10))
        assert result.height == 100

    def test_with_zeros(self):
        """Handles zero values."""
        df = pl.DataFrame({"close": [0.0] * 5 + [100.0] * 95})
        result = df.select(t3("close", length=10))
        assert result.height == 100

    def test_lazy_execution(self, sample_data):
        """Works with LazyFrame."""
        lazy_df = sample_data["pl_df"].lazy()
        result = lazy_df.select(t3("close", length=10)).collect()
        assert "T3_10_0.7" in result.columns
