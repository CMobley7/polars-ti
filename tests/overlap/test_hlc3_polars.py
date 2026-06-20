# -*- coding: utf-8 -*-
"""Tests for pl_hlc3."""

import numpy as np
import polars as pl
import pytest
from polars_ti.overlap.hlc3 import hlc3


class TestPlHlc3:
    @pytest.fixture
    def sample_df(self) -> pl.DataFrame:
        np.random.seed(42)
        return pl.DataFrame(
            {
                "high": 102 + np.random.randn(100),
                "low": 98 + np.random.randn(100),
                "close": 100 + np.random.randn(100),
            }
        )

    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        high = 102 + np.random.randn(100)
        low = 98 + np.random.randn(100)
        close = 100 + np.random.randn(100)
        return {
            "pd_high": high,
            "pd_low": low,
            "pd_close": close,
            "pl_df": pl.DataFrame({"high": high, "low": low, "close": close}),
        }

    def test_returns_correct_column(self, sample_df):
        result = sample_df.select(hlc3("high", "low", "close"))
        assert "HLC3" in result.columns

    def test_formula_correct(self, sample_df):
        result = sample_df.select(hlc3("high", "low", "close"))
        expected = (sample_df["high"] + sample_df["low"] + sample_df["close"]) / 3
        np.testing.assert_array_almost_equal(result["HLC3"].to_numpy(), expected.to_numpy())

    def test_with_expressions(self, sample_df):
        result = sample_df.select(hlc3(pl.col("high"), pl.col("low"), pl.col("close")))
        assert "HLC3" in result.columns

    def test_with_null_values(self):
        """Handles null values gracefully."""
        df = pl.DataFrame(
            {
                "high": [None] + [102.0] * 29,
                "low": [None] + [98.0] * 29,
                "close": [None] + [100.0] * 29,
            }
        )
        result = df.select(hlc3("high", "low", "close"))
        assert result.height == 30

    def test_with_zeros(self):
        """Handles zero values."""
        df = pl.DataFrame(
            {
                "high": [0.0] * 5 + [102.0] * 25,
                "low": [0.0] * 5 + [98.0] * 25,
                "close": [0.0] * 5 + [100.0] * 25,
            }
        )
        result = df.select(hlc3("high", "low", "close"))
        assert result.height == 30

    def test_lazy_execution(self, sample_df):
        """Works with LazyFrame."""
        lazy_df = sample_df.lazy()
        result = lazy_df.select(hlc3("high", "low", "close")).collect()
        assert "HLC3" in result.columns

    def test_talib_parameter(self, sample_data):
        """TA-Lib toggle produces valid results."""
        # Test with talib=False (pure Polars)
        result_polars = sample_data["pl_df"].select(hlc3("high", "low", "close", talib=False))
        assert "HLC3" in result_polars.columns

        # Test with talib=True (uses TA-Lib if available)
        result_talib = sample_data["pl_df"].select(hlc3("high", "low", "close", talib=True))
        assert "HLC3" in result_talib.columns

        # Results should be identical (same formula)
        np.testing.assert_array_almost_equal(
            result_polars["HLC3"].to_numpy(),
            result_talib["HLC3"].to_numpy(),
            decimal=10,
        )

    def test_offset_parameter(self, sample_df):
        """Offset shifts results correctly."""
        result_no_offset = sample_df.select(hlc3("high", "low", "close", offset=0))
        result_offset_2 = sample_df.select(hlc3("high", "low", "close", offset=2))

        # With offset=2, first 2 values should be null
        assert result_offset_2["HLC3"][0] is None
        assert result_offset_2["HLC3"][1] is None

        # Values should be shifted by 2
        np.testing.assert_array_almost_equal(
            result_no_offset["HLC3"].to_numpy()[:-2],
            result_offset_2["HLC3"].to_numpy()[2:],
            decimal=10,
        )
