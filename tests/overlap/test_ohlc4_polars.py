# -*- coding: utf-8 -*-
"""Tests for pl_ohlc4."""

import numpy as np
import polars as pl
import pytest
from polars_ti.overlap.ohlc4 import ohlc4


class TestPlOhlc4:
    @pytest.fixture
    def sample_df(self) -> pl.DataFrame:
        np.random.seed(42)
        return pl.DataFrame(
            {
                "open": 100 + np.random.randn(100),
                "high": 102 + np.random.randn(100),
                "low": 98 + np.random.randn(100),
                "close": 101 + np.random.randn(100),
            }
        )

    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        n = 100
        open_arr = 100 + np.random.randn(n)
        high_arr = 102 + np.random.randn(n)
        low_arr = 98 + np.random.randn(n)
        close_arr = 101 + np.random.randn(n)
        return {
            "pd_open": open_arr,
            "pd_high": high_arr,
            "pd_low": low_arr,
            "pd_close": close_arr,
            "pl_df": pl.DataFrame(
                {
                    "open": open_arr,
                    "high": high_arr,
                    "low": low_arr,
                    "close": close_arr,
                }
            ),
        }

    def test_returns_correct_column(self, sample_df):
        result = sample_df.select(ohlc4("open", "high", "low", "close"))
        assert "OHLC4" in result.columns

    def test_formula_correct(self, sample_df):
        result = sample_df.select(ohlc4("open", "high", "low", "close"))
        expected = (sample_df["open"] + sample_df["high"] + sample_df["low"] + sample_df["close"]) / 4
        np.testing.assert_array_almost_equal(result["OHLC4"].to_numpy(), expected.to_numpy())

    def test_with_expressions(self, sample_df):
        result = sample_df.select(ohlc4(pl.col("open"), pl.col("high"), pl.col("low"), pl.col("close")))
        assert "OHLC4" in result.columns

    def test_with_null_values(self):
        """Handles null values gracefully."""
        df = pl.DataFrame(
            {
                "open": [None] + [100.0] * 29,
                "high": [None] + [102.0] * 29,
                "low": [None] + [98.0] * 29,
                "close": [None] + [101.0] * 29,
            }
        )
        result = df.select(ohlc4("open", "high", "low", "close"))
        assert result.height == 30

    def test_with_zeros(self):
        """Handles zero values."""
        df = pl.DataFrame(
            {
                "open": [0.0] * 5 + [100.0] * 25,
                "high": [0.0] * 5 + [102.0] * 25,
                "low": [0.0] * 5 + [98.0] * 25,
                "close": [0.0] * 5 + [101.0] * 25,
            }
        )
        result = df.select(ohlc4("open", "high", "low", "close"))
        assert result.height == 30

    def test_lazy_execution(self, sample_df):
        """Works with LazyFrame."""
        lazy_df = sample_df.lazy()
        result = lazy_df.select(ohlc4("open", "high", "low", "close")).collect()
        assert "OHLC4" in result.columns
