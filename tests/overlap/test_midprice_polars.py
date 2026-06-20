# -*- coding: utf-8 -*-
"""Tests for pl_midprice."""

import numpy as np
import polars as pl
import pytest
from polars_ti.overlap.midprice import midprice


class TestPlMidprice:
    @pytest.fixture
    def sample_df(self) -> pl.DataFrame:
        np.random.seed(42)
        return pl.DataFrame(
            {
                "high": 102 + np.cumsum(np.random.randn(100) * 0.3),
                "low": 98 + np.cumsum(np.random.randn(100) * 0.3),
            }
        )

    def test_returns_correct_column(self, sample_df):
        result = sample_df.select(midprice("high", "low", length=14))
        assert "MIDPRICE_14" in result.columns

    def test_formula_correct_pure(self, sample_df):
        result = sample_df.select(midprice("high", "low", length=14, talib=False))
        expected = (sample_df["high"].rolling_max(14) + sample_df["low"].rolling_min(14)) / 2
        np.testing.assert_array_almost_equal(result["MIDPRICE_14"].to_numpy(), expected.to_numpy())

    def test_talib_matches_pure(self, sample_df):
        r_pure = sample_df.select(midprice("high", "low", length=14, talib=False))
        r_talib = sample_df.select(midprice("high", "low", length=14, talib=True))

        mask = ~np.isnan(r_pure["MIDPRICE_14"].to_numpy()) & ~np.isnan(r_talib["MIDPRICE_14"].to_numpy())
        np.testing.assert_array_almost_equal(
            r_pure["MIDPRICE_14"].to_numpy()[mask],
            r_talib["MIDPRICE_14"].to_numpy()[mask],
        )

    def test_different_lengths(self, sample_df):
        r14 = sample_df.select(midprice("high", "low", length=14, talib=False))
        r7 = sample_df.select(midprice("high", "low", length=7, talib=False))
        assert "MIDPRICE_14" in r14.columns
        assert "MIDPRICE_7" in r7.columns

    def test_with_null_values(self):
        """Handles null values gracefully."""
        df = pl.DataFrame(
            {
                "high": [None] + [102.0] * 29,
                "low": [None] + [98.0] * 29,
            }
        )
        result = df.select(midprice("high", "low", talib=False))
        assert result.height == 30

    def test_with_zeros(self):
        """Handles zero values."""
        df = pl.DataFrame(
            {
                "high": [0.0] * 5 + [102.0] * 25,
                "low": [0.0] * 5 + [98.0] * 25,
            }
        )
        result = df.select(midprice("high", "low", talib=False))
        assert result.height == 30

    def test_lazy_execution(self, sample_df):
        """Works with LazyFrame."""
        lazy_df = sample_df.lazy()
        result = lazy_df.select(midprice("high", "low", talib=False)).collect()
        assert "MIDPRICE_2" in result.columns
