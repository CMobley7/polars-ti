# -*- coding: utf-8 -*-
"""Tests for pl_wcp."""
import numpy as np
import polars as pl
import pytest
from polars_ti.overlap.wcp import pl_wcp


class TestPlWcp:
    @pytest.fixture
    def sample_df(self) -> pl.DataFrame:
        np.random.seed(42)
        return pl.DataFrame({
            'high': 102 + np.random.randn(100),
            'low': 98 + np.random.randn(100),
            'close': 100 + np.random.randn(100),
        })

    def test_returns_correct_column(self, sample_df):
        result = sample_df.select(pl_wcp("high", "low", "close"))
        assert "WCP" in result.columns

    def test_formula_correct_pure(self, sample_df):
        result = sample_df.select(pl_wcp("high", "low", "close", talib=False))
        expected = (sample_df["high"] + sample_df["low"] + 2 * sample_df["close"]) / 4
        np.testing.assert_array_almost_equal(
            result["WCP"].to_numpy(), expected.to_numpy()
        )

    def test_talib_matches_pure(self, sample_df):
        r_pure = sample_df.select(pl_wcp("high", "low", "close", talib=False))
        r_talib = sample_df.select(pl_wcp("high", "low", "close", talib=True))
        np.testing.assert_array_almost_equal(
            r_pure["WCP"].to_numpy(), r_talib["WCP"].to_numpy()
        )

    def test_offset(self, sample_df):
        result = sample_df.select(pl_wcp("high", "low", "close", offset=5, talib=False))
        arr = result["WCP"].to_numpy()
        assert np.isnan(arr[:5]).all()

    def test_with_null_values(self):
        """Handles null values gracefully."""
        df = pl.DataFrame({
            "high": [None] + [102.0] * 49,
            "low": [None] + [98.0] * 49,
            "close": [None] + [100.0] * 49,
        })
        result = df.select(pl_wcp("high", "low", "close"))
        assert result.height == 50

    def test_with_zeros(self):
        """Handles zero values."""
        df = pl.DataFrame({
            "high": [0.0] * 5 + [102.0] * 45,
            "low": [0.0] * 5 + [98.0] * 45,
            "close": [0.0] * 5 + [100.0] * 45,
        })
        result = df.select(pl_wcp("high", "low", "close"))
        assert result.height == 50

    def test_lazy_execution(self, sample_df):
        """Works with LazyFrame."""
        lazy_df = sample_df.lazy()
        result = lazy_df.select(pl_wcp("high", "low", "close")).collect()
        assert "WCP" in result.columns

