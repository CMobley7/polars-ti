# -*- coding: utf-8 -*-
"""Tests for pl_smma."""

import numpy as np
import polars as pl
import pytest
from polars_ti.overlap.smma import pl_smma


class TestPlSmma:
    @pytest.fixture
    def sample_df(self) -> pl.DataFrame:
        np.random.seed(42)
        return pl.DataFrame(
            {
                "close": 100 + np.cumsum(np.random.randn(100) * 0.5),
            }
        )

    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        close = 100 + np.random.randn(200).cumsum()
        return {
            "pd_series": close,
            "pl_df": pl.DataFrame({"close": close}),
        }

    def test_returns_correct_column(self, sample_df):
        result = sample_df.select(pl_smma("close", length=7))
        assert "SMMA_7" in result.columns

    def test_different_lengths(self, sample_df):
        r7 = sample_df.select(pl_smma("close", length=7))
        r14 = sample_df.select(pl_smma("close", length=14))
        assert "SMMA_7" in r7.columns
        assert "SMMA_14" in r14.columns

    def test_nan_before_length(self, sample_df):
        result = sample_df.select(pl_smma("close", length=7))
        arr = result["SMMA_7"].to_numpy()
        assert np.isnan(arr[:6]).all()
        assert not np.isnan(arr[6])

    def test_offset(self, sample_df):
        result = sample_df.select(pl_smma("close", length=7, offset=5))
        arr = result["SMMA_7"].to_numpy()
        assert np.isnan(arr[:11]).all()

    def test_values_are_numeric(self, sample_df):
        result = sample_df.select(pl_smma("close", length=7))
        arr = result["SMMA_7"].to_numpy()
        mask = ~np.isnan(arr)
        assert mask.sum() > 50

    def test_with_null_values(self):
        """Handles null values gracefully."""
        df = pl.DataFrame({"close": [None] + [100.0] * 29})
        result = df.select(pl_smma("close", length=7))
        assert result.height == 30

    def test_with_zeros(self):
        """Handles zero values."""
        df = pl.DataFrame({"close": [0.0] * 5 + [100.0] * 25})
        result = df.select(pl_smma("close", length=7))
        assert result.height == 30

    def test_lazy_execution(self, sample_df):
        """Works with LazyFrame."""
        lazy_df = sample_df.lazy()
        result = lazy_df.select(pl_smma("close", length=7)).collect()
        assert "SMMA_7" in result.columns
