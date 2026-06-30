# -*- coding: utf-8 -*-
"""Tests for pl_mcgd."""

import numpy as np
import polars as pl
import pytest
from polars_ti.overlap.mcgd import mcgd


class TestPlMcgd:
    @pytest.fixture
    def sample_df(self) -> pl.DataFrame:
        np.random.seed(42)
        return pl.DataFrame({"close": 100 + np.cumsum(np.random.randn(100) * 0.5)})

    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        close = 100 + np.random.randn(100).cumsum()
        return {
            "pd_close": close,
            "pl_df": pl.DataFrame({"close": close}),
        }

    def test_returns_correct_column(self, sample_df):
        result = sample_df.select(mcgd("close"))
        assert "MCGD_10" in result.columns

    def test_custom_length(self, sample_df):
        result = sample_df.select(mcgd("close", length=14))
        assert "MCGD_14" in result.columns

    def test_has_valid_values(self, sample_df):
        result = sample_df.select(mcgd("close"))
        assert (~np.isnan(result["MCGD_10"].to_numpy())).sum() > 80

    def test_with_null_values(self):
        """Handles null values gracefully."""
        df = pl.DataFrame({"close": [None] + [100.0] * 29})
        result = df.select(mcgd("close"))
        assert result.height == 30

    def test_with_zeros(self):
        """Handles zero values."""
        df = pl.DataFrame({"close": [0.0] * 5 + [100.0] * 25})
        result = df.select(mcgd("close"))
        assert result.height == 30

    def test_lazy_execution(self, sample_df):
        """Works with LazyFrame."""
        lazy_df = sample_df.lazy()
        result = lazy_df.select(mcgd("close")).collect()
        assert "MCGD_10" in result.columns
