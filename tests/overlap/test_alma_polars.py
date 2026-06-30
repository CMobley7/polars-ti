# -*- coding: utf-8 -*-
"""Tests for pl_alma."""

import numpy as np
import polars as pl
import pytest
from polars_ti.overlap.alma import alma


class TestPlAlma:
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
        result = sample_df.select(alma("close"))
        assert "ALMA_9_6.0_0.85" in result.columns

    def test_custom_parameters(self, sample_df):
        result = sample_df.select(alma("close", length=14, sigma=4.0, dist_offset=0.9))
        assert "ALMA_14_4.0_0.9" in result.columns

    def test_nan_before_length(self, sample_df):
        result = sample_df.select(alma("close", length=9))
        arr = result["ALMA_9_6.0_0.85"].to_numpy()
        assert np.isnan(arr[:8]).all()
        assert not np.isnan(arr[8])

    def test_offset(self, sample_df):
        result = sample_df.select(alma("close", offset=5))
        arr = result["ALMA_9_6.0_0.85"].to_numpy()
        assert np.isnan(arr[:13]).all()

    def test_has_valid_values(self, sample_df):
        result = sample_df.select(alma("close"))
        arr = result["ALMA_9_6.0_0.85"].to_numpy()
        mask = ~np.isnan(arr)
        assert mask.sum() > 80

    def test_with_null_values(self):
        """Handles null values gracefully."""
        df = pl.DataFrame({"close": [None] + [100.0] * 29})
        result = df.select(alma("close", length=9))
        assert result.height == 30

    def test_with_zeros(self):
        """Handles zero values."""
        df = pl.DataFrame({"close": [0.0] * 5 + [100.0] * 25})
        result = df.select(alma("close", length=9))
        assert result.height == 30

    def test_lazy_execution(self, sample_df):
        """Works with LazyFrame."""
        lazy_df = sample_df.lazy()
        result = lazy_df.select(alma("close", length=9)).collect()
        assert "ALMA_9_6.0_0.85" in result.columns
