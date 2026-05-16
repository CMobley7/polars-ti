# -*- coding: utf-8 -*-
"""Tests for pl_ssf3."""
import numpy as np
import polars as pl
import pytest
from polars_ti.overlap.ssf3 import pl_ssf3


class TestPlSsf3:
    @pytest.fixture
    def sample_df(self) -> pl.DataFrame:
        np.random.seed(42)
        return pl.DataFrame({'close': 100 + np.cumsum(np.random.randn(100) * 0.5)})

    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        close = 100 + np.random.randn(100).cumsum()
        return {
            'pd_close': close,
            'pl_df': pl.DataFrame({'close': close}),
        }

    def test_returns_correct_column(self, sample_df):
        result = sample_df.select(pl_ssf3("close"))
        assert "SSF3_20" in result.columns

    def test_custom_length(self, sample_df):
        result = sample_df.select(pl_ssf3("close", length=14))
        assert "SSF3_14" in result.columns

    def test_has_valid_values(self, sample_df):
        result = sample_df.select(pl_ssf3("close"))
        assert (~np.isnan(result["SSF3_20"].to_numpy())).sum() == 100

    def test_with_null_values(self):
        """Handles null values gracefully."""
        df = pl.DataFrame({"close": [None] + [100.0] * 49})
        result = df.select(pl_ssf3("close"))
        assert result.height == 50

    def test_with_zeros(self):
        """Handles zero values."""
        df = pl.DataFrame({"close": [0.0] * 5 + [100.0] * 45})
        result = df.select(pl_ssf3("close"))
        assert result.height == 50

    def test_lazy_execution(self, sample_df):
        """Works with LazyFrame."""
        lazy_df = sample_df.lazy()
        result = lazy_df.select(pl_ssf3("close")).collect()
        assert "SSF3_20" in result.columns

