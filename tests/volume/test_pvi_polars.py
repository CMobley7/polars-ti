# -*- coding: utf-8 -*-
"""Tests for pl_pvi."""

import numpy as np
import polars as pl
import pytest
from polars_ti.volume.pvi import pl_pvi


class TestPlPvi:
    @pytest.fixture
    def sample_df(self) -> pl.DataFrame:
        np.random.seed(42)
        n = 300
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        volume = np.abs(np.random.randn(n) * 1000) + 100
        return pl.DataFrame(
            {
                "close": close,
                "volume": volume,
            }
        )

    def test_returns_list_of_expressions(self, sample_df):
        exprs = pl_pvi("close", "volume")
        assert isinstance(exprs, list)
        assert len(exprs) == 2

    def test_output_has_correct_columns(self, sample_df):
        exprs = pl_pvi("close", "volume")
        result = sample_df.select(exprs)
        assert "PVI" in result.columns
        assert "PVIe_255" in result.columns

    def test_offset_shifts_result(self, sample_df):
        exprs = pl_pvi("close", "volume", offset=5)
        result = sample_df.select(exprs)
        arr = result["PVI"].to_numpy()
        assert all(np.isnan(arr[:5]))

    def test_with_null_values(self):
        df = pl.DataFrame({"close": [None] + [100.0] * 299, "volume": [None] + [1000.0] * 299})
        exprs = pl_pvi("close", "volume")
        result = df.select(exprs)
        assert result.height == 300

    def test_lazy_execution(self, sample_df):
        lazy_df = sample_df.lazy()
        exprs = pl_pvi("close", "volume")
        result = lazy_df.select(exprs).collect()
        assert "PVI" in result.columns

    def test_custom_parameters(self, sample_df):
        exprs = pl_pvi("close", "volume", length=100, mamode="sma")
        result = sample_df.select(exprs)
        assert "PVI" in result.columns
        assert "PVIs_100" in result.columns
