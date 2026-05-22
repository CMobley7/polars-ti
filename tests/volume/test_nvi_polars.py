# -*- coding: utf-8 -*-
"""Tests for pl_nvi."""

import numpy as np
import polars as pl
import pytest
from polars_ti.volume.nvi import pl_nvi


class TestPlNvi:
    @pytest.fixture
    def sample_df(self) -> pl.DataFrame:
        np.random.seed(42)
        n = 100
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        volume = np.abs(np.random.randn(n) * 1000) + 100
        return pl.DataFrame(
            {
                "close": close,
                "volume": volume,
            }
        )

    def test_returns_expression(self, sample_df):
        result = sample_df.select(pl_nvi("close", "volume"))
        assert result.height == 100

    def test_output_has_correct_alias(self, sample_df):
        result = sample_df.select(pl_nvi("close", "volume", length=2))
        assert "NVI_2" in result.columns

    def test_offset_shifts_result(self, sample_df):
        result = sample_df.select(pl_nvi("close", "volume", offset=5))
        arr = result[result.columns[0]].to_numpy()
        assert all(np.isnan(arr[:5]))

    def test_with_null_values(self):
        df = pl.DataFrame({"close": [None] + [100.0] * 49, "volume": [None] + [1000.0] * 49})
        result = df.select(pl_nvi("close", "volume"))
        assert result.height == 50

    def test_lazy_execution(self, sample_df):
        lazy_df = sample_df.lazy()
        result = lazy_df.select(pl_nvi("close", "volume")).collect()
        assert "NVI_1" in result.columns

    def test_initial_parameter(self, sample_df):
        result = sample_df.select(pl_nvi("close", "volume", initial=500.0))
        arr = result[result.columns[0]].to_numpy()
        assert arr[0] == 500.0
