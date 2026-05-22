# -*- coding: utf-8 -*-
"""Tests for pl_cmf."""

import numpy as np
import polars as pl
import pytest
from polars_ti.volume.cmf import pl_cmf


class TestPlCmf:
    @pytest.fixture
    def sample_df(self) -> pl.DataFrame:
        np.random.seed(42)
        n = 100
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n) * 0.3)
        low = close - np.abs(np.random.randn(n) * 0.3)
        volume = np.abs(np.random.randn(n) * 1000) + 100
        return pl.DataFrame(
            {
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            }
        )

    def test_returns_expression(self, sample_df):
        result = sample_df.select(pl_cmf("high", "low", "close", "volume"))
        assert result.height == 100

    def test_output_has_correct_alias(self, sample_df):
        result = sample_df.select(pl_cmf("high", "low", "close", "volume", length=10))
        assert "CMF_10" in result.columns

    def test_offset_shifts_result(self, sample_df):
        result = sample_df.select(pl_cmf("high", "low", "close", "volume", offset=5))
        arr = result[result.columns[0]].to_numpy()
        assert all(np.isnan(arr[:5]))

    def test_with_null_values(self):
        df = pl.DataFrame(
            {
                "high": [None] + [101.0] * 49,
                "low": [None] + [99.0] * 49,
                "close": [None] + [100.0] * 49,
                "volume": [None] + [1000.0] * 49,
            }
        )
        result = df.select(pl_cmf("high", "low", "close", "volume", length=10))
        assert result.height == 50

    def test_lazy_execution(self, sample_df):
        lazy_df = sample_df.lazy()
        result = lazy_df.select(pl_cmf("high", "low", "close", "volume")).collect()
        assert "CMF_20" in result.columns
