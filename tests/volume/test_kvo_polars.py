# -*- coding: utf-8 -*-
"""Tests for pl_kvo."""

import numpy as np
import polars as pl
import pytest
from polars_ti.volume.kvo import kvo


class TestPlKvo:
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

    def test_returns_list_of_expressions(self, sample_df):
        exprs = kvo("high", "low", "close", "volume")
        assert isinstance(exprs, list)
        assert len(exprs) == 2

    def test_output_has_correct_columns(self, sample_df):
        exprs = kvo("high", "low", "close", "volume")
        result = sample_df.select(exprs)
        assert "KVO_34_55_13" in result.columns
        assert "KVOs_34_55_13" in result.columns

    def test_offset_shifts_result(self, sample_df):
        exprs = kvo("high", "low", "close", "volume", offset=5)
        result = sample_df.select(exprs)
        arr = result["KVO_34_55_13"].to_numpy()
        assert all(np.isnan(arr[:5]))

    def test_with_null_values(self):
        df = pl.DataFrame(
            {
                "high": [None] + [101.0] * 79,
                "low": [None] + [99.0] * 79,
                "close": [None] + [100.0] * 79,
                "volume": [None] + [1000.0] * 79,
            }
        )
        exprs = kvo("high", "low", "close", "volume")
        result = df.select(exprs)
        assert result.height == 80

    def test_lazy_execution(self, sample_df):
        lazy_df = sample_df.lazy()
        exprs = kvo("high", "low", "close", "volume")
        result = lazy_df.select(exprs).collect()
        assert "KVO_34_55_13" in result.columns

    def test_custom_parameters(self, sample_df):
        exprs = kvo("high", "low", "close", "volume", fast=20, slow=40, mamode="sma")
        result = sample_df.select(exprs)
        assert "KVO_20_40_13" in result.columns
