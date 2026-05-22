# -*- coding: utf-8 -*-
"""Tests for pl_avwap."""

import numpy as np
import polars as pl
import pytest
from polars_ti.volume.avwap import pl_avwap


class TestPlAvwap:
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
        exprs = pl_avwap("high", "low", "close", "volume")
        assert isinstance(exprs, list)
        assert len(exprs) == 2

    def test_output_has_correct_columns(self, sample_df):
        exprs = pl_avwap("high", "low", "close", "volume")
        result = sample_df.select(exprs)
        assert "AVWAPH_5_5" in result.columns
        assert "AVWAPL_5_5" in result.columns

    def test_offset_shifts_result(self, sample_df):
        exprs = pl_avwap("high", "low", "close", "volume", offset=5)
        result = sample_df.select(exprs)
        arr = result["AVWAPH_5_5"].to_numpy()
        assert all(np.isnan(arr[:5]))

    def test_with_null_values(self):
        n = 50
        np.random.seed(42)
        df = pl.DataFrame(
            {
                "high": [None] + [101.0 + np.random.rand() for _ in range(n - 1)],
                "low": [None] + [99.0 + np.random.rand() for _ in range(n - 1)],
                "close": [None] + [100.0 + np.random.rand() for _ in range(n - 1)],
                "volume": [None] + [1000.0 + np.random.rand() * 100 for _ in range(n - 1)],
            }
        )
        exprs = pl_avwap("high", "low", "close", "volume", left_strength=2, right_strength=2)
        result = df.select(exprs)
        assert result.height == 50

    def test_lazy_execution(self, sample_df):
        lazy_df = sample_df.lazy()
        exprs = pl_avwap("high", "low", "close", "volume")
        result = lazy_df.select(exprs).collect()
        assert "AVWAPH_5_5" in result.columns

    def test_custom_parameters(self, sample_df):
        exprs = pl_avwap("high", "low", "close", "volume", left_strength=3, right_strength=3)
        result = sample_df.select(exprs)
        assert "AVWAPH_3_3" in result.columns
        assert "AVWAPL_3_3" in result.columns
