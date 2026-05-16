# -*- coding: utf-8 -*-
"""Tests for pl_obv."""
import numpy as np
import polars as pl
import pytest
from polars_ti.volume.obv import pl_obv


class TestPlObv:
    @pytest.fixture
    def sample_df(self) -> pl.DataFrame:
        np.random.seed(42)
        n = 100
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        volume = np.abs(np.random.randn(n) * 1000) + 100
        return pl.DataFrame({
            'close': close,
            'volume': volume,
        })

    def test_returns_expression(self, sample_df):
        result = sample_df.select(pl_obv("close", "volume"))
        assert result.height == 100

    def test_output_has_correct_alias(self, sample_df):
        result = sample_df.select(pl_obv("close", "volume"))
        assert "OBV" in result.columns

    def test_offset_shifts_result(self, sample_df):
        result = sample_df.select(pl_obv("close", "volume", offset=5))
        arr = result[result.columns[0]].to_numpy()
        assert all(np.isnan(arr[:5]))

    def test_talib_parameter(self, sample_df):
        """TA-Lib path produces valid results."""
        result = sample_df.select(pl_obv("close", "volume", talib=True))
        arr = result[result.columns[0]].to_numpy()
        valid = ~np.isnan(arr)
        assert valid.sum() > 50

    def test_with_null_values(self):
        df = pl.DataFrame({
            "close": [None] + [100.0] * 49,
            "volume": [None] + [1000.0] * 49
        })
        result = df.select(pl_obv("close", "volume"))
        assert result.height == 50

    def test_lazy_execution(self, sample_df):
        lazy_df = sample_df.lazy()
        result = lazy_df.select(pl_obv("close", "volume")).collect()
        assert "OBV" in result.columns
