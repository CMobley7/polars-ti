# -*- coding: utf-8 -*-
"""Tests for pl_pvo."""
import numpy as np
import polars as pl
import pytest
from polars_ti.volume.pvo import pl_pvo


class TestPlPvo:
    @pytest.fixture
    def sample_df(self) -> pl.DataFrame:
        np.random.seed(42)
        n = 100
        volume = np.abs(np.random.randn(n) * 1000) + 100
        return pl.DataFrame({
            'volume': volume,
        })

    def test_returns_list_of_expressions(self, sample_df):
        exprs = pl_pvo("volume")
        assert isinstance(exprs, list)
        assert len(exprs) == 3

    def test_output_has_correct_columns(self, sample_df):
        exprs = pl_pvo("volume")
        result = sample_df.select(exprs)
        assert "PVO_12_26_9" in result.columns
        assert "PVOh_12_26_9" in result.columns
        assert "PVOs_12_26_9" in result.columns

    def test_offset_shifts_result(self, sample_df):
        exprs = pl_pvo("volume", offset=5)
        result = sample_df.select(exprs)
        arr = result["PVO_12_26_9"].to_numpy()
        assert all(np.isnan(arr[:5]))

    def test_with_null_values(self):
        df = pl.DataFrame({
            "volume": [None] + [1000.0] * 49
        })
        exprs = pl_pvo("volume")
        result = df.select(exprs)
        assert result.height == 50

    def test_lazy_execution(self, sample_df):
        lazy_df = sample_df.lazy()
        exprs = pl_pvo("volume")
        result = lazy_df.select(exprs).collect()
        assert "PVO_12_26_9" in result.columns

    def test_custom_parameters(self, sample_df):
        exprs = pl_pvo("volume", fast=10, slow=20, signal=5)
        result = sample_df.select(exprs)
        assert "PVO_10_20_5" in result.columns
        assert "PVOh_10_20_5" in result.columns
        assert "PVOs_10_20_5" in result.columns
