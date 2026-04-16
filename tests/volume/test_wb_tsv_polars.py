# -*- coding: utf-8 -*-
"""Tests for pl_wb_tsv."""
import numpy as np
import polars as pl
import pytest
from polars_ti.volume.wb_tsv import pl_wb_tsv


class TestPlWbTsv:
    @pytest.fixture
    def sample_df(self) -> pl.DataFrame:
        np.random.seed(42)
        n = 100
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        volume = np.abs(np.random.randn(n) * 1000) + 100
        return pl.DataFrame({'close': close, 'volume': volume})

    def test_returns_list_of_expressions(self, sample_df):
        exprs = pl_wb_tsv("close", "volume")
        assert isinstance(exprs, list)
        assert len(exprs) == 3

    def test_output_has_correct_columns(self, sample_df):
        exprs = pl_wb_tsv("close", "volume")
        result = sample_df.select(exprs)
        assert "TSV_18_10" in result.columns
        assert "TSVs_18_10" in result.columns
        assert "TSVr_18_10" in result.columns

    def test_offset_shifts_result(self, sample_df):
        exprs = pl_wb_tsv("close", "volume", offset=5)
        result = sample_df.select(exprs)
        arr = result["TSV_18_10"].to_numpy()
        assert all(np.isnan(arr[:5]))

    def test_with_null_values(self):
        df = pl.DataFrame({
            "close": [None] + [100.0] * 99,
            "volume": [None] + [1000.0] * 99
        })
        exprs = pl_wb_tsv("close", "volume")
        result = df.select(exprs)
        assert result.height == 100

    def test_lazy_execution(self, sample_df):
        lazy_df = sample_df.lazy()
        exprs = pl_wb_tsv("close", "volume")
        result = lazy_df.select(exprs).collect()
        assert "TSV_18_10" in result.columns

    def test_custom_parameters(self, sample_df):
        exprs = pl_wb_tsv("close", "volume", length=20, signal=5)
        result = sample_df.select(exprs)
        assert "TSV_20_5" in result.columns
