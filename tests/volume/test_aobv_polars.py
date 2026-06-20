# -*- coding: utf-8 -*-
"""Tests for pl_aobv."""

import numpy as np
import polars as pl
import pytest
from polars_ti.volume.aobv import aobv


class TestPlAobv:
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

    def test_returns_list_of_expressions(self, sample_df):
        exprs = aobv("close", "volume")
        assert isinstance(exprs, list)
        assert len(exprs) == 7

    def test_output_has_correct_columns(self, sample_df):
        exprs = aobv("close", "volume")
        result = sample_df.select(exprs)
        expected_cols = [
            "OBV",
            "OBV_min_2",
            "OBV_max_2",
            "OBVe_4",
            "OBVe_12",
            "AOBV_LR_2",
            "AOBV_SR_2",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_offset_shifts_result(self, sample_df):
        exprs = aobv("close", "volume", offset=5)
        result = sample_df.select(exprs)
        arr = result["OBV"].to_numpy()
        assert all(np.isnan(arr[:5]))

    def test_with_null_values(self):
        df = pl.DataFrame({"close": [None] + [100.0] * 49, "volume": [None] + [1000.0] * 49})
        exprs = aobv("close", "volume")
        result = df.select(exprs)
        assert result.height == 50

    def test_lazy_execution(self, sample_df):
        lazy_df = sample_df.lazy()
        exprs = aobv("close", "volume")
        result = lazy_df.select(exprs).collect()
        assert "OBV" in result.columns

    def test_custom_parameters(self, sample_df):
        exprs = aobv("close", "volume", fast=5, slow=15, mamode="sma")
        result = sample_df.select(exprs)
        assert "OBVs_5" in result.columns
        assert "OBVs_15" in result.columns
