# -*- coding: utf-8 -*-
"""Tests for pl_vfi."""

import numpy as np
import polars as pl
import pytest
from polars_ti.volume.vfi import vfi


class TestPlVfi:
    @pytest.fixture
    def sample_df(self) -> pl.DataFrame:
        np.random.seed(42)
        n = 200
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        volume = np.abs(np.random.randn(n) * 1000) + 100
        return pl.DataFrame({"close": close, "volume": volume})

    def test_returns_expression(self, sample_df):
        result = sample_df.select(vfi("close", "volume"))
        assert result.height == 200

    def test_output_has_correct_alias(self, sample_df):
        result = sample_df.select(vfi("close", "volume"))
        assert "VFI_130" in result.columns

    def test_offset_shifts_result(self, sample_df):
        result = sample_df.select(vfi("close", "volume", offset=5))
        arr = result["VFI_130"].to_numpy()
        assert all(np.isnan(arr[:5]))

    def test_with_null_values(self):
        df = pl.DataFrame({"close": [None] + [100.0] * 199, "volume": [None] + [1000.0] * 199})
        result = df.select(vfi("close", "volume"))
        assert result.height == 200

    def test_lazy_execution(self, sample_df):
        lazy_df = sample_df.lazy()
        result = lazy_df.select(vfi("close", "volume")).collect()
        assert "VFI_130" in result.columns

    def test_custom_length(self, sample_df):
        result = sample_df.select(vfi("close", "volume", length=50))
        assert "VFI_50" in result.columns
