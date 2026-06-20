# -*- coding: utf-8 -*-
"""Tests for pl_vhm."""

import numpy as np
import polars as pl
import pytest
from polars_ti.volume.vhm import vhm


class TestPlVhm:
    @pytest.fixture
    def sample_df(self) -> pl.DataFrame:
        np.random.seed(42)
        n = 700
        volume = np.abs(np.random.randn(n) * 1000) + 100
        return pl.DataFrame({"volume": volume})

    def test_returns_expression(self, sample_df):
        result = sample_df.select(vhm("volume"))
        assert result.height == 700

    def test_output_has_correct_alias(self, sample_df):
        result = sample_df.select(vhm("volume"))
        assert "VHM_610" in result.columns

    def test_offset_shifts_result(self, sample_df):
        result = sample_df.select(vhm("volume", offset=5))
        arr = result["VHM_610"].to_numpy()
        assert all(np.isnan(arr[:5]))

    def test_with_null_values(self):
        df = pl.DataFrame({"volume": [None] + [1000.0] * 699})
        result = df.select(vhm("volume"))
        assert result.height == 700

    def test_lazy_execution(self, sample_df):
        lazy_df = sample_df.lazy()
        result = lazy_df.select(vhm("volume")).collect()
        assert "VHM_610" in result.columns

    def test_custom_slength(self, sample_df):
        result = sample_df.select(vhm("volume", length=100, slength=50))
        assert "VHM_100_50" in result.columns
