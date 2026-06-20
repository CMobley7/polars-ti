# -*- coding: utf-8 -*-
"""Tests for pl_pvr."""

import numpy as np
import polars as pl
import pytest
from polars_ti.volume.pvr import pvr


class TestPlPvr:
    @pytest.fixture
    def sample_df(self) -> pl.DataFrame:
        np.random.seed(42)
        n = 100
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        volume = np.abs(np.random.randn(n) * 1000) + 100
        return pl.DataFrame({"close": close, "volume": volume})

    def test_returns_expression(self, sample_df):
        result = sample_df.select(pvr("close", "volume"))
        assert result.height == 100

    def test_output_has_correct_alias(self, sample_df):
        result = sample_df.select(pvr("close", "volume"))
        assert "PVR" in result.columns

    def test_values_are_categorical_1_to_4(self, sample_df):
        result = sample_df.select(pvr("close", "volume"))
        arr = result["PVR"].to_numpy()
        valid = ~np.isnan(arr)
        unique_vals = set(arr[valid].astype(int))
        assert unique_vals.issubset({1, 2, 3, 4})

    def test_with_null_values(self):
        df = pl.DataFrame({"close": [None] + [100.0] * 49, "volume": [None] + [1000.0] * 49})
        result = df.select(pvr("close", "volume"))
        assert result.height == 50

    def test_lazy_execution(self, sample_df):
        lazy_df = sample_df.lazy()
        result = lazy_df.select(pvr("close", "volume")).collect()
        assert "PVR" in result.columns

    def test_drift_parameter(self, sample_df):
        result = sample_df.select(pvr("close", "volume", drift=2))
        arr = result["PVR"].to_numpy()
        # First 2 values should be null due to drift
        assert np.isnan(arr[0]) and np.isnan(arr[1])
