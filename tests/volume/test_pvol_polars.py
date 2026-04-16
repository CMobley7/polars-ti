# -*- coding: utf-8 -*-
"""Tests for pl_pvol."""
import numpy as np
import polars as pl
import pytest
from polars_ti.volume.pvol import pl_pvol


class TestPlPvol:
    @pytest.fixture
    def sample_df(self) -> pl.DataFrame:
        np.random.seed(42)
        n = 100
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        volume = np.abs(np.random.randn(n) * 1000) + 100
        return pl.DataFrame({'close': close, 'volume': volume})

    def test_returns_expression(self, sample_df):
        result = sample_df.select(pl_pvol("close", "volume"))
        assert result.height == 100

    def test_output_has_correct_alias(self, sample_df):
        result = sample_df.select(pl_pvol("close", "volume"))
        assert "PVOL" in result.columns

    def test_signed_parameter(self, sample_df):
        result_unsigned = sample_df.select(pl_pvol("close", "volume", signed=False))
        result_signed = sample_df.select(pl_pvol("close", "volume", signed=True))
        # Results should be different when signed
        arr_unsigned = result_unsigned["PVOL"].to_numpy()
        arr_signed = result_signed["PVOL"].to_numpy()
        assert not np.allclose(arr_unsigned[1:], arr_signed[1:], equal_nan=True)

    def test_offset_shifts_result(self, sample_df):
        result = sample_df.select(pl_pvol("close", "volume", offset=5))
        arr = result["PVOL"].to_numpy()
        assert all(np.isnan(arr[:5]))

    def test_with_null_values(self):
        df = pl.DataFrame({
            "close": [None] + [100.0] * 49,
            "volume": [None] + [1000.0] * 49
        })
        result = df.select(pl_pvol("close", "volume"))
        assert result.height == 50

    def test_lazy_execution(self, sample_df):
        lazy_df = sample_df.lazy()
        result = lazy_df.select(pl_pvol("close", "volume")).collect()
        assert "PVOL" in result.columns
