# -*- coding: utf-8 -*-
"""Tests for pl_decreasing."""

import numpy as np
import polars as pl
import pytest
from polars_ti.trend.decreasing import decreasing


class TestPlDecreasing:
    @pytest.fixture
    def sample_df(self) -> pl.DataFrame:
        np.random.seed(42)
        return pl.DataFrame(
            {
                "close": 100 + np.cumsum(np.random.randn(100) * 0.5),
            }
        )

    def test_returns_expression(self, sample_df):
        result = sample_df.select(decreasing("close"))
        assert result.height == 100

    def test_output_has_correct_alias(self, sample_df):
        result = sample_df.select(decreasing("close", length=5))
        assert "DEC_5" in result.columns

    def test_offset_shifts_result(self, sample_df):
        result = sample_df.select(decreasing("close", offset=5))
        arr = result[result.columns[0]].to_numpy()
        assert all(v is None or np.isnan(float(v)) for v in arr[:5] if v is not None)

    def test_asint_parameter(self, sample_df):
        r_int = sample_df.select(decreasing("close", asint=True))
        r_bool = sample_df.select(decreasing("close", asint=False))
        assert r_int[r_int.columns[0]].dtype == pl.Int64
        assert r_bool[r_bool.columns[0]].dtype == pl.Boolean

    def test_with_null_values(self):
        df = pl.DataFrame({"close": [None] + [100.0 - i for i in range(49)]})
        result = df.select(decreasing("close", length=2))
        assert result.height == 50

    def test_with_zeros(self):
        df = pl.DataFrame({"close": [50.0 - i * 0.5 for i in range(50)]})
        result = df.select(decreasing("close", length=2))
        assert result.height == 50

    def test_lazy_execution(self, sample_df):
        lazy_df = sample_df.lazy()
        result = lazy_df.select(decreasing("close", length=3)).collect()
        assert "DEC_3" in result.columns
