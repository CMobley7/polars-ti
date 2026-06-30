# -*- coding: utf-8 -*-
"""Tests for pl_increasing."""

import numpy as np
import polars as pl
import pytest
from polars_ti.trend.increasing import increasing


class TestPlIncreasing:
    @pytest.fixture
    def sample_df(self) -> pl.DataFrame:
        np.random.seed(42)
        return pl.DataFrame(
            {
                "close": 100 + np.cumsum(np.random.randn(100) * 0.5),
            }
        )

    def test_returns_expression(self, sample_df):
        result = sample_df.select(increasing("close"))
        assert result.height == 100

    def test_output_has_correct_alias(self, sample_df):
        result = sample_df.select(increasing("close", length=5))
        assert "INC_5" in result.columns

    def test_offset_shifts_result(self, sample_df):
        result = sample_df.select(increasing("close", offset=5))
        arr = result[result.columns[0]].to_numpy()
        assert all(v is None or np.isnan(float(v)) for v in arr[:5] if v is not None)

    def test_asint_parameter(self, sample_df):
        r_int = sample_df.select(increasing("close", asint=True))
        r_bool = sample_df.select(increasing("close", asint=False))
        assert r_int[r_int.columns[0]].dtype == pl.Int64
        assert r_bool[r_bool.columns[0]].dtype == pl.Boolean

    def test_with_null_values(self):
        df = pl.DataFrame({"close": [None] + [100.0 + i for i in range(49)]})
        result = df.select(increasing("close", length=2))
        assert result.height == 50

    def test_with_zeros(self):
        df = pl.DataFrame({"close": [0.0] * 5 + [float(i) for i in range(45)]})
        result = df.select(increasing("close", length=2))
        assert result.height == 50

    def test_lazy_execution(self, sample_df):
        lazy_df = sample_df.lazy()
        result = lazy_df.select(increasing("close", length=3)).collect()
        assert "INC_3" in result.columns
