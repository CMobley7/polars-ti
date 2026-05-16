# -*- coding: utf-8 -*-
"""Tests for pl_roc."""
import numpy as np
import polars as pl
import pytest
from polars_ti.momentum.roc import pl_roc


class TestPlRoc:
    @pytest.fixture
    def sample_df(self) -> pl.DataFrame:
        np.random.seed(42)
        return pl.DataFrame({
            'close': 100 + np.cumsum(np.random.randn(100) * 0.5),
        })

    def test_returns_expression(self, sample_df):
        result = sample_df.select(pl_roc("close"))
        assert result.height == 100

    def test_output_has_correct_alias(self, sample_df):
        result = sample_df.select(pl_roc("close", length=5))
        assert "ROC_5" in result.columns

    def test_offset_shifts_result(self, sample_df):
        result = sample_df.select(pl_roc("close", offset=5))
        arr = result[result.columns[0]].to_numpy()
        assert all(np.isnan(arr[:5]))

    def test_talib_parameter(self, sample_df):
        """TA-Lib path produces valid results."""
        result = sample_df.select(pl_roc("close", length=10, talib=True))
        arr = result[result.columns[0]].to_numpy()
        valid = ~np.isnan(arr)
        assert valid.sum() > 50

    def test_with_null_values(self):
        df = pl.DataFrame({"close": [None] + [100.0 + i for i in range(49)]})
        result = df.select(pl_roc("close", length=5))
        assert result.height == 50

    def test_with_zeros(self):
        df = pl.DataFrame({"close": [100.0] * 5 + [float(i+100) for i in range(45)]})
        result = df.select(pl_roc("close", length=5))
        assert result.height == 50

    def test_lazy_execution(self, sample_df):
        lazy_df = sample_df.lazy()
        result = lazy_df.select(pl_roc("close", length=10)).collect()
        assert "ROC_10" in result.columns

    def test_roc_calculation_correct(self):
        """Verify ROC = scalar * (close - close.shift(n)) / close.shift(n)."""
        df = pl.DataFrame({"close": [100.0, 110.0]})
        result = df.select(pl_roc("close", length=1, scalar=100, talib=False))
        # (110-100)/100 * 100 = 10%
        assert abs(result["ROC_1"][1] - 10.0) < 1e-10

    def test_custom_scalar(self):
        df = pl.DataFrame({"close": [100.0, 110.0]})
        result = df.select(pl_roc("close", length=1, scalar=1, talib=False))
        # (110-100)/100 * 1 = 0.1
        assert abs(result["ROC_1"][1] - 0.1) < 1e-10
