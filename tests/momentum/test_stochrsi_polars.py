# -*- coding: utf-8 -*-
"""Tests for pl_stochrsi - Polars + Numba Stochastic RSI with TA-Lib support."""
import numpy as np
import polars as pl
import pytest
from polars_ti.momentum.stochrsi import pl_stochrsi


class TestPlStochrsi:
    @pytest.fixture
    def sample_df(self):
        np.random.seed(42)
        n = 100
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        return pl.DataFrame({'close': close})

    def test_returns_expr(self):
        expr = pl_stochrsi("close")
        assert isinstance(expr, pl.Expr)

    def test_has_stochrsi_column(self, sample_df):
        result = sample_df.select(pl_stochrsi("close"))
        assert "STOCHRSI" in result.columns

    def test_struct_has_k_d_fields(self, sample_df):
        result = sample_df.select(pl_stochrsi("close"))
        stochrsi = result["STOCHRSI"]
        assert "STOCHRSIk_14_14_3_3" in stochrsi.struct.fields
        assert "STOCHRSId_14_14_3_3" in stochrsi.struct.fields

    def test_has_valid_values(self, sample_df):
        result = sample_df.select(pl_stochrsi("close"))
        stochrsi = result["STOCHRSI"]
        k_vals = stochrsi.struct.field("STOCHRSIk_14_14_3_3")
        # After warmup period, values should be valid
        assert k_vals[40:].null_count() == 0

    def test_offset_parameter(self, sample_df):
        result = sample_df.select(pl_stochrsi("close", offset=5))
        stochrsi = result["STOCHRSI"]
        k_vals = stochrsi.struct.field("STOCHRSIk_14_14_3_3")
        assert k_vals[:5].null_count() == 5

    def test_lazy_execution(self, sample_df):
        result = sample_df.lazy().select(pl_stochrsi("close")).collect()
        assert "STOCHRSI" in result.columns

    def test_talib_true(self, sample_df):
        result = sample_df.select(pl_stochrsi("close", talib=True))
        assert "STOCHRSI" in result.columns

    def test_talib_false(self, sample_df):
        result = sample_df.select(pl_stochrsi("close", talib=False))
        assert "STOCHRSI" in result.columns

    def test_custom_length_parameter(self, sample_df):
        result = sample_df.select(pl_stochrsi("close", length=20))
        stochrsi = result["STOCHRSI"]
        assert "STOCHRSIk_20_14_3_3" in stochrsi.struct.fields

    def test_custom_rsi_length_parameter(self, sample_df):
        result = sample_df.select(pl_stochrsi("close", rsi_length=10))
        stochrsi = result["STOCHRSI"]
        assert "STOCHRSIk_14_10_3_3" in stochrsi.struct.fields

    def test_custom_k_parameter(self, sample_df):
        result = sample_df.select(pl_stochrsi("close", k=5))
        stochrsi = result["STOCHRSI"]
        assert "STOCHRSIk_14_14_5_3" in stochrsi.struct.fields

    def test_custom_d_parameter(self, sample_df):
        result = sample_df.select(pl_stochrsi("close", d=5))
        stochrsi = result["STOCHRSI"]
        assert "STOCHRSId_14_14_3_5" in stochrsi.struct.fields

    def test_with_null_values(self):
        df = pl.DataFrame({
            'close': [100.0, None, 102.0] + [100.0 + i * 0.1 for i in range(60)]
        })
        result = df.select(pl_stochrsi("close", talib=False))
        assert result.height == 63

