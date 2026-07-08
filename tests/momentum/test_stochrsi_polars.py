# -*- coding: utf-8 -*-
"""Tests for pl_stochrsi - Polars + Numba Stochastic RSI with TA-Lib support."""

import numpy as np
import polars as pl
import pytest
from polars_ti.momentum.stochrsi import stochrsi as stochrsi_indicator


class TestPlStochrsi:
    @pytest.fixture
    def sample_df(self):
        np.random.seed(42)
        n = 100
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        return pl.DataFrame({"close": close})

    def test_returns_expr(self):
        expr = stochrsi_indicator("close")
        assert isinstance(expr, pl.Expr)

    def test_has_stochrsi_column(self, sample_df):
        result = sample_df.select(stochrsi_indicator("close"))
        assert "STOCHRSI" in result.columns

    def test_struct_has_k_d_fields(self, sample_df):
        result = sample_df.select(stochrsi_indicator("close"))
        stochrsi = result["STOCHRSI"]
        assert "STOCHRSIk_14_14_3_3" in stochrsi.struct.fields
        assert "STOCHRSId_14_14_3_3" in stochrsi.struct.fields

    def test_has_valid_values(self, sample_df):
        result = sample_df.select(stochrsi_indicator("close"))
        stochrsi = result["STOCHRSI"]
        k_vals = stochrsi.struct.field("STOCHRSIk_14_14_3_3")
        # After warmup period, values should be valid
        assert k_vals[40:].null_count() == 0

    def test_offset_parameter(self, sample_df):
        result = sample_df.select(stochrsi_indicator("close", offset=5))
        stochrsi = result["STOCHRSI"]
        k_vals = stochrsi.struct.field("STOCHRSIk_14_14_3_3")
        assert k_vals[:5].null_count() == 5

    def test_lazy_execution(self, sample_df):
        result = sample_df.lazy().select(stochrsi_indicator("close")).collect()
        assert "STOCHRSI" in result.columns

    def test_talib_true(self, sample_df):
        result = sample_df.select(stochrsi_indicator("close", talib=True))
        assert "STOCHRSI" in result.columns

    def test_talib_false(self, sample_df):
        result = sample_df.select(stochrsi_indicator("close", talib=False))
        assert "STOCHRSI" in result.columns

    def test_custom_length_parameter(self, sample_df):
        result = sample_df.select(stochrsi_indicator("close", length=20))
        stochrsi = result["STOCHRSI"]
        assert "STOCHRSIk_20_14_3_3" in stochrsi.struct.fields

    def test_custom_rsi_length_parameter(self, sample_df):
        result = sample_df.select(stochrsi_indicator("close", rsi_length=10))
        stochrsi = result["STOCHRSI"]
        assert "STOCHRSIk_14_10_3_3" in stochrsi.struct.fields

    def test_custom_k_parameter(self, sample_df):
        result = sample_df.select(stochrsi_indicator("close", k=5))
        stochrsi = result["STOCHRSI"]
        assert "STOCHRSIk_14_14_5_3" in stochrsi.struct.fields

    def test_custom_d_parameter(self, sample_df):
        result = sample_df.select(stochrsi_indicator("close", d=5))
        stochrsi = result["STOCHRSI"]
        assert "STOCHRSId_14_14_3_5" in stochrsi.struct.fields

    def test_with_null_values(self):
        df = pl.DataFrame({"close": [100.0, None, 102.0] + [100.0 + i * 0.1 for i in range(60)]})
        result = df.select(stochrsi_indicator("close", talib=False))
        assert result.height == 63


class TestPlStochrsiMamodeAndSmaRegression:
    """Native %K/%D honour ``mamode`` and the SMA kernel recovers after interior NaN."""

    @pytest.fixture
    def sample_df(self):
        np.random.seed(42)
        n = 100
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        return pl.DataFrame({"close": close})

    def test_mamode_default_unchanged_and_ema_changes(self, sample_df):
        k = "STOCHRSIk_14_14_3_3"
        base = sample_df.select(stochrsi_indicator("close", talib=False)).unnest("STOCHRSI")[k].to_numpy()
        sma = sample_df.select(stochrsi_indicator("close", talib=False, mamode="sma")).unnest("STOCHRSI")[k].to_numpy()
        ema = sample_df.select(stochrsi_indicator("close", talib=False, mamode="ema")).unnest("STOCHRSI")[k].to_numpy()

        # Default equals explicit "sma".
        assert np.array_equal(base, sma, equal_nan=True)
        # Non-default mamode changes the output.
        assert np.nanmax(np.abs(ema - base)) > 0.0

    def test_sma_kernel_recovers_after_interior_nan(self):
        """Regression: the SMA kernel must match rolling(length).mean() semantics
        (min_periods=length) — an interior NaN breaks only the spanning windows
        and must NOT produce wrong values or poison the whole tail."""
        from polars_ti.momentum.stochrsi import _sma_numba

        x = np.array([1, 2, 3, 4, 5, np.nan, 7, 8, 9, 10], dtype=np.float64)
        got = _sma_numba(x, 3)
        # pandas Series(x).rolling(3, min_periods=3).mean()
        expected = np.array([np.nan, np.nan, 2.0, 3.0, 4.0, np.nan, np.nan, np.nan, 8.0, 9.0])
        assert np.array_equal(got, expected, equal_nan=True)
