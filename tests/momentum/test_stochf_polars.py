# -*- coding: utf-8 -*-
"""Tests for pl_stochf - Polars + Numba Fast Stochastic Oscillator with TA-Lib support."""

import numpy as np
import polars as pl
import pytest
from polars_ti.momentum.stochf import stochf as stochf_indicator


class TestPlStochf:
    @pytest.fixture
    def sample_df(self):
        np.random.seed(42)
        n = 100
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n) * 0.5)
        low = close - np.abs(np.random.randn(n) * 0.5)
        return pl.DataFrame({"high": high, "low": low, "close": close})

    def test_returns_expr(self):
        expr = stochf_indicator("high", "low", "close")
        assert isinstance(expr, pl.Expr)

    def test_has_stochf_column(self, sample_df):
        result = sample_df.select(stochf_indicator("high", "low", "close"))
        assert "STOCHF" in result.columns

    def test_struct_has_k_d_fields(self, sample_df):
        result = sample_df.select(stochf_indicator("high", "low", "close"))
        stochf = result["STOCHF"]
        assert "STOCHFk_14_3" in stochf.struct.fields
        assert "STOCHFd_14_3" in stochf.struct.fields

    def test_has_valid_values(self, sample_df):
        result = sample_df.select(stochf_indicator("high", "low", "close"))
        stochf = result["STOCHF"]
        k_vals = stochf.struct.field("STOCHFk_14_3")
        assert k_vals[20:].null_count() == 0

    def test_offset_parameter(self, sample_df):
        result = sample_df.select(stochf_indicator("high", "low", "close", offset=5))
        stochf = result["STOCHF"]
        k_vals = stochf.struct.field("STOCHFk_14_3")
        assert k_vals[:5].null_count() == 5

    def test_lazy_execution(self, sample_df):
        result = sample_df.lazy().select(stochf_indicator("high", "low", "close")).collect()
        assert "STOCHF" in result.columns

    def test_talib_true(self, sample_df):
        result = sample_df.select(stochf_indicator("high", "low", "close", talib=True))
        assert "STOCHF" in result.columns

    def test_talib_false(self, sample_df):
        result = sample_df.select(stochf_indicator("high", "low", "close", talib=False))
        assert "STOCHF" in result.columns

    def test_custom_k_parameter(self, sample_df):
        result = sample_df.select(stochf_indicator("high", "low", "close", k=20))
        stochf = result["STOCHF"]
        assert "STOCHFk_20_3" in stochf.struct.fields

    def test_custom_d_parameter(self, sample_df):
        result = sample_df.select(stochf_indicator("high", "low", "close", d=5))
        stochf = result["STOCHF"]
        assert "STOCHFd_14_5" in stochf.struct.fields

    def test_with_null_values(self):
        df = pl.DataFrame(
            {
                "high": [101.0, None, 103.0] + [102.0] * 50,
                "low": [99.0, 100.0, None] + [98.0] * 50,
                "close": [100.0, 101.0, 102.0] + [100.0] * 50,
            }
        )
        result = df.select(stochf_indicator("high", "low", "close", talib=False))
        assert result.height == 53

    def test_with_zeros(self):
        df = pl.DataFrame(
            {
                "high": [0.0] * 10 + [102.0] * 50,
                "low": [0.0] * 10 + [98.0] * 50,
                "close": [0.0] * 10 + [100.0] * 50,
            }
        )
        result = df.select(stochf_indicator("high", "low", "close", talib=False))
        assert result.height == 60

    def test_talib_parity(self):
        """Verify TA-Lib path produces same results as direct TA-Lib call."""
        try:
            from talib import STOCHF as TALIB_STOCHF
        except ImportError:
            pytest.skip("TA-Lib not installed")

        np.random.seed(42)
        n = 500
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n) * 0.5)
        low = close - np.abs(np.random.randn(n) * 0.5)

        pldf = pl.DataFrame({"high": high, "low": low, "close": close})

        talib_k, talib_d = TALIB_STOCHF(high, low, close, 14, 3, 0)

        polars_result = pldf.select(stochf_indicator("high", "low", "close", talib=True))
        stochf = polars_result["STOCHF"]
        polars_k = stochf.struct.field("STOCHFk_14_3").to_numpy()

        valid_mask = ~np.isnan(talib_k) & ~np.isnan(polars_k)
        max_diff = np.max(np.abs(talib_k[valid_mask] - polars_k[valid_mask]))
        assert max_diff < 1e-6, f"Max diff {max_diff} exceeds tolerance"
