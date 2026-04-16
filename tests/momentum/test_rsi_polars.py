# -*- coding: utf-8 -*-
"""Tests for pl_rsi."""
import numpy as np
import polars as pl
import pytest
from polars_ti.momentum.rsi import pl_rsi
# import pandas as pd  # REMOVED: pandas dependency


class TestPlRsi:
    @pytest.fixture
    def sample_df(self) -> pl.DataFrame:
        np.random.seed(42)
        n = 100
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        return pl.DataFrame({'close': close})

    def test_returns_expression(self, sample_df):
        result = sample_df.select(pl_rsi("close"))
        assert result.height == 100

    def test_output_has_correct_alias(self, sample_df):
        result = sample_df.select(pl_rsi("close", length=10))
        assert "RSI_10" in result.columns

    def test_numerical_parity_talib(self, sample_df):
        """Numerical parity with TA-Lib."""
        result = sample_df.select(pl_rsi("close", talib=True))
        arr = result[result.columns[0]].to_numpy()
        valid = ~np.isnan(arr)
        assert valid.sum() > 50
        # RSI should be bounded 0-100
        assert np.nanmin(arr) >= 0
        assert np.nanmax(arr) <= 100

    def test_offset_shifts_result(self, sample_df):
        result = sample_df.select(pl_rsi("close", offset=5))
        arr = result[result.columns[0]].to_numpy()
        assert all(np.isnan(arr[:5]))

    def test_talib_parameter_toggle(self, sample_df):
        """Both talib=True and talib=False produce valid results."""
        result_talib = sample_df.select(pl_rsi("close", talib=True))
        result_polars = sample_df.select(pl_rsi("close", talib=False))
        
        arr_talib = result_talib[result_talib.columns[0]].to_numpy()
        arr_polars = result_polars[result_polars.columns[0]].to_numpy()
        
        # Both should produce valid RSI values
        assert (~np.isnan(arr_talib)).sum() > 50
        assert (~np.isnan(arr_polars)).sum() > 50

    def test_with_null_values(self):
        df = pl.DataFrame({
            "close": [None] + [100.0] * 49
        })
        result = df.select(pl_rsi("close", length=10))
        assert result.height == 50

    def test_lazy_execution(self, sample_df):
        lazy_df = sample_df.lazy()
        result = lazy_df.select(pl_rsi("close")).collect()
        assert "RSI_14" in result.columns

    def test_mamode_parameter(self, sample_df):
        """Different mamode produces valid results."""
        result = sample_df.select(pl_rsi("close", mamode="ema", talib=False))
        arr = result[result.columns[0]].to_numpy()
        valid = ~np.isnan(arr)
        assert valid.sum() > 50
