# -*- coding: utf-8 -*-
"""Tests for pl_mom."""
import numpy as np
import polars as pl
import pytest
from polars_ti.momentum.mom import pl_mom
import pandas as pd  # REMOVED: pandas dependency  # Restored for fixtures


class TestPlMom:
    @pytest.fixture
    def sample_df(self) -> pl.DataFrame:
        np.random.seed(42)
        return pl.DataFrame({
            'close': 100 + np.cumsum(np.random.randn(100) * 0.5),
        })

    def test_returns_expression(self, sample_df):
        result = sample_df.select(pl_mom("close"))
        assert result.height == 100

    def test_output_has_correct_alias(self, sample_df):
        result = sample_df.select(pl_mom("close", length=5))
        assert "MOM_5" in result.columns

    def test_numerical_parity(self, sample_df):
        """Numerical parity with Pandas implementation."""
        pytest.skip("Pandas implementation removed in Phase 4 purge")
        pd_close = pd.Series(sample_df["close"].to_numpy())
        pd_result = mom(pd_close, length=10, talib=False)
        
        pl_result = sample_df.select(pl_mom("close", length=10, talib=False))
        pl_arr = pl_result[pl_result.columns[0]].to_numpy()
        pd_arr = pd_result.to_numpy()
        
        mask = ~np.isnan(pd_arr) & ~np.isnan(pl_arr)
        assert np.allclose(pl_arr[mask], pd_arr[mask], rtol=1e-6)

    def test_offset_shifts_result(self, sample_df):
        result = sample_df.select(pl_mom("close", offset=5))
        arr = result[result.columns[0]].to_numpy()
        assert all(np.isnan(arr[:5]))

    def test_talib_parameter(self, sample_df):
        """TA-Lib path produces valid results."""
        result = sample_df.select(pl_mom("close", length=10, talib=True))
        arr = result[result.columns[0]].to_numpy()
        valid = ~np.isnan(arr)
        assert valid.sum() > 50

    def test_with_null_values(self):
        df = pl.DataFrame({"close": [None] + [100.0 + i for i in range(49)]})
        result = df.select(pl_mom("close", length=5))
        assert result.height == 50

    def test_with_zeros(self):
        df = pl.DataFrame({"close": [0.0] * 5 + [float(i) for i in range(45)]})
        result = df.select(pl_mom("close", length=5))
        assert result.height == 50

    def test_lazy_execution(self, sample_df):
        lazy_df = sample_df.lazy()
        result = lazy_df.select(pl_mom("close", length=10)).collect()
        assert "MOM_10" in result.columns

    def test_mom_calculation_correct(self):
        """Verify MOM = close - close.shift(length)."""
        df = pl.DataFrame({"close": [float(i) for i in range(1, 21)]})
        result = df.select(pl_mom("close", length=10, talib=False))
        # MOM at index 10 = close[10] - close[0] = 11 - 1 = 10
        assert result["MOM_10"][10] == 10.0
