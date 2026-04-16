# -*- coding: utf-8 -*-
"""Tests for pl_decreasing."""
import numpy as np
import polars as pl
import pytest
from polars_ti.trend.decreasing import pl_decreasing
import pandas as pd  # REMOVED: pandas dependency  # Restored for fixtures


class TestPlDecreasing:
    @pytest.fixture
    def sample_df(self) -> pl.DataFrame:
        np.random.seed(42)
        return pl.DataFrame({
            'close': 100 + np.cumsum(np.random.randn(100) * 0.5),
        })

    def test_returns_expression(self, sample_df):
        result = sample_df.select(pl_decreasing("close"))
        assert result.height == 100

    def test_output_has_correct_alias(self, sample_df):
        result = sample_df.select(pl_decreasing("close", length=5))
        assert "DEC_5" in result.columns

    def test_numerical_parity_non_strict(self, sample_df):
        """Numerical parity with Pandas implementation (non-strict)."""
        pytest.skip("Pandas implementation removed in Phase 4 purge")
        pd_close = pd.Series(sample_df["close"].to_numpy())
        pd_result = decreasing(pd_close, length=3, strict=False, asint=True)
        
        pl_result = sample_df.select(pl_decreasing("close", length=3, strict=False, asint=True))
        pl_arr = pl_result[pl_result.columns[0]].to_numpy()
        pd_arr = pd_result.to_numpy()
        
        mask = ~np.isnan(pd_arr) & ~np.isnan(pl_arr.astype(float))
        assert np.allclose(pl_arr[mask], pd_arr[mask])

    def test_offset_shifts_result(self, sample_df):
        result = sample_df.select(pl_decreasing("close", offset=5))
        arr = result[result.columns[0]].to_numpy()
        assert all(v is None or np.isnan(float(v)) for v in arr[:5] if v is not None)

    def test_asint_parameter(self, sample_df):
        r_int = sample_df.select(pl_decreasing("close", asint=True))
        r_bool = sample_df.select(pl_decreasing("close", asint=False))
        assert r_int[r_int.columns[0]].dtype == pl.Int64
        assert r_bool[r_bool.columns[0]].dtype == pl.Boolean

    def test_with_null_values(self):
        df = pl.DataFrame({"close": [None] + [100.0 - i for i in range(49)]})
        result = df.select(pl_decreasing("close", length=2))
        assert result.height == 50

    def test_with_zeros(self):
        df = pl.DataFrame({"close": [50.0 - i*0.5 for i in range(50)]})
        result = df.select(pl_decreasing("close", length=2))
        assert result.height == 50

    def test_lazy_execution(self, sample_df):
        lazy_df = sample_df.lazy()
        result = lazy_df.select(pl_decreasing("close", length=3)).collect()
        assert "DEC_3" in result.columns
