# -*- coding: utf-8 -*-
"""Tests for pl_increasing."""
import numpy as np
import polars as pl
import pytest
from polars_ti.trend.increasing import pl_increasing
import pandas as pd  # REMOVED: pandas dependency  # Restored for fixtures


class TestPlIncreasing:
    @pytest.fixture
    def sample_df(self) -> pl.DataFrame:
        np.random.seed(42)
        return pl.DataFrame({
            'close': 100 + np.cumsum(np.random.randn(100) * 0.5),
        })

    def test_returns_expression(self, sample_df):
        result = sample_df.select(pl_increasing("close"))
        assert result.height == 100

    def test_output_has_correct_alias(self, sample_df):
        result = sample_df.select(pl_increasing("close", length=5))
        assert "INC_5" in result.columns

    def test_numerical_parity_non_strict(self, sample_df):
        """Numerical parity with Pandas implementation (non-strict)."""
        pytest.skip("Pandas implementation removed in Phase 4 purge")
        pd_close = pd.Series(sample_df["close"].to_numpy())
        pd_result = increasing(pd_close, length=3, strict=False, asint=True)
        
        pl_result = sample_df.select(pl_increasing("close", length=3, strict=False, asint=True))
        pl_arr = pl_result[pl_result.columns[0]].to_numpy()
        pd_arr = pd_result.to_numpy()
        
        # Filter valid values
        mask = ~np.isnan(pd_arr) & ~np.isnan(pl_arr.astype(float))
        assert np.allclose(pl_arr[mask], pd_arr[mask]), f"Max diff: {np.max(np.abs(pl_arr[mask] - pd_arr[mask]))}"

    def test_offset_shifts_result(self, sample_df):
        result = sample_df.select(pl_increasing("close", offset=5))
        arr = result[result.columns[0]].to_numpy()
        assert all(v is None or np.isnan(float(v)) for v in arr[:5] if v is not None)

    def test_asint_parameter(self, sample_df):
        r_int = sample_df.select(pl_increasing("close", asint=True))
        r_bool = sample_df.select(pl_increasing("close", asint=False))
        assert r_int[r_int.columns[0]].dtype == pl.Int64
        assert r_bool[r_bool.columns[0]].dtype == pl.Boolean

    def test_with_null_values(self):
        df = pl.DataFrame({"close": [None] + [100.0 + i for i in range(49)]})
        result = df.select(pl_increasing("close", length=2))
        assert result.height == 50

    def test_with_zeros(self):
        df = pl.DataFrame({"close": [0.0] * 5 + [float(i) for i in range(45)]})
        result = df.select(pl_increasing("close", length=2))
        assert result.height == 50

    def test_lazy_execution(self, sample_df):
        lazy_df = sample_df.lazy()
        result = lazy_df.select(pl_increasing("close", length=3)).collect()
        assert "INC_3" in result.columns
