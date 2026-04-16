# -*- coding: utf-8 -*-
"""Tests for pl_remap."""
import numpy as np
import pandas as pd  # REMOVED: pandas dependency  # Restored for fixtures
import polars as pl
import pytest
from polars_ti.transform.remap import pl_remap


class TestPlRemap:
    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        return np.random.rand(100) * 100

    def test_returns_expression(self, sample_data):
        result = pl_remap("close")
        assert isinstance(result, pl.Expr)

    def test_output_has_correct_alias(self, sample_data):
        df = pl.DataFrame({"close": sample_data})
        result = df.select(pl_remap("close", from_min=0, from_max=100, to_min=-1, to_max=1))
        assert "REMAP_0" in result.columns[0]

    def test_numerical_parity(self, sample_data):
        pytest.skip("Pandas implementation removed in Phase 4 purge")
        pd_close = pd.Series(sample_data)
        pl_df = pl.DataFrame({"close": sample_data})
        pd_result = remap(pd_close, from_min=0, from_max=100, to_min=-1, to_max=1, offset=0)
        pl_result = pl_df.select(pl_remap("close", from_min=0, from_max=100, to_min=-1, to_max=1, offset=0))
        pd_vals = pd_result.to_numpy()
        pl_vals = pl_result[pl_result.columns[0]].to_numpy()
        mask = ~np.isnan(pd_vals) & ~np.isnan(pl_vals)
        max_diff = np.abs(pd_vals[mask] - pl_vals[mask]).max()
        assert max_diff < 1e-6

    def test_with_null_values(self, sample_data):
        data = sample_data.copy()
        data[10:15] = np.nan
        df = pl.DataFrame({"close": data})
        result = df.select(pl_remap("close"))
        assert result.height == 100

    def test_with_zeros(self, sample_data):
        data = sample_data.copy()
        data[20:25] = 0.0
        df = pl.DataFrame({"close": data})
        result = df.select(pl_remap("close"))
        assert result.height == 100

    def test_lazy_execution(self, sample_data):
        df = pl.DataFrame({"close": sample_data})
        result = df.lazy().select(pl_remap("close")).collect()
        assert result.height == 100
