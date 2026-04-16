# -*- coding: utf-8 -*-
"""Tests for pl_fvg."""
import numpy as np
import pandas as pd  # REMOVED: pandas dependency  # Restored for fixtures
import polars as pl
import pytest
from polars_ti.volatility.fvg import pl_fvg


class TestPlFvg:
    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        n = 100
        open_ = 100 + np.cumsum(np.random.randn(n) * 0.5)
        close = open_ + np.random.randn(n) * 0.8
        high = np.maximum(open_, close) + np.abs(np.random.randn(n) * 0.3)
        low = np.minimum(open_, close) - np.abs(np.random.randn(n) * 0.3)
        return {"open": open_, "high": high, "low": low, "close": close}

    def test_returns_expression(self, sample_data):
        result = pl_fvg("open", "high", "low", "close")
        assert isinstance(result, pl.Expr)

    def test_output_has_correct_alias(self, sample_data):
        df = pl.DataFrame(sample_data)
        result = df.select(pl_fvg("open", "high", "low", "close"))
        assert "FVG" in result.columns[0]

    def test_numerical_parity(self, sample_data):
        pytest.skip("Pandas implementation removed in Phase 4 purge")
        pd_open = pd.Series(sample_data["open"])
        pd_high = pd.Series(sample_data["high"])
        pd_low = pd.Series(sample_data["low"])
        pd_close = pd.Series(sample_data["close"])
        pl_df = pl.DataFrame(sample_data)
        pd_result = fvg(pd_open, pd_high, pd_low, pd_close, min_gap=0)
        pl_result = pl_df.select(pl_fvg("open", "high", "low", "close", min_gap=0)).unnest("FVG_0")
        pd_type = pd_result["FVGt_0"].to_numpy()
        pl_type = pl_result["fvg_type"].to_numpy()
        # Count matches
        match_count = sum(1 for i in range(len(pd_type)) if (np.isnan(pd_type[i]) and np.isnan(pl_type[i])) or pd_type[i] == pl_type[i])
        assert match_count == len(pd_type)

    def test_with_null_values(self, sample_data):
        data = sample_data.copy()
        data["close"] = data["close"].copy()
        data["close"][10:15] = np.nan
        df = pl.DataFrame(data)
        result = df.select(pl_fvg("open", "high", "low", "close"))
        assert result.height == 100

    def test_with_zeros(self, sample_data):
        data = sample_data.copy()
        data["close"] = data["close"].copy()
        data["close"][50:55] = 50.0
        df = pl.DataFrame(data)
        result = df.select(pl_fvg("open", "high", "low", "close"))
        assert result.height == 100

    def test_lazy_execution(self, sample_data):
        df = pl.DataFrame(sample_data)
        result = df.lazy().select(pl_fvg("open", "high", "low", "close")).collect()
        assert result.height == 100
