# -*- coding: utf-8 -*-
"""Tests for pl_rma."""
import numpy as np
import pandas as pd  # REMOVED: pandas dependency  # Restored for fixtures
import polars as pl
import pytest
from polars_ti.overlap.rma import pl_rma


class TestPlRma:
    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        n = 100
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        return {"close": close}

    def test_returns_expression(self, sample_data):
        result = pl_rma("close")
        assert isinstance(result, pl.Expr)

    def test_output_has_correct_alias(self, sample_data):
        df = pl.DataFrame(sample_data)
        result = df.select(pl_rma("close", length=14))
        assert result.columns[0] == "RMA_14"

    def test_numerical_parity(self, sample_data):
        pytest.skip("Pandas implementation removed in Phase 4 purge")
        pd_close = pd.Series(sample_data["close"])
        pl_df = pl.DataFrame(sample_data)
        pd_result = rma(pd_close, length=14)
        pl_result = pl_df.select(pl_rma("close", length=14))
        warmup = 20
        pd_vals = pd_result.to_numpy()[warmup:]
        pl_vals = pl_result[pl_result.columns[0]].to_numpy()[warmup:]
        mask = np.isfinite(pd_vals) & np.isfinite(pl_vals)
        if mask.sum() > 0:
            max_diff = np.abs(pd_vals[mask] - pl_vals[mask]).max()
            assert max_diff < 1e-6

    def test_presma_true_mode(self, sample_data):
        """Test that presma=True mode starts at index length-1."""
        df = pl.DataFrame(sample_data)
        result = df.select(pl_rma("close", length=14, presma=True))
        vals = result[result.columns[0]].to_numpy()
        # First 13 values should be NaN
        assert np.all(np.isnan(vals[:13]))
        # Value at index 13 should be valid
        assert np.isfinite(vals[13])

    def test_with_null_values(self, sample_data):
        data = sample_data.copy()
        data["close"] = data["close"].copy()
        data["close"][10:15] = np.nan
        df = pl.DataFrame(data)
        result = df.select(pl_rma("close"))
        assert result.height == 100

    def test_with_zeros(self, sample_data):
        data = sample_data.copy()
        data["close"] = data["close"].copy()
        data["close"][50:55] = 0.0
        df = pl.DataFrame(data)
        result = df.select(pl_rma("close"))
        assert result.height == 100

    def test_lazy_execution(self, sample_data):
        df = pl.DataFrame(sample_data)
        result = df.lazy().select(pl_rma("close")).collect()
        assert result.height == 100
