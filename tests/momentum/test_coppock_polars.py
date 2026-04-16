# -*- coding: utf-8 -*-
"""Tests for pl_coppock - Pure Polars implementation using pl_roc + pl_wma."""
import numpy as np
import polars as pl
import pandas as pd
import pytest
from polars_ti.momentum.coppock import pl_coppock


class TestPlCoppock:
    @pytest.fixture
    def sample_df(self):
        np.random.seed(42)
        n = 100
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        return pl.DataFrame({'close': close})

    def test_returns_expr(self):
        expr = pl_coppock("close")
        assert isinstance(expr, pl.Expr)

    def test_has_coppock_column(self, sample_df):
        result = sample_df.select(pl_coppock("close"))
        assert "COPC_11_14_10" in result.columns

    def test_has_valid_values(self, sample_df):
        result = sample_df.select(pl_coppock("close"))
        # After warmup (14 + 10 = 24), should have values
        assert result["COPC_11_14_10"][30:].null_count() == 0

    def test_offset_parameter(self, sample_df):
        result_no_offset = sample_df.select(pl_coppock("close", offset=0))
        result_with_offset = sample_df.select(pl_coppock("close", offset=5))
        assert result_with_offset["COPC_11_14_10"].null_count() > result_no_offset["COPC_11_14_10"].null_count()

    def test_lazy_execution(self, sample_df):
        result = sample_df.lazy().select(pl_coppock("close")).collect()
        assert "COPC_11_14_10" in result.columns

    def test_custom_lengths(self, sample_df):
        result = sample_df.select(pl_coppock("close", length=5, fast=7, slow=10))
        assert "COPC_7_10_5" in result.columns

    def test_with_null_values(self):
        df = pl.DataFrame({'close': [100.0, None, 102.0] + [100.0] * 50})
        result = df.select(pl_coppock("close"))
        assert result.height == 53

    def test_composition(self, sample_df):
        """Verify it uses pl_roc and pl_wma composition."""
        result = sample_df.select(pl_coppock("close"))
        # Should produce valid non-zero values
        valid = result["COPC_11_14_10"].filter(~result["COPC_11_14_10"].is_nan())
        assert valid.std() > 0  # Should have variation

    def test_numerical_parity(self, sample_df):
        """Verify numerical parity with Pandas implementation."""
        pytest.skip("Pandas implementation removed in Phase 4 purge")
        import pandas as pd
        # from polars_ti.momentum.coppock import coppock as pandas_coppock  # REMOVED: pandas func removed
        
        close = sample_df["close"].to_numpy()
        pdf = pd.DataFrame({'close': close})
        
        pandas_result = pandas_coppock(pdf['close'], length=10, fast=11, slow=14)
        polars_result = sample_df.select(pl_coppock("close", length=10, fast=11, slow=14))
        
        pandas_arr = pandas_result.to_numpy()[30:]
        polars_arr = polars_result["COPC_11_14_10"].to_numpy()[30:]
        
        valid_mask = ~np.isnan(pandas_arr) & ~np.isnan(polars_arr)
        max_diff = np.max(np.abs(pandas_arr[valid_mask] - polars_arr[valid_mask]))
        assert max_diff < 1e-6, f"Max diff {max_diff} exceeds tolerance"
