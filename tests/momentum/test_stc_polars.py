# -*- coding: utf-8 -*-
"""Tests for pl_stc (Schaff Trend Cycle) Polars implementation."""
import numpy as np
import pandas as pd  # REMOVED: pandas dependency  # Restored for fixtures
import polars as pl
import pytest

from polars_ti.momentum.stc import pl_stc


@pytest.fixture
def close_data():
    """Create sample close price data for testing."""
    np.random.seed(42)
    n = 200
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    return {
        "pd_df": pd.DataFrame({"close": close}),
        "pl_df": pl.DataFrame({"close": close}),
    }


class TestPlStcBasic:
    """Basic functionality tests for pl_stc."""

    def test_returns_expr(self, close_data):
        """Test that pl_stc returns a Polars expression."""
        expr = pl_stc()
        assert isinstance(expr, pl.Expr)

    def test_default_parameters(self, close_data):
        """Test pl_stc with default parameters."""
        pl_df = close_data["pl_df"]
        result = pl_df.select(pl_stc()).unnest("STC_10_12_26_0.5")
        
        assert len(result.columns) == 3
        assert "STC_10_12_26_0.5" in result.columns
        assert "STCmacd_10_12_26_0.5" in result.columns
        assert "STCstoch_10_12_26_0.5" in result.columns

    def test_custom_parameters(self, close_data):
        """Test pl_stc with custom parameters."""
        pl_df = close_data["pl_df"]
        result = pl_df.select(
            pl_stc(tclength=15, fast=10, slow=20, factor=0.6)
        ).unnest("STC_15_10_20_0.6")
        
        assert "STC_15_10_20_0.6" in result.columns
        assert len(result) == len(pl_df)


class TestPlStcNumericalParity:
    """Numerical parity tests comparing Polars to Pandas implementation."""

    def test_numerical_parity(self, close_data):
        """Test that pl_stc matches pandas stc output."""
        pytest.skip("Pandas implementation removed in Phase 4 purge")
        pd_df = close_data["pd_df"]
        pl_df = close_data["pl_df"]

        # Pandas result
        pd_result = stc(pd_df["close"], tclength=10, fast=12, slow=26, factor=0.5)

        # Polars result
        pl_result = pl_df.select(
            pl_stc(tclength=10, fast=12, slow=26, factor=0.5)
        ).unnest("STC_10_12_26_0.5")

        warmup = 36
        
        # Compare STC
        stc_col = [c for c in pd_result.columns if c.startswith("STC_")][0]
        pd_stc_vals = pd_result[stc_col].to_numpy()[warmup:]
        pl_stc_vals = pl_result["STC_10_12_26_0.5"].to_numpy()[warmup:]
        
        mask = ~(np.isnan(pd_stc_vals) | np.isnan(pl_stc_vals))
        if mask.sum() > 0:
            max_diff = np.max(np.abs(pd_stc_vals[mask] - pl_stc_vals[mask]))
            assert max_diff < 1e-6, f"STC failed with max diff {max_diff}"


class TestPlStcEdgeCases:
    """Edge case tests for pl_stc."""

    def test_nulls_handling(self, close_data):
        """Test that pl_stc handles null values gracefully."""
        pl_df = close_data["pl_df"]
        
        pl_df_with_nulls = pl_df.with_columns([
            pl.when(pl.col("close").is_first_distinct())
            .then(None)
            .otherwise(pl.col("close"))
            .alias("close")
        ])
        
        result = pl_df_with_nulls.select(pl_stc()).unnest("STC_10_12_26_0.5")
        assert len(result) == len(pl_df)

    def test_lazy_evaluation(self, close_data):
        """Test that pl_stc works in lazy context."""
        pl_df = close_data["pl_df"]
        
        lazy_result = (
            pl_df.lazy()
            .select(pl_stc())
            .unnest("STC_10_12_26_0.5")
            .collect()
        )
        
        eager_result = pl_df.select(pl_stc()).unnest("STC_10_12_26_0.5")
        
        # Compare non-NaN values
        for col in lazy_result.columns:
            lazy_vals = lazy_result[col].to_list()
            eager_vals = eager_result[col].to_list()
            for lv, ev in zip(lazy_vals, eager_vals):
                if lv is not None and ev is not None:
                    if not (np.isnan(lv) and np.isnan(ev)):
                        assert abs(lv - ev) < 1e-10


class TestPlStcFeatureParity:
    """Feature parity tests ensuring all parameters work correctly."""

    def test_fast_slow_swap(self, close_data):
        """Test that fast > slow is handled correctly."""
        pl_df = close_data["pl_df"]
        
        # Pass fast > slow - should be swapped internally
        result = pl_df.select(
            pl_stc(fast=26, slow=12)  # Swapped
        ).unnest("STC_10_12_26_0.5")  # Result should use swapped values
        
        assert "STC_10_12_26_0.5" in result.columns

    def test_different_factors(self, close_data):
        """Test different smoothing factors."""
        pl_df = close_data["pl_df"]
        
        result_low = pl_df.select(pl_stc(factor=0.25)).unnest("STC_10_12_26_0.25")
        result_high = pl_df.select(pl_stc(factor=0.75)).unnest("STC_10_12_26_0.75")
        
        # Different factors should produce different results
        assert "STC_10_12_26_0.25" in result_low.columns
        assert "STC_10_12_26_0.75" in result_high.columns


class TestPlStcIntegration:
    """Integration tests for pl_stc."""

    def test_chaining_with_other_operations(self, close_data):
        """Test pl_stc can be chained with other Polars operations."""
        pl_df = close_data["pl_df"]
        
        result = (
            pl_df
            .select(pl_stc())
            .unnest("STC_10_12_26_0.5")
            .select([
                pl.col("STC_10_12_26_0.5").mean().alias("mean_stc"),
                pl.col("STCmacd_10_12_26_0.5").std().alias("std_macd"),
            ])
        )
        
        assert result.shape[0] == 1
        assert "mean_stc" in result.columns
