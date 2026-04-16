# -*- coding: utf-8 -*-
"""Tests for pl_squeeze (TTM Squeeze) Polars implementation."""
import numpy as np
import pandas as pd  # REMOVED: pandas dependency  # Restored for fixtures
import polars as pl
import pytest

from polars_ti.momentum.squeeze import pl_squeeze


@pytest.fixture
def ohlc_data():
    """Create sample OHLC data for testing."""
    np.random.seed(42)
    n = 200
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n) * 0.3)
    low = close - np.abs(np.random.randn(n) * 0.3)
    return {
        "pd_df": pd.DataFrame({"high": high, "low": low, "close": close}),
        "pl_df": pl.DataFrame({"high": high, "low": low, "close": close}),
    }


class TestPlSqueezeBasic:
    """Basic functionality tests for pl_squeeze."""

    def test_returns_expr(self, ohlc_data):
        """Test that pl_squeeze returns a Polars expression."""
        expr = pl_squeeze()
        assert isinstance(expr, pl.Expr)

    def test_default_parameters(self, ohlc_data):
        """Test pl_squeeze with default parameters."""
        pl_df = ohlc_data["pl_df"]
        result = pl_df.select(pl_squeeze()).unnest("SQZ_20_2.0_20_1.5")
        
        assert len(result.columns) == 4
        assert "SQZ_20_2.0_20_1.5" in result.columns
        assert "SQZ_ON" in result.columns
        assert "SQZ_OFF" in result.columns
        assert "SQZ_NO" in result.columns

    def test_custom_parameters(self, ohlc_data):
        """Test pl_squeeze with custom parameters."""
        pl_df = ohlc_data["pl_df"]
        result = pl_df.select(
            pl_squeeze(bb_length=10, bb_std=1.5, kc_length=15, kc_scalar=2.0)
        ).unnest("SQZ_10_1.5_15_2.0")
        
        assert "SQZ_10_1.5_15_2.0" in result.columns
        assert len(result) == len(pl_df)


class TestPlSqueezeNumericalParity:
    """Numerical parity tests comparing Polars to Pandas implementation."""

    def test_numerical_parity(self, ohlc_data):
        """Test that pl_squeeze matches pandas squeeze output."""
        pytest.skip("Pandas implementation removed in Phase 4 purge")
        pd_df = ohlc_data["pd_df"]
        pl_df = ohlc_data["pl_df"]

        # Pandas result
        pd_result = squeeze(
            pd_df["high"], pd_df["low"], pd_df["close"],
            bb_length=20, bb_std=2.0, kc_length=20, kc_scalar=1.5,
            mom_length=12, mom_smooth=6, mamode="sma", tr=True
        )

        # Polars result
        pl_result = pl_df.select(
            pl_squeeze(
                bb_length=20, bb_std=2.0, kc_length=20, kc_scalar=1.5,
                mom_length=12, mom_smooth=6, mamode="sma", use_tr=True
            )
        ).unnest("SQZ_20_2.0_20_1.5")

        warmup = 30
        sqz_col = [c for c in pd_result.columns if c.startswith("SQZ_")][0]
        
        pd_sqz = pd_result[sqz_col].to_numpy()[warmup:]
        pl_sqz = pl_result["SQZ_20_2.0_20_1.5"].to_numpy()[warmup:]
        
        mask = ~(np.isnan(pd_sqz) | np.isnan(pl_sqz))
        if mask.sum() > 0:
            max_diff = np.max(np.abs(pd_sqz[mask] - pl_sqz[mask]))
            assert max_diff < 1e-6, f"SQZ failed with max diff {max_diff}"

        # Compare flags
        for col in ["SQZ_ON", "SQZ_OFF", "SQZ_NO"]:
            pd_vals = pd_result[col].to_numpy()[warmup:].astype(float)
            pl_vals = pl_result[col].to_numpy()[warmup:].astype(float)
            max_diff = np.max(np.abs(pd_vals - pl_vals))
            assert max_diff < 1e-6, f"{col} failed with max diff {max_diff}"


class TestPlSqueezeEdgeCases:
    """Edge case tests for pl_squeeze."""

    def test_nulls_handling(self, ohlc_data):
        """Test that pl_squeeze handles null values gracefully."""
        pl_df = ohlc_data["pl_df"]
        
        pl_df_with_nulls = pl_df.with_columns([
            pl.when(pl.col("close").is_first_distinct())
            .then(None)
            .otherwise(pl.col("close"))
            .alias("close")
        ])
        
        result = pl_df_with_nulls.select(pl_squeeze()).unnest("SQZ_20_2.0_20_1.5")
        assert len(result) == len(pl_df)

    def test_lazy_evaluation(self, ohlc_data):
        """Test that pl_squeeze works in lazy context."""
        pl_df = ohlc_data["pl_df"]
        
        lazy_result = (
            pl_df.lazy()
            .select(pl_squeeze())
            .unnest("SQZ_20_2.0_20_1.5")
            .collect()
        )
        
        eager_result = pl_df.select(pl_squeeze()).unnest("SQZ_20_2.0_20_1.5")
        
        for col in ["SQZ_ON", "SQZ_OFF", "SQZ_NO"]:
            assert lazy_result[col].to_list() == eager_result[col].to_list()


class TestPlSqueezeFeatureParity:
    """Feature parity tests ensuring all parameters work correctly."""

    def test_mamode_ema(self, ohlc_data):
        """Test with EMA mode."""
        pl_df = ohlc_data["pl_df"]
        
        result = pl_df.select(
            pl_squeeze(mamode="ema")
        ).unnest("SQZ_20_2.0_20_1.5")
        
        assert len(result) == len(pl_df)

    def test_use_tr_false(self, ohlc_data):
        """Test with use_tr=False."""
        pl_df = ohlc_data["pl_df"]
        
        result = pl_df.select(
            pl_squeeze(use_tr=False)
        ).unnest("SQZhlr_20_2.0_20_1.5")
        
        assert "SQZhlr_20_2.0_20_1.5" in result.columns

    def test_asint_false(self, ohlc_data):
        """Test asint=False returns boolean types."""
        pl_df = ohlc_data["pl_df"]
        
        result = pl_df.select(
            pl_squeeze(asint=False)
        ).unnest("SQZ_20_2.0_20_1.5")
        
        assert result["SQZ_ON"].dtype == pl.Boolean
        assert result["SQZ_OFF"].dtype == pl.Boolean
        assert result["SQZ_NO"].dtype == pl.Boolean


class TestPlSqueezeIntegration:
    """Integration tests for pl_squeeze."""

    def test_chaining_with_other_operations(self, ohlc_data):
        """Test pl_squeeze can be chained with other Polars operations."""
        pl_df = ohlc_data["pl_df"]
        
        result = (
            pl_df
            .select(pl_squeeze())
            .unnest("SQZ_20_2.0_20_1.5")
            .select([
                pl.col("SQZ_ON").sum().alias("total_on"),
                pl.col("SQZ_OFF").sum().alias("total_off"),
            ])
        )
        
        assert result.shape[0] == 1
        assert "total_on" in result.columns
