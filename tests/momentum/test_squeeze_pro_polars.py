# -*- coding: utf-8 -*-
"""Tests for pl_squeeze_pro (Squeeze PRO) Polars implementation."""
import numpy as np
import pandas as pd  # REMOVED: pandas dependency  # Restored for fixtures
import polars as pl
import pytest

from polars_ti.momentum.squeeze_pro import pl_squeeze_pro


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


class TestPlSqueezeProBasic:
    """Basic functionality tests for pl_squeeze_pro."""

    def test_returns_expr(self, ohlc_data):
        """Test that pl_squeeze_pro returns a Polars expression."""
        expr = pl_squeeze_pro()
        assert isinstance(expr, pl.Expr)

    def test_default_parameters(self, ohlc_data):
        """Test pl_squeeze_pro with default parameters."""
        pl_df = ohlc_data["pl_df"]
        result = pl_df.select(pl_squeeze_pro()).unnest("SQZPRO_20_2.0_20_2.0_1.5_1.0")
        
        assert len(result.columns) == 6
        assert "SQZPRO_20_2.0_20_2.0_1.5_1.0" in result.columns
        assert "SQZPRO_ON_WIDE" in result.columns
        assert "SQZPRO_ON_NORMAL" in result.columns
        assert "SQZPRO_ON_NARROW" in result.columns
        assert "SQZPRO_OFF" in result.columns
        assert "SQZPRO_NO" in result.columns

    def test_custom_parameters(self, ohlc_data):
        """Test pl_squeeze_pro with custom parameters."""
        pl_df = ohlc_data["pl_df"]
        result = pl_df.select(
            pl_squeeze_pro(
                bb_length=15, bb_std=1.5, kc_length=15,
                kc_scalar_wide=2.5, kc_scalar_normal=1.8, kc_scalar_narrow=1.2
            )
        ).unnest("SQZPRO_15_1.5_15_2.5_1.8_1.2")
        
        assert "SQZPRO_15_1.5_15_2.5_1.8_1.2" in result.columns
        assert len(result) == len(pl_df)


class TestPlSqueezeProNumericalParity:
    """Numerical parity tests comparing Polars to Pandas implementation."""

    def test_numerical_parity(self, ohlc_data):
        """Test that pl_squeeze_pro matches pandas squeeze_pro output."""
        pytest.skip("Pandas implementation removed in Phase 4 purge")
        pd_df = ohlc_data["pd_df"]
        pl_df = ohlc_data["pl_df"]

        # Pandas result
        pd_result = squeeze_pro(
            pd_df["high"], pd_df["low"], pd_df["close"],
            bb_length=20, bb_std=2.0, kc_length=20,
            kc_scalar_wide=2.0, kc_scalar_normal=1.5, kc_scalar_narrow=1.0,
            mom_length=12, mom_smooth=6, mamode="sma", tr=True
        )

        # Polars result
        pl_result = pl_df.select(
            pl_squeeze_pro(
                bb_length=20, bb_std=2.0, kc_length=20,
                kc_scalar_wide=2.0, kc_scalar_normal=1.5, kc_scalar_narrow=1.0,
                mom_length=12, mom_smooth=6, mamode="sma", use_tr=True
            )
        ).unnest("SQZPRO_20_2.0_20_2.0_1.5_1.0")

        warmup = 30
        sqzpro_col = [c for c in pd_result.columns if c.startswith("SQZPRO_") and not c.startswith("SQZPRO_ON") and not c.startswith("SQZPRO_OFF") and not c.startswith("SQZPRO_NO")][0]
        
        pd_sqz = pd_result[sqzpro_col].to_numpy()[warmup:]
        pl_sqz = pl_result["SQZPRO_20_2.0_20_2.0_1.5_1.0"].to_numpy()[warmup:]
        
        mask = ~(np.isnan(pd_sqz) | np.isnan(pl_sqz))
        if mask.sum() > 0:
            max_diff = np.max(np.abs(pd_sqz[mask] - pl_sqz[mask]))
            assert max_diff < 1e-6, f"SQZPRO failed with max diff {max_diff}"


class TestPlSqueezeProEdgeCases:
    """Edge case tests for pl_squeeze_pro."""

    def test_lazy_evaluation(self, ohlc_data):
        """Test that pl_squeeze_pro works in lazy context."""
        pl_df = ohlc_data["pl_df"]
        
        lazy_result = (
            pl_df.lazy()
            .select(pl_squeeze_pro())
            .unnest("SQZPRO_20_2.0_20_2.0_1.5_1.0")
            .collect()
        )
        
        eager_result = pl_df.select(pl_squeeze_pro()).unnest("SQZPRO_20_2.0_20_2.0_1.5_1.0")
        
        for col in ["SQZPRO_ON_WIDE", "SQZPRO_OFF", "SQZPRO_NO"]:
            assert lazy_result[col].to_list() == eager_result[col].to_list()

    def test_invalid_kc_scalars(self, ohlc_data):
        """Test that invalid kc scalars return None."""
        # wide must be > normal > narrow
        result = pl_squeeze_pro(kc_scalar_wide=1.0, kc_scalar_normal=1.5, kc_scalar_narrow=2.0)
        assert result is None


class TestPlSqueezeProFeatureParity:
    """Feature parity tests ensuring all parameters work correctly."""

    def test_mamode_ema(self, ohlc_data):
        """Test with EMA mode."""
        pl_df = ohlc_data["pl_df"]
        
        result = pl_df.select(
            pl_squeeze_pro(mamode="ema")
        ).unnest("SQZPRO_20_2.0_20_2.0_1.5_1.0")
        
        assert len(result) == len(pl_df)

    def test_asint_false(self, ohlc_data):
        """Test asint=False returns boolean types."""
        pl_df = ohlc_data["pl_df"]
        
        result = pl_df.select(
            pl_squeeze_pro(asint=False)
        ).unnest("SQZPRO_20_2.0_20_2.0_1.5_1.0")
        
        assert result["SQZPRO_ON_WIDE"].dtype == pl.Boolean
        assert result["SQZPRO_OFF"].dtype == pl.Boolean
        assert result["SQZPRO_NO"].dtype == pl.Boolean


class TestPlSqueezeProIntegration:
    """Integration tests for pl_squeeze_pro."""

    def test_chaining_with_other_operations(self, ohlc_data):
        """Test pl_squeeze_pro can be chained with other Polars operations."""
        pl_df = ohlc_data["pl_df"]
        
        result = (
            pl_df
            .select(pl_squeeze_pro())
            .unnest("SQZPRO_20_2.0_20_2.0_1.5_1.0")
            .select([
                pl.col("SQZPRO_ON_WIDE").sum().alias("total_wide"),
                pl.col("SQZPRO_ON_NARROW").sum().alias("total_narrow"),
            ])
        )
        
        assert result.shape[0] == 1
        assert "total_wide" in result.columns
