# -*- coding: utf-8 -*-
"""Tests for pl_squeeze_pro (Squeeze PRO) Polars implementation."""

import numpy as np
import polars as pl
import pytest

from polars_ti.momentum.squeeze_pro import squeeze_pro


@pytest.fixture
def ohlc_data():
    """Create sample OHLC data for testing."""
    np.random.seed(42)
    n = 200
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n) * 0.3)
    low = close - np.abs(np.random.randn(n) * 0.3)
    return {
        "pd_df": pl.DataFrame({"high": high, "low": low, "close": close}),
        "pl_df": pl.DataFrame({"high": high, "low": low, "close": close}),
    }


class TestPlSqueezeProBasic:
    """Basic functionality tests for pl_squeeze_pro."""

    def test_returns_expr(self, ohlc_data):
        """Test that pl_squeeze_pro returns a Polars expression."""
        expr = squeeze_pro()
        assert isinstance(expr, pl.Expr)

    def test_default_parameters(self, ohlc_data):
        """Test pl_squeeze_pro with default parameters."""
        pl_df = ohlc_data["pl_df"]
        result = pl_df.select(squeeze_pro()).unnest("SQZPRO_20_2.0_20_2.0_1.5_1.0")

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
            squeeze_pro(
                bb_length=15,
                bb_std=1.5,
                kc_length=15,
                kc_scalar_wide=2.5,
                kc_scalar_normal=1.8,
                kc_scalar_narrow=1.2,
            )
        ).unnest("SQZPRO_15_1.5_15_2.5_1.8_1.2")

        assert "SQZPRO_15_1.5_15_2.5_1.8_1.2" in result.columns
        assert len(result) == len(pl_df)


class TestPlSqueezeProNumericalParity:
    """Numerical parity tests comparing Polars to Pandas implementation."""


class TestPlSqueezeProEdgeCases:
    """Edge case tests for pl_squeeze_pro."""

    def test_lazy_evaluation(self, ohlc_data):
        """Test that pl_squeeze_pro works in lazy context."""
        pl_df = ohlc_data["pl_df"]

        lazy_result = pl_df.lazy().select(squeeze_pro()).unnest("SQZPRO_20_2.0_20_2.0_1.5_1.0").collect()

        eager_result = pl_df.select(squeeze_pro()).unnest("SQZPRO_20_2.0_20_2.0_1.5_1.0")

        for col in ["SQZPRO_ON_WIDE", "SQZPRO_OFF", "SQZPRO_NO"]:
            assert lazy_result[col].to_list() == eager_result[col].to_list()

    def test_invalid_kc_scalars(self, ohlc_data):
        """Test that invalid kc scalars return None."""
        # wide must be > normal > narrow
        result = squeeze_pro(kc_scalar_wide=1.0, kc_scalar_normal=1.5, kc_scalar_narrow=2.0)
        assert result is None


class TestPlSqueezeProFeatureParity:
    """Feature parity tests ensuring all parameters work correctly."""

    def test_mamode_ema(self, ohlc_data):
        """Test with EMA mode."""
        pl_df = ohlc_data["pl_df"]

        result = pl_df.select(squeeze_pro(mamode="ema")).unnest("SQZPRO_20_2.0_20_2.0_1.5_1.0")

        assert len(result) == len(pl_df)

    def test_asint_false(self, ohlc_data):
        """Test asint=False returns boolean types."""
        pl_df = ohlc_data["pl_df"]

        result = pl_df.select(squeeze_pro(asint=False)).unnest("SQZPRO_20_2.0_20_2.0_1.5_1.0")

        assert result["SQZPRO_ON_WIDE"].dtype == pl.Boolean
        assert result["SQZPRO_OFF"].dtype == pl.Boolean
        assert result["SQZPRO_NO"].dtype == pl.Boolean


class TestPlSqueezeProIntegration:
    """Integration tests for pl_squeeze_pro."""

    def test_chaining_with_other_operations(self, ohlc_data):
        """Test pl_squeeze_pro can be chained with other Polars operations."""
        pl_df = ohlc_data["pl_df"]

        result = (
            pl_df.select(squeeze_pro())
            .unnest("SQZPRO_20_2.0_20_2.0_1.5_1.0")
            .select(
                [
                    pl.col("SQZPRO_ON_WIDE").sum().alias("total_wide"),
                    pl.col("SQZPRO_ON_NARROW").sum().alias("total_narrow"),
                ]
            )
        )

        assert result.shape[0] == 1
        assert "total_wide" in result.columns
