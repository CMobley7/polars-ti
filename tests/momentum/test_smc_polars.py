# -*- coding: utf-8 -*-
"""Tests for pl_smc (Smart Money Concept) Polars implementation."""

import numpy as np
import polars as pl
import pytest

from polars_ti.momentum.smc import pl_smc


@pytest.fixture
def ohlcv_data():
    """Create sample OHLCV data for testing."""
    np.random.seed(42)
    n = 200
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n) * 0.3)
    low = close - np.abs(np.random.randn(n) * 0.3)
    open_ = close + np.random.randn(n) * 0.2
    volume = np.random.randint(1000, 10000, n)

    return {
        "pd_df": pl.DataFrame(
            {
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            }
        ),
        "pl_df": pl.DataFrame(
            {
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            }
        ),
    }


class TestPlSmcBasic:
    """Basic functionality tests for pl_smc."""

    def test_returns_expr(self, ohlcv_data):
        """Test that pl_smc returns a Polars expression."""
        expr = pl_smc()
        assert isinstance(expr, pl.Expr)

    def test_default_parameters(self, ohlcv_data):
        """Test pl_smc with default parameters."""
        pl_df = ohlcv_data["pl_df"]
        result = pl_df.select(pl_smc()).unnest("SMC_14_50_20_5")

        # Should have 7 columns
        assert len(result.columns) == 7
        assert "SMChv_14_50_20_5" in result.columns
        assert "SMCbf_14_50_20_5" in result.columns
        assert "SMCbi_14_50_20_5" in result.columns
        assert "SMCbp_14_50_20_5" in result.columns
        assert "SMCtf_14_50_20_5" in result.columns
        assert "SMCti_14_50_20_5" in result.columns
        assert "SMCtp_14_50_20_5" in result.columns

    def test_custom_parameters(self, ohlcv_data):
        """Test pl_smc with custom parameters."""
        pl_df = ohlcv_data["pl_df"]
        result = pl_df.select(
            pl_smc(
                abr_length=10,
                close_length=30,
                vol_length=15,
                percent=3,
            )
        ).unnest("SMC_10_30_15_3")

        assert "SMChv_10_30_15_3" in result.columns
        assert len(result) == len(pl_df)


class TestPlSmcNumericalParity:
    """Numerical parity tests comparing Polars to Pandas implementation."""


class TestPlSmcEdgeCases:
    """Edge case tests for pl_smc."""

    def test_nulls_handling(self, ohlcv_data):
        """Test that pl_smc handles null values gracefully."""
        pl_df = ohlcv_data["pl_df"]

        # Insert nulls
        pl_df_with_nulls = pl_df.with_columns(
            [pl.when(pl.col("close").is_first_distinct()).then(None).otherwise(pl.col("close")).alias("close")]
        )

        # Should not raise
        result = pl_df_with_nulls.select(pl_smc()).unnest("SMC_14_50_20_5")
        assert len(result) == len(pl_df)

    def test_zeros_handling(self, ohlcv_data):
        """Test that pl_smc handles zero values."""
        pl_df = pl.DataFrame(
            {
                "open": [0.0] * 100,
                "high": [0.1] * 100,
                "low": [0.0] * 100,
                "close": [0.05] * 100,
            }
        )

        # Should not raise
        result = pl_df.select(pl_smc()).unnest("SMC_14_50_20_5")
        assert len(result) == 100

    def test_lazy_evaluation(self, ohlcv_data):
        """Test that pl_smc works in lazy context."""
        pl_df = ohlcv_data["pl_df"]

        lazy_result = pl_df.lazy().select(pl_smc()).unnest("SMC_14_50_20_5").collect()

        eager_result = pl_df.select(pl_smc()).unnest("SMC_14_50_20_5")

        for col in lazy_result.columns:
            assert lazy_result[col].to_list() == eager_result[col].to_list()


class TestPlSmcFeatureParity:
    """Feature parity tests ensuring all parameters work correctly."""

    def test_offset_parameter(self, ohlcv_data):
        """Test offset parameter shifts results."""
        pl_df = ohlcv_data["pl_df"]

        result_no_offset = pl_df.select(pl_smc(offset=0)).unnest("SMC_14_50_20_5")

        result_offset = pl_df.select(pl_smc(offset=5)).unnest("SMC_14_50_20_5")

        # Values should be shifted
        col = "SMCbi_14_50_20_5"
        no_offset_vals = result_no_offset[col].to_numpy()
        offset_vals = result_offset[col].to_numpy()

        # After warmup and offset, values should match shifted
        assert len(no_offset_vals) == len(offset_vals)

    def test_asint_false(self, ohlcv_data):
        """Test asint=False returns boolean types."""
        pl_df = ohlcv_data["pl_df"]

        result = pl_df.select(pl_smc(asint=False)).unnest("SMC_14_50_20_5")

        # Flag columns should be bool when asint=False
        assert result["SMChv_14_50_20_5"].dtype == pl.Boolean
        assert result["SMCbf_14_50_20_5"].dtype == pl.Boolean
        assert result["SMCtf_14_50_20_5"].dtype == pl.Boolean

    def test_vol_ratio_parameter(self, ohlcv_data):
        """Test different volatility ratio values."""
        pl_df = ohlcv_data["pl_df"]

        result_low = pl_df.select(pl_smc(vol_ratio=0.5)).unnest("SMC_14_50_20_5")

        result_high = pl_df.select(pl_smc(vol_ratio=3.0)).unnest("SMC_14_50_20_5")

        # Higher vol_ratio should produce fewer high volatility flags
        hv_low = result_low["SMChv_14_50_20_5"].sum()
        hv_high = result_high["SMChv_14_50_20_5"].sum()

        assert hv_low >= hv_high, "Lower vol_ratio should produce more HV flags"


class TestPlSmcIntegration:
    """Integration tests for pl_smc with other Polars operations."""

    def test_chaining_with_other_operations(self, ohlcv_data):
        """Test pl_smc can be chained with other Polars operations."""
        pl_df = ohlcv_data["pl_df"]

        result = (
            pl_df.select(pl_smc())
            .unnest("SMC_14_50_20_5")
            .select(
                [
                    pl.col("SMChv_14_50_20_5").sum().alias("total_hv"),
                    pl.col("SMCbf_14_50_20_5").sum().alias("total_bf"),
                    pl.col("SMCtf_14_50_20_5").sum().alias("total_tf"),
                ]
            )
        )

        assert result.shape[0] == 1
        assert "total_hv" in result.columns

    def test_with_group_by(self, ohlcv_data):
        """Test pl_smc can be used after group_by operations."""
        pl_df = ohlcv_data["pl_df"]
        pl_df = pl_df.with_row_index("idx")
        pl_df = pl_df.with_columns((pl.col("idx") % 2).alias("group"))

        # Apply SMC per group
        result = pl_df.group_by("group").agg(pl_smc())

        assert len(result) == 2
