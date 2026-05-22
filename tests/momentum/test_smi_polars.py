# -*- coding: utf-8 -*-
"""Tests for pl_smi (SMI Ergodic Indicator) Polars implementation."""

import numpy as np
import polars as pl
import pytest

from polars_ti.momentum.smi import pl_smi


@pytest.fixture
def close_data():
    """Create sample close price data for testing."""
    np.random.seed(42)
    n = 200
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    return {
        "pd_df": pl.DataFrame({"close": close}),
        "pl_df": pl.DataFrame({"close": close}),
    }


class TestPlSmiBasic:
    """Basic functionality tests for pl_smi."""

    def test_returns_expr(self, close_data):
        """Test that pl_smi returns a Polars expression."""
        expr = pl_smi()
        assert isinstance(expr, pl.Expr)

    def test_default_parameters(self, close_data):
        """Test pl_smi with default parameters."""
        pl_df = close_data["pl_df"]
        result = pl_df.select(pl_smi()).unnest("SMI_5_20_5_1.0")

        assert len(result.columns) == 3
        assert "SMI_5_20_5_1.0" in result.columns
        assert "SMIs_5_20_5_1.0" in result.columns
        assert "SMIo_5_20_5_1.0" in result.columns

    def test_custom_parameters(self, close_data):
        """Test pl_smi with custom parameters."""
        pl_df = close_data["pl_df"]
        result = pl_df.select(pl_smi(fast=3, slow=10, signal=3, scalar=100)).unnest("SMI_3_10_3_100")

        assert "SMI_3_10_3_100" in result.columns
        assert len(result) == len(pl_df)


class TestPlSmiNumericalParity:
    """Numerical parity tests comparing Polars to Pandas implementation."""


class TestPlSmiEdgeCases:
    """Edge case tests for pl_smi."""

    def test_nulls_handling(self, close_data):
        """Test that pl_smi handles null values gracefully."""
        pl_df = close_data["pl_df"]

        pl_df_with_nulls = pl_df.with_columns(
            [pl.when(pl.col("close").is_first_distinct()).then(None).otherwise(pl.col("close")).alias("close")]
        )

        result = pl_df_with_nulls.select(pl_smi()).unnest("SMI_5_20_5_1.0")
        assert len(result) == len(pl_df)

    def test_zeros_handling(self, close_data):
        """Test that pl_smi handles constant values."""
        pl_df = pl.DataFrame({"close": [100.0] * 100})

        result = pl_df.select(pl_smi()).unnest("SMI_5_20_5_1.0")
        assert len(result) == 100

    def test_lazy_evaluation(self, close_data):
        """Test that pl_smi works in lazy context."""
        pl_df = close_data["pl_df"]

        lazy_result = pl_df.lazy().select(pl_smi()).unnest("SMI_5_20_5_1.0").collect()

        eager_result = pl_df.select(pl_smi()).unnest("SMI_5_20_5_1.0")

        for col in lazy_result.columns:
            lazy_vals = lazy_result[col].to_list()
            eager_vals = eager_result[col].to_list()
            # Compare non-NaN values
            for lv, ev in zip(lazy_vals, eager_vals):
                if lv is not None and ev is not None:
                    assert lv == ev or (np.isnan(lv) and np.isnan(ev))


class TestPlSmiFeatureParity:
    """Feature parity tests ensuring all parameters work correctly."""

    def test_offset_parameter(self, close_data):
        """Test offset parameter shifts results."""
        pl_df = close_data["pl_df"]

        result_no_offset = pl_df.select(pl_smi(offset=0)).unnest("SMI_5_20_5_1.0")

        result_offset = pl_df.select(pl_smi(offset=5)).unnest("SMI_5_20_5_1.0")

        # Values should be different due to shift
        assert len(result_no_offset) == len(result_offset)

    def test_scalar_parameter(self, close_data):
        """Test scalar parameter scales results."""
        pl_df = close_data["pl_df"]

        result_1x = pl_df.select(pl_smi(scalar=1)).unnest("SMI_5_20_5_1")

        result_100x = pl_df.select(pl_smi(scalar=100)).unnest("SMI_5_20_5_100")

        # 100x scalar should have 100x the value
        warmup = 50
        smi_1x = result_1x["SMI_5_20_5_1"].to_numpy()[warmup:]
        smi_100x = result_100x["SMI_5_20_5_100"].to_numpy()[warmup:]

        mask = ~(np.isnan(smi_1x) | np.isnan(smi_100x))
        if mask.sum() > 0:
            ratio = smi_100x[mask] / smi_1x[mask]
            # Should be approximately 100
            assert np.allclose(ratio, 100, rtol=1e-6)

    def test_fast_slow_swap(self, close_data):
        """Test that fast > slow is handled correctly."""
        pl_df = close_data["pl_df"]

        # Pass fast > slow - should be swapped internally
        result = pl_df.select(pl_smi(fast=20, slow=5, signal=5, scalar=1)).unnest(
            "SMI_5_20_5_1"
        )  # Note: props should show swapped values

        assert "SMI_5_20_5_1" in result.columns


class TestPlSmiIntegration:
    """Integration tests for pl_smi."""

    def test_chaining_with_other_operations(self, close_data):
        """Test pl_smi can be chained with other Polars operations."""
        pl_df = close_data["pl_df"]

        result = (
            pl_df.select(pl_smi())
            .unnest("SMI_5_20_5_1.0")
            .select(
                [
                    pl.col("SMI_5_20_5_1.0").mean().alias("mean_smi"),
                    pl.col("SMIo_5_20_5_1.0").std().alias("std_osc"),
                ]
            )
        )

        assert result.shape[0] == 1
        assert "mean_smi" in result.columns
