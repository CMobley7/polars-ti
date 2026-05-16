# -*- coding: utf-8 -*-
"""Tests for pl_cfo - Polars Chande Forecast Oscillator."""
import numpy as np
import polars as pl
import pytest
from polars_ti.momentum.cfo import pl_cfo


class TestPlCfo:
    @pytest.fixture
    def sample_df(self) -> pl.DataFrame:
        np.random.seed(42)
        n = 100
        return pl.DataFrame({
            'close': 100 + np.cumsum(np.random.randn(n) * 0.5),
        })

    def test_returns_expr(self):
        """pl_cfo returns a Polars expression."""
        expr = pl_cfo("close")
        assert isinstance(expr, pl.Expr)

    def test_has_cfo_column(self, sample_df):
        """Result contains CFO column with correct name."""
        result = sample_df.select(pl_cfo("close", length=9))
        assert "CFO_9" in result.columns

    def test_custom_length(self, sample_df):
        """Custom length parameter works."""
        result = sample_df.select(pl_cfo("close", length=20))
        assert "CFO_20" in result.columns

    def test_has_valid_values(self, sample_df):
        """Result has valid non-NaN values after warmup."""
        result = sample_df.select(pl_cfo("close", length=9))
        arr = result["CFO_9"].to_numpy()
        # After warmup (9 rows), should have valid values
        assert not np.isnan(arr[10:]).any()

    def test_with_null_values(self, sample_df):
        """Handles null values gracefully."""
        df_with_nulls = sample_df.with_columns(
            pl.when(pl.col("close").is_first_distinct())
            .then(None)
            .otherwise(pl.col("close"))
            .alias("close")
        )
        result = df_with_nulls.select(pl_cfo("close", length=9))
        assert result.height == sample_df.height

    def test_with_zeros(self):
        """Handles zero values (division check)."""
        df = pl.DataFrame({'close': [0.0] * 5 + [100.0] * 95})
        result = df.select(pl_cfo("close", length=9))
        # Should not crash, zeros might produce NaN/inf
        assert result.height == 100

    def test_lazy_execution(self, sample_df):
        """Works with LazyFrame."""
        lazy_df = sample_df.lazy()
        result = lazy_df.select(pl_cfo("close", length=9)).collect()
        assert "CFO_9" in result.columns
        assert result.height == sample_df.height

    def test_offset_parameter(self, sample_df):
        """Offset parameter shifts results."""
        result_no_offset = sample_df.select(pl_cfo("close", length=9, offset=0))
        result_offset = sample_df.select(pl_cfo("close", length=9, offset=2))
        # First 2 values of offset should be NaN
        assert np.isnan(result_offset["CFO_9"].to_numpy()[0])
        assert np.isnan(result_offset["CFO_9"].to_numpy()[1])

    def test_scalar_parameter(self, sample_df):
        """Scalar parameter scales results."""
        result_100 = sample_df.select(pl_cfo("close", length=9, scalar=100.0))
        result_1 = sample_df.select(pl_cfo("close", length=9, scalar=1.0))
        # At index 15, values should differ by factor of 100
        val_100 = result_100["CFO_9"].to_numpy()[15]
        val_1 = result_1["CFO_9"].to_numpy()[15]
        assert abs(val_100 / val_1 - 100) < 1e-10

