# -*- coding: utf-8 -*-
"""Tests for pl_po (Projection Oscillator)."""
import numpy as np
import polars as pl
import pytest
import pandas as pd  # REMOVED: pandas dependency  # Restored for fixtures
from polars_ti.momentum.po import pl_po


class TestPlPo:
    @pytest.fixture
    def sample_df(self) -> pl.DataFrame:
        np.random.seed(42)
        n = 100
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        return pl.DataFrame({'close': close})

    def test_returns_expression(self, sample_df):
        """Test that pl_po returns a valid expression."""
        result = sample_df.select(pl_po("close"))
        assert result.height == 100

    def test_output_has_correct_alias(self, sample_df):
        """Test that output column has correct name."""
        result = sample_df.select(pl_po("close", length=10))
        assert "PO_10" in result.columns

    def test_numerical_parity_pandas(self, sample_df):
        """Numerical parity with Pandas implementation."""
        # Create pandas data directly
        pytest.skip("Pandas implementation removed in Phase 4 purge")
        np.random.seed(42)
        n = 100
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        sample_pdf = pd.DataFrame({'close': close})
        
        # Polars result
        polars_result = sample_df.select(pl_po("close", length=14))
        polars_arr = polars_result.to_numpy().flatten()
        
        # Pandas result
        pandas_result = po(sample_pdf['close'], length=14)
        pandas_arr = pandas_result.to_numpy()
        
        # Compare valid values after warmup
        warmup = 14
        valid_polars = polars_arr[warmup:]
        valid_pandas = pandas_arr[warmup:]
        
        mask = ~np.isnan(valid_polars) & ~np.isnan(valid_pandas)
        max_diff = np.max(np.abs(valid_polars[mask] - valid_pandas[mask]))
        
        assert max_diff < 1e-6, f"Max diff {max_diff} exceeds tolerance 1e-6"


    def test_offset_shifts_result(self, sample_df):
        """Test that offset parameter shifts results correctly."""
        result = sample_df.select(pl_po("close", offset=5))
        arr = result[result.columns[0]].to_numpy()
        # First 5 values should be null due to offset
        assert all(np.isnan(arr[:5]))

    def test_with_null_values(self):
        """Test handling of null values."""
        df = pl.DataFrame({
            "close": [None] + [100.0] * 49
        })
        result = df.select(pl_po("close", length=10))
        assert result.height == 50

    def test_with_zero_values(self):
        """Test handling of zero prices (division protection)."""
        df = pl.DataFrame({
            "close": [0.0] * 20 + [100.0] * 30
        })
        result = df.select(pl_po("close", length=10))
        arr = result[result.columns[0]].to_numpy()
        # Should handle zeros gracefully without inf
        assert not np.any(np.isinf(arr))

    def test_lazy_execution(self, sample_df):
        """Test that pl_po works with lazy frames."""
        lazy_df = sample_df.lazy()
        result = lazy_df.select(pl_po("close")).collect()
        assert "PO_14" in result.columns

    def test_different_lengths(self, sample_df):
        """Test with different length parameters."""
        for length in [5, 10, 20, 50]:
            result = sample_df.select(pl_po("close", length=length))
            assert f"PO_{length}" in result.columns
            # First length-1 values should be NaN
            arr = result[result.columns[0]].to_numpy()
            assert all(np.isnan(arr[:length-1]))
