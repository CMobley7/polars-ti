# -*- coding: utf-8 -*-
"""Unit tests for polars_ti/overlap/tema.py Polars implementation."""
import numpy as np
import pandas as pd  # REMOVED: pandas dependency  # Restored for fixtures
import polars as pl
import pytest

from polars_ti.overlap.tema import pl_tema


class TestPlTema:
    """Tests for pl_tema - Triple Exponential Moving Average."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        np.random.seed(42)
        close = 100 + np.random.randn(200).cumsum()
        return {
            'pd_series': pd.Series(close, name='close'),
            'pl_df': pl.DataFrame({'close': close}),
        }

    def test_numerical_parity_length_10(self, sample_data):
        """Test numerical parity with Pandas for length=10."""
        pytest.skip("Pandas implementation removed in Phase 4 purge")
        pd_result = tema(sample_data['pd_series'], length=10)
        pl_result = sample_data['pl_df'].select(pl_tema('close', length=10)).to_series()
        
        warmup = 30
        pd_vals = pd_result.iloc[warmup:].values
        pl_vals = pl_result[warmup:].to_numpy()
        
        valid = ~np.isnan(pd_vals) & ~np.isnan(pl_vals)
        diff = np.abs(pd_vals[valid] - pl_vals[valid])
        assert np.max(diff) < 1e-10, f"Max diff: {np.max(diff)}"

    def test_output_has_correct_alias(self, sample_data):
        """Test that output column has correct alias."""
        result = sample_data['pl_df'].select(pl_tema('close', length=10))
        assert result.columns[0] == 'TEMA_10'

    def test_offset_shifts_result(self, sample_data):
        """Test that offset parameter shifts the result."""
        no_offset = sample_data['pl_df'].select(pl_tema('close', length=10)).to_series()
        with_offset = sample_data['pl_df'].select(pl_tema('close', length=10, offset=5)).to_series()
        
        for i in range(30, 50):
            if not np.isnan(no_offset[i]):
                assert no_offset[i] == with_offset[i + 5], f"Offset mismatch at {i}"

    def test_warmup_period_has_nan(self, sample_data):
        """Test that warmup period contains NaN values."""
        result = sample_data['pl_df'].select(pl_tema('close', length=10)).to_series()
        # TEMA needs 3*length-2 warmup
        assert result[:27].is_nan().all()

    def test_with_null_values(self):
        """Handles null values gracefully."""
        df = pl.DataFrame({"close": [None] + [100.0] * 59})
        result = df.select(pl_tema("close", length=10))
        assert result.height == 60

    def test_with_zeros(self):
        """Handles zero values."""
        df = pl.DataFrame({"close": [0.0] * 5 + [100.0] * 55})
        result = df.select(pl_tema("close", length=10))
        assert result.height == 60

    def test_lazy_execution(self, sample_data):
        """Works with LazyFrame."""
        lazy_df = sample_data['pl_df'].lazy()
        result = lazy_df.select(pl_tema("close", length=10)).collect()
        assert "TEMA_10" in result.columns
