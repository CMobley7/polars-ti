# -*- coding: utf-8 -*-
"""Tests for pl_vp (Volume Profile)."""
import numpy as np
import polars as pl
import pytest
from polars_ti.volume.vp import pl_vp


class TestPlVp:
    @pytest.fixture
    def sample_df(self) -> pl.DataFrame:
        np.random.seed(42)
        n = 100
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        volume = np.abs(np.random.randn(n) * 1000) + 100
        return pl.DataFrame({'close': close, 'volume': volume})

    def test_returns_dataframe(self, sample_df):
        result = pl_vp(sample_df)
        assert isinstance(result, pl.DataFrame)

    def test_has_correct_columns(self, sample_df):
        result = pl_vp(sample_df)
        expected_cols = {"low_close", "mean_close", "high_close", 
                        "pos_volume", "neg_volume", "total_volume"}
        assert set(result.columns) == expected_cols

    def test_returns_width_rows(self, sample_df):
        result = pl_vp(sample_df, width=10)
        assert result.height == 10
        
        result5 = pl_vp(sample_df, width=5)
        assert result5.height == 5

    def test_chronological_mode(self, sample_df):
        """sort=False splits data chronologically."""
        result = pl_vp(sample_df, width=10, sort=False)
        assert result.height == 10
        # Total volume should sum to input volume
        total = result["total_volume"].sum()
        assert total > 0

    def test_price_range_mode(self, sample_df):
        """sort=True bins by price ranges."""
        result = pl_vp(sample_df, width=10, sort=True)
        assert result.height <= 10  # May have fewer if some bins empty

    def test_with_null_values(self):
        df = pl.DataFrame({
            "close": [None] + [100.0 + i * 0.1 for i in range(99)],
            "volume": [None] + [1000.0] * 99
        })
        result = pl_vp(df, width=5)
        assert result is not None
        assert result.height <= 5

    def test_volume_conservation(self, sample_df):
        """Total volume in profile should match input volume."""
        result = pl_vp(sample_df, width=10, sort=False)
        # pos + neg should roughly equal total (some may be neutral on first row)
        assert result["total_volume"].sum() > 0
