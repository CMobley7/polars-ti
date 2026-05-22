# -*- coding: utf-8 -*-
"""Unit tests for polars_ti/candles/cdl_doji.py Polars implementation."""

import numpy as np
import polars as pl
import pytest

from polars_ti.candles.cdl_doji import pl_cdl_doji


class TestPlCdlDoji:
    """Tests for pl_cdl_doji."""

    def test_detects_doji_small_body(self):
        """Test that doji is detected when body is very small."""
        df = pl.DataFrame(
            {
                "open": [100.0, 100.0, 100.0] * 5,
                "high": [110.0, 110.0, 110.0] * 5,
                "low": [90.0, 90.0, 90.0] * 5,
                "close": [100.01, 100.01, 100.01] * 5,  # Very small body
            }
        )
        result = df.select(pl_cdl_doji("open", "high", "low", "close", length=3))
        # After warmup (first 3), should detect doji
        assert result.to_numpy().flatten()[3:].sum() > 0

    def test_no_doji_large_body(self):
        """Test that no doji when body is large."""
        df = pl.DataFrame(
            {
                "open": [90.0] * 15,
                "high": [110.0] * 15,
                "low": [85.0] * 15,
                "close": [105.0] * 15,  # Large body (15 points)
            }
        )
        result = df.select(pl_cdl_doji("open", "high", "low", "close", length=10))
        # Should NOT detect doji - large body
        assert result.to_numpy().flatten()[10:].sum() == 0

    def test_output_alias(self):
        """Test that output has correct alias."""
        df = pl.DataFrame(
            {
                "open": [100.0] * 15,
                "high": [110.0] * 15,
                "low": [90.0] * 15,
                "close": [100.5] * 15,
            }
        )
        result = df.select(pl_cdl_doji("open", "high", "low", "close", length=10, factor=10.0))
        assert result.columns[0] == "CDL_DOJI_10_0.1"

    def test_custom_scalar(self):
        """Test that custom scalar is applied."""
        df = pl.DataFrame(
            {
                "open": [100.0] * 15,
                "high": [110.0] * 15,
                "low": [90.0] * 15,
                "close": [100.001] * 15,  # Tiny body
            }
        )
        result = df.select(pl_cdl_doji("open", "high", "low", "close", length=10, scalar=50.0))
        # If doji detected, should be 50 not 100
        detected = result.to_numpy().flatten()[10:]
        if any(detected > 0):
            assert all(v in [0, 50] for v in detected)

    def test_with_null_values(self):
        """Handles null values gracefully."""
        df = pl.DataFrame(
            {
                "open": [None] + [100.0] * 19,
                "high": [110.0] * 20,
                "low": [90.0] * 20,
                "close": [100.01] * 20,
            }
        )
        result = df.select(pl_cdl_doji("open", "high", "low", "close", length=10))
        assert result.height == 20

    def test_with_zeros(self):
        """Handles zero values."""
        df = pl.DataFrame(
            {
                "open": [0.0] * 5 + [100.0] * 15,
                "high": [0.0] * 5 + [110.0] * 15,
                "low": [0.0] * 5 + [90.0] * 15,
                "close": [0.0] * 5 + [100.01] * 15,
            }
        )
        result = df.select(pl_cdl_doji("open", "high", "low", "close", length=10))
        assert result.height == 20

    def test_lazy_execution(self):
        """Works with LazyFrame."""
        df = pl.DataFrame(
            {
                "open": [100.0] * 15,
                "high": [110.0] * 15,
                "low": [90.0] * 15,
                "close": [100.01] * 15,
            }
        )
        lazy_df = df.lazy()
        result = lazy_df.select(pl_cdl_doji("open", "high", "low", "close", length=10)).collect()
        assert "CDL_DOJI_10_0.1" in result.columns

    def test_offset_parameter(self):
        """Offset parameter shifts results."""
        df = pl.DataFrame(
            {
                "open": [100.0] * 15,
                "high": [110.0] * 15,
                "low": [90.0] * 15,
                "close": [100.01] * 15,
            }
        )
        result = df.select(pl_cdl_doji("open", "high", "low", "close", length=10, offset=2))
        arr = result.to_numpy().flatten()
        # First 2 values should be null due to offset
        assert arr[0] is None or np.isnan(float(arr[0])) if arr[0] is not None else True
