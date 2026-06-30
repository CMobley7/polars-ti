# -*- coding: utf-8 -*-
"""Unit tests for polars_ti/candles/ha.py Polars implementation."""

import numpy as np
import polars as pl
import pytest

from polars_ti.candles.ha import ha, ha_apply


class TestPlHa:
    """Tests for pl_ha and pl_ha_apply."""

    @pytest.fixture
    def sample_df(self):
        """Create sample OHLC DataFrame."""
        np.random.seed(42)
        n = 20
        high = np.random.randn(n).cumsum() + 102
        low = high - np.abs(np.random.randn(n) * 2)
        open_ = low + np.random.rand(n) * (high - low)
        close = low + np.random.rand(n) * (high - low)
        return pl.DataFrame({"open": open_, "high": high, "low": low, "close": close})

    def test_ha_returns_expression(self):
        """Test that ha() returns a Polars expression."""
        expr = ha("open", "high", "low", "close")
        assert isinstance(expr, pl.Expr)

    def test_pl_ha_apply_adds_columns(self, sample_df):
        """Test that pl_ha_apply adds HA columns."""
        result = ha_apply(sample_df)
        assert "HA_open" in result.columns
        assert "HA_high" in result.columns
        assert "HA_low" in result.columns
        assert "HA_close" in result.columns

    def test_ha_close_is_ohlc_average(self, sample_df):
        """Test that HA_close is average of OHLC."""
        result = ha_apply(sample_df)
        expected_ha_close = (sample_df["open"] + sample_df["high"] + sample_df["low"] + sample_df["close"]) / 4
        diff = (result["HA_close"] - expected_ha_close).abs().max()
        assert diff < 1e-10

    def test_ha_high_ge_ha_open_close(self, sample_df):
        """Test that HA_high >= max(HA_open, HA_close)."""
        result = ha_apply(sample_df)
        ha_high = result["HA_high"].to_numpy()
        ha_open = result["HA_open"].to_numpy()
        ha_close = result["HA_close"].to_numpy()
        assert (ha_high >= np.maximum(ha_open, ha_close) - 1e-10).all()

    def test_ha_low_le_ha_open_close(self, sample_df):
        """Test that HA_low <= min(HA_open, HA_close)."""
        result = ha_apply(sample_df)
        ha_low = result["HA_low"].to_numpy()
        ha_open = result["HA_open"].to_numpy()
        ha_close = result["HA_close"].to_numpy()
        assert (ha_low <= np.minimum(ha_open, ha_close) + 1e-10).all()

    def test_with_null_values(self):
        """Handles null values gracefully (may produce NaN in results)."""
        df = pl.DataFrame(
            {
                "open": [None] + [100.0] * 19,
                "high": [110.0] * 20,
                "low": [90.0] * 20,
                "close": [105.0] * 20,
            }
        )
        # Should not crash; result may have NaNs
        result = ha_apply(df)
        assert result.height == 20
        assert "HA_close" in result.columns

    def test_with_zeros(self):
        """Handles zero values."""
        df = pl.DataFrame(
            {
                "open": [0.0] * 5 + [100.0] * 15,
                "high": [0.0] * 5 + [110.0] * 15,
                "low": [0.0] * 5 + [90.0] * 15,
                "close": [0.0] * 5 + [105.0] * 15,
            }
        )
        result = ha_apply(df)
        assert result.height == 20

    def test_preserves_original_columns(self, sample_df):
        """Original columns are preserved."""
        result = ha_apply(sample_df)
        assert "open" in result.columns
        assert "high" in result.columns
        assert "low" in result.columns
        assert "close" in result.columns
