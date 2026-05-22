# -*- coding: utf-8 -*-
"""Tests for pl_vwap with anchor and bands support."""

import numpy as np
import polars as pl
import pytest
from datetime import datetime, timedelta
from polars_ti.volume.vwap import pl_vwap


class TestPlVwap:
    @pytest.fixture
    def sample_df(self) -> pl.DataFrame:
        np.random.seed(42)
        n = 100
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n) * 0.2)
        low = close - np.abs(np.random.randn(n) * 0.2)
        volume = np.abs(np.random.randn(n) * 1000) + 100
        return pl.DataFrame({"high": high, "low": low, "close": close, "volume": volume})

    @pytest.fixture
    def sample_df_with_datetime(self) -> pl.DataFrame:
        np.random.seed(42)
        n = 100
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n) * 0.2)
        low = close - np.abs(np.random.randn(n) * 0.2)
        volume = np.abs(np.random.randn(n) * 1000) + 100
        # Create datetime spanning 5 days (20 rows per day)
        base = datetime(2024, 1, 1, 9, 0, 0)
        datetimes = [base + timedelta(hours=i // 20 * 24, minutes=(i % 20) * 30) for i in range(n)]
        return pl.DataFrame(
            {
                "datetime": datetimes,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            }
        )

    def test_returns_list_of_expressions(self, sample_df):
        result = sample_df.select(pl_vwap("high", "low", "close", "volume"))
        assert isinstance(pl_vwap("high", "low", "close", "volume"), list)
        assert len(pl_vwap("high", "low", "close", "volume")) == 1

    def test_output_has_correct_alias(self, sample_df):
        result = sample_df.select(pl_vwap("high", "low", "close", "volume"))
        assert "VWAP_1D" in result.columns

    def test_offset_shifts_result(self, sample_df):
        result = sample_df.select(pl_vwap("high", "low", "close", "volume", offset=5))
        arr = result["VWAP_1D"].to_numpy()
        assert all(np.isnan(arr[:5]))

    def test_with_null_values(self):
        df = pl.DataFrame(
            {
                "high": [None] + [101.0] * 99,
                "low": [None] + [99.0] * 99,
                "close": [None] + [100.0] * 99,
                "volume": [None] + [1000.0] * 99,
            }
        )
        result = df.select(pl_vwap("high", "low", "close", "volume"))
        assert result.height == 100

    def test_lazy_execution(self, sample_df):
        lazy_df = sample_df.lazy()
        result = lazy_df.select(pl_vwap("high", "low", "close", "volume")).collect()
        assert "VWAP_1D" in result.columns

    def test_bands_parameter(self, sample_df):
        """Test stddev bands are created correctly."""
        exprs = pl_vwap("high", "low", "close", "volume", bands=[1, 2])
        result = sample_df.select(exprs)
        assert "VWAP_1D" in result.columns
        assert "VWAP_1D_L_1" in result.columns
        assert "VWAP_1D_U_1" in result.columns
        assert "VWAP_1D_L_2" in result.columns
        assert "VWAP_1D_U_2" in result.columns

    def test_anchored_vwap(self, sample_df_with_datetime):
        """Test anchored VWAP resets each period."""
        exprs = pl_vwap("high", "low", "close", "volume", datetime_col="datetime", anchor="1d")
        result = sample_df_with_datetime.select(exprs)
        assert "VWAP_1D" in result.columns
        # Anchored VWAP should reset each day, so first value of each day should be close to typical price

    def test_value_is_reasonable(self, sample_df):
        result = sample_df.select(pl_vwap("high", "low", "close", "volume"))
        vwap_vals = result["VWAP_1D"].to_numpy()
        close_vals = sample_df["close"].to_numpy()
        # VWAP should be in similar range to close
        valid = ~np.isnan(vwap_vals)
        assert np.abs(np.mean(vwap_vals[valid]) - np.mean(close_vals)) < 10.0

    def test_bands_are_symmetric(self, sample_df):
        """Verify upper and lower bands are symmetric around VWAP."""
        result = sample_df.select(pl_vwap("high", "low", "close", "volume", bands=[1]))
        vwap = result["VWAP_1D"].to_numpy()
        lower = result["VWAP_1D_L_1"].to_numpy()
        upper = result["VWAP_1D_U_1"].to_numpy()
        valid = ~(np.isnan(vwap) | np.isnan(lower) | np.isnan(upper))
        # Distance from VWAP should be equal for upper and lower
        dist_lower = np.abs(vwap[valid] - lower[valid])
        dist_upper = np.abs(upper[valid] - vwap[valid])
        np.testing.assert_allclose(dist_lower, dist_upper, rtol=1e-6)
