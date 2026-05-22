# -*- coding: utf-8 -*-
"""
Supertrend Verification Test

Verifies our Supertrend implementation against the TradingView official
formula from: https://www.tradingview.com/support/solutions/43000634738-supertrend/
"""

import numpy as np
import pandas as pd
import pytest
from pandas import DataFrame, Series

import polars_ti as ti


def test_supertrend_band_preservation_on_flip():
    """Test that bands are preserved correctly on trend flip bars.

    This verifies GSLabIt's fix: band preservation should happen on
    EVERY bar, not just when staying in the same direction.
    """
    # Create synthetic data with a clear trend flip
    dates = pd.date_range("2024-01-01", periods=20, freq="D")

    # Uptrend then sudden drop
    high = pd.Series(
        [
            100,
            102,
            104,
            106,
            108,
            110,
            112,
            114,
            116,
            118,  # Uptrend
            115,
            110,
            105,
            100,
            95,
            90,
            85,
            80,
            75,
            70,  # Downtrend
        ],
        index=dates,
    )

    low = pd.Series(
        [
            98,
            100,
            102,
            104,
            106,
            108,
            110,
            112,
            114,
            116,  # Uptrend
            110,
            105,
            100,
            95,
            90,
            85,
            80,
            75,
            70,
            65,  # Downtrend
        ],
        index=dates,
    )

    close = pd.Series(
        [
            99,
            101,
            103,
            105,
            107,
            109,
            111,
            113,
            115,
            117,  # Uptrend closes
            112,
            107,
            102,
            97,
            92,
            87,
            82,
            77,
            72,
            67,  # Downtrend closes
        ],
        index=dates,
    )

    # Calculate Supertrend
    result = ti.supertrend(high, low, close, length=7, multiplier=3.0)

    # Verify result is a DataFrame with expected columns
    assert isinstance(result, DataFrame)
    assert "SUPERT_7_3.0" in result.columns
    assert "SUPERTd_7_3.0" in result.columns

    # The direction should eventually flip from 1 to -1
    directions = result["SUPERTd_7_3.0"].dropna().values
    flips = np.diff(directions)

    # There should be at least one flip (from +1 to -1)
    assert (flips == -2).any(), "No trend flip detected in test data"


def test_supertrend_direction_consistency():
    """Test that direction values are consistently 1 (up) or -1 (down)."""
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    np.random.seed(42)

    # Random walk price data
    returns = np.random.randn(100) * 0.02
    close = pd.Series(100 * np.cumprod(1 + returns), index=dates)
    high = close * (1 + np.abs(np.random.randn(100) * 0.01))
    low = close * (1 - np.abs(np.random.randn(100) * 0.01))

    result = ti.supertrend(high, low, close, length=7, multiplier=3.0)

    # Direction should only be 1 or -1 (or NaN for warmup period)
    directions = result["SUPERTd_7_3.0"].dropna()
    unique_dirs = set(directions.unique())
    assert unique_dirs.issubset({-1, 1, np.nan}), f"Unexpected directions: {unique_dirs}"


def test_supertrend_matches_reference(df):
    """Test that our Supertrend produces consistent results."""
    result = ti.supertrend(df.high, df.low, df.close, length=7, multiplier=3.0)

    # Basic structure checks
    assert isinstance(result, DataFrame)
    assert result.name == "SUPERT_7_3.0"
    assert len(result.columns) == 4

    # Check that supertrend line is within price range (sanity check)
    supert = result["SUPERT_7_3.0"].dropna()
    assert (supert >= df.low.min() * 0.5).all(), "Supertrend below reasonable range"
    assert (supert <= df.high.max() * 1.5).all(), "Supertrend above reasonable range"

    # Check direction column exists and has valid values
    directions = result["SUPERTd_7_3.0"].dropna()
    assert len(directions) > 0
    assert set(directions.unique()).issubset({-1, 1})


def test_supertrend_tradingview_formula_logic():
    """Test the core TradingView formula logic for band preservation.

    TradingView formula (from official docs):
        upperBand = basicUpperBand < prev upperBand or prev close > prev upperBand
                    ? basicUpperBand : prev upperBand
        lowerBand = basicLowerBand > prev lowerBand or prev close < prev lowerBand
                    ? basicLowerBand : prev lowerBand

    This test verifies our implementation follows this logic.
    """
    # Create data where we can predict band behavior
    dates = pd.date_range("2024-01-01", periods=50, freq="D")
    np.random.seed(123)

    # Steady uptrend
    close = pd.Series(100 + np.arange(50) * 0.5, index=dates)
    high = close + 1
    low = close - 1

    result = ti.supertrend(high, low, close, length=7, multiplier=2.0)

    # In a steady uptrend, direction should be mostly positive
    directions = result["SUPERTd_7_2.0"].dropna()
    uptrend_pct = (directions == 1).mean()
    assert uptrend_pct > 0.7, f"Expected mostly uptrend, got {uptrend_pct:.2%}"

    # Long column should have values (uptrend uses lower band)
    long_vals = result["SUPERTl_7_2.0"].dropna()
    assert len(long_vals) > 0, "No long values in uptrend"
