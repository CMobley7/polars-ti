# -*- coding: utf-8 -*-
"""Unit tests for polars_ti/utils/_metrics.py Polars utilities."""
import numpy as np
import polars as pl
import pytest

from polars_ti.utils._metrics import (
    pl_log_return,
    pl_percent_return,
    pl_cumulative_return,
    pl_rolling_volatility,
    pl_drawdown,
    pl_max_drawdown,
)


class TestPlLogReturn:
    """Tests for pl_log_return."""

    def test_basic_calculation(self):
        """Test basic log return calculation."""
        df = pl.DataFrame({"close": [100.0, 110.0, 121.0]})
        result = df.select(pl_log_return("close"))["log_return"]
        # log(110/100) ≈ 0.0953, log(121/110) ≈ 0.0953
        values = result.to_numpy()
        assert np.isnan(values[0])  # First is NaN
        assert abs(values[1] - np.log(110/100)) < 1e-10
        assert abs(values[2] - np.log(121/110)) < 1e-10


class TestPlPercentReturn:
    """Tests for pl_percent_return."""

    def test_basic_calculation(self):
        """Test basic percent return calculation."""
        df = pl.DataFrame({"close": [100.0, 110.0, 99.0]})
        result = df.select(pl_percent_return("close"))["pct_return"]
        values = result.to_numpy()
        assert np.isnan(values[0])  # First is NaN
        assert abs(values[1] - 0.10) < 1e-10  # 10% increase
        assert abs(values[2] - (-0.10)) < 1e-10  # 10% decrease


class TestPlCumulativeReturn:
    """Tests for pl_cumulative_return."""

    def test_basic_calculation(self):
        """Test cumulative return from first value."""
        df = pl.DataFrame({"close": [100.0, 110.0, 121.0]})
        result = df.select(pl_cumulative_return("close"))["cum_return"]
        values = result.to_numpy()
        assert values[0] == 0.0  # No return at start
        assert abs(values[1] - 0.10) < 1e-10  # 10%
        assert abs(values[2] - 0.21) < 1e-10  # 21%


class TestPlDrawdown:
    """Tests for pl_drawdown."""

    def test_basic_calculation(self):
        """Test drawdown from peak calculation."""
        df = pl.DataFrame({"close": [100.0, 110.0, 100.0, 90.0]})
        result = df.select(pl_drawdown("close"))["drawdown"]
        values = result.to_numpy()
        assert values[0] == 0.0  # At peak
        assert values[1] == 0.0  # New peak
        assert abs(values[2] - (-0.0909)) < 0.01  # Down from 110
        assert abs(values[3] - (-0.1818)) < 0.01  # Down from 110


class TestPlMaxDrawdown:
    """Tests for pl_max_drawdown."""

    def test_basic_calculation(self):
        """Test max drawdown calculation."""
        df = pl.DataFrame({"close": [100.0, 110.0, 100.0, 90.0, 95.0]})
        result = pl_max_drawdown(df, "close")
        # Max drawdown is -18.18% (from 110 to 90)
        assert abs(result - (-0.1818)) < 0.01


class TestPlRollingVolatility:
    """Tests for pl_rolling_volatility."""

    def test_basic_calculation(self):
        """Test rolling volatility calculation."""
        np.random.seed(42)
        close = np.random.randn(50).cumsum() + 100
        df = pl.DataFrame({"close": close.tolist()})
        result = df.select(pl_rolling_volatility("close", length=10))["volatility_10"]
        values = result.to_numpy()
        # First 9 should be null, rest should be positive
        assert np.isnan(values[:9]).all()
        assert (values[10:] > 0).all()
