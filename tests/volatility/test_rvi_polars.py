# -*- coding: utf-8 -*-
"""Tests for pl_rvi."""

import numpy as np
import polars as pl
import pytest
from polars_ti.volatility.rvi import rvi


class TestPlRvi:
    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        n = 100
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n) * 0.3)
        low = close - np.abs(np.random.randn(n) * 0.3)
        return {"close": close, "high": high, "low": low}

    def test_returns_expression(self, sample_data):
        result = rvi("close")
        assert isinstance(result, pl.Expr)

    def test_output_has_correct_alias(self, sample_data):
        df = pl.DataFrame(sample_data)
        result = df.select(rvi("close"))
        assert "RVI" in result.columns[0]

    def test_with_null_values(self, sample_data):
        data = sample_data.copy()
        data["close"] = np.array(data["close"])
        data["close"][10:15] = np.nan
        df = pl.DataFrame(data)
        result = df.select(rvi("close"))
        assert result.height == 100

    def test_with_zeros(self, sample_data):
        data = sample_data.copy()
        data["close"] = np.array(data["close"])
        data["close"][50:55] = data["close"][49]  # Flat region (0 diff)
        df = pl.DataFrame(data)
        result = df.select(rvi("close"))
        assert result.height == 100

    def test_lazy_execution(self, sample_data):
        df = pl.DataFrame(sample_data)
        result = df.lazy().select(rvi("close")).collect()
        assert result.height == 100

    def test_mamode_dispatch_restored(self, sample_data):
        """Regression: non-sma/ema mamodes route through the ma() dispatcher.

        Previously any mamode other than 'sma'/'ema' silently fell back to EMA.
        A non-default MA type must now change the output, while the default
        (mamode='ema') remains byte-identical.
        """
        df = pl.DataFrame(sample_data)
        default = df.select(rvi("close")).to_numpy().ravel()
        explicit_ema = df.select(rvi("close", mamode="ema")).to_numpy().ravel()
        wma = df.select(rvi("close", mamode="wma")).to_numpy().ravel()
        rma = df.select(rvi("close", mamode="rma")).to_numpy().ravel()

        # Default output unchanged (ema path preserved byte-for-byte).
        np.testing.assert_array_equal(np.isnan(default), np.isnan(explicit_ema))
        m = ~np.isnan(default)
        assert np.array_equal(default[m], explicit_ema[m])

        # Non-default MA types now actually change the result.
        assert not np.allclose(default, wma, equal_nan=True)
        assert not np.allclose(default, rma, equal_nan=True)
        # wma is no longer a silent alias of the ema fallback.
        assert not np.allclose(explicit_ema, wma, equal_nan=True)
