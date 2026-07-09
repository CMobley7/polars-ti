# -*- coding: utf-8 -*-
"""Tests for pl_inertia - Polars implementation."""

import numpy as np
import polars as pl
import pytest
from polars_ti.momentum.inertia import inertia


class TestPlInertia:
    @pytest.fixture
    def sample_df(self):
        np.random.seed(42)
        n = 200
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        return pl.DataFrame({"close": close})

    def test_returns_expr(self):
        expr = inertia("close")
        assert isinstance(expr, pl.Expr)

    def test_has_inertia_column(self, sample_df):
        result = sample_df.select(inertia("close"))
        assert "INERTIA_20_14" in result.columns

    def test_has_valid_values(self, sample_df):
        result = sample_df.select(inertia("close"))
        assert result["INERTIA_20_14"][55:].null_count() == 0

    def test_offset_parameter(self, sample_df):
        result_no_offset = sample_df.select(inertia("close", offset=0))
        result_with_offset = sample_df.select(inertia("close", offset=5))
        # Offset should add more nulls
        assert result_with_offset["INERTIA_20_14"].null_count() > result_no_offset["INERTIA_20_14"].null_count()

    def test_lazy_execution(self, sample_df):
        result = sample_df.lazy().select(inertia("close")).collect()
        assert "INERTIA_20_14" in result.columns

    def test_custom_lengths(self, sample_df):
        result = sample_df.select(inertia("close", length=10, rvi_length=7))
        assert "INERTIA_10_7" in result.columns


class TestPlInertiaAccessorModes:
    """Regression: the df.ti.inertia accessor must resolve high/low so the
    refined=True and thirds=True modes work (previously raised
    ColumnNotFoundError because high/low were never passed)."""

    @pytest.fixture
    def ohlc_df(self):
        np.random.seed(11)
        n = 200
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n)) * 0.3
        low = close - np.abs(np.random.randn(n)) * 0.3
        return pl.DataFrame({"high": high, "low": low, "close": close})

    def test_accessor_default(self, ohlc_df):
        out = ohlc_df.ti.inertia()
        assert out.height == 200

    def test_accessor_refined(self, ohlc_df):
        out = ohlc_df.ti.inertia(refined=True)
        assert out.height == 200

    def test_accessor_thirds(self, ohlc_df):
        out = ohlc_df.ti.inertia(thirds=True)
        assert out.height == 200
