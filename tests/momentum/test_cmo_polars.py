# -*- coding: utf-8 -*-
"""Tests for pl_cmo - Pure Polars + TA-Lib implementation."""

import numpy as np
import polars as pl
import pytest
from polars_ti.momentum.cmo import cmo


class TestPlCmo:
    @pytest.fixture
    def sample_df(self):
        np.random.seed(42)
        n = 100
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        return pl.DataFrame({"close": close})

    def test_returns_expr(self):
        expr = cmo("close")
        assert isinstance(expr, pl.Expr)

    def test_has_cmo_column(self, sample_df):
        result = sample_df.select(cmo("close"))
        assert "CMO_14" in result.columns

    def test_has_valid_values(self, sample_df):
        result = sample_df.select(cmo("close"))
        assert result["CMO_14"][20:].null_count() == 0

    def test_offset_parameter(self, sample_df):
        result_no_offset = sample_df.select(cmo("close", offset=0, talib=False))
        result_with_offset = sample_df.select(cmo("close", offset=5, talib=False))
        # With offset 5, first 5 values are shifted so more nulls
        assert result_with_offset["CMO_14"].null_count() > result_no_offset["CMO_14"].null_count()

    def test_lazy_execution(self, sample_df):
        result = sample_df.lazy().select(cmo("close")).collect()
        assert "CMO_14" in result.columns

    def test_talib_true(self, sample_df):
        result = sample_df.select(cmo("close", talib=True))
        assert "CMO_14" in result.columns

    def test_talib_false(self, sample_df):
        result = sample_df.select(cmo("close", talib=False))
        assert "CMO_14" in result.columns

    def test_custom_length(self, sample_df):
        result = sample_df.select(cmo("close", length=20))
        assert "CMO_20" in result.columns

    def test_scalar_parameter(self, sample_df):
        result = sample_df.select(cmo("close", scalar=50.0, talib=False))
        assert "CMO_14" in result.columns

    def test_values_in_range(self, sample_df):
        """CMO should be between -100 and 100."""
        result = sample_df.select(cmo("close", talib=False))
        valid = result["CMO_14"].filter(~result["CMO_14"].is_nan())
        assert valid.min() >= -100
        assert valid.max() <= 100


class TestCmoTalibParity:
    """Native (talib=False) CMO must equal TA-Lib CMO.

    TA-Lib Wilder-smooths the up/down moves (``CMO == 2*RSI - 100``); the OLD
    native path used a flat rolling sum and diverged by ~63. The classic port
    switches the native path to Wilder smoothing so it matches TA-Lib exactly.
    """

    @pytest.fixture
    def spy(self):
        return pl.read_csv("data/SPY_D.csv", try_parse_dates=True).head(1500)

    def test_native_matches_talib_cmo(self, spy):
        pytest.importorskip("talib")
        import talib

        native = spy.select(cmo("close", talib=False))["CMO_14"].to_numpy()
        ref = talib.CMO(spy["close"].to_numpy().astype("float64"), timeperiod=14)
        m = ~np.isnan(native) & ~np.isnan(ref)
        assert m.sum() > 1000
        assert np.max(np.abs(native[m] - ref[m])) < 1e-6
