# -*- coding: utf-8 -*-
"""Tests for pl_dm - Pure Polars + TA-Lib implementation."""

import numpy as np
import polars as pl
import pytest
from polars_ti.momentum.dm import dm


class TestPlDm:
    @pytest.fixture
    def sample_df(self):
        np.random.seed(42)
        n = 100
        high = 101 + np.cumsum(np.random.randn(n) * 0.5)
        low = high - np.abs(np.random.randn(n) * 0.3)
        return pl.DataFrame({"high": high, "low": low})

    def test_returns_list_of_expr(self):
        result = dm("high", "low")
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(e, pl.Expr) for e in result)

    def test_has_dm_columns(self, sample_df):
        result = sample_df.select(dm("high", "low"))
        assert "DMP_14" in result.columns
        assert "DMN_14" in result.columns

    def test_has_valid_values(self, sample_df):
        result = sample_df.select(dm("high", "low"))
        assert result["DMP_14"][20:].is_nan().sum() == 0
        assert result["DMN_14"][20:].is_nan().sum() == 0

    def test_offset_parameter(self, sample_df):
        result_no_offset = sample_df.select(dm("high", "low", offset=0))
        result_with_offset = sample_df.select(dm("high", "low", offset=5))
        assert result_with_offset["DMP_14"].null_count() > result_no_offset["DMP_14"].null_count()

    def test_lazy_execution(self, sample_df):
        result = sample_df.lazy().select(dm("high", "low")).collect()
        assert "DMP_14" in result.columns

    def test_talib_true(self, sample_df):
        result = sample_df.select(dm("high", "low", talib=True))
        assert "DMP_14" in result.columns

    def test_talib_false(self, sample_df):
        result = sample_df.select(dm("high", "low", talib=False))
        assert "DMP_14" in result.columns

    def test_values_non_negative(self, sample_df):
        """DM values should be non-negative."""
        result = sample_df.select(dm("high", "low", talib=False))
        valid_dmp = result["DMP_14"].filter(~result["DMP_14"].is_null())
        valid_dmn = result["DMN_14"].filter(~result["DMN_14"].is_null())
        assert valid_dmp.min() >= 0
        assert valid_dmn.min() >= 0

    def test_drift_parameter(self, sample_df):
        """Restored ``drift`` param (native path): default unchanged, drift=2 changes output."""
        dmp_def, _ = dm("high", "low", talib=False)
        dmp_one, _ = dm("high", "low", talib=False, drift=1)
        dmp_two, _ = dm("high", "low", talib=False, drift=2)

        a_def = sample_df.select(dmp_def)["DMP_14"].to_numpy()
        a_one = sample_df.select(dmp_one)["DMP_14"].to_numpy()
        a_two = sample_df.select(dmp_two)["DMP_14"].to_numpy()

        assert np.array_equal(a_def, a_one, equal_nan=True)
        assert np.nanmax(np.abs(a_two - a_def)) > 0.0


class TestDmTalibParity:
    """Native (talib=False) +DM/-DM must equal TA-Lib PLUS_DM/MINUS_DM.

    The OLD native path smoothed on the average scale via ``ma('rma')`` and
    diverged from TA-Lib by tens of points; the classic port switches the native
    path to Wilder sum-smoothing so it matches TA-Lib exactly.
    """

    @pytest.fixture
    def spy(self):
        return pl.read_csv("data/SPY_D.csv", try_parse_dates=True).head(1500)

    def test_native_dmp_matches_talib_plus_dm(self, spy):
        pytest.importorskip("talib")
        import talib

        native = spy.select(dm("high", "low", talib=False))["DMP_14"].to_numpy()
        ref = talib.PLUS_DM(
            spy["high"].to_numpy().astype("float64"),
            spy["low"].to_numpy().astype("float64"),
            timeperiod=14,
        )
        m = ~np.isnan(native) & ~np.isnan(ref)
        assert m.sum() > 1000
        assert np.max(np.abs(native[m] - ref[m])) < 1e-6

    def test_native_dmn_matches_talib_minus_dm(self, spy):
        pytest.importorskip("talib")
        import talib

        native = spy.select(dm("high", "low", talib=False))["DMN_14"].to_numpy()
        ref = talib.MINUS_DM(
            spy["high"].to_numpy().astype("float64"),
            spy["low"].to_numpy().astype("float64"),
            timeperiod=14,
        )
        m = ~np.isnan(native) & ~np.isnan(ref)
        assert m.sum() > 1000
        assert np.max(np.abs(native[m] - ref[m])) < 1e-6
