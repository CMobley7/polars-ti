# -*- coding: utf-8 -*-
"""Tests for pl_hl2."""

import numpy as np
import polars as pl
import pytest
from polars_ti.overlap.hl2 import hl2


class TestPlHl2:
    """Test suite for pl_hl2 Polars implementation."""

    @pytest.fixture
    def sample_df(self) -> pl.DataFrame:
        np.random.seed(42)
        return pl.DataFrame(
            {
                "high": 102 + np.random.randn(100),
                "low": 98 + np.random.randn(100),
            }
        )

    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        high = 102 + np.random.randn(100)
        low = 98 + np.random.randn(100)
        return {
            "pd_high": high,
            "pd_low": low,
            "pl_df": pl.DataFrame({"high": high, "low": low}),
        }

    def test_returns_correct_column(self, sample_df):
        result = sample_df.select(hl2("high", "low"))
        assert "HL2" in result.columns

    def test_formula_correct(self, sample_df):
        result = sample_df.select(hl2("high", "low"))
        expected = (sample_df["high"] + sample_df["low"]) / 2
        np.testing.assert_array_almost_equal(result["HL2"].to_numpy(), expected.to_numpy())

    def test_with_expressions(self, sample_df):
        result = sample_df.select(hl2(pl.col("high"), pl.col("low")))
        assert "HL2" in result.columns

    def test_with_null_values(self):
        """Handles null values gracefully."""
        df = pl.DataFrame(
            {
                "high": [None] + [102.0] * 29,
                "low": [None] + [98.0] * 29,
            }
        )
        result = df.select(hl2("high", "low"))
        assert result.height == 30

    def test_with_zeros(self):
        """Handles zero values."""
        df = pl.DataFrame(
            {
                "high": [0.0] * 5 + [102.0] * 25,
                "low": [0.0] * 5 + [98.0] * 25,
            }
        )
        result = df.select(hl2("high", "low"))
        assert result.height == 30

    def test_lazy_execution(self, sample_df):
        """Works with LazyFrame."""
        lazy_df = sample_df.lazy()
        result = lazy_df.select(hl2("high", "low")).collect()
        assert "HL2" in result.columns

    def test_talib_routes_to_medprice(self, sample_df):
        """talib=True routes to talib.MEDPRICE exactly."""
        talib = pytest.importorskip("talib")
        got = sample_df.select(hl2("high", "low", talib=True))["HL2"].to_numpy()
        ref = talib.MEDPRICE(
            sample_df["high"].to_numpy().astype(np.float64),
            sample_df["low"].to_numpy().astype(np.float64),
        )
        assert np.nanmax(np.abs(got - ref)) == 0.0

    def test_default_output_unchanged_by_talib(self, sample_df):
        """Default (talib=True) output equals the native (talib=False) formula."""
        default = sample_df.select(hl2("high", "low"))["HL2"].to_numpy()
        native = sample_df.select(hl2("high", "low", talib=False))["HL2"].to_numpy()
        np.testing.assert_array_equal(default, native)

    @pytest.mark.parametrize("use_talib", [True, False])
    def test_offset_shifts_result(self, sample_df, use_talib):
        """offset shifts the series by N; the first N rows become null."""
        if use_talib:
            pytest.importorskip("talib")
        base = sample_df.select(hl2("high", "low", talib=use_talib))["HL2"]
        shifted = sample_df.select(hl2("high", "low", talib=use_talib, offset=1))["HL2"]
        assert shifted[0] is None
        np.testing.assert_array_equal(shifted[1:].to_numpy(), base[:-1].to_numpy())

    def test_offset_zero_is_default(self, sample_df):
        """offset=0 leaves the output identical to the unshifted default."""
        default = sample_df.select(hl2("high", "low"))["HL2"].to_numpy()
        zero = sample_df.select(hl2("high", "low", offset=0))["HL2"].to_numpy()
        np.testing.assert_array_equal(default, zero)
