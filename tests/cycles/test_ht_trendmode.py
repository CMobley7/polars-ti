# -*- coding: utf-8 -*-
"""Unit tests for polars_ti/cycles/ht_trendmode.py."""

import numpy as np
import polars as pl
import pytest

from polars_ti.cycles.ht_trendmode import ht_trendmode


def _price_series(n=200, seed=42):
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.standard_normal(n)) + 100.0


class TestHtTrendmode:
    def test_returns_expression(self):
        assert isinstance(ht_trendmode("close"), pl.Expr)

    def test_column_name(self):
        df = pl.DataFrame({"close": _price_series()})
        result = df.select(ht_trendmode("close", talib=False))
        assert "HT_TRENDMODE" in result.columns

    def test_output_dtype(self):
        df = pl.DataFrame({"close": _price_series()})
        result = df.select(ht_trendmode("close", talib=False))
        assert result["HT_TRENDMODE"].dtype == pl.Int32

    def test_values_binary(self):
        """Native trendmode must contain only 0 or 1."""
        df = pl.DataFrame({"close": _price_series(200)})
        vals = df.select(ht_trendmode("close", talib=False))["HT_TRENDMODE"].to_numpy()
        unique = set(vals.tolist())
        assert unique.issubset({0, 1}), f"Unexpected values: {unique - {0, 1}}"

    def test_no_nans(self):
        """Trendmode is integer-initialized; should have no NaN/null."""
        df = pl.DataFrame({"close": _price_series(200)})
        result = df.select(ht_trendmode("close", talib=False))
        assert result["HT_TRENDMODE"].null_count() == 0

    def test_offset(self):
        df = pl.DataFrame({"close": _price_series(200)})
        v0 = df.select(ht_trendmode("close", talib=False, offset=0))["HT_TRENDMODE"].to_numpy()
        v1 = df.select(ht_trendmode("close", talib=False, offset=1))["HT_TRENDMODE"].to_numpy()
        np.testing.assert_array_equal(v1[1:], v0[:-1])

    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("talib"),
        reason="TA-Lib not installed",
    )
    def test_talib_exact_match(self):
        """TA-Lib path must exactly equal talib.HT_TRENDMODE."""
        import talib

        arr = _price_series(200)
        df = pl.DataFrame({"close": arr})
        result = df.select(ht_trendmode("close", talib=True))["HT_TRENDMODE"].to_numpy()
        expected = talib.HT_TRENDMODE(arr).astype(np.int32)
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("talib"),
        reason="TA-Lib not installed",
    )
    def test_native_is_binary(self):
        """Native path (talib=False) must return only 0/1 even when TA-Lib is present."""
        df = pl.DataFrame({"close": _price_series(200)})
        vals = df.select(ht_trendmode("close", talib=False))["HT_TRENDMODE"].to_numpy()
        unique = set(vals.tolist())
        assert unique.issubset({0, 1})
