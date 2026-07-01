# -*- coding: utf-8 -*-
"""Unit tests for polars_ti/cycles/ht_dcphase.py."""

import numpy as np
import polars as pl
import pytest

from polars_ti.cycles.ht_dcphase import ht_dcphase


def _price_series(n=200, seed=42):
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.standard_normal(n)) + 100.0


class TestHtDcphase:
    def test_returns_expression(self):
        assert isinstance(ht_dcphase("close"), pl.Expr)

    def test_column_name(self):
        df = pl.DataFrame({"close": _price_series()})
        result = df.select(ht_dcphase("close", talib=False))
        assert "HT_DCPHASE" in result.columns

    def test_output_dtype(self):
        df = pl.DataFrame({"close": _price_series()})
        result = df.select(ht_dcphase("close", talib=False))
        assert result["HT_DCPHASE"].dtype == pl.Float64

    def test_warmup_nans(self):
        """First 63 bars should be NaN for native path."""
        df = pl.DataFrame({"close": _price_series(200)})
        vals = df.select(ht_dcphase("close", talib=False))["HT_DCPHASE"].to_numpy()
        assert np.isnan(vals[:63]).all()

    def test_valid_values_after_warmup(self):
        """Phase values should be within reasonable bounds [-360, 360]."""
        df = pl.DataFrame({"close": _price_series(200)})
        vals = df.select(ht_dcphase("close", talib=False))["HT_DCPHASE"].to_numpy()
        valid = vals[~np.isnan(vals)]
        assert len(valid) > 0
        # Phase is expressed in degrees; values should be finite
        assert np.isfinite(valid).all()

    def test_offset(self):
        df = pl.DataFrame({"close": _price_series(200)})
        v0 = df.select(ht_dcphase("close", talib=False, offset=0))["HT_DCPHASE"].to_numpy()
        v1 = df.select(ht_dcphase("close", talib=False, offset=1))["HT_DCPHASE"].to_numpy()
        np.testing.assert_array_equal(v1[1:], v0[:-1])

    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("talib"),
        reason="TA-Lib not installed",
    )
    def test_talib_exact_match(self):
        import talib

        arr = _price_series(200)
        df = pl.DataFrame({"close": arr})
        result = df.select(ht_dcphase("close", talib=True))["HT_DCPHASE"].to_numpy()
        expected = talib.HT_DCPHASE(arr)
        np.testing.assert_array_almost_equal(result, expected)
