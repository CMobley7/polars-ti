# -*- coding: utf-8 -*-
"""Unit tests for polars_ti/cycles/ht_dcperiod.py."""

import numpy as np
import polars as pl
import pytest

from polars_ti.cycles.ht_dcperiod import ht_dcperiod


def _price_series(n=200, seed=42):
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.standard_normal(n)) + 100.0


class TestHtDcperiod:
    def test_returns_expression(self):
        assert isinstance(ht_dcperiod("close"), pl.Expr)

    def test_column_name(self):
        df = pl.DataFrame({"close": _price_series()})
        result = df.select(ht_dcperiod("close", talib=False))
        assert "HT_DCPERIOD" in result.columns

    def test_output_dtype(self):
        df = pl.DataFrame({"close": _price_series()})
        result = df.select(ht_dcperiod("close", talib=False))
        assert result["HT_DCPERIOD"].dtype == pl.Float64

    def test_warmup_nans(self):
        """First 32 bars should be NaN for native path."""
        df = pl.DataFrame({"close": _price_series(200)})
        vals = df.select(ht_dcperiod("close", talib=False))["HT_DCPERIOD"].to_numpy()
        assert np.isnan(vals[:32]).all()

    def test_valid_values_after_warmup(self):
        """Values after warmup should be in plausible cycle-period range [6, 50]."""
        df = pl.DataFrame({"close": _price_series(200)})
        vals = df.select(ht_dcperiod("close", talib=False))["HT_DCPERIOD"].to_numpy()
        valid = vals[~np.isnan(vals)]
        assert len(valid) > 0
        assert (valid >= 6.0).all()
        assert (valid <= 50.0).all()

    def test_offset(self):
        df = pl.DataFrame({"close": _price_series(200)})
        v0 = df.select(ht_dcperiod("close", talib=False, offset=0))["HT_DCPERIOD"].to_numpy()
        v1 = df.select(ht_dcperiod("close", talib=False, offset=1))["HT_DCPERIOD"].to_numpy()
        # Shift-1 means v1[1] == v0[0], etc.
        np.testing.assert_array_equal(v1[1:], v0[:-1])

    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("talib"),
        reason="TA-Lib not installed",
    )
    def test_talib_exact_match(self):
        """TA-Lib path must exactly equal talib.HT_DCPERIOD."""
        import talib

        arr = _price_series(200)
        df = pl.DataFrame({"close": arr})
        result = df.select(ht_dcperiod("close", talib=True))["HT_DCPERIOD"].to_numpy()
        expected = talib.HT_DCPERIOD(arr)
        np.testing.assert_array_almost_equal(result, expected)

    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("talib"),
        reason="TA-Lib not installed",
    )
    def test_native_close_to_talib_after_warmup(self):
        """Native path should be numerically close to TA-Lib after warmup."""
        import talib

        arr = _price_series(500)
        df = pl.DataFrame({"close": arr})
        native = df.select(ht_dcperiod("close", talib=False))["HT_DCPERIOD"].to_numpy()
        expected = talib.HT_DCPERIOD(arr)
        # Compare only in the stable region
        mask = ~np.isnan(native) & ~np.isnan(expected)
        stable = np.where(mask)[0]
        if len(stable) > 20:
            start = stable[20]  # skip first 20 valid bars
            # The native path is an approximation of TA-Lib; allow looser tolerance
            # (~15% relative, ~5 absolute) since the HT pipeline starts from index 6
            # while TA-Lib uses a longer warmup that produces slightly different values.
            np.testing.assert_allclose(
                native[start:],
                expected[start:],
                rtol=0.15,
                atol=5.0,
                err_msg="Native HT_DCPERIOD deviates too far from TA-Lib",
            )
