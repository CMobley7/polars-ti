# -*- coding: utf-8 -*-
"""Unit tests for polars_ti/cycles/ht_sine.py."""

import numpy as np
import polars as pl
import pytest

from polars_ti.cycles.ht_sine import ht_sine


def _price_series(n=200, seed=42):
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.standard_normal(n)) + 100.0


class TestHtSine:
    def test_returns_expression(self):
        assert isinstance(ht_sine("close"), pl.Expr)

    def test_struct_fields(self):
        df = pl.DataFrame({"close": _price_series()})
        result = df.select(ht_sine("close", talib=False)).unnest("HT_SINE")
        assert "sine" in result.columns
        assert "leadsine" in result.columns

    def test_output_dtype(self):
        df = pl.DataFrame({"close": _price_series()})
        result = df.select(ht_sine("close", talib=False))
        schema = result.schema["HT_SINE"]
        assert isinstance(schema, pl.Struct)
        fields = {f.name: f.dtype for f in schema.fields}
        assert fields["sine"] == pl.Float64
        assert fields["leadsine"] == pl.Float64

    def test_warmup_nans(self):
        """First 63 bars should be NaN for native path."""
        df = pl.DataFrame({"close": _price_series(200)})
        result = df.select(ht_sine("close", talib=False)).unnest("HT_SINE")
        s = result["sine"].to_numpy()
        ls = result["leadsine"].to_numpy()
        assert np.isnan(s[:63]).all()
        assert np.isnan(ls[:63]).all()

    def test_values_bounded(self):
        """Sine values should be in [-1, 1]."""
        df = pl.DataFrame({"close": _price_series(200)})
        result = df.select(ht_sine("close", talib=False)).unnest("HT_SINE")
        s = result["sine"].to_numpy()
        ls = result["leadsine"].to_numpy()
        valid_s = s[~np.isnan(s)]
        valid_ls = ls[~np.isnan(ls)]
        assert (np.abs(valid_s) <= 1.0 + 1e-10).all()
        assert (np.abs(valid_ls) <= 1.0 + 1e-10).all()

    def test_leadsine_leads_sine(self):
        """leadSine = sin(dcphase + 45deg) should differ from sine = sin(dcphase)."""
        df = pl.DataFrame({"close": _price_series(200)})
        result = df.select(ht_sine("close", talib=False)).unnest("HT_SINE")
        s = result["sine"].to_numpy()
        ls = result["leadsine"].to_numpy()
        mask = ~np.isnan(s) & ~np.isnan(ls)
        # They should not be identical
        assert not np.allclose(s[mask], ls[mask])

    def test_offset(self):
        df = pl.DataFrame({"close": _price_series(200)})
        r0 = df.select(ht_sine("close", talib=False, offset=0)).unnest("HT_SINE")
        r1 = df.select(ht_sine("close", talib=False, offset=1)).unnest("HT_SINE")
        np.testing.assert_array_equal(r1["sine"].to_numpy()[1:], r0["sine"].to_numpy()[:-1])

    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("talib"),
        reason="TA-Lib not installed",
    )
    def test_talib_exact_match(self):
        import talib

        arr = _price_series(200)
        df = pl.DataFrame({"close": arr})
        result = df.select(ht_sine("close", talib=True)).unnest("HT_SINE")
        exp_sine, exp_leadsine = talib.HT_SINE(arr)
        np.testing.assert_array_almost_equal(result["sine"].to_numpy(), exp_sine)
        np.testing.assert_array_almost_equal(result["leadsine"].to_numpy(), exp_leadsine)
