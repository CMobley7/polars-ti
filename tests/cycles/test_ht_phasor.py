# -*- coding: utf-8 -*-
"""Unit tests for polars_ti/cycles/ht_phasor.py."""

import numpy as np
import polars as pl
import pytest

from polars_ti.cycles.ht_phasor import ht_phasor


def _price_series(n=200, seed=42):
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.standard_normal(n)) + 100.0


class TestHtPhasor:
    def test_returns_expression(self):
        assert isinstance(ht_phasor("close"), pl.Expr)

    def test_struct_fields(self):
        df = pl.DataFrame({"close": _price_series()})
        result = df.select(ht_phasor("close", talib=False))
        # Unnest to check field names
        unnested = result.unnest("HT_PHASOR")
        assert "inphase" in unnested.columns
        assert "quadrature" in unnested.columns

    def test_output_dtype(self):
        df = pl.DataFrame({"close": _price_series()})
        result = df.select(ht_phasor("close", talib=False))
        schema = result.schema["HT_PHASOR"]
        assert isinstance(schema, pl.Struct)
        fields = {f.name: f.dtype for f in schema.fields}
        assert fields["inphase"] == pl.Float64
        assert fields["quadrature"] == pl.Float64

    def test_warmup_nans(self):
        """First 32 bars should be NaN for native path."""
        df = pl.DataFrame({"close": _price_series(200)})
        result = df.select(ht_phasor("close", talib=False)).unnest("HT_PHASOR")
        ip = result["inphase"].to_numpy()
        qd = result["quadrature"].to_numpy()
        assert np.isnan(ip[:32]).all()
        assert np.isnan(qd[:32]).all()

    def test_valid_values_finite(self):
        df = pl.DataFrame({"close": _price_series(200)})
        result = df.select(ht_phasor("close", talib=False)).unnest("HT_PHASOR")
        ip = result["inphase"].to_numpy()
        qd = result["quadrature"].to_numpy()
        valid_ip = ip[~np.isnan(ip)]
        valid_qd = qd[~np.isnan(qd)]
        assert np.isfinite(valid_ip).all()
        assert np.isfinite(valid_qd).all()

    def test_offset(self):
        df = pl.DataFrame({"close": _price_series(200)})
        r0 = df.select(ht_phasor("close", talib=False, offset=0)).unnest("HT_PHASOR")
        r1 = df.select(ht_phasor("close", talib=False, offset=1)).unnest("HT_PHASOR")
        np.testing.assert_array_equal(r1["inphase"].to_numpy()[1:], r0["inphase"].to_numpy()[:-1])

    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("talib"),
        reason="TA-Lib not installed",
    )
    def test_talib_exact_match(self):
        import talib

        arr = _price_series(200)
        df = pl.DataFrame({"close": arr})
        result = df.select(ht_phasor("close", talib=True)).unnest("HT_PHASOR")
        exp_ip, exp_qd = talib.HT_PHASOR(arr)
        np.testing.assert_array_almost_equal(result["inphase"].to_numpy(), exp_ip)
        np.testing.assert_array_almost_equal(result["quadrature"].to_numpy(), exp_qd)

    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("talib"),
        reason="TA-Lib not installed",
    )
    def test_native_matches_talib_after_warmup(self):
        """Native phasor matches TA-Lib EXACTLY from the lookback (32) onward.

        The native Hilbert pipeline is a direct port of the TA-Lib C loop, so
        the raw I1 / Q1 phasor components are bit-for-bit identical (up to float
        rounding) once TA-Lib emits values at bar 32.
        """
        import talib

        arr = _price_series(500)
        df = pl.DataFrame({"close": arr})
        result = df.select(ht_phasor("close", talib=False)).unnest("HT_PHASOR")
        exp_ip, exp_qd = talib.HT_PHASOR(arr)
        start = 32  # TA-Lib HT_PHASOR lookback
        np.testing.assert_allclose(result["inphase"].to_numpy()[start:], exp_ip[start:], rtol=0, atol=1e-8)
        np.testing.assert_allclose(result["quadrature"].to_numpy()[start:], exp_qd[start:], rtol=0, atol=1e-8)
