# -*- coding: utf-8 -*-
"""Tests for adx - native Polars/Numba path + TA-Lib parity.

The native ADX kernel exposes its DMP/DMN struct fields as the Wilder
sum-smoothed directional movement, matching TA-Lib PLUS_DM/MINUS_DM (the same
convention the talib path returns). The ADX line itself is pinned to TA-Lib via
the ``match_talib`` parity exception and converges to ``talib.ADX`` after the
warmup transient.
"""

import numpy as np
import polars as pl
import pytest

from polars_ti.maps import Imports
from polars_ti.trend.adx import adx


@pytest.fixture
def spy() -> pl.DataFrame:
    return pl.read_csv("data/SPY_D.csv", try_parse_dates=True).head(1500)


def test_returns_struct_columns(spy):
    result = spy.select(adx("high", "low", "close")).unnest("ADX_14")
    for col in ("ADX_14", "ADXR_14_2", "DMP_14", "DMN_14"):
        assert col in result.columns


def test_native_dmp_dmn_match_talib(spy):
    pytest.importorskip("talib")
    import talib

    result = spy.select(adx("high", "low", "close", talib=False)).unnest("ADX_14")
    high = spy["high"].to_numpy().astype("float64")
    low = spy["low"].to_numpy().astype("float64")

    dmp = result["DMP_14"].to_numpy()
    dmn = result["DMN_14"].to_numpy()
    ref_p = talib.PLUS_DM(high, low, timeperiod=14)
    ref_n = talib.MINUS_DM(high, low, timeperiod=14)

    mp = ~np.isnan(dmp) & ~np.isnan(ref_p)
    mn = ~np.isnan(dmn) & ~np.isnan(ref_n)
    assert mp.sum() > 1000 and mn.sum() > 1000
    assert np.max(np.abs(dmp[mp] - ref_p[mp])) < 1e-6
    assert np.max(np.abs(dmn[mn] - ref_n[mn])) < 1e-6


def test_native_adx_converges_to_talib(spy):
    """The native ADX line matches TA-Lib once the RMA seeding transient decays."""
    pytest.importorskip("talib")
    import talib

    result = spy.select(adx("high", "low", "close", talib=False)).unnest("ADX_14")
    native = result["ADX_14"].to_numpy()
    ref = talib.ADX(
        spy["high"].to_numpy().astype("float64"),
        spy["low"].to_numpy().astype("float64"),
        spy["close"].to_numpy().astype("float64"),
        timeperiod=14,
    )
    # Post-warmup (well past the 2*length seeding), native tracks TA-Lib.
    assert np.nanmax(np.abs(native[200:] - ref[200:])) < 1e-3


def test_lensig_defaults_to_length(spy):
    """Omitting lensig defaults it to length: columns and the TA-Lib fast path
    must key off ``length``, not a hardcoded 14."""
    result = spy.select(adx("high", "low", "close", length=21)).unnest("ADX_21")
    for col in ("ADX_21", "ADXR_21_2", "DMP_21", "DMN_21"):
        assert col in result.columns


@pytest.mark.skipif(not Imports["talib"], reason="requires TA-Lib")
def test_lensig_default_routes_talib_at_length(spy):
    """With lensig omitted, adx(length=21, talib=True) must hit the TA-Lib fast
    path and match talib.ADX(21) exactly (previously mislabeled/native)."""
    import talib

    result = spy.select(adx("high", "low", "close", length=21, talib=True)).unnest("ADX_21")
    ref = talib.ADX(
        spy["high"].to_numpy().astype("float64"),
        spy["low"].to_numpy().astype("float64"),
        spy["close"].to_numpy().astype("float64"),
        timeperiod=21,
    )
    vals = result["ADX_21"].to_numpy()
    m = ~np.isnan(vals) & ~np.isnan(ref)
    assert m.sum() > 1000
    assert np.max(np.abs(vals[m] - ref[m])) == 0.0


def test_explicit_lensig_differs_from_length_uses_native(spy):
    """An explicit lensig != length is honored (native), labeling by lensig."""
    result = spy.select(adx("high", "low", "close", length=21, lensig=10)).unnest("ADX_10")
    assert "ADX_10" in result.columns
    assert "DMP_21" in result.columns


@pytest.mark.skipif(not Imports["talib"], reason="requires TA-Lib")
def test_talib_path_matches_talib(spy):
    import talib

    result = spy.select(adx("high", "low", "close", talib=True)).unnest("ADX_14")
    high = spy["high"].to_numpy().astype("float64")
    low = spy["low"].to_numpy().astype("float64")
    close = spy["close"].to_numpy().astype("float64")

    for col, ref in (
        ("ADX_14", talib.ADX(high, low, close, timeperiod=14)),
        ("DMP_14", talib.PLUS_DM(high, low, timeperiod=14)),
        ("DMN_14", talib.MINUS_DM(high, low, timeperiod=14)),
    ):
        vals = result[col].to_numpy()
        m = ~np.isnan(vals) & ~np.isnan(ref)
        assert np.max(np.abs(vals[m] - ref[m])) < 1e-8
