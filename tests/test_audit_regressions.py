# -*- coding: utf-8 -*-
"""Regression tests locking the Stage-3 audit fixes.

Each test guards a specific, previously-verified defect. Where a fix restores a
parameter's effect on the default TA-Lib path, the test asserts the parameter now
changes the output (it did NOT before the fix).
"""

import numpy as np
import polars as pl
import pytest

from polars_ti.maps import Imports

requires_talib = pytest.mark.skipif(not Imports["talib"], reason="requires TA-Lib")


def _ohlcv(n: int = 300) -> pl.DataFrame:
    """Deterministic synthetic OHLCV (no RNG state leakage)."""
    t = np.arange(n, dtype=float)
    close = 100 + 10 * np.sin(t / 13.0) + 0.02 * t + 2 * np.sin(t / 2.7) * np.cos(t / 5.1)
    open_ = close + np.sin(t / 3.3)
    high = np.maximum(open_, close) + np.abs(np.sin(t / 1.9)) + 0.5
    low = np.minimum(open_, close) - np.abs(np.cos(t / 2.3)) - 0.5
    vol = 1_000_000 + 50_000 * np.abs(np.sin(t / 7.0)) + 100 * t
    return pl.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vol})


def _trix_line(df, **kw):
    from polars_ti.momentum.trix import trix

    length = kw.get("length", 18)
    signal = kw.get("signal", 9)
    return df.select(trix("close", **kw).alias("t")).unnest("t")[f"TRIX_{length}_{signal}"].to_numpy()


# --- H1: trix/trixh honor scalar & drift on the default TA-Lib path -----------
@requires_talib
def test_trix_talib_honors_scalar():
    df = _ohlcv()
    a = _trix_line(df, length=18, signal=9, scalar=100.0, talib=True)
    b = _trix_line(df, length=18, signal=9, scalar=50.0, talib=True)
    assert np.nanmax(np.abs(a - b)) > 1e-6  # scalar had NO effect before the fix


@requires_talib
def test_trix_talib_honors_drift():
    df = _ohlcv()
    a = _trix_line(df, length=18, signal=9, drift=1, talib=True)
    c = _trix_line(df, length=18, signal=9, drift=3, talib=True)
    assert np.nanmax(np.abs(a - c)) > 1e-6


@requires_talib
def test_trix_talib_default_matches_raw_talib():
    """Default (scalar=100, drift=1) must still equal talib.TRIX exactly."""
    import talib

    df = _ohlcv()
    got = _trix_line(df, length=18, signal=9, talib=True)
    ref = talib.TRIX(df["close"].to_numpy().astype(float), timeperiod=18)
    assert np.nanmax(np.abs(got - ref)) == 0.0


@requires_talib
def test_trixh_talib_honors_scalar():
    from polars_ti.momentum.trixh import trixh

    df = _ohlcv()
    a = df.select(trixh("close", length=18, signal=9, scalar=100.0, talib=True))["TRIX_18_9"].to_numpy()
    b = df.select(trixh("close", length=18, signal=9, scalar=50.0, talib=True))["TRIX_18_9"].to_numpy()
    assert np.nanmax(np.abs(a - b)) > 1e-6


# --- M1: roc/cmo/ppo honor scalar on the default TA-Lib path ------------------
@requires_talib
def test_roc_talib_honors_scalar():
    from polars_ti.momentum.roc import roc

    df = _ohlcv()
    a = df.select(roc("close", length=10, scalar=100.0, talib=True)).to_series().to_numpy()
    b = df.select(roc("close", length=10, scalar=50.0, talib=True)).to_series().to_numpy()
    assert np.nanmax(np.abs(a - b)) > 1e-6


@requires_talib
def test_cmo_talib_honors_scalar():
    from polars_ti.momentum.cmo import cmo

    df = _ohlcv()
    a = df.select(cmo("close", length=14, scalar=100.0, talib=True)).to_series().to_numpy()
    b = df.select(cmo("close", length=14, scalar=50.0, talib=True)).to_series().to_numpy()
    assert np.nanmax(np.abs(a - b)) > 1e-6


@requires_talib
def test_ppo_talib_honors_scalar():
    from polars_ti.momentum.ppo import ppo

    df = _ohlcv()
    a = df.select(ppo("close", scalar=100.0, talib=True)).to_series().to_numpy()
    b = df.select(ppo("close", scalar=50.0, talib=True)).to_series().to_numpy()
    assert np.nanmax(np.abs(a - b)) > 1e-6


# --- H2: trama mamode restored ------------------------------------------------
def test_trama_mamode_changes_output_and_does_not_raise():
    from polars_ti.trend.trama import trama

    df = _ohlcv()
    sma = df.select(trama("close", length=10, mamode="sma"))["TRAMA_10"].to_numpy()
    ema = df.select(trama("close", length=10, mamode="ema"))["TRAMA_10"].to_numpy()
    assert np.nanmax(np.abs(sma - ema)) > 1e-9  # mamode now has an effect


def test_trama_default_is_sma():
    from polars_ti.trend.trama import trama

    df = _ohlcv()
    default = df.select(trama("close", length=10))["TRAMA_10"].to_numpy()
    sma = df.select(trama("close", length=10, mamode="sma"))["TRAMA_10"].to_numpy()
    assert np.nanmax(np.abs(default - sma)) == 0.0


# --- H3: nb_non_zero_range guards isolated flat bars element-wise --------------
def test_nb_non_zero_range_element_wise_flat_bar():
    from polars_ti.utils._core import nb_non_zero_range

    x = np.array([10.0, 10.0, 10.0])
    y = np.array([9.0, 10.0, 8.0])  # middle bar is flat (diff == 0)
    out = nb_non_zero_range(x, y)
    eps = np.finfo(np.float64).eps
    assert out[0] == 1.0 and out[2] == 2.0  # non-flat bars unchanged
    assert 0.0 < out[1] <= eps  # the isolated flat bar is now epsilon, not 0


# --- M2: psar does not perform out-of-bounds access for n < 2 -----------------
@pytest.mark.parametrize("n", [0, 1])
def test_psar_short_input_no_crash(n):
    from polars_ti.trend.psar import psar

    df = _ohlcv(50).head(n)
    result = df.select(psar("high", "low", "close"))
    assert result.height == n


# --- M6: vfi nulls a zero denominator instead of a huge finite spike ----------
def test_vfi_zero_volume_yields_null_not_spike():
    from polars_ti.volume.vfi import vfi

    df = _ohlcv(30).with_columns(pl.lit(0.0).alias("volume"))  # all-zero volume
    out = df.select(vfi("close", "volume", length=5)).to_series()
    finite = out.drop_nulls().to_numpy()
    finite = finite[~np.isnan(finite)]
    # No astronomically large finite spike from dividing by ~eps.
    assert finite.size == 0 or np.max(np.abs(finite)) < 1e6


# --- L1: stderr clamps ddof >= length instead of returning all-NaN ------------
def test_stderr_ddof_ge_length_clamped():
    from polars_ti.statistics.stderr import stderr

    df = _ohlcv(30)
    out = df.select(stderr("close", length=5, ddof=5)).to_series().to_numpy()
    assert np.isfinite(out).any()  # was entirely NaN before the clamp


# --- M4: cdl_z full=True differs from the rolling default ---------------------
def test_cdl_z_full_differs_from_rolling():
    from polars_ti.candles.cdl_z import cdl_z

    df = _ohlcv(120)
    rolling = df.select(cdl_z("open", "high", "low", "close", length=30, full=False))
    full = df.select(cdl_z("open", "high", "low", "close", length=30, full=True))
    a = rolling["close_Z_30_1"].to_numpy()
    b = full["close_Z_30_1"].to_numpy()
    assert np.nanmax(np.abs(a - b)) > 1e-6


# --- M7: signed_series honors its initial argument ----------------------------
def test_signed_series_uses_initial():
    from polars_ti.utils._core import signed_series

    s = pl.Series("x", [3.0, 2.0, 2.0, 1.0, 1.0, 5.0, 6.0, 6.0, 7.0, 5.0])
    out = signed_series(s, initial=99, lag=1)
    assert out[0] == 99  # first element was always None/NaN before the fix


# --- F12 (Stage-4 crash sweep): kama OOB write / SIGABRT on short input -------
def test_kama_short_frame_no_crash():
    """kama on a frame shorter than length must return all-NaN, not corrupt the
    heap (nb_kama wrote result[length-1] out of bounds -> free(): invalid next
    size / SIGABRT)."""
    from polars_ti.overlap.kama import kama

    df = pl.DataFrame({"close": [100.0, 101.0, 102.0]})  # n=3 < default length 10
    out = df.select(kama("close", talib=False)).to_series().to_numpy()
    assert out.shape[0] == 3
    assert bool(np.all(np.isnan(out)))


# --- Stage-4 r3: pvr row 0 is deterministic category 1 (was null) ------------
def test_pvr_first_row_is_category_one():
    """pvr's first bar must be category 1 (baseline .diff().fillna(0)), not null."""
    from polars_ti.volume.pvr import pvr

    df = pl.DataFrame({"close": [100.0, 100.5, 101.0, 100.0, 101.5], "volume": [500.0, 600, 400, 300, 700]})
    out = df.select(pvr("close", "volume")).to_series().to_numpy()
    assert out[0] == 1.0


# --- Stage-4 r4: kvo interior flat bar must not poison the TA-Lib EMA --------
def test_kvo_flat_bar_does_not_truncate_series():
    """An unchanged hlc3 bar (diff==0) must map to 0.0, not null. A null flows
    into the default TA-Lib EMA and NaN-poisons the entire tail (df.ti.kvo()
    collapsed to ~50 valid values on real data). Row 0 (null diff) stays null."""
    from polars_ti.volume.kvo import kvo

    t = np.arange(200.0)
    c = 100 + np.sin(t / 9) + 0.05 * t
    h, low = c + 1, c - 1
    for col in (c, h, low):
        col[120] = col[119]  # flat hlc3 bar mid-series
    df = pl.DataFrame({"high": h, "low": low, "close": c, "volume": 1e6 + 50 * t})
    arr = df.select(kvo("high", "low", "close", "volume")).to_series(0).to_numpy()
    valid = ~np.isnan(arr)
    assert int(np.max(np.where(valid)[0])) >= 190  # not truncated at the flat bar
