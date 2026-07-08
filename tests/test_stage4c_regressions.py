# -*- coding: utf-8 -*-
"""Regression tests locking the Stage-4 (batch C) audit fixes C1-C6.

Each test guards a specific, previously-verified defect. Every test fails on the
pre-fix code and passes afterwards. Frames are small and deterministic (no RNG
ordering dependence beyond a fixed seed).
"""

import warnings

import numpy as np
import polars as pl

import polars_ti as ti  # noqa: F401  (registers the .ti namespace)
from polars_ti.overlap.kama import kama
from polars_ti.overlap.rainbow import rainbow
from polars_ti.overlap.wcp import wcp
from polars_ti.volatility.bbands import bbands
from polars_ti.volatility.ui import ui
from polars_ti.utils._study import CommonStudy


def _ohlc(n: int) -> pl.DataFrame:
    """Deterministic synthetic OHLC frame (no RNG state)."""
    t = np.arange(n, dtype=float)
    close = 100.0 + 10.0 * np.sin(t / 9.0) + 0.05 * t
    high = close + 0.5 + np.abs(np.sin(t / 2.0))
    low = close - 0.5 - np.abs(np.cos(t / 2.5))
    vol = 1000.0 + 100.0 * np.abs(np.sin(t / 4.0)) + t
    return pl.DataFrame({"high": high, "low": low, "close": close, "volume": vol})


# --- C1: negative offset must NaN-fill the wrapped tail (not leak real values) -
def test_c1_negative_offset_nan_fills_both_signs():
    df = _ohlc(80)
    tail = df.select(wcp("high", "low", "close", offset=-3)).to_series().to_numpy()
    head = df.select(wcp("high", "low", "close", offset=3)).to_series().to_numpy()
    base = df.select(wcp("high", "low", "close", offset=0)).to_series().to_numpy()
    # negative offset -> last |offset| rows are NaN (previously wrapped head vals)
    assert np.all(np.isnan(tail[-3:]))
    # and the wrapped values are NOT the leaked head of the base series
    assert not np.array_equal(np.nan_to_num(tail[-3:]), np.nan_to_num(base[:3]))
    # positive offset still NaN-fills the head (unchanged behaviour)
    assert np.all(np.isnan(head[:3]))
    # interior of a negative offset equals base shifted left by 3
    assert np.allclose(tail[:-3], base[3:], equal_nan=True)


# --- C2: kama default (talib) path must honour non-default fast/slow -----------
def test_c2_kama_honours_fast_slow():
    rng = np.random.default_rng(11)
    df = pl.DataFrame({"close": np.cumsum(rng.standard_normal(200)) + 100.0})
    default = df.select(kama("close", fast=2, slow=30)).to_series().to_numpy()
    tuned = df.select(kama("close", fast=5, slow=40)).to_series().to_numpy()
    both = ~np.isnan(default) & ~np.isnan(tuned)
    assert np.nanmax(np.abs(default[both] - tuned[both])) > 1e-6


# --- C3: rainbow offset must not compound across ribbons -----------------------
def test_c3_rainbow_offset_no_compounding():
    rng = np.random.default_rng(5)
    df = pl.DataFrame({"close": np.cumsum(rng.standard_normal(120)) + 100.0})
    r0 = df.select(rainbow("close", num_ribbons=4, offset=0)).to_series().struct.unnest()
    r1 = df.select(rainbow("close", num_ribbons=4, offset=1)).to_series().struct.unnest()
    a = r0["RAINBOW_3"].to_numpy()
    b = r1["RAINBOW_3"].to_numpy()
    shifted = np.full_like(a, np.nan)
    shifted[1:] = a[:-1]
    m = ~np.isnan(b) & ~np.isnan(shifted)
    assert m.sum() > 0
    # ribbon 3 with offset=1 is shifted exactly once (not three times)
    assert np.allclose(b[m], shifted[m])


# --- C4: universal prefix/suffix + CommonStudy VOL column ----------------------
def test_c4_prefix_suffix_direct():
    df = _ohlc(60)
    out = df.ti.sma(prefix="VOL", suffix="x")
    assert out.columns == ["VOL_SMA_10_x"]


def test_c4_commonstudy_yields_vol_column_without_skip():
    df = _ohlc(60)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any skipped-indicator RuntimeWarning fails
        out = df.ti.study(CommonStudy)
    assert "VOL_SMA_20" in out.columns


# --- C5: bbands native BBB/BBP guard divide-by-zero (NaN, not inf) -------------
def test_c5_bbands_native_zero_mean_no_inf():
    # Zero-mean oscillator: SMA (mid) == 0 over even windows -> unguarded BBB=inf.
    close = np.array([(-1.0) ** i for i in range(40)], dtype=float)
    df = pl.DataFrame({"close": close})
    st = df.select(bbands("close", length=4, talib=False)).to_series().struct.unnest()
    bbb = st[[c for c in st.columns if c.startswith("BBB")][0]].to_numpy()
    assert not np.any(np.isinf(bbb))


# --- C6: ui warmup uses a full window (min_samples=length) ---------------------
def test_c6_ui_first_valid_full_window():
    rng = np.random.default_rng(7)
    df = pl.DataFrame({"close": np.cumsum(rng.standard_normal(120)) + 200.0})
    length = 14
    vals = df.select(ui("close", length=length)).to_series().to_numpy()
    first_valid = int(np.argmax(~np.isnan(vals)))
    # highest_close needs `length` rows, then rolling_sum needs `length` d2 values
    assert first_valid == 2 * length - 2
    assert np.all(np.isnan(vals[: 2 * length - 2]))
