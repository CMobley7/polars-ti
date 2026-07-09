# -*- coding: utf-8 -*-
"""Native-path == TA-Lib alignment guarantees.

The native EMA warmup seed was corrected to match TA-Lib on leading-NaN and
cascaded inputs (previously it seeded one bar early). As a consequence a family
of EMA-driven indicators now produce, in native (``talib=False``) mode, the SAME
values as their own ``talib=True`` output — and, for DEMA, the SAME values as the
direct TA-Lib reference.

These indicators have NO dedicated TA-Lib column in ``talib_reference.parquet``
(except DEMA), so they are excluded from the OLD-golden parity grading in
``parity_exceptions.py`` (``match_talib``/``intentional``). This module is the
real guarantee behind that exclusion: it pins ``native == talib`` post-warmup for
every aligned indicator, so a regression in either path turns the suite red.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

import polars_ti as ti  # noqa: F401 — registers the 'ti' accessor

SLICE_ROWS = 1500
FIXTURES = "tests/fixtures"
# Wilder/EMA seed transients decay quickly; grade well past every warmup.
WARMUP = 300
TOL = 1e-6


@pytest.fixture(scope="module")
def spy() -> pl.DataFrame:
    return pl.read_csv("data/SPY_D.csv", try_parse_dates=True).head(SLICE_ROWS)


def _max_abs(native: np.ndarray, talib_: np.ndarray, warmup: int = WARMUP) -> float:
    mask = ~np.isnan(native) & ~np.isnan(talib_)
    mask[:warmup] = False
    assert mask.sum() > 100, "insufficient overlap to grade native==talib"
    return float(np.max(np.abs(native[mask] - talib_[mask])))


def _select(df: pl.DataFrame, exprs) -> pl.DataFrame:
    return df.select(exprs)


def test_dema_native_matches_talib_reference(spy):
    """DEMA is a pure triple-of-EMA transform: native == talib.DEMA (and the
    talib_reference column, which IS talib.DEMA)."""
    talib = pytest.importorskip("talib")
    from polars_ti.overlap.dema import dema

    close = spy["close"].to_numpy().astype(float)
    ref = talib.DEMA(close, timeperiod=10)

    native = _select(spy, dema("close", 10, talib=False)).to_series().to_numpy()
    talib_mode = _select(spy, dema("close", 10, talib=True)).to_series().to_numpy()

    assert _max_abs(native, ref) < TOL, "native DEMA != talib.DEMA"
    assert _max_abs(talib_mode, ref) < TOL, "talib-mode DEMA != talib.DEMA"

    # And the committed TA-Lib reference fixture agrees (this is the match_talib pin).
    ref_col = pl.read_parquet(f"{FIXTURES}/talib_reference.parquet")["DEMA_10"].to_numpy()
    assert _max_abs(native, ref_col) < TOL


def test_zlma_native_matches_talib(spy):
    from polars_ti.overlap.zlma import zlma

    native = _select(spy, zlma("close", 10, talib=False)).to_series().to_numpy()
    talib_mode = _select(spy, zlma("close", 10, talib=True)).to_series().to_numpy()
    assert _max_abs(native, talib_mode) < TOL


def test_tsi_native_matches_talib(spy):
    from polars_ti.momentum.tsi import tsi

    native = _select(spy, tsi("close", talib=False))
    talib_mode = _select(spy, tsi("close", talib=True))
    for col in ("TSI_13_25_13", "TSIs_13_25_13"):
        assert _max_abs(native[col].to_numpy(), talib_mode[col].to_numpy()) < TOL, col


def test_tmo_native_matches_talib(spy):
    from polars_ti.momentum.tmo import tmo

    native = _select(spy, tmo("open", "close", talib=False)).unnest("TMO")
    talib_mode = _select(spy, tmo("open", "close", talib=True)).unnest("TMO")
    for col in ("TMO_14_5_3", "TMOs_14_5_3", "TMOM_14_5_3", "TMOMs_14_5_3"):
        assert _max_abs(native[col].to_numpy(), talib_mode[col].to_numpy()) < TOL, col


def test_efi_native_matches_talib(spy):
    from polars_ti.volume.efi import efi

    native = _select(spy, efi("close", "volume", talib=False)).to_series().to_numpy()
    talib_mode = _select(spy, efi("close", "volume", talib=True)).to_series().to_numpy()
    assert _max_abs(native, talib_mode) < TOL


def test_kvo_native_matches_talib(spy):
    from polars_ti.volume.kvo import kvo

    native = _select(spy, kvo("high", "low", "close", "volume", talib=False))
    talib_mode = _select(spy, kvo("high", "low", "close", "volume", talib=True))
    for col in ("KVO_34_55_13", "KVOs_34_55_13"):
        assert _max_abs(native[col].to_numpy(), talib_mode[col].to_numpy()) < TOL, col


def test_atr_native_matches_talib_reference(spy):
    """Native Wilder ATR now seeds like TA-Lib (SMA of TR[1..length] at index
    length): native == talib.ATR == the talib_reference column."""
    talib = pytest.importorskip("talib")
    from polars_ti.volatility.atr import atr

    h = spy["high"].to_numpy().astype(float)
    low_ = spy["low"].to_numpy().astype(float)
    c = spy["close"].to_numpy().astype(float)
    ref = talib.ATR(h, low_, c, timeperiod=14)

    native = _select(spy, atr("high", "low", "close", 14, talib=False)).to_series().to_numpy()
    talib_mode = _select(spy, atr("high", "low", "close", 14, talib=True)).to_series().to_numpy()
    # Whole overlap, not just post-warmup: the seed itself now matches TA-Lib.
    assert _max_abs(native, ref, warmup=20) < TOL, "native ATR != talib.ATR"
    assert _max_abs(talib_mode, ref, warmup=20) == 0.0

    ref_col = pl.read_parquet(f"{FIXTURES}/talib_reference.parquet")["ATRr_14"].to_numpy()
    assert _max_abs(native, ref_col, warmup=20) < TOL


def test_natr_native_matches_talib_reference(spy):
    talib = pytest.importorskip("talib")
    from polars_ti.volatility.natr import natr

    h = spy["high"].to_numpy().astype(float)
    low_ = spy["low"].to_numpy().astype(float)
    c = spy["close"].to_numpy().astype(float)
    ref = talib.NATR(h, low_, c, timeperiod=14)
    native = _select(spy, natr("high", "low", "close", length=14, talib=False)).to_series().to_numpy()
    assert _max_abs(native, ref) < TOL, "native NATR != talib.NATR"


def test_cksp_rma_native_matches_talib(spy):
    """CKSP's TradingView mode (Wilder/rma ATR) now has native == talib-mode.
    (Book mode uses an SMA ATR that legitimately differs from TA-Lib's Wilder
    ATR, so it stays an intentional native divergence.)"""
    from polars_ti.trend.cksp import cksp

    native = _select(spy, cksp("high", "low", "close", tvmode=True, talib=False)).unnest("CKSP_10_1_9")
    talib_mode = _select(spy, cksp("high", "low", "close", tvmode=True, talib=True)).unnest("CKSP_10_1_9")
    for col in ("CKSPl_10_1_9", "CKSPs_10_1_9"):
        assert _max_abs(native[col].to_numpy(), talib_mode[col].to_numpy()) < TOL, col


def test_chandelier_and_rwi_native_match_talib(spy):
    """chandelier_exit and RWI both smooth an internal Wilder ATR; native == talib."""
    from polars_ti.trend.rwi import rwi
    from polars_ti.volatility.chandelier_exit import chandelier_exit

    n_rwi = _select(spy, rwi("high", "low", "close", talib=False)).unnest("RWI_14")
    t_rwi = _select(spy, rwi("high", "low", "close", talib=True)).unnest("RWI_14")
    for col in ("RWIh_14", "RWIl_14"):
        assert _max_abs(n_rwi[col].to_numpy(), t_rwi[col].to_numpy()) < TOL, col

    n_ch = _select(spy, chandelier_exit("high", "low", "close", talib=False))
    t_ch = _select(spy, chandelier_exit("high", "low", "close", talib=True))
    n_ch = n_ch.unnest(n_ch.columns[0])
    t_ch = t_ch.unnest(t_ch.columns[0])
    for col in ("long", "short"):
        assert _max_abs(n_ch[col].to_numpy(), t_ch[col].to_numpy()) < TOL, col


def test_trix_native_matches_talib(spy):
    """TRIX now routes both paths through ``ema()``; native == talib.TRIX."""
    talib = pytest.importorskip("talib")
    from polars_ti.momentum.trix import trix

    ref = talib.TRIX(spy["close"].to_numpy().astype(float), timeperiod=30)
    native = _select(spy, trix("close", talib=False)).to_series().struct.field("TRIX_30_9").to_numpy()
    talib_mode = _select(spy, trix("close", talib=True)).to_series().struct.field("TRIX_30_9").to_numpy()
    # talib mode is byte-identical to talib.TRIX; native matches to float noise.
    assert _max_abs(talib_mode, ref) == 0.0
    assert _max_abs(native, ref) < TOL
