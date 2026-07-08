# -*- coding: utf-8 -*-
"""Regression tests locking the Stage-4 audit fixes.

Each test guards a specific, previously-verified defect (crash / OOB write /
wrong value / broken public accessor). Every test fails or crashes on the
pre-fix code and passes afterwards. Frames are small and deterministic (no RNG
ordering dependence).
"""

import numpy as np
import polars as pl

import polars_ti as ti  # noqa: F401  (registers the .ti namespace)
from polars_ti.momentum.fisher import fisher
from polars_ti.momentum.stoch import stoch
from polars_ti.momentum.stochf import stochf
from polars_ti.overlap.pivots import pivots
from polars_ti.trend.adx import adx


def _ohlcv(n: int) -> pl.DataFrame:
    """Deterministic synthetic OHLCV frame (no RNG state)."""
    t = np.arange(n, dtype=float)
    close = 100.0 + 10.0 * np.sin(t / 9.0) + 0.05 * t
    open_ = close + 0.3 * np.sin(t / 3.0)
    high = np.maximum(open_, close) + 0.5 + np.abs(np.sin(t / 2.0))
    low = np.minimum(open_, close) - 0.5 - np.abs(np.cos(t / 2.5))
    vol = 1000.0 + 100.0 * np.abs(np.sin(t / 4.0)) + t
    return pl.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vol})


def _all_nan(df: pl.DataFrame) -> bool:
    return all(df[c].is_nan().all() for c in df.columns)


# ---------------------------------------------------------------------------
# F1 — adx native kernel OOB when n < length (SIGSEGV / heap corruption)
# ---------------------------------------------------------------------------
def test_f1_adx_short_frame_no_crash():
    for n in (3, 13):  # 13 == length - 1
        df = _ohlcv(n)
        out = df.select(adx("high", "low", "close", talib=False)).unnest("ADX_14")
        assert out.height == n
        assert _all_nan(out)


# ---------------------------------------------------------------------------
# F3 — pivots DeMark must select the per-bar branch (not whole-series .all())
# ---------------------------------------------------------------------------
def test_f3_pivot_demark_per_bar_branch():
    # bar0 up (c>o), bar1 down (c<o), bar2 equal (c==o)
    o = np.array([10.0, 10.0, 10.0])
    h = np.array([12.0, 11.0, 13.0])
    lo = np.array([9.0, 8.0, 10.0])
    c = np.array([11.0, 9.0, 10.0])
    df = pl.DataFrame({"open": o, "high": h, "low": lo, "close": c})
    expr = pivots("high", "low", "close", open_="open", method="demark")
    res = df.select(expr)
    res = res.unnest(res.columns[0])
    tp_col = next(cc for cc in res.columns if cc.endswith("_P"))
    tp = res[tp_col].to_numpy()  # pivot is shifted forward one bar

    exp_up = 0.25 * (2 * h[0] + lo[0] + c[0])  # up bar -> 2*high + low + close
    exp_down = 0.25 * (h[1] + 2 * lo[1] + c[1])  # down bar -> high + 2*low + close
    assert tp[1] == exp_up
    assert tp[2] == exp_down
    # A whole-series .all() collapse would have made every bar use the down formula.
    assert tp[1] != 0.25 * (h[0] + 2 * lo[0] + c[0])


# ---------------------------------------------------------------------------
# F4 — xsignals accessor must pass numeric thresholds through (not as columns)
# ---------------------------------------------------------------------------
def test_f4_xsignals_accessor_numeric_thresholds():
    n = 40
    df = pl.DataFrame({"rsicol": 50.0 + 20.0 * np.sin(np.arange(n) / 3.0)})
    out = df.ti.xsignals(signal="rsicol", xa=70, xb=30)
    assert out.height == n
    assert out.width >= 1


# ---------------------------------------------------------------------------
# F5 — nb_fisher OOB write when n < length
# ---------------------------------------------------------------------------
def test_f5_fisher_short_frame_no_crash():
    df = _ohlcv(5)  # length default 9 > 5
    out = df.select(fisher("high", "low", length=9))
    assert out.height == 5
    assert _all_nan(out)


# ---------------------------------------------------------------------------
# F6 — vp accessor must pass the DataFrame (uses df.height), not an Expr
# ---------------------------------------------------------------------------
def test_f6_vp_accessor():
    df = _ohlcv(100)
    out = df.ti.vp()
    assert out is not None
    assert out.height == 10  # default width
    assert "total_volume" in out.columns


# ---------------------------------------------------------------------------
# F7 — smma / alligator IndexError when n <= length
# ---------------------------------------------------------------------------
def test_f7_smma_and_alligator_short_frame_no_crash():
    df5 = _ohlcv(5)
    smma_out = df5.select(ti.smma("close", length=7))
    assert smma_out.height == 5
    assert _all_nan(smma_out)

    df10 = _ohlcv(10)  # alligator jaw=13 > 10
    alg = df10.ti.alligator()  # must not raise
    assert alg.height == 10


# ---------------------------------------------------------------------------
# F10 — tos_stdevall with length must preserve input height (append safe)
# ---------------------------------------------------------------------------
def test_f10_tos_stdevall_length_preserves_height():
    df = _ohlcv(50)
    out = df.ti.tos_stdevall(length=20)
    assert out.height == 50
    # append=True previously raised ShapeError due to truncated output.
    appended = df.ti.tos_stdevall(length=20, append=True)
    assert appended.height == 50


# ---------------------------------------------------------------------------
# F11 — native stoch/stochf must honor mamode (ema != sma)
# ---------------------------------------------------------------------------
def test_f11_stoch_native_mamode_honored():
    df = _ohlcv(80)
    sma = df.select(stoch("high", "low", "close", talib=False, mamode="sma")).unnest("STOCH")
    ema = df.select(stoch("high", "low", "close", talib=False, mamode="ema")).unnest("STOCH")
    diff = (sma["STOCHk_14_3_3"].fill_nan(None) - ema["STOCHk_14_3_3"].fill_nan(None)).abs().max()
    assert diff is not None and diff > 1e-6


def test_f11_stochf_native_mamode_honored():
    df = _ohlcv(80)
    sma = df.select(stochf("high", "low", "close", talib=False, mamode="sma")).unnest("STOCHF")
    ema = df.select(stochf("high", "low", "close", talib=False, mamode="ema")).unnest("STOCHF")
    diff = (sma["STOCHFd_14_3"].fill_nan(None) - ema["STOCHFd_14_3"].fill_nan(None)).abs().max()
    assert diff is not None and diff > 1e-6
