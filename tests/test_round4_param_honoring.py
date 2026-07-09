# -*- coding: utf-8 -*-
"""Regression tests for the round-4 audit fixes.

Two classes of defect are locked here:

A. Accessor OHLC column overrides. Six accessor methods (bop, brar, avgprice,
   ohlc4, fvg, pdist) used ``kw.setdefault`` to inject default OHLC column names,
   then resolved them with an ``arg or kw.pop(...)`` short-circuit. When the
   caller supplied the column explicitly the pop was skipped, leaking the
   setdefault-injected key into ``**kw`` and raising ``TypeError``.

B. Param-honoring stragglers. Completing the "honor the non-default param the
   TA-Lib C function cannot represent" policy: bop ``scalar`` and cci ``c`` are
   linear rescales; accbands ``c``/``mamode``, sma ``min_periods`` and uo
   weights/``drift`` fall through to the native path. Defaults must stay
   byte-identical to the TA-Lib golden.
"""

import numpy as np
import polars as pl
import pytest

import polars_ti as ti  # noqa: F401 — registers the 'ti' namespace
from polars_ti.maps import Imports

requires_talib = pytest.mark.skipif(not Imports["talib"], reason="requires TA-Lib")


@pytest.fixture(scope="module")
def df():
    return pl.read_csv("data/SPY_D.csv", try_parse_dates=True).head(1500)


def _col(df, expr):
    return df.select(expr.alias("x")).get_column("x").to_numpy()


# --------------------------------------------------------------------------- A
@pytest.mark.parametrize("method", ["bop", "brar", "avgprice", "ohlc4", "fvg", "pdist"])
def test_accessor_ohlc_override_by_kwarg(df, method):
    """An explicit OHLC column override must not leak into the native call."""
    renamed = df.rename({"high": "my_high"})
    # Before the fix this raised: TypeError got multiple values for argument 'high'.
    result = getattr(renamed.ti, method)(high="my_high")
    assert result is not None


@pytest.mark.parametrize("method", ["bop", "brar", "avgprice", "ohlc4", "fvg", "pdist"])
def test_accessor_open_positional_override(df, method):
    """``open_=`` override must resolve without leaking an ``open`` kwarg."""
    result = getattr(df.ti, method)(open_="open")
    assert result is not None


# --------------------------------------------------------------------------- B
@requires_talib
def test_bop_talib_honors_scalar(df):
    base = _col(df, ti.bop("open", "high", "low", "close", scalar=1.0))
    scaled = _col(df, ti.bop("open", "high", "low", "close", scalar=2.0))
    assert np.nanmax(np.abs(scaled - 2.0 * base)) < 1e-9


@requires_talib
def test_cci_talib_honors_c(df):
    base = _col(df, ti.cci("high", "low", "close", c=0.015))
    scaled = _col(df, ti.cci("high", "low", "close", c=0.030))
    assert np.nanmax(np.abs(scaled - 0.5 * base)) < 1e-9


@requires_talib
def test_accbands_talib_honors_c_and_mamode(df):
    default = df.select(ti.accbands("high", "low", "close").alias("s")).unnest("s")
    c8 = df.select(ti.accbands("high", "low", "close", c=8.0).alias("s")).unnest("s")
    ema = df.select(ti.accbands("high", "low", "close", mamode="ema").alias("s")).unnest("s")
    assert not np.allclose(default["ACCBU_20"].to_numpy(), c8["ACCBU_20"].to_numpy(), equal_nan=True)
    assert not np.allclose(default["ACCBM_20"].to_numpy(), ema["ACCBM_20"].to_numpy(), equal_nan=True)


@requires_talib
def test_sma_talib_honors_min_periods(df):
    default = _col(df, ti.sma("close", length=10))
    partial = _col(df, ti.sma("close", length=10, min_periods=3))
    # min_periods=3 emits a value at index 3 where the full-window path is still NaN.
    assert np.isnan(default[3]) and not np.isnan(partial[3])


@requires_talib
def test_uo_talib_honors_weights_and_drift(df):
    default = _col(df, ti.uo("high", "low", "close"))
    equal_w = _col(df, ti.uo("high", "low", "close", fast_w=1.0, medium_w=1.0, slow_w=1.0))
    drift2 = _col(df, ti.uo("high", "low", "close", drift=2))
    assert not np.allclose(default, equal_w, equal_nan=True)
    assert not np.allclose(default, drift2, equal_nan=True)


# --- Defaults must remain byte-identical to the TA-Lib golden ----------------
@requires_talib
@pytest.mark.parametrize(
    "expr_fn, golden_col",
    [
        (lambda: ti.bop("open", "high", "low", "close"), "BOP"),
        (lambda: ti.cci("high", "low", "close"), "CCI_14_0.015"),
        (lambda: ti.sma("close", length=10), "SMA_10"),
        (lambda: ti.uo("high", "low", "close"), "UO_7_14_28"),
    ],
)
def test_default_matches_talib_golden(df, expr_fn, golden_col):
    golden = pl.read_parquet("tests/fixtures/old_talib.parquet").get_column(golden_col).to_numpy()
    got = _col(df, expr_fn())
    assert np.allclose(got, golden, equal_nan=True, atol=1e-6)


@requires_talib
def test_accbands_default_matches_talib_golden(df):
    golden = pl.read_parquet("tests/fixtures/old_talib.parquet")
    got = df.select(ti.accbands("high", "low", "close").alias("s")).unnest("s")
    assert np.allclose(got["ACCBU_20"].to_numpy(), golden.get_column("ACCBU_20").to_numpy(), equal_nan=True, atol=1e-6)


# --- round 5: further drift / prenan stragglers -------------------------------
@requires_talib
def test_cmo_talib_honors_drift(df):
    base = _col(df, ti.cmo("close", drift=1))
    shifted = _col(df, ti.cmo("close", drift=2))
    assert not np.allclose(base, shifted, equal_nan=True)


@requires_talib
def test_dm_talib_honors_drift(df):
    e1 = ti.dm("high", "low", length=14, drift=1)
    e2 = ti.dm("high", "low", length=14, drift=2)
    d1 = df.select(e1[0].alias("p")).get_column("p").to_numpy()
    d2 = df.select(e2[0].alias("p")).get_column("p").to_numpy()
    assert not np.allclose(d1, d2, equal_nan=True)


@requires_talib
def test_vidya_talib_honors_drift(df):
    base = _col(df, ti.vidya("close", drift=1))
    shifted = _col(df, ti.vidya("close", drift=4))
    assert not np.allclose(base, shifted, equal_nan=True)


@requires_talib
def test_mama_talib_honors_prenan(df):
    """prenan is a leading-NaN mask; a larger prenan blanks bars the default kept."""
    small = df.select(ti.mama("close", prenan=3).alias("s")).unnest("s")["MAMA_0.5_0.05"].to_numpy()
    large = df.select(ti.mama("close", prenan=40).alias("s")).unnest("s")["MAMA_0.5_0.05"].to_numpy()
    assert not np.isnan(small[35]) and np.isnan(large[35])


@requires_talib
def test_ht_trendline_talib_honors_prenan(df):
    small = _col(df, ti.ht_trendline("close", prenan=63))
    large = _col(df, ti.ht_trendline("close", prenan=80))
    assert not np.isnan(small[70]) and np.isnan(large[70])


@requires_talib
def test_cmo_default_matches_talib_golden(df):
    """The drift guard is a no-op at drift=1: CMO_14 default stays pinned to talib."""
    golden = pl.read_parquet("tests/fixtures/old_talib.parquet").get_column("CMO_14").to_numpy()
    assert np.allclose(_col(df, ti.cmo("close")), golden, equal_nan=True, atol=1e-6)


@requires_talib
@pytest.mark.parametrize("prenan", [3, 63])
def test_mama_ht_default_prenan_no_op(df, prenan):
    """At default prenan (<= TA-Lib's own warmup) the mask adds no NaNs vs raw talib."""
    import talib

    close = df.get_column("close").to_numpy().astype(np.float64)
    mama_default = df.select(ti.mama("close").alias("s")).unnest("s")["MAMA_0.5_0.05"].to_numpy()
    raw_mama, _ = talib.MAMA(close, 0.5, 0.05)
    assert np.allclose(mama_default, raw_mama, equal_nan=True, atol=1e-9)
    ht_default = _col(df, ti.ht_trendline("close"))
    assert np.allclose(ht_default, talib.HT_TRENDLINE(close), equal_nan=True, atol=1e-9)
