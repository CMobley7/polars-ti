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
