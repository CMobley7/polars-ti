# -*- coding: utf-8 -*-
"""Unit tests for polars_ti/trend/psar.py Polars implementation.

Pins the classic 9258bf6 PSAR fixes:
  * the reversal test uses the GUARDED SAR (after clamping), not the raw
    projection;
  * the guard spans the prior TWO bars' high/low;
  * long/short are reclassified from the combined SAR using close
    (SAR < close -> long, SAR >= close -> short), matching TA-Lib.

TA-Lib's SAR is a single series, so the per-bar combined SAR line
(PSARl where long, else PSARs) is validated against talib.SAR.
"""

import numpy as np
import polars as pl
import pytest

import polars_ti as ti  # noqa: F401 — registers the 'ti' accessor
from polars_ti.trend.psar import psar


def _combined(df: pl.DataFrame) -> np.ndarray:
    out = df.select(psar("high", "low", "close")).unnest("PSAR_0.02_0.2")
    psl = out["PSARl_0.02_0.2"].to_numpy()
    pss = out["PSARs_0.02_0.2"].to_numpy()
    return np.where(~np.isnan(psl), psl, pss), psl, pss


def test_returns_struct_columns():
    df = pl.read_csv("data/SPY_D.csv", try_parse_dates=True).head(200)
    out = df.select(psar("high", "low", "close")).unnest("PSAR_0.02_0.2")
    assert set(out.columns) == {
        "PSARl_0.02_0.2",
        "PSARs_0.02_0.2",
        "PSARaf_0.02_0.2",
        "PSARr_0.02_0.2",
    }


def test_combined_sar_matches_talib():
    """The combined PSAR line == talib.SAR exactly after the warmup transient.

    Only the first ~10 bars differ (TA-Lib seeds the first SAR differently from
    our close-based seed); structure and reversals match TA-Lib thereafter."""
    talib = pytest.importorskip("talib")
    df = pl.read_csv("data/SPY_D.csv", try_parse_dates=True).head(1500)
    h = df["high"].to_numpy().astype(float)
    low_ = df["low"].to_numpy().astype(float)
    ref = talib.SAR(h, low_, acceleration=0.02, maximum=0.2)
    comb, _, _ = _combined(df)
    mask = ~np.isnan(comb) & ~np.isnan(ref)
    mask[:11] = False  # skip the seed-driven warmup transient
    assert mask.sum() > 1000
    assert np.max(np.abs(comb[mask] - ref[mask])) < 1e-9


def test_warmup_transient_is_small_and_local():
    """The pre-fix bug produced a large (>7) global divergence from talib.SAR;
    after the fix the only divergence is a tiny, local warmup transient."""
    talib = pytest.importorskip("talib")
    df = pl.read_csv("data/SPY_D.csv", try_parse_dates=True).head(1500)
    h = df["high"].to_numpy().astype(float)
    low_ = df["low"].to_numpy().astype(float)
    ref = talib.SAR(h, low_, acceleration=0.02, maximum=0.2)
    comb, _, _ = _combined(df)
    diff = np.abs(comb - ref)
    diff[np.isnan(diff)] = 0.0
    # All divergences > 0.01 are confined to the early warmup (< index 11).
    assert np.all(np.where(diff > 0.01)[0] < 11)
    assert np.max(diff) < 0.1


def test_long_short_close_reclassification():
    """No bar is classified as both long and short, and the split follows close:
    long where combined SAR < close, short where SAR >= close (classic 9258bf6)."""
    df = pl.read_csv("data/SPY_D.csv", try_parse_dates=True).head(1500)
    comb, psl, pss = _combined(df)
    close = df["close"].to_numpy().astype(float)
    both = (~np.isnan(psl)) & (~np.isnan(pss))
    assert int(both.sum()) == 0
    # Where long is set, SAR < close; where short is set, SAR >= close.
    long_mask = ~np.isnan(psl)
    short_mask = ~np.isnan(pss)
    assert np.all(psl[long_mask] < close[long_mask])
    assert np.all(pss[short_mask] >= close[short_mask])
