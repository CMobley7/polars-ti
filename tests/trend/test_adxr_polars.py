# -*- coding: utf-8 -*-
"""Tests for adxr - Native Polars pl.Expr API + TA-Lib parity."""

import numpy as np
import polars as pl
import pytest

from polars_ti.maps import Imports
from polars_ti.trend.adxr import adxr


@pytest.fixture
def spy() -> pl.DataFrame:
    return pl.read_csv("data/SPY_D.csv", try_parse_dates=True).head(1500)


def test_returns_expr():
    assert isinstance(adxr("high", "low", "close"), pl.Expr)


def test_column_name(spy):
    result = spy.select(adxr("high", "low", "close"))
    assert "ADXR_14" in result.columns


def test_native_matches_talib(spy):
    """Native ADXR is derived from the native ADX line, whose Wilder warmup seed
    converges to TA-Lib's exponentially. Post deep-warmup the two agree to float
    noise, so grade from a warmed-up offset."""
    pytest.importorskip("talib")
    import talib

    native = spy.select(adxr("high", "low", "close", talib=False)).to_series().to_numpy()
    ref = talib.ADXR(
        spy["high"].to_numpy().astype("float64"),
        spy["low"].to_numpy().astype("float64"),
        spy["close"].to_numpy().astype("float64"),
        timeperiod=14,
    )
    warm = 400
    native, ref = native[warm:], ref[warm:]
    m = ~np.isnan(native) & ~np.isnan(ref)
    assert m.sum() > 500
    assert np.max(np.abs(native[m] - ref[m])) < 1e-6


@pytest.mark.skipif(not Imports["talib"], reason="requires TA-Lib")
def test_talib_path_matches_talib(spy):
    import talib

    native = spy.select(adxr("high", "low", "close", talib=True)).to_series().to_numpy()
    ref = talib.ADXR(
        spy["high"].to_numpy().astype("float64"),
        spy["low"].to_numpy().astype("float64"),
        spy["close"].to_numpy().astype("float64"),
        timeperiod=14,
    )
    m = ~np.isnan(native) & ~np.isnan(ref)
    assert np.max(np.abs(native[m] - ref[m])) < 1e-8
