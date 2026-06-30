# -*- coding: utf-8 -*-
"""Tests for correl - Native Polars pl.Expr API + TA-Lib parity."""

import numpy as np
import polars as pl
import pytest

from polars_ti.maps import Imports
from polars_ti.statistics.correl import correl


@pytest.fixture
def spy() -> pl.DataFrame:
    return pl.read_csv("data/SPY_D.csv", try_parse_dates=True).head(1500)


def test_returns_expr():
    assert isinstance(correl("high", "low"), pl.Expr)


def test_column_name(spy):
    result = spy.select(correl("high", "low"))
    assert "CORREL_30" in result.columns


@pytest.mark.parametrize("length", [5, 30])
def test_native_matches_talib(spy, length):
    pytest.importorskip("talib")
    import talib

    native = spy.select(correl("high", "low", length=length, talib=False)).to_series().to_numpy()
    ref = talib.CORREL(
        spy["high"].to_numpy().astype("float64"),
        spy["low"].to_numpy().astype("float64"),
        timeperiod=length,
    )
    m = ~np.isnan(native) & ~np.isnan(ref)
    assert m.sum() > 100
    assert np.max(np.abs(native[m] - ref[m])) < 1e-7


@pytest.mark.skipif(not Imports["talib"], reason="requires TA-Lib")
def test_talib_path_matches_talib(spy):
    import talib

    native = spy.select(correl("high", "low", length=30, talib=True)).to_series().to_numpy()
    ref = talib.CORREL(
        spy["high"].to_numpy().astype("float64"),
        spy["low"].to_numpy().astype("float64"),
        timeperiod=30,
    )
    m = ~np.isnan(native) & ~np.isnan(ref)
    assert np.max(np.abs(native[m] - ref[m])) < 1e-7
