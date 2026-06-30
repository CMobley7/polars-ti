# -*- coding: utf-8 -*-
"""Tests for rocr100 - Native Polars pl.Expr API + TA-Lib parity."""

import numpy as np
import polars as pl
import pytest

from polars_ti.maps import Imports
from polars_ti.momentum.rocr100 import rocr100


@pytest.fixture
def spy() -> pl.DataFrame:
    return pl.read_csv("data/SPY_D.csv", try_parse_dates=True).head(1500)


def test_returns_expr():
    assert isinstance(rocr100("close"), pl.Expr)


def test_column_name(spy):
    result = spy.select(rocr100("close"))
    assert "ROCR100_10" in result.columns


def test_native_matches_talib(spy):
    pytest.importorskip("talib")
    import talib

    native = spy.select(rocr100("close", talib=False)).to_series().to_numpy()
    ref = talib.ROCR100(spy["close"].to_numpy().astype("float64"), timeperiod=10)
    m = ~np.isnan(native) & ~np.isnan(ref)
    assert m.sum() > 100
    assert np.max(np.abs(native[m] - ref[m])) < 1e-8


@pytest.mark.skipif(not Imports["talib"], reason="requires TA-Lib")
def test_talib_path_matches_talib(spy):
    import talib

    native = spy.select(rocr100("close", talib=True)).to_series().to_numpy()
    ref = talib.ROCR100(spy["close"].to_numpy().astype("float64"), timeperiod=10)
    m = ~np.isnan(native) & ~np.isnan(ref)
    assert np.max(np.abs(native[m] - ref[m])) < 1e-8
