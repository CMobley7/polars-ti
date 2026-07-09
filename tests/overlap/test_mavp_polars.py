# -*- coding: utf-8 -*-
"""Tests for mavp - Native Polars pl.Expr API + TA-Lib parity."""

import numpy as np
import polars as pl
import pytest

from polars_ti.maps import Imports
from polars_ti.overlap.mavp import mavp


@pytest.fixture
def spy() -> pl.DataFrame:
    df = pl.read_csv("data/SPY_D.csv", try_parse_dates=True).head(1500)
    # Deterministic, varied per-bar period spanning the [2, 30] clamp range.
    return df.with_columns((pl.col("close") % 25 + 3).alias("periods"))


def test_returns_expr():
    assert isinstance(mavp("close", "periods"), pl.Expr)


def test_column_name(spy):
    result = spy.select(mavp("close", "periods"))
    assert "MAVP_2_30" in result.columns


def test_native_matches_talib(spy):
    pytest.importorskip("talib")
    import talib

    native = spy.select(mavp("close", "periods", talib=False)).to_series().to_numpy()
    ref = talib.MAVP(
        spy["close"].to_numpy().astype("float64"),
        spy["periods"].to_numpy().astype("float64"),
        minperiod=2,
        maxperiod=30,
        matype=0,
    )
    m = ~np.isnan(native) & ~np.isnan(ref)
    assert m.sum() > 1000
    assert np.max(np.abs(native[m] - ref[m])) < 1e-9


@pytest.mark.skipif(not Imports["talib"], reason="requires TA-Lib")
def test_talib_path_matches_talib(spy):
    import talib

    native = spy.select(mavp("close", "periods", talib=True)).to_series().to_numpy()
    ref = talib.MAVP(
        spy["close"].to_numpy().astype("float64"),
        spy["periods"].to_numpy().astype("float64"),
        minperiod=2,
        maxperiod=30,
        matype=0,
    )
    m = ~np.isnan(native) & ~np.isnan(ref)
    assert np.max(np.abs(native[m] - ref[m])) < 1e-12


def test_native_matype_nonzero_raises():
    """The native kernel implements SMA only; non-zero matype must be rejected
    when TA-Lib is unavailable/disabled."""
    df = pl.DataFrame({"close": [1.0] * 40, "periods": [5.0] * 40})
    with pytest.raises(ValueError, match="matype=0"):
        df.select(mavp("close", "periods", matype=1, talib=False))
