# -*- coding: utf-8 -*-
"""Parity gate for every native candlestick pattern against TA-Lib.

For each ``cdl_<name>`` pattern this asserts that BOTH the native NumPy path
(``talib=False``) and the TA-Lib fast-path (``talib=True``) reproduce
``talib.CDL<NAME>`` exactly (0 mismatches) on SPY daily data. TA-Lib is the
reference, so the module is skipped when TA-Lib is not installed.
"""

import numpy as np
import polars as pl
import pytest

import importlib

from polars_ti.maps import Imports

_cdl_pattern_mod = importlib.import_module("polars_ti.candles.cdl_pattern")

pytestmark = pytest.mark.skipif(not Imports["talib"], reason="requires TA-Lib (the parity reference)")

SLICE_ROWS = 1500

# Patterns whose TA-Lib function (and native detector) take a `penetration`
# parameter — pass the matching default to the TA-Lib reference.
PENETRATION = {
    "abandonedbaby": 0.3,
    "darkcloudcover": 0.5,
    "eveningstar": 0.3,
    "eveningdojistar": 0.3,
    "morningstar": 0.3,
    "morningdojistar": 0.3,
    "mathold": 0.5,
}

# All natively-implemented framework patterns (name -> cdl_<name> function).
NATIVE_PATTERNS = _cdl_pattern_mod.NATIVE_PATTERNS


@pytest.fixture(scope="module")
def ohlc():
    df = pl.read_csv("data/SPY_D.csv", try_parse_dates=True).head(SLICE_ROWS)
    return (
        df,
        df["open"].to_numpy().astype(float),
        df["high"].to_numpy().astype(float),
        df["low"].to_numpy().astype(float),
        df["close"].to_numpy().astype(float),
    )


@pytest.mark.parametrize("name", sorted(NATIVE_PATTERNS))
@pytest.mark.parametrize("talib", [False, True])
def test_cdl_matches_talib(name, talib, ohlc):
    import talib as talib_mod

    df, o, h, low_, c = ohlc
    fn = NATIVE_PATTERNS[name]
    suffix = name.upper()
    ref_fn = getattr(talib_mod, f"CDL{suffix}")

    pen = PENETRATION.get(name)
    if pen is not None:
        ref = ref_fn(o, h, low_, c, penetration=pen).astype(float)
        got = df.select(fn("open", "high", "low", "close", penetration=pen, talib=talib)).to_series().to_numpy()
    else:
        ref = ref_fn(o, h, low_, c).astype(float)
        got = df.select(fn("open", "high", "low", "close", talib=talib)).to_series().to_numpy()

    mismatches = int(np.sum(got != ref))
    assert mismatches == 0, f"CDL_{suffix} (talib={talib}): {mismatches} mismatches vs talib.CDL{suffix}"


def test_all_talib_patterns_covered():
    """Every TA-Lib CDL pattern in ALL_PATTERNS is natively implemented."""
    import talib as talib_mod

    for name in _cdl_pattern_mod.ALL_PATTERNS:
        if name in ("doji", "inside"):
            continue  # parametrised native implementations, not framework ports
        assert name in NATIVE_PATTERNS, f"{name} missing from NATIVE_PATTERNS"
        assert hasattr(talib_mod, f"CDL{name.upper()}"), f"no talib.CDL{name.upper()}"
