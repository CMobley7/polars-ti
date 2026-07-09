# -*- coding: utf-8 -*-
"""WS5 study-wide completeness tests.

Lock the AllStudy output to a committed column manifest and guarantee no
indicator silently drops or emits an all-NaN/all-null column, in BOTH the
talib=True and talib=False code paths.
"""

import json
import warnings

import numpy as np
import polars as pl
import pytest

import polars_ti as ti  # noqa: F401 — registers 'ti'
from _parity import flatten_structs
from polars_ti.maps import Imports

FIXTURES = "tests/fixtures"
SLICE_ROWS = 1500
BASE = {"date", "open", "high", "low", "close", "volume", "dividends", "stock splits"}

# In a no-TA-Lib environment the talib=True study falls back to native output,
# so only the talib=False expectations are valid; skip the talib=True cases.
_TALIB_MODES = [False] if not Imports["talib"] else [True, False]


@pytest.fixture(scope="module")
def manifest():
    with open(f"{FIXTURES}/expected_columns.json") as f:
        return json.load(f)


def _study(talib: bool) -> pl.DataFrame:
    df = pl.read_csv("data/SPY_D.csv", try_parse_dates=True).head(SLICE_ROWS)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return df.ti.study(ti.AllStudy, cores=0, talib=talib, errors="ignore")


@pytest.mark.parametrize("talib", _TALIB_MODES)
def test_study_column_manifest(manifest, talib):
    """The AllStudy output (flattened) must match the committed column manifest
    exactly in each mode — no silent drops, no unexpected additions. With the
    native candlestick suite, the talib=False set now matches the talib=True set
    (both include the ~60 candle patterns)."""
    flat = flatten_structs(_study(talib))
    got = sorted(flat.columns)
    key = "flat_columns_talib" if talib else "flat_columns_notalib"
    expected = sorted(manifest[key])
    missing = sorted(set(expected) - set(got))
    extra = sorted(set(got) - set(expected))
    assert not missing, f"talib={talib}: columns dropped from study: {missing}"
    assert not extra, f"talib={talib}: unexpected new columns: {extra}"


@pytest.mark.parametrize("talib", _TALIB_MODES)
def test_no_all_nan_study_columns(talib):
    """No indicator column may be entirely NaN/null in either mode."""
    flat = flatten_structs(_study(talib))
    all_nan = []
    for c in flat.columns:
        if c in BASE:
            continue
        s = flat[c]
        if s.dtype.is_numeric():
            if bool(np.all(np.isnan(s.to_numpy().astype(float)))):
                all_nan.append(c)
        elif s.null_count() == s.len():
            all_nan.append(c)
    assert not all_nan, f"talib={talib}: all-NaN/all-null columns: {all_nan}"
