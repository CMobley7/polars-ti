# -*- coding: utf-8 -*-
"""Validation tests for Tulip-parity indicators against the classic fork and TA-Lib.

Compares polars_ti native outputs to:
  - pandas-ta-classic (classic fork) for all new tulip-parity indicators
  - TA-Lib directly for avgprice (AVGPRICE), medprice (MEDPRICE), typprice (TYPPRICE)

The comparison uses identical float64 arrays sourced from the same Polars frame
to avoid 1-ULP CSV-parse mismatches.
"""

import sys

import numpy as np
import polars as pl
import pytest

import polars_ti as ti  # noqa: F401

sys.path.insert(0, "tmp/pandas-ta-classic")
import pandas_ta_classic as pta  # noqa: E402

from polars_ti.maps import Imports  # noqa: E402

SLICE_ROWS = 500
ABS_TOL = 1e-10  # tight float64 tolerance


@pytest.fixture(scope="module")
def data():
    """Load SPY and return both Polars and numpy arrays (no pyarrow needed)."""
    df = pl.read_csv("data/SPY_D.csv", try_parse_dates=True).head(SLICE_ROWS)
    arrs = {c: df[c].to_numpy().astype(float) for c in ["open", "high", "low", "close", "volume"]}
    return df, arrs


def _pd_series(arr, arrs):
    """Wrap a numpy array in a pandas Series with matching index."""
    import pandas as pd

    return pd.Series(arr)


def _assert_close(name, got, ref, tol=ABS_TOL):
    mask = ~(np.isnan(got) | np.isnan(ref))
    assert mask.sum() > 0, f"{name}: no valid (non-NaN) rows to compare"
    max_err = float(np.max(np.abs(got[mask] - ref[mask])))
    assert max_err < tol, f"{name}: max_abs_error={max_err:.2e} exceeds tol={tol:.2e}"


# ---------------------------------------------------------------------------
# avgprice / medprice / typprice — vs classic fork AND TA-Lib
# ---------------------------------------------------------------------------


def test_avgprice_vs_classic(data):
    df, arrs = data
    import pandas as pd

    r_native = df.ti.avgprice(talib=False).to_series().to_numpy()
    r_classic = pta.avgprice(
        pd.Series(arrs["open"]),
        pd.Series(arrs["high"]),
        pd.Series(arrs["low"]),
        pd.Series(arrs["close"]),
        talib=False,
    ).values
    _assert_close("avgprice vs classic", r_native, r_classic)


@pytest.mark.skipif(not Imports["talib"], reason="requires TA-Lib")
def test_avgprice_talib_path(data):
    df, arrs = data
    import talib

    r_polars = df.ti.avgprice(talib=True).to_series().to_numpy()
    r_talib = talib.AVGPRICE(arrs["open"], arrs["high"], arrs["low"], arrs["close"])
    _assert_close("avgprice talib-path vs talib.AVGPRICE", r_polars, r_talib)


def test_medprice_vs_classic(data):
    df, arrs = data
    import pandas as pd

    r_native = df.ti.medprice(talib=False).to_series().to_numpy()
    r_classic = pta.medprice(pd.Series(arrs["high"]), pd.Series(arrs["low"]), talib=False).values
    _assert_close("medprice vs classic", r_native, r_classic)


@pytest.mark.skipif(not Imports["talib"], reason="requires TA-Lib")
def test_medprice_talib_path(data):
    df, arrs = data
    import talib

    r_polars = df.ti.medprice(talib=True).to_series().to_numpy()
    r_talib = talib.MEDPRICE(arrs["high"], arrs["low"])
    _assert_close("medprice talib-path vs talib.MEDPRICE", r_polars, r_talib)


def test_typprice_vs_classic(data):
    df, arrs = data
    import pandas as pd

    r_native = df.ti.typprice(talib=False).to_series().to_numpy()
    r_classic = pta.typprice(
        pd.Series(arrs["high"]), pd.Series(arrs["low"]), pd.Series(arrs["close"]), talib=False
    ).values
    _assert_close("typprice vs classic", r_native, r_classic)


@pytest.mark.skipif(not Imports["talib"], reason="requires TA-Lib")
def test_typprice_talib_path(data):
    df, arrs = data
    import talib

    r_polars = df.ti.typprice(talib=True).to_series().to_numpy()
    r_talib = talib.TYPPRICE(arrs["high"], arrs["low"], arrs["close"])
    _assert_close("typprice talib-path vs talib.TYPPRICE", r_polars, r_talib)


# ---------------------------------------------------------------------------
# msw — cycles
# ---------------------------------------------------------------------------


def test_msw_vs_classic(data):
    df, arrs = data
    import pandas as pd

    # Run polars msw — returns a struct, expand it
    msw_out = df.select(df.ti.msw()).to_series().struct.unnest()
    sine_native = msw_out["sine"].to_numpy()
    lead_native = msw_out["lead"].to_numpy()

    r_classic = pta.msw(pd.Series(arrs["close"]))
    sine_classic = r_classic["MSW_SINE_5"].values
    lead_classic = r_classic["MSW_LEAD_5"].values

    _assert_close("msw sine vs classic", sine_native, sine_classic)
    _assert_close("msw lead vs classic", lead_native, lead_classic)


# ---------------------------------------------------------------------------
# cvi — volatility
# ---------------------------------------------------------------------------


def test_cvi_vs_classic(data):
    df, arrs = data
    import pandas as pd

    r_native = df.ti.cvi().to_series().to_numpy()
    r_classic = pta.cvi(pd.Series(arrs["high"]), pd.Series(arrs["low"])).values
    _assert_close("cvi vs classic", r_native, r_classic)


# ---------------------------------------------------------------------------
# hvol — volatility
# ---------------------------------------------------------------------------


def test_hvol_vs_classic(data):
    df, arrs = data
    import pandas as pd

    r_native = df.ti.hvol().to_series().to_numpy()
    r_classic = pta.hvol(pd.Series(arrs["close"])).values
    _assert_close("hvol vs classic", r_native, r_classic)


# ---------------------------------------------------------------------------
# avolume — volatility
# ---------------------------------------------------------------------------


def test_avolume_vs_classic(data):
    df, arrs = data
    import pandas as pd

    r_native = df.ti.avolume().to_series().to_numpy()
    r_classic = pta.avolume(pd.Series(arrs["close"])).values
    _assert_close("avolume vs classic", r_native, r_classic)


# ---------------------------------------------------------------------------
# marketfi — volume
# ---------------------------------------------------------------------------


def test_marketfi_vs_classic(data):
    df, arrs = data
    import pandas as pd

    r_native = df.ti.marketfi().to_series().to_numpy()
    r_classic = pta.marketfi(pd.Series(arrs["high"]), pd.Series(arrs["low"]), pd.Series(arrs["volume"])).values
    _assert_close("marketfi vs classic", r_native, r_classic)


# ---------------------------------------------------------------------------
# vosc — volume
# ---------------------------------------------------------------------------


def test_vosc_vs_classic(data):
    df, arrs = data
    import pandas as pd

    r_native = df.ti.vosc().to_series().to_numpy()
    r_classic = pta.vosc(pd.Series(arrs["volume"])).values
    _assert_close("vosc vs classic", r_native, r_classic)


# ---------------------------------------------------------------------------
# wad — volume
# ---------------------------------------------------------------------------


def test_wad_vs_classic(data):
    df, arrs = data
    import pandas as pd

    r_native = df.ti.wad().to_series().to_numpy()
    r_classic = pta.wad(pd.Series(arrs["high"]), pd.Series(arrs["low"]), pd.Series(arrs["close"])).values
    _assert_close("wad vs classic", r_native, r_classic)


# ---------------------------------------------------------------------------
# emv — volume (raw tulip variant)
# ---------------------------------------------------------------------------


def test_emv_vs_classic(data):
    df, arrs = data
    import pandas as pd

    r_native = df.ti.emv().to_series().to_numpy()
    r_classic = pta.emv(pd.Series(arrs["high"]), pd.Series(arrs["low"]), pd.Series(arrs["volume"])).values
    _assert_close("emv vs classic", r_native, r_classic)


# ---------------------------------------------------------------------------
# fosc — momentum
# ---------------------------------------------------------------------------


def test_fosc_vs_classic(data):
    df, arrs = data
    import pandas as pd

    r_native = df.ti.fosc().to_series().to_numpy()
    r_classic = pta.fosc(pd.Series(arrs["close"])).values
    _assert_close("fosc vs classic", r_native, r_classic, tol=1e-9)


# ---------------------------------------------------------------------------
# stderr — statistics
# ---------------------------------------------------------------------------


def test_stderr_vs_classic(data):
    df, arrs = data
    import pandas as pd

    r_native = df.ti.stderr().to_series().to_numpy()
    r_classic = pta.stderr(pd.Series(arrs["close"])).values
    _assert_close("stderr vs classic", r_native, r_classic)


# ---------------------------------------------------------------------------
# md — statistics (mean deviation)
# ---------------------------------------------------------------------------


def test_md_vs_classic(data):
    df, arrs = data
    import pandas as pd

    r_native = df.ti.md().to_series().to_numpy()
    r_classic = pta.md(pd.Series(arrs["close"])).values
    _assert_close("md vs classic", r_native, r_classic)
