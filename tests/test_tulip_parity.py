# -*- coding: utf-8 -*-
"""Validation tests for Tulip-parity indicators against golden fixtures and TA-Lib.

Compares polars_ti native outputs to:
  - Committed golden fixtures generated once from pandas-ta-classic (the classic
    fork).  This lets the cross-validation run in CI *without* the clone, which
    is gitignored and absent there.  Regenerate with
    ``tests/fixtures/_generate_classic_parity.py`` when the reference changes.
  - TA-Lib directly for avgprice (AVGPRICE), medprice (MEDPRICE),
    typprice (TYPPRICE).

The fixtures were produced from the *same* float64 SPY_D slice these tests load,
so the comparison is not polluted by ULP-level CSV-parse differences.
"""

import os

import numpy as np
import polars as pl
import pytest

import polars_ti as ti  # noqa: F401
from polars_ti.maps import Imports

SLICE_ROWS = 500
ABS_TOL = 1e-10  # tight float64 tolerance
FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "classic_tulip_parity.parquet")


@pytest.fixture(scope="module")
def data():
    """Load SPY and return both Polars and numpy arrays (no pyarrow needed)."""
    df = pl.read_csv("data/SPY_D.csv", try_parse_dates=True).head(SLICE_ROWS)
    arrs = {c: df[c].to_numpy().astype(float) for c in ["open", "high", "low", "close", "volume"]}
    return df, arrs


@pytest.fixture(scope="module")
def golden():
    """Committed classic-fork reference outputs (generated once from the clone)."""
    return pl.read_parquet(FIXTURE)


def _assert_close(name, got, ref, tol=ABS_TOL):
    got = np.asarray(got, dtype=float)
    ref = np.asarray(ref, dtype=float)
    mask = ~(np.isnan(got) | np.isnan(ref))
    assert mask.sum() > 0, f"{name}: no valid (non-NaN) rows to compare"
    max_err = float(np.max(np.abs(got[mask] - ref[mask])))
    assert max_err < tol, f"{name}: max_abs_error={max_err:.2e} exceeds tol={tol:.2e}"


# ---------------------------------------------------------------------------
# avgprice / medprice / typprice — vs classic fixture AND TA-Lib
# ---------------------------------------------------------------------------


def test_avgprice_vs_classic(data, golden):
    df, _ = data
    r_native = df.ti.avgprice(talib=False).to_series().to_numpy()
    _assert_close("avgprice vs classic", r_native, golden["avgprice"].to_numpy())


@pytest.mark.skipif(not Imports["talib"], reason="requires TA-Lib")
def test_avgprice_talib_path(data):
    df, arrs = data
    import talib

    r_polars = df.ti.avgprice(talib=True).to_series().to_numpy()
    r_talib = talib.AVGPRICE(arrs["open"], arrs["high"], arrs["low"], arrs["close"])
    _assert_close("avgprice talib-path vs talib.AVGPRICE", r_polars, r_talib)


def test_medprice_vs_classic(data, golden):
    df, _ = data
    r_native = df.ti.medprice(talib=False).to_series().to_numpy()
    _assert_close("medprice vs classic", r_native, golden["medprice"].to_numpy())


@pytest.mark.skipif(not Imports["talib"], reason="requires TA-Lib")
def test_medprice_talib_path(data):
    df, arrs = data
    import talib

    r_polars = df.ti.medprice(talib=True).to_series().to_numpy()
    r_talib = talib.MEDPRICE(arrs["high"], arrs["low"])
    _assert_close("medprice talib-path vs talib.MEDPRICE", r_polars, r_talib)


def test_typprice_vs_classic(data, golden):
    df, _ = data
    r_native = df.ti.typprice(talib=False).to_series().to_numpy()
    _assert_close("typprice vs classic", r_native, golden["typprice"].to_numpy())


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


def test_msw_vs_classic(data, golden):
    df, _ = data
    msw_out = df.select(df.ti.msw()).to_series().struct.unnest()
    _assert_close("msw sine vs classic", msw_out["sine"].to_numpy(), golden["msw_sine"].to_numpy())
    _assert_close("msw lead vs classic", msw_out["lead"].to_numpy(), golden["msw_lead"].to_numpy())


# ---------------------------------------------------------------------------
# cvi — volatility
# ---------------------------------------------------------------------------


def test_cvi_vs_classic(data, golden):
    df, _ = data
    _assert_close("cvi vs classic", df.ti.cvi().to_series().to_numpy(), golden["cvi"].to_numpy())


# ---------------------------------------------------------------------------
# hvol — volatility
# ---------------------------------------------------------------------------


def test_hvol_vs_classic(data, golden):
    df, _ = data
    _assert_close("hvol vs classic", df.ti.hvol().to_series().to_numpy(), golden["hvol"].to_numpy())


# ---------------------------------------------------------------------------
# avolume — volatility
# ---------------------------------------------------------------------------


def test_avolume_vs_classic(data, golden):
    df, _ = data
    _assert_close("avolume vs classic", df.ti.avolume().to_series().to_numpy(), golden["avolume"].to_numpy())


# ---------------------------------------------------------------------------
# marketfi — volume
# ---------------------------------------------------------------------------


def test_marketfi_vs_classic(data, golden):
    df, _ = data
    _assert_close("marketfi vs classic", df.ti.marketfi().to_series().to_numpy(), golden["marketfi"].to_numpy())


# ---------------------------------------------------------------------------
# vosc — volume
# ---------------------------------------------------------------------------


def test_vosc_vs_classic(data, golden):
    df, _ = data
    _assert_close("vosc vs classic", df.ti.vosc().to_series().to_numpy(), golden["vosc"].to_numpy())


# ---------------------------------------------------------------------------
# wad — volume
# ---------------------------------------------------------------------------


def test_wad_vs_classic(data, golden):
    df, _ = data
    _assert_close("wad vs classic", df.ti.wad().to_series().to_numpy(), golden["wad"].to_numpy())


# ---------------------------------------------------------------------------
# emv — volume (raw tulip variant)
# ---------------------------------------------------------------------------


def test_emv_vs_classic(data, golden):
    df, _ = data
    _assert_close("emv vs classic", df.ti.emv().to_series().to_numpy(), golden["emv"].to_numpy())


# ---------------------------------------------------------------------------
# fosc — momentum
# ---------------------------------------------------------------------------


def test_fosc_vs_classic(data, golden):
    df, _ = data
    _assert_close("fosc vs classic", df.ti.fosc().to_series().to_numpy(), golden["fosc"].to_numpy(), tol=1e-9)


# ---------------------------------------------------------------------------
# stderr — statistics
# ---------------------------------------------------------------------------


def test_stderr_vs_classic(data, golden):
    df, _ = data
    _assert_close("stderr vs classic", df.ti.stderr().to_series().to_numpy(), golden["stderr"].to_numpy())


# ---------------------------------------------------------------------------
# md — statistics (mean deviation)
# ---------------------------------------------------------------------------


def test_md_vs_classic(data, golden):
    df, _ = data
    _assert_close("md vs classic", df.ti.md().to_series().to_numpy(), golden["md"].to_numpy())
