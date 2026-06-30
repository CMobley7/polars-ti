# -*- coding: utf-8 -*-
"""Tests for smc_sweep - Native Polars pl.Expr API + classic-fork parity.

smc_sweep has no TA-Lib equivalent, so it is validated against the
pandas-ta-classic reference implementation cloned at tmp/pandas-ta-classic.
"""

import os
import sys

import numpy as np
import polars as pl
import pytest

from polars_ti.momentum.smc_sweep import smc_sweep

CLASSIC = os.path.join(os.getcwd(), "tmp", "pandas-ta-classic")


@pytest.fixture
def spy() -> pl.DataFrame:
    return pl.read_csv("data/SPY_D.csv", try_parse_dates=True).head(1500)


def test_returns_expr():
    assert isinstance(smc_sweep("open", "high", "low", "close"), pl.Expr)


def test_column_name(spy):
    result = spy.select(smc_sweep("open", "high", "low", "close"))
    assert "SMC_SWEEP_15_1.5" in result.columns


def test_values_are_ternary(spy):
    arr = spy.select(smc_sweep("open", "high", "low", "close")).to_series().to_numpy()
    assert set(np.unique(arr)).issubset({-1, 0, 1})


@pytest.mark.parametrize("length,wick_mult", [(15, 1.5), (10, 2.0), (20, 1.0), (20, 2.5)])
def test_matches_classic_fork(spy, length, wick_mult):
    """Native Polars output must exactly match the pandas-ta-classic reference.

    Both implementations are fed the *same* float64 inputs (sourced from the
    Polars frame) so the comparison is not polluted by ULP-level differences
    between the pandas and Polars CSV parsers — which otherwise create spurious
    one-bar disagreements exactly on the ``wick > body * wick_mult`` boundary.
    """
    if not os.path.isdir(CLASSIC):
        pytest.skip("pandas-ta-classic fork not available at tmp/pandas-ta-classic")
    if CLASSIC not in sys.path:
        sys.path.insert(0, CLASSIC)
    pd = pytest.importorskip("pandas")
    try:
        from pandas_ta_classic.momentum.smc_sweep import smc_sweep as classic_smc_sweep
    except Exception as exc:  # pragma: no cover - environment guard
        pytest.skip(f"cannot import classic smc_sweep: {exc}")

    # Identical inputs for both engines.
    o = pd.Series(spy["open"].to_numpy())
    h = pd.Series(spy["high"].to_numpy())
    lo = pd.Series(spy["low"].to_numpy())
    c = pd.Series(spy["close"].to_numpy())
    ref = classic_smc_sweep(o, h, lo, c, length=length, wick_mult=wick_mult).to_numpy()

    native = (
        spy.select(smc_sweep("open", "high", "low", "close", length=length, wick_mult=wick_mult)).to_series().to_numpy()
    )

    m = ~pd.isna(ref)
    assert m.sum() == len(ref)  # classic emits 0 (not NaN) everywhere
    assert np.max(np.abs(native.astype(float)[m] - ref.astype(float)[m])) == 0.0
