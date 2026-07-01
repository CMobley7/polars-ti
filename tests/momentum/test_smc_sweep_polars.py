# -*- coding: utf-8 -*-
"""Tests for smc_sweep - Native Polars pl.Expr API + classic-fork parity.

smc_sweep has no TA-Lib equivalent, so it is validated against committed golden
fixtures generated once from the pandas-ta-classic reference fork.  The fixtures
live at ``tests/fixtures/classic_smc_sweep.parquet`` and let this validation run
in CI without the (gitignored, CI-absent) clone.  Regenerate them with
``tests/fixtures/_generate_classic_parity.py`` when the reference changes.
"""

import os

import numpy as np
import polars as pl
import pytest

from polars_ti.momentum.smc_sweep import smc_sweep

SMC_ROWS = 1500
FIXTURE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "fixtures", "classic_smc_sweep.parquet")


@pytest.fixture
def spy() -> pl.DataFrame:
    return pl.read_csv("data/SPY_D.csv", try_parse_dates=True).head(SMC_ROWS)


@pytest.fixture(scope="module")
def golden() -> pl.DataFrame:
    return pl.read_parquet(FIXTURE)


def test_returns_expr():
    assert isinstance(smc_sweep("open", "high", "low", "close"), pl.Expr)


def test_column_name(spy):
    result = spy.select(smc_sweep("open", "high", "low", "close"))
    assert "SMC_SWEEP_15_1.5" in result.columns


def test_values_are_ternary(spy):
    arr = spy.select(smc_sweep("open", "high", "low", "close")).to_series().to_numpy()
    assert set(np.unique(arr)).issubset({-1, 0, 1})


@pytest.mark.parametrize("length,wick_mult", [(15, 1.5), (10, 2.0), (20, 1.0), (20, 2.5)])
def test_matches_classic_fork(spy, golden, length, wick_mult):
    """Native Polars output must exactly match the committed classic-fork golden.

    The fixture was generated from the *same* float64 SPY_D slice loaded here, so
    the comparison is not polluted by ULP-level differences between the pandas
    and Polars CSV parsers — which otherwise create spurious one-bar
    disagreements exactly on the ``wick > body * wick_mult`` boundary.
    """
    ref = golden[f"smc_{length}_{wick_mult}"].to_numpy()

    native = (
        spy.select(smc_sweep("open", "high", "low", "close", length=length, wick_mult=wick_mult)).to_series().to_numpy()
    )

    m = ~np.isnan(ref)
    assert m.sum() == len(ref)  # classic emits 0 (not NaN) everywhere
    assert np.max(np.abs(native.astype(float)[m] - ref.astype(float)[m])) == 0.0
