# -*- coding: utf-8 -*-
"""WS3 — native EMA/RMA presma-seed parity.

Pins the fix for the native presma-seeding bug (COMPARISON_REPORT §5–§7,
REMEDIATION_PLAN WS3 + Decision 4: *native = pandas-ta, talib = TA-Lib*).

A leading null/NaN in the input used to poison the recursive Numba EMA/RMA seed
(``presma=True``), turning ``atr``/``natr``/``dema``/``zlma``/aberration's ATR
bands into all-NaN in native mode and drifting ``kdj``/``tsi``/``tmo``/``efi``.

These tests assert, for the full 1500-row SPY slice:
  * native (``talib=False``) study output matches the OLD native golden
    (``old_notalib.parquet``) for the repaired columns, within float tol;
  * the talib branch still matches the OLD talib golden / TA-Lib reference
    (we did not touch the talib path);
  * no all-NaN columns remain in the repaired native cluster.
"""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl
import pytest

import polars_ti as ti  # noqa: F401 — registers the 'ti' accessor
from _parity import assert_column, compare_frames

FIXTURES = "tests/fixtures"
SLICE_ROWS = 1500

# Repaired native all-NaN cluster (§5): go all-NaN -> match OLD native golden.
# NB: NATR_14 was here but classic b914429 changed its native default mamode to
# rma (Wilder), so it now intentionally diverges from the OLD native golden
# (which baked in the buggy ema default). It is graded against talib.NATR
# instead in test_natr_default_mamode_matches_talib.
#
# DEMA_10 and ZL_EMA_10 were also here, but the native EMA warmup seed was later
# corrected to match TA-Lib on leading-NaN/cascaded inputs (was 1 bar early), so
# their native output now MATCHES TA-Lib and diverges from the OLD native golden.
# DEMA_10 is graded vs the TA-Lib reference (match_talib); ZL_EMA_10 (no TA-Lib
# equiv) is intentional. Both are pinned native==talib by
# tests/overlap/test_native_talib_alignment.py.
#
# ATRr_14 was also here, but the native Wilder/RMA seed was later aligned to
# TA-Lib's ATR warmup (SMA of TR[1..length] at index length), so native ATR now
# MATCHES talib.ATR and diverges from the OLD native golden. ATRr_14 is graded
# vs the TA-Lib reference (match_talib) and pinned native==talib by
# tests/overlap/test_native_talib_alignment.py.
NATIVE_CLUSTER = ["ABER_ZG_5_15"]
# Still validated for the leading-NaN seed (not-all-NaN), just not vs the OLD
# golden. NATR_14's seed is exercised by the dedicated talib comparison below.

# Value-drift indicators (§7): align to the OLD native golden.
# NB: TSI/TSIs/TMO/TMOs/EFI were here, but the native EMA seed correction (see
# above) moved their native output onto the SAME indicator's talib=True output.
# They have no TA-Lib reference column, so they are registered "intentional" in
# parity_exceptions and pinned native==talib by test_native_talib_alignment.py.
VALUE_DRIFT = [
    "K_9_3",
    "D_9_3",
    "J_9_3",
]


def _study(talib: bool) -> pl.DataFrame:
    df = pl.read_csv("data/SPY_D.csv", try_parse_dates=True).head(SLICE_ROWS)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return df.ti.study(ti.AllStudy, cores=0, talib=talib, errors="ignore")


@pytest.fixture(scope="module")
def native_report():
    new = _study(talib=False)
    golden = pl.read_parquet(f"{FIXTURES}/old_notalib.parquet")
    return compare_frames(new, golden)


@pytest.fixture(scope="module")
def talib_report():
    from polars_ti.maps import Imports

    if not Imports["talib"]:
        pytest.skip("requires TA-Lib (grades the talib study against the TA-Lib golden)")
    new = _study(talib=True)
    golden = pl.read_parquet(f"{FIXTURES}/old_talib.parquet")
    return compare_frames(new, golden)


@pytest.mark.parametrize("col", NATIVE_CLUSTER + VALUE_DRIFT)
def test_native_matches_old_golden(native_report, col):
    """Native-mode parity vs the OLD native golden (Decision 4)."""
    assert_column(native_report, col)


@pytest.mark.parametrize("col", NATIVE_CLUSTER)
def test_native_cluster_not_all_nan(col):
    """The repaired cluster must no longer be all-NaN in native mode."""
    rep = compare_frames(_study(talib=False), pl.read_parquet(f"{FIXTURES}/old_notalib.parquet"))
    assert rep[col]["cls"] != "no-overlap", f"{col}: still all-NaN in native mode"
    assert rep[col]["overlap"] > 0


@pytest.mark.parametrize("col", ["ATRr_14", "DEMA_10", "ABER_ATR_5_15"])
def test_talib_branch_unchanged(talib_report, col):
    """The talib branch is untouched: still matches the OLD talib golden."""
    assert_column(talib_report, col)


def test_natr_default_mamode_matches_talib():
    """classic b914429: NATR's default native mamode is now rma (Wilder), which
    aligns with ATR and TA-Lib. Validate:
      * talib-mode NATR == talib.NATR exactly, and
      * native (rma) NATR converges to talib.NATR (the residual is purely the
        Wilder seed transient, identical to ATR's; gone after warmup).
    The OLD goldens baked in the buggy ema default, so NATR_14 is registered in
    TALIB_DIVERGENCE / NATIVE_DIVERGENCE instead of matching them."""
    talib = pytest.importorskip("talib")
    from polars_ti.volatility.natr import natr

    df = pl.read_csv("data/SPY_D.csv", try_parse_dates=True).head(SLICE_ROWS)
    h = df["high"].to_numpy().astype(float)
    low_ = df["low"].to_numpy().astype(float)
    c = df["close"].to_numpy().astype(float)
    ref = talib.NATR(h, low_, c, timeperiod=14)

    nat_talib = df.select(natr("high", "low", "close", length=14, talib=True)).to_series().to_numpy()
    nat_native = df.select(natr("high", "low", "close", length=14, talib=False)).to_series().to_numpy()

    m = ~np.isnan(ref) & ~np.isnan(nat_talib)
    assert m.sum() > 100
    assert np.max(np.abs(ref[m] - nat_talib[m])) < 1e-9, "talib NATR not exact vs talib.NATR"

    # Native rma converges to TA-Lib after the Wilder seed transient.
    m2 = ~np.isnan(ref) & ~np.isnan(nat_native)
    m2[:500] = False
    assert np.max(np.abs(ref[m2] - nat_native[m2])) < 1e-6, "native rma NATR did not converge to talib.NATR"


def test_aberration_atr_bands_match_talib_reference():
    """ABER ATR bands: OLD native golden captured TA-Lib ATR (an OLD talib
    non-propagation bug), so NEW talib-mode ATR must match the TA-Lib reference;
    NEW native is the correct pandas-ta value (intentional divergence)."""
    new_t = _study(talib=True)
    ref = pl.read_parquet(f"{FIXTURES}/talib_reference.parquet")
    rep = compare_frames(new_t, ref)
    # ABER_ATR maps onto the TA-Lib ATR(15) reference if present; otherwise the
    # talib-mode parity vs old_talib (asserted above) already covers it.
    if "ATR_15" in rep:
        assert_column(rep, "ATR_15")


def test_native_ema_seed_tolerant_of_leading_nan():
    """Unit guard: a leading NaN must not poison the native EMA/RMA recursion,
    AND the native EMA must now match ``talib.EMA`` on a leading-NaN input.

    The seed was corrected to average the first ``length`` FINITE values and land
    at ``first_finite + length - 1`` (== TA-Lib's warmup), one bar later than the
    old NaN-skipping window that jumped the gun by a bar.
    """
    talib = pytest.importorskip("talib")
    from polars_ti.overlap.ema import _ema_numba
    from polars_ti.overlap.rma import _rma_numba

    x = np.concatenate([[np.nan], np.arange(1.0, 31.0)])  # first finite at index 1
    ema_out = _ema_numba(x, 14, True, False)
    rma_out = _rma_numba(x, 14, True)
    # Anti-poison: a single leading NaN must not wipe out the whole column.
    assert not np.all(np.isnan(ema_out)), "EMA poisoned by leading NaN"
    assert not np.all(np.isnan(rma_out)), "RMA poisoned by leading NaN"
    # Seed lands at first_finite + length - 1 == 14 (TA-Lib warmup), not earlier.
    assert np.isnan(ema_out[13])
    assert not np.isnan(ema_out[14])
    # Native EMA now matches talib.EMA on the same leading-NaN input.
    ref = talib.EMA(x, 14)
    mask = ~np.isnan(ema_out) & ~np.isnan(ref)
    assert mask.sum() > 10
    assert np.max(np.abs(ema_out[mask] - ref[mask])) < 1e-9, "native EMA != talib.EMA on leading-NaN input"


def test_native_ema_survives_long_leading_nan_run():
    """A leading-NaN run longer than ``length`` (cascaded-EMA warmup) must still
    re-seed (not go all-NaN) and land at ``first_finite + length - 1``, matching
    TA-Lib's warmup on the finite tail."""
    talib = pytest.importorskip("talib")
    from polars_ti.overlap.ema import _ema_numba

    x = np.concatenate([[np.nan] * 25, np.arange(25.0, 60.0)])  # first finite at index 25
    out = _ema_numba(x, 13, True, False)
    finite = np.where(~np.isnan(out))[0]
    assert finite.size > 0, "cascaded EMA all-NaN — leading-NaN run poisoned it"
    # Seed averages the first 13 finite values, landing at 25 + 13 - 1 == 37.
    assert finite[0] == 37
    # And the recursion matches talib.EMA over the finite tail.
    ref = talib.EMA(x, 13)
    mask = ~np.isnan(out) & ~np.isnan(ref)
    assert mask.sum() > 5
    assert np.max(np.abs(out[mask] - ref[mask])) < 1e-9
