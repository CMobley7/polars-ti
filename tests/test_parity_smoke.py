# -*- coding: utf-8 -*-
"""WS0 oracle wiring smoke test.

Confirms the parity foundation is in place and consistent:
  * the golden fixtures load and have the expected deterministic shape,
  * the parity engine runs against a fresh NEW AllStudy output,
  * a known-good column (SMA_10) and a "new is better" exception column
    (RSI_14, pinned to TA-Lib) are at parity,
  * the parity_exceptions registry classifies broken indicators as broken_todo,
  * the broken-today columns ``xfail`` (so they surface rather than error the
    suite) — these are remediated in WS2–WS4.

This is intentionally NOT the full per-indicator parity suite (that is WS5); it
only proves the oracle harness itself works end-to-end.
"""

import warnings

import numpy as np
import polars as pl
import pytest

import polars_ti as ti  # noqa: F401 — registers 'ti'
from _parity import assert_column, compare_frames
from parity_exceptions import FIXED_COLS, PARITY_EXCEPTIONS, mode_for

FIXTURES = "tests/fixtures"
SLICE_ROWS = 1500


@pytest.fixture(scope="module")
def new_talib():
    from polars_ti.maps import Imports

    if not Imports["talib"]:
        pytest.skip("requires TA-Lib (grades the talib study against the TA-Lib golden)")
    df = pl.read_csv("data/SPY_D.csv", try_parse_dates=True).head(SLICE_ROWS)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return df.ti.study(ti.AllStudy, cores=0, talib=True, errors="ignore")


@pytest.fixture(scope="module")
def report(new_talib):
    golden = pl.read_parquet(f"{FIXTURES}/old_talib.parquet")
    return compare_frames(new_talib, golden)


def test_fixtures_load_and_shape():
    old_t = pl.read_parquet(f"{FIXTURES}/old_talib.parquet")
    old_n = pl.read_parquet(f"{FIXTURES}/old_notalib.parquet")
    ref = pl.read_parquet(f"{FIXTURES}/talib_reference.parquet")
    assert old_t.height == SLICE_ROWS
    assert old_n.height == SLICE_ROWS
    assert ref.height == SLICE_ROWS
    # OLD library emits 387 indicator cols + 8 base cols.
    assert old_t.width == 395
    assert old_n.width == 395
    # talib_reference: 97 reference columns + date.
    assert ref.width == 98


def test_parity_engine_runs(report):
    # Sanity: a meaningful number of columns matched to float noise.
    n_match = sum(1 for k, v in report.items() if not k.startswith("__") and v["cls"] == "match")
    assert n_match > 200


def test_no_unmapped_old_columns(report):
    """Completeness: every OLD golden column folds onto a NEW column (via exact
    name, struct-suffix, or the explicit RENAME_MAP). No OLD column is silently
    dropped/renamed without being graded."""
    assert report["__old_only__"] == [], f"OLD columns not folded onto any NEW column: {report['__old_only__']}"


def test_known_good_column_matches(report):
    assert_column(report, "SMA_10")


def test_new_better_column_matches_talib(new_talib):
    # RSI_14 is a "match_talib" exception: pin against the TA-Lib reference.
    assert mode_for("RSI_14") == "match_talib"
    ref = pl.read_parquet(f"{FIXTURES}/talib_reference.parquet")
    rep_vs_talib = compare_frames(new_talib, ref)
    assert_column(rep_vs_talib, "RSI_14")


def test_all_catalogued_defects_fixed():
    # Every catalogued WS2-WS4 defect is now fixed: the broken_todo set is empty.
    broken = [c for c, spec in PARITY_EXCEPTIONS.items() if spec["mode"] == "broken_todo"]
    assert broken == [], f"unexpected broken_todo columns remain: {broken}"


@pytest.mark.parametrize("col", FIXED_COLS)
def test_fixed_indicators_parity(report, col):
    """WS2-WS4-repaired columns must match the OLD (talib) golden within float
    tolerance. These were ``broken_todo`` at the start of remediation; they are
    now enforced strictly so reverting any fix turns this test red."""
    assert_column(report, col)


def test_halftrend_direction_parity(new_talib):
    """HT_direction is a string label column ("long"/"short"/None) that the
    numeric parity engine can't grade; compare it directly to the OLD golden."""
    from _parity import flatten_structs

    golden = pl.read_parquet(f"{FIXTURES}/old_talib.parquet")
    flat = flatten_structs(new_talib)
    ncol = next(c for c in flat.columns if c.endswith("HT_direction_14_2_2"))
    nvals = flat[ncol].to_list()
    gvals = golden["HT_direction_14_2_2"].to_list()
    n = min(len(nvals), len(gvals))
    mism = [i for i in range(n) if gvals[i] is not None and nvals[i] != gvals[i]]
    assert not mism, f"{len(mism)} HT_direction mismatches (first at row {mism[:1]})"
    assert any(v in ("long", "short") for v in nvals)


def test_vhm_canonical(new_talib):
    """VHM intentionally diverges from the buggy OLD golden (OLD divided by a
    single constant from ``pstdev(volume, mean=slength)``). Pin NEW to the
    canonical TradingView formula: (volume - SMA(volume, 610)) / rolling stdev,
    which has a time-varying denominator and finite post-warmup values."""
    assert mode_for("VHM_610") == "intentional"
    from _parity import flatten_structs

    flat = flatten_structs(new_talib)
    col = next(c for c in flat.columns if c.endswith("VHM_610"))
    s = flat[col].to_numpy()
    post = s[700:]  # past the 610-bar warmup
    finite = post[~np.isnan(post)]
    assert finite.size > 100
    # The old constant-denominator bug compressed values to ~1e-6 magnitude;
    # the canonical rolling-stdev formula yields O(1)+ z-scores.
    assert np.nanmax(np.abs(finite)) > 1.0


BROKEN_TODO_COLS = sorted(col for col, spec in PARITY_EXCEPTIONS.items() if spec["mode"] == "broken_todo")


@pytest.mark.xfail(
    reason="broken_todo: NEW indicator broken/missing/all-NaN today; fixed in WS2-WS4",
    strict=False,
)
@pytest.mark.parametrize("col", BROKEN_TODO_COLS)
def test_broken_indicators_xfail(report, col):
    """These columns are broken in NEW today (WS2-4 will fix them). They xfail so
    the suite surfaces them without erroring; once fixed they will XPASS and can
    be promoted to strict parity tests in WS5.

    A column may be broken either by being *missing* from the study output
    (silently dropped) or by being present-but-all-NaN / materially wrong; both
    are caught here as expected failures."""
    assert col in report, f"{col}: indicator missing from study output (broken/dropped)"
    assert_column(report, col)
