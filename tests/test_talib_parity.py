# -*- coding: utf-8 -*-
"""WS5 full TA-Lib-mode parity against the pandas (old_talib) golden.

This is the *complete* per-column gate that `test_parity_smoke.py` (a sanity
subset) is not: it asserts that EVERY shared column of the all-study output in
TA-Lib mode matches the pandas golden within float tolerance, except for a small,
explicitly documented set of exceptions. This is what makes the
"differences-from-pandas-ta" doc's parity claims verifiable and prevents an
undocumented divergence from silently creeping in.

Exceptions (everything else must match):
  * ``match_talib`` columns — pinned to the TA-Lib reference because pandas-ta
    was wrong (graded by test_parity_smoke / the TA-Lib reference), so they do
    NOT match the pandas golden.
  * ``intentional`` columns — deliberate convention divergences (ddof, seeds, …).
  * ``TALIB_DIVERGENCE`` — columns whose TA-Lib-mode value legitimately differs
    from the (native-based) pandas golden, documented below.
"""

import warnings

import polars as pl
import pytest

import polars_ti as ti  # noqa: F401 — registers 'ti'
from _parity import assert_column, compare_frames
from parity_exceptions import mode_for
from polars_ti.maps import Imports

# This gate grades the TA-Lib-mode study against the TA-Lib (old_talib) golden,
# so it is only meaningful when TA-Lib is installed. In a no-TA-Lib environment
# the study falls back to native output, which must NOT be compared to the
# TA-Lib golden — skip the whole module (the native path is covered by
# tests/test_native_parity.py).
pytestmark = pytest.mark.skipif(not Imports["talib"], reason="requires TA-Lib (grades vs the TA-Lib golden)")

FIXTURES = "tests/fixtures"
SLICE_ROWS = 1500

# Columns whose TA-Lib-mode value legitimately diverges from the pandas golden.
TALIB_DIVERGENCE = {
    # OLD had no TA-Lib KAMA path, so its golden is native KAMA in both modes.
    # NEW's native path matches the golden (enforced by test_native_parity);
    # the talib path follows TA-Lib's KAMA (native = pandas-ta, talib = TA-Lib).
    "KAMA_10_2_30": "OLD had no TA-Lib KAMA; NEW talib path follows TA-Lib (native matches golden)",
    # String label column — graded by test_halftrend_direction_parity.
    "HT_direction_14_2_2": "string label column; graded by test_halftrend_direction_parity",
    # classic b914429: NATR default mamode is now rma (Wilder, == ATR/TA-Lib).
    # OLD baked in the buggy ema default in BOTH goldens; NEW talib-mode NATR
    # matches talib.NATR exactly (pinned by tests/volatility/test_natr_polars.py).
    "NATR_14": "classic b914429: NATR mamode default rma; NEW talib matches talib.NATR (OLD golden used ema)",
    # classic 1474768 (downstream of the VIDYA SMA-seed fix): OTT defaults to
    # mamode="vidya", and its OTTSL field IS vidya(length=5) exactly. The OLD
    # goldens baked in the buggy zero-seed VIDYA, so OTT/OTTSL/OTTd diverge in
    # BOTH modes. OTTSL == NEW vidya (validated vs the classic fork); OTT/OTTd
    # are deterministic transforms of it.
    "OTT_5_2.4": "classic 1474768: OTTSL == seed-fixed vidya(5); OLD golden used zero-seed vidya",
    "OTTSL_5_2.4": "classic 1474768: OTTSL == seed-fixed vidya(5); OLD golden used zero-seed vidya",
    "OTTd_5_2.4": "classic 1474768: OTT direction derived from seed-fixed vidya; OLD golden used zero-seed vidya",
}


@pytest.fixture(scope="module")
def report():
    df = pl.read_csv("data/SPY_D.csv", try_parse_dates=True).head(SLICE_ROWS)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        new = df.ti.study(ti.AllStudy, cores=0, talib=True, errors="ignore")
    golden = pl.read_parquet(f"{FIXTURES}/old_talib.parquet")
    return compare_frames(new, golden)


def _graded_cols(report):
    cols = []
    for k in report:
        if k.startswith("__"):
            continue
        if mode_for(k) in ("match_talib", "intentional"):
            continue
        if k in TALIB_DIVERGENCE:
            continue
        cols.append(k)
    return sorted(cols)


def test_full_talib_parity_has_broad_coverage(report):
    # The graded set is the large majority of the all-study output. (DMP_14,
    # DMN_14 and CMO_14 are pinned to the TA-Lib reference via ``match_talib``
    # for the classic-port native Wilder-smoothing alignment, so they are graded
    # against talib_reference rather than the pandas golden here.)
    #
    # The EMA-seed alignment reclassified DEMA_10 (match_talib) plus the
    # ZL_EMA/TSI/TMO/EFI/KVO family (intentional) out of pandas-golden grading:
    # their native output now matches TA-Lib and is pinned native==talib by
    # tests/overlap/test_native_talib_alignment.py, so the floor is 340.
    assert len(_graded_cols(report)) >= 340


def test_full_talib_parity(report):
    """Every shared column (minus documented exceptions) matches the pandas
    golden in TA-Lib mode within float tolerance."""
    failures = []
    for col in _graded_cols(report):
        try:
            assert_column(report, col)
        except AssertionError as exc:
            failures.append(str(exc))
    assert not failures, "TA-Lib-mode parity regressions:\n" + "\n".join(failures)


def test_no_undocumented_old_only_columns(report):
    # Completeness: every pandas column folds onto a NEW column.
    assert report["__old_only__"] == [], f"unmapped OLD columns: {report['__old_only__']}"
