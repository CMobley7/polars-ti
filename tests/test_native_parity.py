# -*- coding: utf-8 -*-
"""WS5 native-mode parity (talib=False) against the OLD notalib golden.

IMPORTANT: the OLD ``old_notalib`` golden is NOT a faithful native reference for
a sizeable set of indicators. OLD pervasively failed to propagate ``talib`` into
its internal MA/ATR/RSI/CMO helpers (so the OLD "native" golden actually holds
TA-Lib values), and OLD's own native RSI/ADX/CMO seeds were buggy (pinned to
TA-Lib via the ``match_talib`` exceptions). For every such column the NEW native
path is correct-by-design (Decision 4) and intentionally diverges from the OLD
native golden; these are listed in ``NATIVE_DIVERGENCE``.

``NATIVE_TODO`` holds genuine NEW native discrepancies whose OLD golden *is*
native (OLD propagated talib) and so remain to be root-caused; they xfail.

Everything else must match the OLD native golden within float tolerance — this
guards the ~290 genuinely-native columns against NEW native regressions and is
the test that the no-TA-Lib CI leg relies on.
"""

import warnings

import polars as pl
import pytest

import polars_ti as ti  # noqa: F401 — registers 'ti'
from _parity import assert_column, compare_frames
from parity_exceptions import mode_for

FIXTURES = "tests/fixtures"
SLICE_ROWS = 1500

# Genuine NEW native discrepancies whose OLD golden is a real native reference.
# (Empty — SMC, the last one, was an accessor arg-order bug now fixed.)
NATIVE_TODO: set[str] = set()

# Columns whose OLD native golden is TA-Lib-contaminated (OLD non-propagation)
# or depends on OLD's buggy native RSI/ADX/CMO seed. NEW native is correct by
# Decision 4 and intentionally diverges; not graded against the OLD golden.
NATIVE_DIVERGENCE = {
    "ADXR_14_2",
    "ALPHAT_14_1_50",
    "ALPHATl_14_1_50_2",
    "CFO_9",
    "CKSPl_10_3_20",
    "CKSPs_10_3_20",
    "CRSI_3_2_100",
    "DMP_14",
    "HT_atr_high_14_2_2",
    "HT_atr_low_14_2_2",
    "HT_direction_14_2_2",  # string column; graded by test_halftrend_direction_parity
    "INERTIA_20_14",
    "KCLe_20_2",
    "KCUe_20_2",
    "KVO_34_55_13",
    "KVOs_34_55_13",
    "NATR_14",  # classic b914429: native mamode default rma (Wilder); OLD golden baked in the buggy ema default
    "OTT_5_2.4",
    "OTTd_5_2.4",
    "OTTSL_5_2.4",
    "PGO_14",
    "PMAX_10_3.0",
    "PMAXl_10_3.0",
    "PMAXs_10_3.0",
    "PVOh_12_26_9",
    "PVOs_12_26_9",
    "QQE_14_5_4.236",
    "QQE_14_5_4.236_RSIMA",
    "QQEl_14_5_4.236",
    "QQEs_14_5_4.236",
    "RVI_14",
    "SMI_5_20_5_1.0",
    "SMIo_5_20_5_1.0",
    "SMIs_5_20_5_1.0",
    "SQZ_NO",
    "SQZ_OFF",
    "SQZPRO_NO",
    "SQZPRO_OFF",
    "SUPERT_7_3.0",
    "SUPERTl_7_3.0",
    "SUPERTs_7_3.0",
    "TRIMA_10",  # classic 41c91db: native TRIMA uses ceil/floor windows now
    "TRIX_18_9",
    "TRIXh_18_9",
    "TRIXs_18_9",
    "TRIXs_30_9",
    "VIDYA_14",
}


@pytest.fixture(scope="module")
def native_report():
    df = pl.read_csv("data/SPY_D.csv", try_parse_dates=True).head(SLICE_ROWS)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        new = df.ti.study(ti.AllStudy, cores=0, talib=False, errors="ignore")
    golden = pl.read_parquet(f"{FIXTURES}/old_notalib.parquet")
    return compare_frames(new, golden)


def _gradeable(report):
    """OLD-golden columns that should match NEW native within tolerance."""
    cols = []
    for k, v in report.items():
        if k.startswith("__"):
            continue
        if mode_for(k) in ("match_talib", "intentional"):
            continue
        if k in NATIVE_DIVERGENCE or k in NATIVE_TODO:
            continue
        cols.append(k)
    return sorted(cols)


def test_native_parity_has_broad_coverage(native_report):
    # Sanity: the clean native-gradeable set is large (most of the library).
    assert len(_gradeable(native_report)) > 250


def test_native_parity(native_report):
    """Every gradeable column matches the OLD native golden within tolerance."""
    failures = []
    for col in _gradeable(native_report):
        r = native_report[col]
        if r["cls"] in ("material", "no-overlap", "non-numeric"):
            failures.append((col, r["cls"], r["max_abs"]))
    assert not failures, f"native-mode parity regressions: {failures}"


@pytest.mark.xfail(reason="known NEW native discrepancy (NATIVE_TODO)", strict=False)
@pytest.mark.parametrize("col", sorted(NATIVE_TODO))
def test_native_todo(native_report, col):
    assert_column(native_report, col)
