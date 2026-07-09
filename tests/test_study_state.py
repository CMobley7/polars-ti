# -*- coding: utf-8 -*-
"""Regression: study() must not leak one run's output columns into the next.

Previously study() permanently mutated ``self._df`` by hstacking its output,
and _run skipped recomputing any column already present. Running two studies on
the SAME accessor therefore returned stale results for the second run (it kept
the first run's values). This pins the fix that restores the accessor's working
frame to the original input after each study.
"""

import polars as pl

import polars_ti as ti  # noqa: F401 — registers the 'ti' namespace


def _study():
    return ti.Study(name="Kama", ti=[{"kind": "kama", "length": 10}])


def _df():
    return pl.read_csv("data/SPY_D.csv", try_parse_dates=True).head(300)


def test_sequential_studies_recompute_from_original():
    """talib=True then talib=False on one accessor: the second matches a fresh
    native study rather than reusing the first (talib) run's values."""
    study = _study()

    acc = _df().ti
    acc.study(study, talib=True)
    sequential = acc.study(study, talib=False)

    fresh = _df().ti.study(study, talib=False)

    kama_col = next(c for c in sequential.columns if c.startswith("KAMA"))
    assert kama_col in fresh.columns

    seq_vals = sequential[kama_col].fill_null(0.0)
    fresh_vals = fresh[kama_col].fill_null(0.0)
    max_diff = float((seq_vals - fresh_vals).abs().max())
    assert max_diff < 1e-9


def test_study_does_not_permanently_mutate_accessor_frame():
    """study() returns the accumulated frame but leaves the accessor's own
    working frame at the original input columns."""
    acc = _df().ti
    original_cols = list(acc._df.columns)
    out = acc.study(_study())
    assert any(c.startswith("KAMA") for c in out.columns)
    assert list(acc._df.columns) == original_cols
