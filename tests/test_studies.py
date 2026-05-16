# -*- coding: utf-8 -*-
"""test_studies.py — Polars-TI Study API tests.

Refactored from the original pandas-ti version to use the native Polars
DataFrame extension (``df.ti``).  All pandas idioms have been replaced with
their Polars equivalents:

* ``df.shape[1]``      → ``len(df.columns)``
* ``df.iloc[:n]``      → ``df[:n]``
* ``df["col"]``        → Polars Series (already works)
* ``DataFrame().ti``   → ``pl.DataFrame().ti``
"""
from multiprocessing import cpu_count

import pytest
import polars as pl

import polars_ti as ti


# ---------------------------------------------------------------------------
# Module-level parametrize data — mirrors the original test_studies.py
# ---------------------------------------------------------------------------

categories = pl.DataFrame().ti.categories() + [
    pytest.param(ti.CommonStudy, id="common"),
    pytest.param(ti.AllStudy, id="all"),
]

# +/- when adding/removing indicators
ALL_COLUMNS = 384


# ---------------------------------------------------------------------------
# Study property tests
# ---------------------------------------------------------------------------

def test_all_study_props(all_study):
    s = all_study
    assert s.name == "All"
    assert isinstance(s.description, str)
    assert s.total_ti() == 0  # Only 'study' that is None
    assert len(s.created) > 0
    assert s.cores == cpu_count()


def test_common_study_props(common_study):
    s = common_study
    assert s.name == "Common Price and Volume SMAs"
    assert isinstance(s.description, str)
    assert s.total_ti() == 5
    assert len(s.created) > 0
    assert s.cores == 0


# ---------------------------------------------------------------------------
# Study column-count tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "category,columns",
    [
        ("candles", 70),
        ("cycles", 3),
        ("momentum", 97),
        ("overlap", 75),
        ("performance", 5),
        ("statistics", 16),
        ("transform", 5),
        ("trend", 39),
        ("volatility", 46),
        ("volume", 30),
        pytest.param(ti.AllStudy, ALL_COLUMNS, id=f"all-{ALL_COLUMNS}"),
        pytest.param(ti.CommonStudy, 5, id="common-5"),
    ],
)
def test_study_category_columns(df, category, columns):
    initial_columns = len(df.columns)
    df.ti.study(category, cores=0)
    assert len(df.columns) == initial_columns + columns


@pytest.mark.parametrize("talib", [False, True])
@pytest.mark.parametrize("category", categories)
def test_study_category_talib(df, category, talib):
    initial_columns = len(df.columns)
    df.ti.study(category, cores=0, talib=talib)
    assert len(df.columns) > initial_columns


# ---------------------------------------------------------------------------
# Custom Study tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("talib", [False, True])
def test_study_custom_a_talib(df, custom_study_a, talib):
    initial_columns = len(df.columns)
    df.ti.study(custom_study_a, cores=0, talib=talib)
    assert len(df.columns) > initial_columns


@pytest.mark.parametrize("talib", [False, True])
def test_study_custom_b_talib(df, custom_study_b, talib):
    initial_columns = len(df.columns)
    df.ti.study(custom_study_b, cores=0, talib=talib)
    assert len(df.columns) - initial_columns == 3


@pytest.mark.parametrize("talib", [False, True])
def test_study_custom_c_talib(df, custom_study_c, talib):
    initial_columns = len(df.columns)
    df.ti.study(custom_study_c, cores=0, talib=talib)
    assert len(df.columns) - initial_columns == 5


@pytest.mark.parametrize("talib", [False, True])
def test_study_custom_d_talib(df, custom_study_d, talib):
    initial_columns = len(df.columns)
    df.ti.study(custom_study_d, cores=0, talib=talib)
    assert len(df.columns) - initial_columns == 3


@pytest.mark.parametrize("talib", [False, True])
def test_study_custom_e_talib(df, custom_study_e, talib):
    initial_columns = len(df.columns)
    df.ti.study(custom_study_e, cores=0, talib=talib)
    df.ti.tsignals(trend="AMATe_LR_20_50_2", append=True)
    assert len(df.columns) - initial_columns == 8


# ---------------------------------------------------------------------------
# All-study multi-run and incremental-row tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("talib", [False, True])
def test_study_all_multirun_talib(df, all_study, talib):
    new_columns = 708  # +/- when adding/removing indicators
    initial_columns = len(df.columns)
    df.ti.study(all_study, length=10, cores=0, talib=talib)
    df.ti.study(all_study, length=50, cores=0, talib=talib)
    df.ti.study(all_study, fast=5, slow=10, cores=0, talib=talib)
    assert len(df.columns) == new_columns + initial_columns


# Note: As expected, it will print a VWAP datetime ordered index warning
# when less than 2 rows
@pytest.mark.parametrize("talib", [False, True])
def test_study_all_incremental_rows_talib(df, all_study, talib):
    MAX_ROWS = 90
    df = df[:MAX_ROWS]  # Trim for this test

    for i in range(0, MAX_ROWS):
        _df = df[:i]
        _df.ti.study(all_study, cores=0, talib=talib)
        # Break when max columns reached
        if len(_df.columns) - len(df.columns) == ALL_COLUMNS:
            assert len(_df.columns) > len(df.columns)
            break


@pytest.mark.parametrize("talib", [False, True])
@pytest.mark.parametrize("category", categories)
def test_study_mp_category_talib(df, category, talib):
    cores = cpu_count() - 2
    initial_columns = len(df.columns)
    df.ti.study(category, cores=cores, talib=talib)
    assert len(df.columns) > initial_columns
