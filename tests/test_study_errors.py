# -*- coding: utf-8 -*-
"""Smoke tests for the study() ``errors`` policy (WS0 #4 / Decision 3).

Verifies the three modes:
  * ``ignore``  -> completes silently, no warning, even with a broken indicator.
  * ``warn``    -> completes and emits a RuntimeWarning naming the broken one.
  * ``raise``   -> re-raises the first indicator failure.

A custom Study with a deliberately-broken indicator spec (a kwarg the indicator
cannot accept and the talib-retry cannot rescue) is used to force a failure
without depending on which real indicators happen to be broken today.
"""

import warnings

import polars as pl
import pytest

import polars_ti as ti  # noqa: F401 — registers the 'ti' namespace


@pytest.fixture
def df():
    return pl.read_csv("data/SPY_D.csv", try_parse_dates=True).head(200)


def _broken_study():
    """A Study whose single indicator fails: ``sma`` does not accept a
    ``length="not-an-int"`` window and the talib-retry cannot fix it."""
    return ti.Study(
        name="DeliberatelyBroken",
        ti=[{"kind": "sma", "length": "not-an-int"}],
        description="forces an indicator failure for error-policy tests",
    )


def test_errors_ignore_is_silent(df):
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning would raise
        out = df.ti.study(_broken_study(), errors="ignore")
    assert isinstance(out, pl.DataFrame)
    # The broken indicator produced no column; study still completed.
    assert "SMA_10" not in out.columns


def test_errors_warn_completes_and_warns(df):
    with pytest.warns(RuntimeWarning, match="sma"):
        out = df.ti.study(_broken_study(), errors="warn")
    assert isinstance(out, pl.DataFrame)


def test_errors_warn_is_default(df):
    # Default (no errors= passed) behaves like "warn".
    with pytest.warns(RuntimeWarning):
        df.ti.study(_broken_study())


def test_errors_raise_propagates(df):
    with pytest.raises(Exception):
        df.ti.study(_broken_study(), errors="raise")


def test_invalid_errors_value(df):
    with pytest.raises(ValueError, match="errors must be one of"):
        df.ti.study(_broken_study(), errors="bogus")


def test_clean_study_does_not_warn(df):
    # A study with only a known-good indicator should not warn in warn mode.
    good = ti.Study(name="Good", ti=[{"kind": "sma", "length": 10}])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = df.ti.study(good, errors="warn")
    assert "SMA_10" in out.columns
