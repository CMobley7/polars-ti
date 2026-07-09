# -*- coding: utf-8 -*-
"""WS5 per-indicator parametrized smoke tests.

For every indicator registered in ``maps.Category`` (176 of them), assert that:
  * it runs without raising (with its default arguments),
  * it produces at least one non-empty, not-entirely-NaN output column,
  * it is deterministic (same input -> same output).

Both-modes (talib on/off) coverage is provided study-wide by
``test_study_completeness`` and the parity suites; here we exercise each
indicator individually with its defaults.

A few composition indicators operate on OTHER indicators' output columns rather
than raw OHLCV (``long_run``/``short_run`` need fast/slow MAs, ``tsignals`` a
``trend`` column, ``xsignals`` a ``signal`` column); the input frame is augmented
with those columns so they can be exercised too.
"""

import warnings

import numpy as np
import polars as pl
import pytest

import polars_ti as ti  # noqa: F401 — registers 'ti'
from _parity import flatten_structs
from polars_ti.maps import Category

ALL_INDICATORS = sorted({name for cat in Category.values() for name in cat})
BASE = {
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "dividends",
    "stock splits",
    # synthetic input columns added by the df fixture for composition indicators
    "fast",
    "slow",
    "trend",
    "signal",
    "above",
    "below",
    "benchmark",
    "periods",
}


@pytest.fixture(scope="module")
def df():
    # 700 rows so even the longest default lookback (VHM, length=610) warms up.
    base = pl.read_csv("data/SPY_D.csv", try_parse_dates=True).head(700)
    # Derived columns needed by composition indicators (long_run/short_run need
    # fast/slow MAs; tsignals a 'trend'; xsignals 'signal'/'above'/'below';
    # beta/correl a 'benchmark' series; mavp a per-bar 'periods' series).
    return base.with_columns(
        pl.col("close").rolling_mean(10).alias("fast"),
        pl.col("close").rolling_mean(20).alias("slow"),
        (pl.col("close") > pl.col("close").rolling_mean(20)).cast(pl.Int64).alias("trend"),
        pl.col("close").alias("signal"),
        pl.lit(70.0).alias("above"),
        pl.lit(30.0).alias("below"),
        pl.col("open").alias("benchmark"),
        pl.lit(14.0).alias("periods"),
    )


def _run(df, name):
    fn = getattr(df.ti, name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn()


def _coerce(df, result):
    if isinstance(result, pl.DataFrame):
        return result
    return df.select(result)


@pytest.mark.parametrize("name", ALL_INDICATORS)
def test_indicator_runs_and_has_output(df, name):
    """Indicator runs without raising and yields a non-all-NaN output column."""
    flat = flatten_structs(_coerce(df, _run(df, name)))
    cols = [c for c in flat.columns if c not in BASE]
    assert cols, f"{name} produced no indicator columns"
    # At least one output column must have a finite/non-null value.
    any_finite = False
    for c in cols:
        s = flat[c]
        if s.dtype.is_numeric():
            if bool(np.any(np.isfinite(s.to_numpy().astype(float)))):
                any_finite = True
                break
        elif s.null_count() < s.len():
            any_finite = True
            break
    assert any_finite, f"{name} output is entirely NaN/null"


@pytest.mark.parametrize("name", ALL_INDICATORS)
def test_indicator_deterministic(df, name):
    """Same input -> identical output across two runs."""
    r1 = flatten_structs(_coerce(df, _run(df, name)))
    r2 = flatten_structs(_coerce(df, _run(df, name)))
    assert r1.columns == r2.columns, f"{name}: column set not deterministic"
    for c in r1.columns:
        a, b = r1[c], r2[c]
        if a.dtype.is_numeric():
            assert np.array_equal(a.to_numpy().astype(float), b.to_numpy().astype(float), equal_nan=True), (
                f"{name}: column {c} not deterministic"
            )
        else:
            assert a.to_list() == b.to_list(), f"{name}: column {c} not deterministic"
