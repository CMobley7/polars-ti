# -*- coding: utf-8 -*-
"""Regression tests for four accessor/study machinery fixes in ``core.py``.

A. ``study(errors="raise")`` must not leak accumulated columns onto the cached
   accessor when an indicator raises.
B. ``study()`` must not forward study-wide kwargs to indicators that cannot
   accept them (strict native signatures), silently dropping ~18 indicators.
C. ``reverse(append=True)`` must return the reversed frame, not the original.
D. ``col_names`` / ``col_numbers`` must rename/select produced output columns,
   in both direct calls and Study specs.
"""

import warnings

import polars as pl
import pytest

import polars_ti as ti  # noqa: F401 — registers the 'ti' namespace


@pytest.fixture
def df():
    return pl.read_csv("data/SPY_D.csv", try_parse_dates=True).head(150)


# --------------------------------------------------------------------------- A
def test_raised_study_leaves_accessor_state_clean(df):
    """A raised study must restore the accessor's working frame to the input."""
    acc = df.ti
    original_cols = list(acc._df.columns)
    broken = ti.Study(
        name="Broken",
        ti=[{"kind": "sma", "length": 2}, {"kind": "sma", "length": "not-an-int"}],
    )
    with pytest.raises(Exception):
        acc.study(broken, errors="raise")

    # The partially-accumulated SMA column must NOT persist on the accessor.
    assert list(acc._df.columns) == original_cols

    # A subsequent study on the SAME accessor must recompute correctly.
    out = acc.study(ti.Study(name="Good", ti=[{"kind": "sma", "length": 5}]), errors="warn")
    assert "SMA_5" in out.columns


# --------------------------------------------------------------------------- B
def test_study_category_kwarg_drops_zero(df):
    """``study('momentum', length=10)`` must compute every momentum indicator.

    Indicators whose native signature has no ``length`` param must silently
    absorb the study-wide kwarg (it is filtered out before the call) rather than
    raising TypeError and being dropped under the default ``errors='warn'``.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any "N failed" warning would raise
        out = df.ti.study("momentum", length=10)
    assert isinstance(out, pl.DataFrame)
    # length=10 must still be applied where accepted (e.g. RSI_10, not RSI_14).
    assert any(c.startswith("RSI_10") for c in out.columns)


def test_study_kwarg_filtering_does_not_regress_state(df):
    """Fix B must not reintroduce the fix-A state leak: a fresh vs sequential
    study on the same accessor still match, and the accessor frame is restored."""
    study = ti.Study(name="K", ti=[{"kind": "kama", "length": 10}])
    acc = df.ti
    original_cols = list(acc._df.columns)
    acc.study(study, talib=True)
    sequential = acc.study(study, talib=False)
    fresh = df.ti.study(study, talib=False)

    kama_col = next(c for c in sequential.columns if c.startswith("KAMA"))
    max_diff = float((sequential[kama_col].fill_null(0.0) - fresh[kama_col].fill_null(0.0)).abs().max())
    assert max_diff < 1e-9
    assert list(acc._df.columns) == original_cols


# --------------------------------------------------------------------------- C
def test_reverse_append_true_is_reversed(df):
    """``reverse(append=True)`` must equal ``reverse()`` and actually reverse."""
    plain = df.ti.reverse()
    appended = df.ti.reverse(append=True)
    assert appended.equals(plain)
    assert appended["close"][0] == df["close"][-1]
    assert appended["close"][-1] == df["close"][0]


# --------------------------------------------------------------------------- D
def test_col_names_renames_produced_columns(df):
    """``col_names`` renames the produced (unnested) output columns positionally."""
    names = ("LB", "MB", "UB", "BW", "BP")
    result = df.ti.bbands(length=5, col_names=names)
    assert result.columns == list(names)


def test_col_names_default_unchanged(df):
    """Without col_names the struct output is unchanged."""
    assert df.ti.bbands(length=5).columns == ["BBANDS_5_2.0"]


def test_col_names_composes_with_append_and_prefix(df):
    names = ("LB", "MB", "UB", "BW", "BP")
    appended = df.ti.bbands(length=5, col_names=names, append=True)
    for c in df.columns:
        assert c in appended.columns
    for c in names:
        assert c in appended.columns

    prefixed = df.ti.bbands(length=5, col_names=names, prefix="X", suffix="Y")
    assert prefixed.columns == [f"X_{n}_Y" for n in names]


def test_col_names_too_few_raises(df):
    with pytest.raises(ValueError, match="Not enough col_names"):
        df.ti.bbands(length=5, col_names=("a", "b"))


def test_col_numbers_selects_produced_columns(df):
    result = df.ti.bbands(length=5, col_numbers=(0, 2))
    assert result.width == 2
    assert result.columns == ["BBL_5_2.0", "BBU_5_2.0"]


def test_col_numbers_out_of_range_raises(df):
    with pytest.raises(ValueError, match="out of range"):
        df.ti.bbands(length=5, col_numbers=(0, 99))


def test_col_names_in_study_spec(df):
    names = ("LB", "MB", "UB", "BW", "BP")
    study = ti.Study(name="BB", ti=[{"kind": "bbands", "length": 5, "col_names": names}])
    out = df.ti.study(study)
    for c in names:
        assert c in out.columns


# --------------------------------------------------------------------------- E
# Fix 1: append / study must let a FRESH result overwrite an existing
# output-column name instead of silently keeping the stale value.
@pytest.fixture
def ohlc():
    return pl.DataFrame(
        {
            "open": [10.0, 20, 30, 40, 50, 60],
            "high": [11.0, 21, 31, 41, 51, 61],
            "low": [9.0, 19, 29, 39, 49, 59],
            "close": [1.0, 2, 3, 4, 5, 6],
            "volume": [100.0, 200, 300, 400, 500, 600],
        }
    )


def test_append_recompute_overwrites_stale_column(ohlc):
    """A second append that recomputes the same column name must win.

    ``sma(length=3, close="close")`` then ``sma(length=3, close="open")`` both
    emit ``SMA_3``; the second (open-based) value must overwrite the first.
    """
    d1 = ohlc.ti.sma(length=3, close="close", append=True)
    assert d1["SMA_3"][-1] == pytest.approx(5.0)  # close-based: mean(4,5,6)

    d2 = d1.ti.sma(length=3, close="open", append=True)
    assert d2["SMA_3"][-1] == pytest.approx(50.0)  # open-based: mean(40,50,60)
    # Overwrite happens in place: no duplicate column, position preserved.
    assert d2.columns.count("SMA_3") == 1
    assert d2.columns == ["open", "high", "low", "close", "volume", "SMA_3"]


def test_study_last_spec_wins_on_column_collision(ohlc):
    """Two study specs producing the same column name keep the LAST value."""
    study = ti.Study(
        name="Collide",
        ti=[
            {"kind": "sma", "length": 3, "close": "close"},
            {"kind": "sma", "length": 3, "close": "open"},
        ],
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = ohlc.ti.study(study)
    assert out.columns.count("SMA_3") == 1
    assert out["SMA_3"][-1] == pytest.approx(50.0)  # last spec (open) wins


# --------------------------------------------------------------------------- F
# Fix 2: a study-wide ``open=`` must reach open_-param indicators (candles,
# bop, ohlc4, ...) whose native signature renames the builtin to ``open_``.
def test_study_open_kwarg_resolves_for_candles():
    """``study('candles', open=...)`` must drop zero indicators (incl. cdl_pattern)."""
    import numpy as np

    rng = np.random.default_rng(0)
    base = 100 + np.cumsum(rng.standard_normal(60))
    cdf = pl.DataFrame(
        {
            "OpenPrice": base,
            "High": base + np.abs(rng.standard_normal(60)),
            "Low": base - np.abs(rng.standard_normal(60)),
            "Close": base + rng.standard_normal(60),
        }
    )
    # errors="raise" would surface the historical KeyError/ColumnNotFound.
    out = cdf.ti.study("candles", open="OpenPrice", high="High", low="Low", close="Close", errors="raise")
    assert isinstance(out, pl.DataFrame)
    # Every candle column is produced (0 dropped) and resolves the aliased open.
    assert any(c.startswith("CDL_") for c in out.columns)
    assert out.width - cdf.width >= 60


def test_study_open_alias_does_not_regress_kwarg_filtering(df):
    """Fix 2 must not widen the filter for genuinely-unaccepted kwargs.

    ``study('momentum', length=10)`` still drops zero indicators (fix B)."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = df.ti.study("momentum", length=10)
    assert isinstance(out, pl.DataFrame)
