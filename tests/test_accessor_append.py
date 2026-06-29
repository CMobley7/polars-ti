# -*- coding: utf-8 -*-
"""The df.ti accessor's ``append=`` keyword must hstack indicator columns onto
the original frame instead of leaking into the indicator function."""

import polars as pl
import pytest

import polars_ti as ti  # noqa: F401 — registers 'ti'


@pytest.fixture
def df():
    return pl.read_csv("data/SPY_D.csv", try_parse_dates=True).head(80)


@pytest.mark.parametrize(
    "name,kwargs,expected",
    [
        ("sma", {"length": 20}, ["SMA_20"]),
        ("rsi", {}, ["RSI_14"]),
        ("macd", {}, ["MACD"]),  # struct column
    ],
)
def test_append_true_keeps_original_columns(df, name, kwargs, expected):
    result = getattr(df.ti, name)(append=True, **kwargs)
    # original columns preserved
    for c in df.columns:
        assert c in result.columns
    # indicator output appended
    for c in expected:
        assert c in result.columns
    assert result.height == df.height


@pytest.mark.parametrize("name", ["sma", "rsi", "macd"])
def test_append_false_is_result_only(df, name):
    result = getattr(df.ti, name)()
    # default returns result-only (no base OHLC columns hstacked)
    assert "open" not in result.columns
    assert "close" not in result.columns
