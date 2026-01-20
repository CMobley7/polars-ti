# -*- coding: utf-8 -*-
import pandas.testing as pdt
import talib as tal
from pandas import DataFrame, Series

import polars_ti as ti

from .config import CORRELATION, CORRELATION_THRESHOLD, error_analysis


# TA Lib style Tests
def test_cdl_doji(df):
    result = ti.cdl_doji(df.open, df.high, df.low, df.close)
    assert isinstance(result, Series)
    assert result.name == "CDL_DOJI_10_0.1"

    try:
        expected = tal.CDLDOJI(df.open, df.high, df.low, df.close)
        pdt.assert_series_equal(result, expected, check_names=False, check_dtype=False)
    except AssertionError:
        try:
            corr = ti.utils.df_error_analysis(result, expected)
            print(f"{corr=}")
            assert corr > CORRELATION_THRESHOLD
        except Exception as ex:
            error_analysis(result, CORRELATION, ex)


def test_cdl_inside(df):
    result = ti.cdl_inside(df.open, df.high, df.low, df.close)
    assert isinstance(result, Series)
    assert result.name == "CDL_INSIDE"

    result = ti.cdl_inside(df.open, df.high, df.low, df.close, asbool=True)
    assert isinstance(result, Series)
    assert result.name == "CDL_INSIDE"


def test_cdl_pattern(df):
    result = ti.cdl_pattern(df.open, df.high, df.low, df.close, name="all")
    assert isinstance(result, DataFrame)
    assert len(result.columns) == len(ti.CDL_PATTERN_NAMES)

    result = ti.cdl_pattern(df.open, df.high, df.low, df.close, name="doji")
    assert isinstance(result, DataFrame)

    result = ti.cdl_pattern(df.open, df.high, df.low, df.close, name=["doji", "inside"])
    assert isinstance(result, DataFrame)


def test_cdl_z(df):
    result = ti.cdl_z(df.open, df.high, df.low, df.close)
    assert isinstance(result, DataFrame)
    assert result.name == "CDL_Z_30_1"


def test_ha(df):
    result = ti.ha(df.open, df.high, df.low, df.close)
    assert isinstance(result, DataFrame)
    assert result.name == "Heikin-Ashi"


# DataFrame Extension Tests
def test_ext_cdl_doji(df):
    df.ti.cdl_pattern("doji", append=True)
    assert df.columns[-1] == "CDL_DOJI_10_0.1"


def test_ext_cdl_inside(df):
    df.ti.cdl_pattern("inside", append=True)
    assert df.columns[-1] == "CDL_INSIDE"


def test_ext_cdl_z(df):
    df.ti.cdl_z(append=True)
    assert list(df.columns[-4:]) == [
        "open_Z_30_1",
        "high_Z_30_1",
        "low_Z_30_1",
        "close_Z_30_1",
    ]


def test_ext_ha(df):
    df.ti.ha(append=True)
    assert list(df.columns[-4:]) == ["HA_open", "HA_high", "HA_low", "HA_close"]
