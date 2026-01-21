# -*- coding: utf-8 -*-
from pandas import Series

import polars_ti as ti


# TA Lib style Tests
def test_ebsw(df):
    result = ti.ebsw(df.close)
    assert isinstance(result, Series)
    assert result.name == "EBSW_40_10"


def test_reflex(df):
    result = ti.reflex(df.close)
    assert isinstance(result, Series)
    assert result.name == "REFLEX_20_20_0.04"


# DataFrame Extension Tests
def test_ext_ebsw(df):
    df.ti.ebsw(append=True)
    assert df.columns[-1] == "EBSW_40_10"


def test_ext_reflex(df):
    df.ti.reflex(append=True)
    assert df.columns[-1] == "REFLEX_20_20_0.04"
