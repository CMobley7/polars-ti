# -*- coding: utf-8 -*-
import numpy as np
import pandas.testing as pdt
import talib as tal
from pandas import DataFrame, Series, read_csv

import polars_ti as ti

from .config import CORRELATION, CORRELATION_THRESHOLD, error_analysis

sample_adx_data = read_csv(
    f"data/ADX_D.csv",
    index_col=0,
    parse_dates=True,
    # date_format="%f"
    date_format="%m/%d/%Y %I:%M:%S `%p",
)

expected_tv_adx = DataFrame(
    {
        "ADX_14": [
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            9.874338,
            10.408195,
            10.799274,
        ],
        "DMP_14": [
            None,
            13.686598,
            14.247809,
            13.436449,
            17.946530,
            17.193874,
            19.214901,
            17.860325,
            16.899406,
            16.207983,
            15.998908,
            15.202702,
            14.621306,
            14.303707,
            13.451093,
            12.932243,
            12.840198,
        ],
        "DMN_14": [
            None,
            21.954010,
            21.055379,
            23.102292,
            21.120552,
            20.234781,
            18.827331,
            19.298744,
            19.029033,
            18.250478,
            17.414460,
            16.772558,
            16.131126,
            15.780731,
            19.097781,
            18.361121,
            17.689287,
        ],
    }
)


# TA Lib style Tests
def test_adx(df):
    result = ti.adx(df.high, df.low, df.close, talib=False)
    assert isinstance(result, DataFrame)
    assert result.name == "ADX_14"
    assert len(result.columns) == 4

    try:
        expected = tal.ADX(df.high, df.low, df.close)
        pdt.assert_series_equal(result.iloc[0], expected, check_names=False)
    except AssertionError:
        try:
            corr = ti.utils.df_error_analysis(result, expected)
            print(f"{corr=}")
            assert corr > CORRELATION_THRESHOLD
        except Exception as ex:
            error_analysis(result, CORRELATION, ex)

    result = ti.adx(df.high, df.low, df.close)
    assert isinstance(result, DataFrame)
    assert result.name == "ADX_14"

    # TV Test
    df = sample_adx_data
    result = ti.adx(df.high, df.low, df.close, tvmode=True)
    assert isinstance(result, DataFrame)
    assert result.name == "ADX_14"

    result = result.iloc[13:]
    result = result.drop(result.columns[1], axis=1)
    result = result.reset_index(drop=True)
    pdt.assert_frame_equal(result, expected_tv_adx)


def test_alphatrend(df):
    result = ti.alphatrend(df.open, df.high, df.low, df.close)
    assert isinstance(result, DataFrame)
    assert result.name == "ALPHAT_14_1_50"


def test_amat(df):
    result = ti.amat(df.close)
    assert isinstance(result, DataFrame)
    assert result.name == "AMATe_8_21_2"


def test_aroon(df):
    result = ti.aroon(df.high, df.low, talib=False)
    assert isinstance(result, DataFrame)
    assert result.name == "AROON_14"

    try:
        expected = tal.AROON(df.high, df.low)
        expecteddf = DataFrame({"AROOND_14": expected[0], "AROONU_14": expected[1]})
        pdt.assert_frame_equal(result, expecteddf)
    except AssertionError:
        try:
            aroond_corr = ti.utils.df_error_analysis(
                result.iloc[:, 0], expected.iloc[:, 0]
            )
            print(f"{aroond_corr=}")
            assert aroond_corr > CORRELATION_THRESHOLD
        except Exception as ex:
            error_analysis(result.iloc[:, 0], CORRELATION, ex)

        try:
            aroonu_corr = ti.utils.df_error_analysis(
                result.iloc[:, 1], expected.iloc[:, 1]
            )
            print(f"{aroonu_corr=}")
            assert aroonu_corr > CORRELATION_THRESHOLD
        except Exception as ex:
            error_analysis(result.iloc[:, 1], CORRELATION, ex)

    result = ti.aroon(df.high, df.low)
    assert isinstance(result, DataFrame)
    assert result.name == "AROON_14"


def test_aroon_osc(df):
    result = ti.aroon(df.high, df.low, talib=False)
    assert isinstance(result, DataFrame)
    assert result.name == "AROON_14"

    try:
        expected = tal.AROONOSC(df.high, df.low)
        pdt.assert_series_equal(result.iloc[:, 2], expected, check_names=False)
    except AssertionError:
        try:
            corr = ti.utils.df_error_analysis(result, expected.iloc[:, 2])
            print(f"{corr=}")
            assert corr > CORRELATION_THRESHOLD
        except Exception as ex:
            error_analysis(result, CORRELATION, ex)


def test_chop(df):
    result = ti.chop(df.high, df.low, df.close, ln=False)
    assert isinstance(result, Series)
    assert result.name == "CHOP_14_1_100.0"

    result = ti.chop(df.high, df.low, df.close, ln=True)
    assert isinstance(result, Series)
    assert result.name == "CHOPln_14_1_100.0"


def test_cksp(df):
    result = ti.cksp(df.high, df.low, df.close, tvmode=False)
    assert isinstance(result, DataFrame)
    assert result.name == "CKSP_10_3_20"

    result = ti.cksp(df.high, df.low, df.close, tvmode=True)
    assert isinstance(result, DataFrame)
    assert result.name == "CKSP_10_1_9"


def test_decay(df):
    result = ti.decay(df.close)
    assert isinstance(result, Series)
    assert result.name == "LDECAY_1"

    result = ti.decay(df.close, mode="exp")
    assert isinstance(result, Series)
    assert result.name == "EXPDECAY_1"

    tulip = Series([0, 0, 0, 1, 0, 0, 0, 1, 0, 0])
    expected = Series([0, 0, 0, 1, 0.75, 0.5, 0.25, 1, 0.75, 0.5])
    result = ti.decay(tulip, length=4, mode="linear")
    assert isinstance(result, Series)
    assert result.name == "LDECAY_4"
    pdt.assert_series_equal(result, expected, check_names=False)


def test_decreasing(df):
    result = ti.decreasing(df.close)
    assert isinstance(result, Series)
    assert result.name == "DEC_1"

    result = ti.decreasing(df.close, length=3, strict=True)
    assert isinstance(result, Series)
    assert result.name == "SDEC_3"


def test_dpo(df):
    result = ti.dpo(df.close)
    assert isinstance(result, Series)
    assert result.name == "DPO_20"


def test_ht_trendline(df):
    result = ti.ht_trendline(df.close, talib=False)
    assert isinstance(result, Series)
    assert result.name == "HT_TL"

    try:
        expected = tal.HT_TRENDLINE(df.close)
        corr = ti.utils.df_error_analysis(result, expected)
        pdt.assert_series_equal(result, expected, check_names=False)
    except AssertionError:
        try:
            corr = ti.utils.df_error_analysis(result, expected)
            print(f"{corr=}")
            assert corr > CORRELATION_THRESHOLD
        except Exception as ex:
            error_analysis(result, CORRELATION, ex)

    result = ti.ht_trendline(df.close)
    assert isinstance(result, Series)
    assert result.name == "HT_TL"


def test_increasing(df):
    result = ti.increasing(df.close)
    assert isinstance(result, Series)
    assert result.name == "INC_1"

    result = ti.increasing(df.close, length=3, strict=True)
    assert isinstance(result, Series)
    assert result.name == "SINC_3"


def test_long_run(df):
    result = ti.long_run(df.open, df.close)
    assert isinstance(result, Series)
    assert result.name == "LR_2"


def test_psar(df):
    result = ti.psar(df.high, df.low, talib=False)
    assert isinstance(result, DataFrame)
    assert result.name == "PSAR_0.02_0.2"

    # Combine Long and Short SAR"s into one SAR value
    psar = result[result.columns[:2]].fillna(0)
    psar = psar[psar.columns[0]] + psar[psar.columns[1]]
    psar.iloc[0] = np.nan
    psar.name = result.name

    try:
        expected = tal.SAR(df.high, df.low)
        pdt.assert_series_equal(psar, expected, check_names=False)
    except AssertionError:
        try:
            corr = ti.utils.df_error_analysis(psar, expected)
            print(f"{corr=}")
            assert corr > CORRELATION_THRESHOLD
        except Exception as ex:
            error_analysis(result, CORRELATION, ex)


def test_qstick(df):
    result = ti.qstick(df.open, df.close)
    assert isinstance(result, Series)
    assert result.name == "QS_10"


def test_rwi(df):
    result = ti.rwi(df.high, df.low, df.close)
    assert isinstance(result, DataFrame)
    assert result.name == "RWI_14"


def test_short_run(df):
    result = ti.short_run(df.close, df.open)
    assert isinstance(result, Series)
    assert result.name == "SR_2"


def test_trendflex(df):
    result = ti.trendflex(df.close)
    assert isinstance(result, Series)
    assert result.name == "TRENDFLEX_20_20_0.04"


def test_vhf(df):
    result = ti.vhf(df.high, df.low, df.close)
    assert isinstance(result, Series)
    assert result.name == "VHF_28"


def test_vortex(df):
    result = ti.vortex(df.high, df.low, df.close)
    assert isinstance(result, DataFrame)
    assert result.name == "VTX_14"


def test_zigzag(df):
    result = ti.zigzag(df.high, df.low)
    assert isinstance(result, DataFrame)
    assert result.name == "ZIGZAG_5.0%_10"

    result = ti.zigzag(df.high, df.low, df.close)
    assert isinstance(result, DataFrame)
    assert result.name == "ZIGZAG_5.0%_10"

    notna = result.iloc[:, 0].notna()
    high_pivotsdf = result[notna & (result["ZIGZAGs_5.0%_10"] == 1)]
    assert isinstance(high_pivotsdf, DataFrame)
    assert high_pivotsdf.shape[0] == 1

    low_pivotsdf = result[notna & (result["ZIGZAGs_5.0%_10"] == -1)]
    assert isinstance(low_pivotsdf, DataFrame)
    assert low_pivotsdf.shape[0] == 1

    all_pivotsdf = result[notna]
    assert isinstance(all_pivotsdf, DataFrame)
    assert all_pivotsdf.shape[0] == low_pivotsdf.shape[0] + high_pivotsdf.shape[0]


# DataFrame Extension Tests
def test_ext_adx(df):
    df.ti.adx(append=True)
    assert list(df.columns[-4:]) == ["ADX_14", "ADXR_14_2", "DMP_14", "DMN_14"]


def test_ext_alphatrend(df):
    df.ti.alphatrend(append=True)
    assert list(df.columns[-2:]) == ["ALPHAT_14_1_50", "ALPHATl_14_1_50_2"]


def test_ext_amat(df):
    df.ti.amat(append=True)
    assert list(df.columns[-2:]) == ["AMATe_LR_8_21_2", "AMATe_SR_8_21_2"]


def test_ext_aroon(df):
    df.ti.aroon(append=True)
    assert list(df.columns[-3:]) == ["AROOND_14", "AROONU_14", "AROONOSC_14"]


def test_ext_chop(df):
    df.ti.chop(ln=False, append=True)
    assert df.columns[-1] == "CHOP_14_1_100.0"

    df.ti.chop(ln=True, append=True)
    assert df.columns[-1] == "CHOPln_14_1_100.0"


def test_ext_cksp(df):
    df.ti.cksp(tvmode=False, append=True)
    assert list(df.columns[-2:]) == ["CKSPl_10_3_20", "CKSPs_10_3_20"]

    df.ti.cksp(tvmode=True, append=True)
    assert list(df.columns[-2:]) == ["CKSPl_10_1_9", "CKSPs_10_1_9"]


def test_ext_decay(df):
    df.ti.decay(append=True)
    assert df.columns[-1] == "LDECAY_1"

    df.ti.decay(mode="exp", append=True)
    assert df.columns[-1] == "EXPDECAY_1"


def test_ext_decreasing(df):
    df.ti.decreasing(append=True)
    assert df.columns[-1] == "DEC_1"

    df.ti.decreasing(length=3, strict=True, append=True)
    assert df.columns[-1] == "SDEC_3"


def test_ext_dpo(df):
    df.ti.dpo(append=True)
    assert df.columns[-1] == "DPO_20"


def test_ext_ht_trendline(df):
    df.ti.ht_trendline(append=True)
    assert df.columns[-1] == "HT_TL"


def test_ext_increasing(df):
    df.ti.increasing(append=True)
    assert df.columns[-1] == "INC_1"

    df.ti.increasing(length=3, strict=True, append=True)
    assert df.columns[-1] == "SINC_3"


def test_ext_long_run(df):
    result = df.ti.long_run(append=True)
    assert df.shape == result.shape

    fast, slow = df.ti.ema(8), df.ti.ema(21)
    df.ti.long_run(fast, slow, append=True)
    assert df.columns[-1] == "LR_2"


def test_ext_psar(df):
    df.ti.psar(append=True)
    expected = ["PSARl_0.02_0.2", "PSARs_0.02_0.2", "PSARaf_0.02_0.2", "PSARr_0.02_0.2"]
    assert list(df.columns[-4:]) == expected


def test_ext_qstick(df):
    df.ti.qstick(append=True)
    assert df.columns[-1] == "QS_10"


def test_ext_short_run(df):
    result = df.ti.short_run(append=True)
    assert df.shape == result.shape

    fast, slow = df.ti.ema(8), df.ti.ema(21)
    df.ti.short_run(fast, slow, append=True)
    assert df.columns[-1] == "SR_2"


def test_ext_rwi(df):
    df.ti.rwi(append=True)
    assert list(df.columns[-2:]) == ["RWIh_14", "RWIl_14"]


def test_ext_trendflex(df):
    df.ti.trendflex(append=True)
    assert df.columns[-1] == "TRENDFLEX_20_20_0.04"


def test_ext_vhf(df):
    df.ti.vhf(append=True)
    assert df.columns[-1] == "VHF_28"


def test_ext_vortex(df):
    df.ti.vortex(append=True)
    assert list(df.columns[-2:]) == ["VTXP_14", "VTXM_14"]


def test_ext_zigzag(df):
    df.ti.zigzag(append=True)
    assert list(df.columns[-3:]) == [
        "ZIGZAGs_5.0%_10",
        "ZIGZAGv_5.0%_10",
        "ZIGZAGd_5.0%_10",
    ]
