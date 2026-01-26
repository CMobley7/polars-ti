# -*- coding: utf-8 -*-
import numpy as np
import pandas.testing as pdt
import talib as tal
from pandas import DataFrame, Series
from pytest import mark

import polars_ti as ti

from .config import CORRELATION, CORRELATION_THRESHOLD, error_analysis


# TA Lib style Tests
def test_alligator(df):
    result = ti.alligator(df.high, df.low)
    assert isinstance(result, DataFrame)
    assert result.name == "AG_13_8_5"


def test_alma(df):
    result = ti.alma(df.close)
    assert isinstance(result, Series)
    assert result.name == "ALMA_9_6.0_0.85"


def test_dema(df):
    result = ti.dema(df.close, talib=False)
    assert isinstance(result, Series)
    assert result.name == "DEMA_10"

    try:
        expected = tal.DEMA(df.close, 10)
        pdt.assert_series_equal(result, expected, check_names=False)
    except AssertionError:
        try:
            corr = ti.utils.df_error_analysis(result, expected)
            print(f"{corr=}")
            assert corr > CORRELATION_THRESHOLD
        except Exception as ex:
            error_analysis(result, CORRELATION, ex)

    result = ti.dema(df.close)
    assert isinstance(result, Series)
    assert result.name == "DEMA_10"


def test_ema(df):
    result = ti.ema(df.close, talib=False, presma=True)
    assert isinstance(result, Series)
    assert result.name == "EMA_10"

    try:
        expected = tal.EMA(df.close, 10)
        pdt.assert_series_equal(result, expected, check_names=False)
    except AssertionError:
        try:
            corr = ti.utils.df_error_analysis(result, expected)
            print(f"{corr=}")
            assert corr > CORRELATION_THRESHOLD
        except Exception as ex:
            error_analysis(result, CORRELATION, ex)

    result = ti.ema(df.close, talib=True)
    assert isinstance(result, Series)
    assert result.name == "EMA_10"

    result = ti.ema(df.close, talib=True, presma=False, adjust=False)
    assert isinstance(result, Series)
    assert result.name == "EMA_10"

    result = ti.ema(df.close, talib=True, presma=False, adjust=True)
    assert isinstance(result, Series)
    assert result.name == "EMA_10"

    result = ti.ema(df.close, talib=True, presma=True, adjust=True)
    assert isinstance(result, Series)
    assert result.name == "EMA_10"


def test_fwma(df):
    result = ti.fwma(df.close)
    assert isinstance(result, Series)
    assert result.name == "FWMA_10"


def test_hilo(df):
    result = ti.hilo(df.high, df.low, df.close)
    assert isinstance(result, DataFrame)
    assert result.name == "HILO_13_21"


def test_hl2(df):
    result = ti.hl2(df.high, df.low)
    assert isinstance(result, Series)
    assert result.name == "HL2"


def test_hlc3(df):
    result = ti.hlc3(df.high, df.low, df.close, talib=False)
    assert isinstance(result, Series)
    assert result.name == "HLC3"

    try:
        expected = tal.TYPPRICE(df.high, df.low, df.close)
        pdt.assert_series_equal(result, expected, check_names=False)
    except AssertionError:
        try:
            corr = ti.utils.df_error_analysis(result, expected)
            print(f"{corr=}")
            assert corr > CORRELATION_THRESHOLD
        except Exception as ex:
            error_analysis(result, CORRELATION, ex)

    result = ti.hlc3(df.high, df.low, df.close)
    assert isinstance(result, Series)
    assert result.name == "HLC3"


def test_hma(df):
    result = ti.hma(df.close)
    assert isinstance(result, Series)
    assert result.name == "HMA_10"


def test_hwma(df):
    result = ti.hwma(df.close)
    assert isinstance(result, Series)
    assert result.name == "HWMA_0.2_0.1_0.1"


def test_ichimoku(df):
    result_ichimoku, result_span = ti.ichimoku(df.high, df.low, df.close)
    assert isinstance(result_ichimoku, DataFrame)
    assert isinstance(result_span, DataFrame)
    assert result_ichimoku.name == "ICHIMOKU_9_26_52"
    assert result_span.name == "ICHISPAN_9_26"


def test_jma(df):
    result = ti.jma(df.close)
    assert isinstance(result, Series)
    assert result.name == "JMA_7_0.0"


def test_kama(df):
    result = ti.kama(df.close)
    assert isinstance(result, Series)
    assert result.name == "KAMA_10_2_30"


def test_linreg(df):
    result = ti.linreg(df.close, talib=False)
    assert isinstance(result, Series)
    assert result.name == "LINREG_14"

    try:
        expected = tal.LINEARREG(df.close)
        pdt.assert_series_equal(result, expected, check_names=False)
    except AssertionError:
        try:
            corr = ti.utils.df_error_analysis(result, expected)
            print(f"{corr=}")
            assert corr > CORRELATION_THRESHOLD
        except Exception as ex:
            error_analysis(result, CORRELATION, ex)

    result = ti.linreg(df.close, talib=True)
    assert isinstance(result, Series)
    assert result.name == "LINREG_14"


def test_linreg_angle(df):
    result = ti.linreg(df.close, angle=True, talib=False)
    assert isinstance(result, Series)
    assert result.name == "LINREGa_14"

    try:
        expected = tal.LINEARREG_ANGLE(df.close)
        pdt.assert_series_equal(result, expected, check_names=False)
    except AssertionError:
        try:
            corr = ti.utils.df_error_analysis(result, expected)
            print(f"{corr=}")
            assert corr > CORRELATION_THRESHOLD
        except Exception as ex:
            error_analysis(result, CORRELATION, ex)

    result = ti.linreg(df.close, angle=True, talib=True)
    assert isinstance(result, Series)
    assert result.name == "LINREGa_14"


def test_linreg_intercept(df):
    result = ti.linreg(df.close, intercept=True, talib=False)
    assert isinstance(result, Series)
    assert result.name == "LINREGb_14"

    try:
        expected = tal.LINEARREG_INTERCEPT(df.close)
        pdt.assert_series_equal(result, expected, check_names=False)
    except AssertionError:
        try:
            corr = ti.utils.df_error_analysis(result, expected)
            print(f"{corr=}")
            assert corr > CORRELATION_THRESHOLD
        except Exception as ex:
            error_analysis(result, CORRELATION, ex)

    result = ti.linreg(df.close, intercept=True, talib=True)
    assert isinstance(result, Series)
    assert result.name == "LINREGb_14"


def test_linreg_r(df):
    result = ti.linreg(df.close, r=True)
    assert isinstance(result, Series)
    assert result.name == "LINREGr_14"


def test_linreg_slope(df):
    result = ti.linreg(df.close, slope=True, talib=False)
    assert isinstance(result, Series)
    assert result.name == "LINREGm_14"

    try:
        expected = tal.LINEARREG_SLOPE(df.close)
        pdt.assert_series_equal(result, expected, check_names=False)
    except AssertionError:
        try:
            corr = ti.utils.df_error_analysis(result, expected)
            print(f"{corr=}")
            assert corr > CORRELATION_THRESHOLD
        except Exception as ex:
            error_analysis(result, CORRELATION, ex)

    result = ti.linreg(df.close, slope=True, talib=True)
    assert isinstance(result, Series)
    assert result.name == "LINREGm_14"


def test_ma(df):
    result = ti.ma()
    assert isinstance(result, list)
    assert len(result) > 0

    result = ti.ma("ema", df.close)
    assert isinstance(result, Series)
    assert result.name == "EMA_10"

    result = ti.ma("fwma", df.close, length=4)
    assert isinstance(result, Series)
    assert result.name == "FWMA_4"


def test_mama(df):
    result = ti.mama(df.close, talib=False)
    assert isinstance(result, DataFrame)
    assert result.name == "MAMA_0.5_0.05"

    try:
        expected = tal.MAMA(df.close)
        expecteddf = DataFrame(
            {"MAMA_0.5_0.05": expected[0], "FAMA_0.5_0.05": expected[1]}
        )
        pdt.assert_frame_equal(result, expecteddf)
    except AssertionError:
        try:
            mama_corr = ti.utils.df_error_analysis(
                result.iloc[:, 0], expecteddf.iloc[:, 0]
            )
            assert mama_corr > CORRELATION_THRESHOLD
            print(f"{mama_corr=}")
        except Exception as ex:
            error_analysis(result.iloc[:, 0], CORRELATION, ex)

        try:
            fama_corr = ti.utils.df_error_analysis(
                result.iloc[:, 1], expecteddf.iloc[:, 1]
            )
            assert fama_corr > CORRELATION_THRESHOLD
            print(f"{fama_corr=}")
        except Exception as ex:
            error_analysis(result.iloc[:, 1], CORRELATION, ex)

    result = ti.mama(df.close)
    assert isinstance(result, DataFrame)
    assert result.name == "MAMA_0.5_0.05"


def test_mcgd(df):
    result = ti.mcgd(df.close)
    assert isinstance(result, Series)
    assert result.name == "MCGD_10"


def test_midpoint(df):
    result = ti.midpoint(df.close, talib=False)
    assert isinstance(result, Series)
    assert result.name == "MIDPOINT_2"

    try:
        expected = tal.MIDPOINT(df.close, 2)
        pdt.assert_series_equal(result, expected, check_names=False)
    except AssertionError:
        try:
            corr = ti.utils.df_error_analysis(result, expected)
            print(f"{corr=}")
            assert corr > CORRELATION_THRESHOLD
        except Exception as ex:
            error_analysis(result, CORRELATION, ex)

    result = ti.midpoint(df.close)
    assert isinstance(result, Series)
    assert result.name == "MIDPOINT_2"


def test_midprice(df):
    result = ti.midprice(df.high, df.low, talib=False)
    assert isinstance(result, Series)
    assert result.name == "MIDPRICE_2"

    try:
        expected = tal.MIDPRICE(df.high, df.low, 2)
        pdt.assert_series_equal(result, expected, check_names=False)
    except AssertionError:
        try:
            corr = ti.utils.df_error_analysis(result, expected)
            print(f"{corr=}")
            assert corr > CORRELATION_THRESHOLD
        except Exception as ex:
            error_analysis(result, CORRELATION, ex)

    result = ti.midprice(df.high, df.low)
    assert isinstance(result, Series)
    assert result.name == "MIDPRICE_2"


def test_ohlc4(df):
    result = ti.ohlc4(df.open, df.high, df.low, df.close)
    assert isinstance(result, Series)
    assert result.name == "OHLC4"


def test_ott(df):
    result = ti.ott(df.close)
    assert isinstance(result, DataFrame)
    assert result.name == "OTT_5_2.4"


@mark.parametrize(
    "method,name,columns",
    [
        (None, "PIVOTS_TRAD_D", 9),
        ("camarilla", "PIVOTS_CAMA_D", 9),
        ("classic", "PIVOTS_CLAS_D", 9),
        ("demark", "PIVOTS_DEMA_D", 3),
        ("fibonacci", "PIVOTS_FIBO_D", 7),
        ("traditional", "PIVOTS_TRAD_D", 9),
        ("woodie", "PIVOTS_WOOD_D", 9),
    ],
)
def test_pivots(df, method, name, columns):
    result = ti.pivots(df.open, df.high, df.low, df.close, method=method)
    assert isinstance(result, DataFrame)
    assert result.name == name
    assert result.columns.size == columns


def test_pwma(df):
    result = ti.pwma(df.close)
    assert isinstance(result, Series)
    assert result.name == "PWMA_10"


def test_rma(df):
    result = ti.rma(df.close)
    assert isinstance(result, Series)
    assert result.name == "RMA_10"


def test_sinwma(df):
    result = ti.sinwma(df.close)
    assert isinstance(result, Series)
    assert result.name == "SINWMA_14"


def test_sma(df):
    result = ti.sma(df.close, talib=False)
    assert isinstance(result, Series)
    assert result.name == "SMA_10"

    try:
        expected = tal.SMA(df.close, 10)
        pdt.assert_series_equal(result, expected, check_names=False)
    except AssertionError:
        try:
            corr = ti.utils.df_error_analysis(result, expected)
            print(f"{corr=}")
            assert corr > CORRELATION_THRESHOLD
        except Exception as ex:
            error_analysis(result, CORRELATION, ex)

    result = ti.sma(df.close, talib=True)
    assert isinstance(result, Series)
    assert result.name == "SMA_10"


def test_smma(df):
    result = ti.smma(df.close)
    assert isinstance(result, Series)
    assert result.name == "SMMA_7"


def test_ssf(df):
    result = ti.ssf(df.close)
    assert isinstance(result, Series)
    assert result.name == "SSF_20"

    result = ti.ssf(df.close, pi=np.pi, sqrt2=np.sqrt(2), everget=True)
    assert isinstance(result, Series)
    assert result.name == "SSFe_20"


def test_ssf3(df):
    result = ti.ssf3(df.close)
    assert isinstance(result, Series)
    assert result.name == "SSF3_20"


def test_swma(df):
    result = ti.swma(df.close)
    assert isinstance(result, Series)
    assert result.name == "SWMA_10"


def test_supertrend(df):
    result = ti.supertrend(df.high, df.low, df.close)
    assert isinstance(result, DataFrame)
    assert result.name == "SUPERT_7_3.0"


def test_t3(df):
    result = ti.t3(df.close, talib=False)
    assert isinstance(result, Series)
    assert result.name == "T3_10_0.7"

    try:
        expected = tal.T3(df.close, 10)
        pdt.assert_series_equal(result, expected, check_names=False)
    except AssertionError:
        try:
            corr = ti.utils.df_error_analysis(result, expected)
            print(f"{corr=}")
            assert corr > CORRELATION_THRESHOLD
        except Exception as ex:
            error_analysis(result, CORRELATION, ex)

    result = ti.t3(df.close, talib=True)
    assert isinstance(result, Series)
    assert result.name == "T3_10_0.7"


def test_tema(df):
    result = ti.tema(df.close, talib=False)
    assert isinstance(result, Series)
    assert result.name == "TEMA_10"

    try:
        expected = tal.TEMA(df.close, 10)
        pdt.assert_series_equal(result, expected, check_names=False)
    except AssertionError:
        try:
            corr = ti.utils.df_error_analysis(result, expected)
            print(f"{corr=}")
            assert corr > CORRELATION_THRESHOLD
        except Exception as ex:
            error_analysis(result, CORRELATION, ex)

    result = ti.tema(df.close, talib=True)
    assert isinstance(result, Series)
    assert result.name == "TEMA_10"


def test_trima(df):
    result = ti.trima(df.close, talib=False)
    assert isinstance(result, Series)
    assert result.name == "TRIMA_10"

    try:
        expected = tal.TRIMA(df.close, 10)
        pdt.assert_series_equal(result, expected, check_names=False)
    except AssertionError:
        try:
            corr = ti.utils.df_error_analysis(result, expected)
            print(f"{corr=}")
            assert corr > CORRELATION_THRESHOLD
        except Exception as ex:
            error_analysis(result, CORRELATION, ex)

    result = ti.trima(df.close, talib=True)
    assert isinstance(result, Series)
    assert result.name == "TRIMA_10"


def test_tsf(df):
    result = ti.linreg(df.close, tsf=True, talib=False)
    assert isinstance(result, Series)
    assert result.name == "LINREG_14"

    try:
        expected = tal.TSF(df.close)
        pdt.assert_series_equal(result, expected, check_names=False)
    except AssertionError:
        try:
            corr = ti.utils.df_error_analysis(result, expected)
            print(f"{corr=}")
            assert corr > CORRELATION_THRESHOLD
        except Exception as ex:
            error_analysis(result, CORRELATION, ex)

    result = ti.linreg(df.close, tsf=True, talib=True)
    assert isinstance(result, Series)
    assert result.name == "LINREG_14"


def test_vidya(df):
    result = ti.vidya(df.close)
    assert isinstance(result, Series)
    assert result.name == "VIDYA_14"


def test_vwma(df):
    result = ti.vwma(df.close, df.volume)
    assert isinstance(result, Series)
    assert result.name == "VWMA_10"


def test_wcp(df):
    result = ti.wcp(df.high, df.low, df.close, talib=False)
    assert isinstance(result, Series)
    assert result.name == "WCP"

    try:
        expected = tal.WCLPRICE(df.high, df.low, df.close)
        pdt.assert_series_equal(result, expected, check_names=False)
    except AssertionError:
        try:
            corr = ti.utils.df_error_analysis(result, expected)
            print(f"{corr=}")
            assert corr > CORRELATION_THRESHOLD
        except Exception as ex:
            error_analysis(result, CORRELATION, ex)

    result = ti.wcp(df.high, df.low, df.close, talib=True)
    assert isinstance(result, Series)
    assert result.name == "WCP"


def test_wma(df):
    result = ti.wma(df.close, talib=False)
    assert isinstance(result, Series)
    assert result.name == "WMA_10"

    try:
        expected = tal.WMA(df.close, 10)
        pdt.assert_series_equal(result, expected, check_names=False)
    except AssertionError:
        try:
            corr = ti.utils.df_error_analysis(result, expected)
            print(f"{corr=}")
            assert corr > CORRELATION_THRESHOLD
        except Exception as ex:
            error_analysis(result, CORRELATION, ex)

    result = ti.wma(df.close, talib=True)
    assert isinstance(result, Series)
    assert result.name == "WMA_10"


def test_zlma(df):
    result = ti.zlma(df.close, df.volume)
    assert isinstance(result, Series)
    assert result.name == "ZL_EMA_10"


def test_mmar(df):
    result = ti.mmar(df.close)
    assert isinstance(result, DataFrame)
    assert result.name == "MMAR_10_5_6"

    result = ti.mmar(df.close, length=5, step=3, num_ribbons=4)
    assert isinstance(result, DataFrame)
    assert result.name == "MMAR_5_3_4"


def test_rainbow(df):
    result = ti.rainbow(df.close)
    assert isinstance(result, DataFrame)
    assert result.name == "RAINBOW_2_10"

    result = ti.rainbow(df.close, length=3, num_ribbons=5)
    assert isinstance(result, DataFrame)
    assert result.name == "RAINBOW_3_5"


# DataFrame Extension Tests
def test_ext_alligator(df):
    df.ti.alligator(append=True)
    assert list(df.columns[-3:]) == ["AGj_13_8_5", "AGt_13_8_5", "AGl_13_8_5"]


def test_ext_alma(df):
    df.ti.alma(append=True)
    assert df.columns[-1] == "ALMA_9_6.0_0.85"


def test_ext_dema(df):
    df.ti.dema(append=True)
    assert df.columns[-1] == "DEMA_10"


def test_ext_ema(df):
    df.ti.ema(append=True)
    assert df.columns[-1] == "EMA_10"


def test_ext_fwma(df):
    df.ti.fwma(append=True)
    assert df.columns[-1] == "FWMA_10"


def test_ext_hilo(df):
    df.ti.hilo(append=True)
    assert list(df.columns[-3:]) == ["HILO_13_21", "HILOl_13_21", "HILOs_13_21"]


def test_ext_hl2(df):
    df.ti.hl2(append=True)
    assert df.columns[-1] == "HL2"


def test_ext_hlc3(df):
    df.ti.hlc3(append=True)
    assert df.columns[-1] == "HLC3"


def test_ext_hma(df):
    df.ti.hma(append=True)
    assert df.columns[-1] == "HMA_10"


def test_ext_hwma(df):
    df.ti.hwma(append=True)
    assert df.columns[-1] == "HWMA_0.2_0.1_0.1"


def test_ext_ichimoku(df):
    df.ti.ichimoku(append=True)
    columns = ["ISA_9", "ISB_26", "ITS_9", "IKS_26", "ICS_26"]
    assert list(df.columns[-5:]) == columns


def test_ext_jma(df):
    df.ti.jma(append=True)
    assert df.columns[-1] == "JMA_7_0.0"


def test_ext_kama(df):
    df.ti.kama(append=True)
    assert df.columns[-1] == "KAMA_10_2_30"


def test_ext_linreg(df):
    df.ti.linreg(append=True)
    assert df.columns[-1] == "LINREG_14"


def test_ext_mama(df):
    df.ti.mama(append=True)
    assert list(df.columns[-2:]) == ["MAMA_0.5_0.05", "FAMA_0.5_0.05"]


def test_ext_mcgd(df):
    df.ti.mcgd(append=True)
    assert df.columns[-1] == "MCGD_10"


def test_ext_midpoint(df):
    df.ti.midpoint(append=True)
    assert df.columns[-1] == "MIDPOINT_2"


def test_ext_midprice(df):
    df.ti.midprice(append=True)
    assert df.columns[-1] == "MIDPRICE_2"


def test_ext_ohlc4(df):
    df.ti.ohlc4(append=True)
    assert df.columns[-1] == "OHLC4"


def test_ext_pivots(df):
    df.ti.pivots(append=True)
    assert len(df.columns[-9:]) == 9


def test_ext_hl2(df):
    df.ti.hl2(append=True)
    assert df.columns[-1] == "HL2"


def test_ext_pwma(df):
    df.ti.pwma(append=True)
    assert df.columns[-1] == "PWMA_10"


def test_ext_sinwma(df):
    df.ti.sinwma(append=True)
    assert df.columns[-1] == "SINWMA_14"


def test_ext_sma(df):
    df.ti.sma(append=True)
    assert df.columns[-1] == "SMA_10"


def test_ext_smma(df):
    df.ti.smma(append=True)
    assert df.columns[-1] == "SMMA_7"


def test_ext_ssf(df):
    df.ti.ssf(append=True)
    assert df.columns[-1] == "SSF_20"


def test_ext_ssf3(df):
    df.ti.ssf3(append=True)
    assert df.columns[-1] == "SSF3_20"


def test_ext_swma(df):
    df.ti.swma(append=True)
    assert df.columns[-1] == "SWMA_10"


def test_ext_supertrend(df):
    df.ti.supertrend(append=True)
    columns = ["SUPERT_7_3.0", "SUPERTd_7_3.0", "SUPERTl_7_3.0", "SUPERTs_7_3.0"]
    assert list(df.columns[-4:]) == columns


def test_ext_t3(df):
    df.ti.t3(append=True)
    assert df.columns[-1] == "T3_10_0.7"


def test_ext_tema(df):
    df.ti.tema(append=True)
    assert df.columns[-1] == "TEMA_10"


def test_ext_trima(df):
    df.ti.trima(append=True)
    assert df.columns[-1] == "TRIMA_10"


def test_ext_vidya(df):
    df.ti.vidya(append=True)
    assert df.columns[-1] == "VIDYA_14"


def test_ext_vwap(df):
    df.ti.vwap(append=True)
    assert df.columns[-1] == "VWAP_D"


def test_ext_vwma(df):
    df.ti.vwma(append=True)
    assert df.columns[-1] == "VWMA_10"


def test_ext_wcp(df):
    df.ti.wcp(append=True)
    assert df.columns[-1] == "WCP"


def test_ext_wma(df):
    df.ti.wma(append=True)
    assert df.columns[-1] == "WMA_10"


def test_ext_zlma(df):
    df.ti.zlma(append=True)
    assert df.columns[-1] == "ZL_EMA_10"
