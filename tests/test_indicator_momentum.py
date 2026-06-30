# -*- coding: utf-8 -*-
import pandas.testing as pdt
import talib as tal
from pandas import DataFrame, Series, concat
from pytest import mark

import polars_ti as ti

from .config import CORRELATION, CORRELATION_THRESHOLD, error_analysis


# TA Lib style Tests
def test_ao(df):
    result = ti.ao(df.high, df.low)
    assert isinstance(result, Series)
    assert result.name == "AO_5_34"


def test_apo(df):
    result = ti.apo(df.close, talib=False)
    assert isinstance(result, Series)
    assert result.name == "APO_12_26"

    try:
        expected = tal.APO(df.close)
        pdt.assert_series_equal(result, expected, check_names=False)
    except AssertionError:
        try:
            corr = ti.utils.df_error_analysis(result, expected)
            assert corr > CORRELATION_THRESHOLD
            print(f"{corr=}")
        except Exception as ex:
            error_analysis(result, CORRELATION, ex)

    result = ti.apo(df.close)
    assert isinstance(result, Series)
    assert result.name == "APO_12_26"


def test_bias(df):
    result = ti.bias(df.close)
    assert isinstance(result, Series)
    assert result.name == "BIAS_SMA_26"


def test_bop(df):
    result = ti.bop(df.open, df.high, df.low, df.close, talib=False)
    assert isinstance(result, Series)
    assert result.name == "BOP"

    try:
        expected = tal.BOP(df.open, df.high, df.low, df.close)
        pdt.assert_series_equal(result, expected, check_names=False)
    except AssertionError:
        try:
            corr = ti.utils.df_error_analysis(result, expected)
            assert corr > CORRELATION_THRESHOLD
            print(f"{corr=}")
        except Exception as ex:
            error_analysis(result, CORRELATION, ex)

    result = ti.bop(df.open, df.high, df.low, df.close)
    assert isinstance(result, Series)
    assert result.name == "BOP"


def test_brar(df):
    result = ti.brar(df.open, df.high, df.low, df.close)
    assert isinstance(result, DataFrame)
    assert result.name == "BRAR_26"


def test_cci(df):
    result = ti.cci(df.high, df.low, df.close, talib=False)
    assert isinstance(result, Series)
    assert result.name == "CCI_14_0.015"

    try:
        expected = tal.CCI(df.high, df.low, df.close)
        pdt.assert_series_equal(result, expected, check_names=False)
    except AssertionError:
        try:
            corr = ti.utils.df_error_analysis(result, expected)
            assert corr > CORRELATION_THRESHOLD
            print(f"{corr=}")
        except Exception as ex:
            error_analysis(result, CORRELATION, ex)

    result = ti.cci(df.high, df.low, df.close)
    assert isinstance(result, Series)
    assert result.name == "CCI_14_0.015"


def test_cfo(df):
    result = ti.cfo(df.close)
    assert isinstance(result, Series)
    assert result.name == "CFO_9"


def test_cg(df):
    result = ti.cg(df.close)
    assert isinstance(result, Series)
    assert result.name == "CG_10"


def test_cmo(df):
    result = ti.cmo(df.close, talib=False)
    assert isinstance(result, Series)
    assert result.name == "CMO_14"

    try:
        expected = tal.CMO(df.close)
        pdt.assert_series_equal(result, expected, check_names=False)
    except AssertionError:
        try:
            corr = ti.utils.df_error_analysis(result, expected)
            assert corr > CORRELATION_THRESHOLD
            print(f"{corr=}")
        except Exception as ex:
            error_analysis(result, CORRELATION, ex)

    result = ti.cmo(df.close)
    assert isinstance(result, Series)
    assert result.name == "CMO_14"


def test_coppock(df):
    result = ti.coppock(df.close)
    assert isinstance(result, Series)
    assert result.name == "COPC_11_14_10"


def test_cti(df):
    result = ti.cti(df.close)
    assert isinstance(result, Series)
    assert result.name == "CTI_12"


def test_crsi(df):
    result = ti.crsi(df.close)
    assert isinstance(result, Series)
    assert result.name == "CRSI_3_2_100"


def test_er(df):
    result = ti.er(df.close)
    assert isinstance(result, Series)
    assert result.name == "ER_10"


def test_dm(df):
    result = ti.dm(df.high, df.low, talib=False)
    assert isinstance(result, DataFrame)
    assert result.name == "DM_14"

    try:
        expected_pos = tal.PLUS_DM(df.high, df.low)
        expected_neg = tal.MINUS_DM(df.high, df.low)
        expecteddf = DataFrame({"DMP_14": expected_pos, "DMN_14": expected_neg})
        pdt.assert_frame_equal(result, expecteddf)
    except AssertionError:
        try:
            dmp_corr = ti.utils.df_error_analysis(result.iloc[:, 0], expecteddf.iloc[:, 0])
            assert dmp_corr > CORRELATION_THRESHOLD
            print(f"{dmp_corr=}")
        except Exception as ex:
            error_analysis(result, CORRELATION, ex)

        try:
            dmn_corr = ti.utils.df_error_analysis(result.iloc[:, 1], expecteddf.iloc[:, 1])
            assert dmn_corr > CORRELATION_THRESHOLD
            print(f"{dmn_corr=}")
        except Exception as ex:
            error_analysis(result, CORRELATION, ex)

    result = ti.dm(df.high, df.low)
    assert isinstance(result, DataFrame)
    assert result.name == "DM_14"


def test_eri(df):
    result = ti.eri(df.high, df.low, df.close)
    assert isinstance(result, DataFrame)
    assert result.name == "ERI_13"


def test_exhc(df):
    result = ti.exhc(df.close)
    assert isinstance(result, DataFrame)
    assert result.name == "EXHCa"

    result = ti.exhc(df.close, show_all=False)
    assert isinstance(result, DataFrame)
    assert result.name == "EXHC"


def test_fisher(df):
    result = ti.fisher(df.high, df.low)
    assert isinstance(result, DataFrame)
    assert result.name == "FISHERT_9_1"


def test_imi(df):
    result = ti.imi(df.open, df.close)
    assert isinstance(result, Series)
    assert result.name == "IMI_14"

    # IMI values should be between 0 and 100
    valid_values = result.dropna()
    assert (valid_values >= 0).all() and (valid_values <= 100).all()


def test_inertia(df):
    result = ti.inertia(df.close)
    assert isinstance(result, Series)
    assert result.name == "INERTIA_20_14"

    result = ti.inertia(df.close, df.high, df.low, refined=True)
    assert isinstance(result, Series)
    assert result.name == "INERTIAr_20_14"

    result = ti.inertia(df.close, df.high, df.low, thirds=True)
    assert isinstance(result, Series)
    assert result.name == "INERTIAt_20_14"


def test_kdj(df):
    result = ti.kdj(df.high, df.low, df.close)
    assert isinstance(result, DataFrame)
    assert result.name == "KDJ_9_3"


def test_kst(df):
    result = ti.kst(df.close)
    assert isinstance(result, DataFrame)
    assert result.name == "KST_10_15_20_30_10_10_10_15_9"


def test_macd(df):
    result = ti.macd(df.close, talib=False)
    assert isinstance(result, DataFrame)
    assert result.name == "MACD_12_26_9"

    try:
        expected = tal.MACD(df.close)
        expecteddf = DataFrame(
            {
                "MACDh_12_26_9": expected[2],
                "MACDs_12_26_9": expected[1],
                "MACD_12_26_9": expected[0],
            }
        )
        pdt.assert_frame_equal(result, expecteddf)
    except AssertionError:
        try:
            macd_corr = ti.utils.df_error_analysis(result, expected)
            print(f"{macd_corr=}")
            assert macd_corr > CORRELATION_THRESHOLD
        except Exception as ex:
            error_analysis(result.iloc[:, 0], CORRELATION, ex)

        try:
            history_corr = ti.utils.df_error_analysis(result, expected)
            print(f"{history_corr=}")
            assert history_corr > CORRELATION_THRESHOLD
        except Exception as ex:
            error_analysis(result.iloc[:, 1], CORRELATION, ex)

        try:
            signal_corr = ti.utils.df_error_analysis(result, expected)
            print(f"{signal_corr=}")
            assert signal_corr > CORRELATION_THRESHOLD
        except Exception as ex:
            error_analysis(result.iloc[:, 2], CORRELATION, ex)

    result = ti.macd(df.close)
    assert isinstance(result, DataFrame)
    assert result.name == "MACD_12_26_9"


def test_macdas(df):
    result = ti.macd(df.close, asmode=True)
    assert isinstance(result, DataFrame)
    assert result.name == "MACDAS_12_26_9"


def test_mom(df):
    result = ti.mom(df.close, talib=False)
    assert isinstance(result, Series)
    assert result.name == "MOM_10"

    try:
        expected = tal.MOM(df.close)
        pdt.assert_series_equal(result, expected, check_names=False)
    except AssertionError:
        try:
            corr = ti.utils.df_error_analysis(result, expected)
            print(f"{corr=}")
            assert corr > CORRELATION_THRESHOLD
        except Exception as ex:
            error_analysis(result, CORRELATION, ex)

    result = ti.mom(df.close)
    assert isinstance(result, Series)
    assert result.name == "MOM_10"


def test_pgo(df):
    result = ti.pgo(df.high, df.low, df.close, asmode=True)
    assert isinstance(result, Series)
    assert result.name == "PGO_14"


def test_ppo(df):
    result = ti.ppo(df.close, talib=False)
    assert isinstance(result, DataFrame)
    assert result.name == "PPO_12_26_9"

    try:
        expected = tal.PPO(df.close)
        pdt.assert_series_equal(result.iloc[:, 0], expected, check_names=False)
    except AssertionError:
        try:
            corr = ti.utils.df_error_analysis(result, expected)
            print(f"{corr=}")
            assert corr > CORRELATION_THRESHOLD
        except Exception as ex:
            error_analysis(result, CORRELATION, ex)

    result = ti.ppo(df.close)
    assert isinstance(result, DataFrame)
    assert result.name == "PPO_12_26_9"


def test_psl(df):
    result = ti.psl(df.close, asmode=True)
    assert isinstance(result, Series)
    assert result.name == "PSL_12"


def test_pvo(df):
    result = ti.pvo(df.volume, asmode=True)
    assert isinstance(result, DataFrame)
    assert result.name == "PVO_12_26_9"


def test_qqe(df):
    result = ti.qqe(df.volume, asmode=True)
    assert isinstance(result, DataFrame)
    assert result.name == "QQE_14_5_4.236"


def test_roc(df):
    result = ti.roc(df.close, talib=False)
    assert isinstance(result, Series)
    assert result.name == "ROC_10"

    try:
        expected = tal.ROC(df.close)
        pdt.assert_series_equal(result, expected, check_names=False)
    except AssertionError:
        try:
            corr = ti.utils.df_error_analysis(result, expected)
            print(f"{corr=}")
            assert corr > CORRELATION_THRESHOLD
        except Exception as ex:
            error_analysis(result, CORRELATION, ex)

    result = ti.roc(df.close)
    assert isinstance(result, Series)
    assert result.name == "ROC_10"


def test_rsi(df):
    result = ti.rsi(df.close, talib=False)
    assert isinstance(result, Series)
    assert result.name == "RSI_14"

    try:
        expected = tal.RSI(df.close)
        pdt.assert_series_equal(result, expected, check_names=False)
    except AssertionError:
        try:
            corr = ti.utils.df_error_analysis(result, expected)
            print(f"{corr=}")
            assert corr > CORRELATION_THRESHOLD
        except Exception as ex:
            error_analysis(result, CORRELATION, ex)

    result = ti.rsi(df.close)
    assert isinstance(result, Series)
    assert result.name == "RSI_14"


def test_rsx(df):
    result = ti.rsx(df.close)
    assert isinstance(result, Series)
    assert result.name == "RSX_14"


def test_rmi(df):
    result = ti.rmi(df.close)
    assert isinstance(result, Series)
    assert result.name == "RMI_14_5"

    # Test with custom parameters
    result = ti.rmi(df.close, length=20, momentum=10)
    assert isinstance(result, Series)
    assert result.name == "RMI_20_10"


def test_rvgi(df):
    result = ti.rvgi(df.open, df.high, df.low, df.close)
    assert isinstance(result, DataFrame)
    assert result.name == "RVGI_14_4"


def test_slope(df):
    result = ti.slope(df.close)
    assert isinstance(result, Series)
    assert result.name == "SLOPE_1"

    result = ti.slope(df.close, as_angle=True)
    assert isinstance(result, Series)
    assert result.name == "ANGLEr_1"

    result = ti.slope(df.close, as_angle=True, to_degrees=True)
    assert isinstance(result, Series)
    assert result.name == "ANGLEd_1"


def test_smc(df):
    result = ti.smc(df.open, df.high, df.low, df.close)
    assert isinstance(result, DataFrame)
    assert result.name == "SMC_14_50_20_5"


def test_smi(df):
    result = ti.smi(df.close)
    assert isinstance(result, DataFrame)
    assert result.name == "SMI_5_20_5_1.0"

    result = ti.smi(df.close, scalar=10)
    assert isinstance(result, DataFrame)
    assert result.name == "SMI_5_20_5_10.0"


def test_squeeze(df):
    result = ti.squeeze(df.high, df.low, df.close)
    assert isinstance(result, DataFrame)
    assert result.name == "SQZ_20_2.0_20_1.5"

    result = ti.squeeze(df.high, df.low, df.close, tr=False)
    assert isinstance(result, DataFrame)
    assert result.name == "SQZhlr_20_2.0_20_1.5"

    result = ti.squeeze(df.high, df.low, df.close, lazybear=True)
    assert isinstance(result, DataFrame)
    assert result.name == "SQZ_20_2.0_20_1.5_LB"

    result = ti.squeeze(df.high, df.low, df.close, tr=False, lazybear=True)
    assert isinstance(result, DataFrame)
    assert result.name == "SQZhlr_20_2.0_20_1.5_LB"


def test_squeeze_pro(df):
    result = ti.squeeze_pro(df.high, df.low, df.close)
    assert isinstance(result, DataFrame)
    assert result.name == "SQZPRO_20_2.0_20_2.0_1.5_1.0"

    result = ti.squeeze_pro(df.high, df.low, df.close, tr=False)
    assert isinstance(result, DataFrame)
    assert result.name == "SQZPROhlr_20_2.0_20_2.0_1.5_1.0"

    result = ti.squeeze_pro(df.high, df.low, df.close, 20, 2, 20, 3, 2, 1)
    assert isinstance(result, DataFrame)
    assert result.name == "SQZPRO_20_2_20_3.0_2.0_1.0"

    result = ti.squeeze_pro(df.high, df.low, df.close, 20, 2, 20, 3, 2, 1, tr=False)
    assert isinstance(result, DataFrame)
    assert result.name == "SQZPROhlr_20_2_20_3.0_2.0_1.0"


def test_stc(df):
    result = ti.stc(df.close)
    assert isinstance(result, DataFrame)
    assert result.name == "STC_10_12_26_0.5"


def test_stoch(df):
    result = ti.stoch(df.high, df.low, df.close, talib=False)
    assert isinstance(result, DataFrame)
    assert result.name == "STOCH_14_3_3"

    try:
        expected = tal.STOCH(df.high, df.low, df.close, 14, 3, 0, 3, 0)
        expecteddf = DataFrame({"STOCHk_14_3_0_3_0": expected[0], "STOCHd_14_3_0_3": expected[1]})
        pdt.assert_frame_equal(result, expecteddf)
    except AssertionError:
        try:
            corr = ti.utils.df_error_analysis(result, expected)
            print(f"{corr=}")
            assert corr > CORRELATION_THRESHOLD
        except Exception as ex:
            error_analysis(result, CORRELATION, ex)

    result = ti.stoch(df.high, df.low, df.close)
    assert isinstance(result, DataFrame)
    assert result.name == "STOCH_14_3_3"


def test_stochf(df):
    result = ti.stochf(df.high, df.low, df.close, talib=False)
    assert isinstance(result, DataFrame)
    assert result.name == "STOCHF_14_3"

    try:
        expected = tal.STOCHF(df.high, df.low, df.close, 14, 3, 0)
        expecteddf = DataFrame({"STOCHFk_14_3_0": expected[0], "STOCHFd_14_3_0": expected[1]})
        pdt.assert_frame_equal(result, expecteddf)
    except AssertionError:
        try:
            stochk_corr = ti.utils.df_error_analysis(result.iloc[:, 0], expected.iloc[:, 0])
            print(f"{stochk_corr=}")
            assert stochk_corr > CORRELATION_THRESHOLD
        except Exception as ex:
            error_analysis(result.iloc[:, 0], CORRELATION, ex)

        try:
            stochd_corr = ti.utils.df_error_analysis(result.iloc[:, 1], expected.iloc[:, 1])
            print(f"{stochd_corr=}")
            assert stochd_corr > CORRELATION_THRESHOLD
        except Exception as ex:
            error_analysis(result.iloc[:, 1], CORRELATION, ex)

    result = ti.stochf(df.high, df.low, df.close)
    assert isinstance(result, DataFrame)
    assert result.name == "STOCHF_14_3"


def test_stochrsi(df):
    # TV Correlation
    result = ti.stochrsi(df.close, talib=False)
    assert isinstance(result, DataFrame)
    assert result.name == "STOCHRSI_14_14_3_3"

    try:
        expected = tal.STOCHRSI(df.close, 14, 14, 3, 0)
        expecteddf = DataFrame({"STOCHRSIk_14_14_0_3": expected[0], "STOCHRSId_14_14_3_0": expected[1]})
        pdt.assert_frame_equal(result, expecteddf)
    except AssertionError:
        try:
            corr = ti.utils.df_error_analysis(result.iloc[:, 0], expecteddf.iloc[:, 1])
            print(f"{corr=}")
            assert corr > CORRELATION_THRESHOLD
        except Exception as ex:
            error_analysis(result.iloc[:, 0], CORRELATION, ex, newline=False)


def test_tmo(df):
    result = ti.tmo(df.open, df.close)
    assert isinstance(result, DataFrame)
    assert result.name == "TMO_14_5_3"

    result = ti.tmo(df.open, df.close, compute_momentum=True)
    assert isinstance(result, DataFrame)
    assert result.name == "TMO_14_5_3"


def test_trix(df):
    result = ti.trix(df.close)
    assert isinstance(result, DataFrame)
    assert result.name == "TRIX_30_9"


def test_tsi(df):
    result = ti.tsi(df.close)
    assert isinstance(result, DataFrame)
    assert result.name == "TSI_13_25_13"


def test_uo(df):
    result = ti.uo(df.high, df.low, df.close, talib=False)
    assert isinstance(result, Series)
    assert result.name == "UO_7_14_28"

    try:
        expected = tal.ULTOSC(df.high, df.low, df.close)
        pdt.assert_series_equal(result, expected, check_names=False)
    except AssertionError:
        try:
            corr = ti.utils.df_error_analysis(result, expected)
            print(f"{corr=}")
            assert corr > CORRELATION_THRESHOLD
        except Exception as ex:
            error_analysis(result, CORRELATION, ex)

    result = ti.uo(df.high, df.low, df.close)
    assert isinstance(result, Series)
    assert result.name == "UO_7_14_28"


def test_willr(df):
    result = ti.willr(df.high, df.low, df.close, talib=False)
    assert isinstance(result, Series)
    assert result.name == "WILLR_14"

    try:
        expected = tal.WILLR(df.high, df.low, df.close)
        pdt.assert_series_equal(result, expected, check_names=False)
    except AssertionError:
        try:
            corr = ti.utils.df_error_analysis(result, expected)
            print(f"{corr=}")
            assert corr > CORRELATION_THRESHOLD
        except Exception as ex:
            error_analysis(result, CORRELATION, ex)

    result = ti.willr(df.high, df.low, df.close)
    assert isinstance(result, Series)
    assert result.name == "WILLR_14"


def test_lrsi(df):
    result = ti.lrsi(df.close)
    assert isinstance(result, Series)
    assert result.name == "LRSI_14"

    result = ti.lrsi(df.close, gamma=0.7)
    assert isinstance(result, Series)
    assert result.name == "LRSI_14"


def test_po(df):
    result = ti.po(df.close)
    assert isinstance(result, Series)
    assert result.name == "PO_14"


def test_trixh(df):
    result = ti.trixh(df.close)
    assert isinstance(result, DataFrame)
    assert result.name == "TRIXH_18_9"


def test_vwmacd(df):
    result = ti.vwmacd(df.close, df.volume)
    assert isinstance(result, DataFrame)
    assert result.name == "VWMACD_12_26_9"


# DataFrame Extension Tests
def test_ext_ao(df):
    df.ti.ao(append=True)
    assert df.columns[-1] == "AO_5_34"


def test_ext_apo(df):
    df.ti.apo(append=True)
    assert df.columns[-1] == "APO_12_26"


def test_ext_bias(df):
    df.ti.bias(append=True)
    assert df.columns[-1] == "BIAS_SMA_26"


def test_ext_bop(df):
    df.ti.bop(append=True)
    assert df.columns[-1] == "BOP"


def test_ext_brar(df):
    df.ti.brar(append=True)
    assert df.columns[-1] == "BR_26"


def test_ext_cci(df):
    df.ti.cci(append=True)
    assert df.columns[-1] == "CCI_14_0.015"


def test_ext_cfo(df):
    df.ti.cfo(append=True)
    assert df.columns[-1] == "CFO_9"


def test_ext_cg(df):
    df.ti.cg(append=True)
    assert df.columns[-1] == "CG_10"


def test_ext_cmo(df):
    df.ti.cmo(append=True)
    assert df.columns[-1] == "CMO_14"


def test_ext_coppock(df):
    df.ti.coppock(append=True)
    assert df.columns[-1] == "COPC_11_14_10"


def test_ext_crsi(df):
    df.ti.crsi(append=True)
    assert df.columns[-1] == "CRSI_3_2_100"


def test_ext_cti(df):
    df.ti.cti(append=True)
    assert df.columns[-1] == "CTI_12"


def test_ext_crsi(df):
    df.ti.crsi(append=True)
    assert df.columns[-1] == "CRSI_3_2_100"


def test_ext_er(df):
    df.ti.er(append=True)
    assert df.columns[-1] == "ER_10"


def test_ext_eri(df):
    df.ti.eri(append=True)
    assert list(df.columns[-2:]) == ["BULLP_13", "BEARP_13"]


def test_ext_exhc(df):
    df.ti.exhc(append=True)
    assert list(df.columns[-2:]) == ["EXHC_DNa", "EXHC_UPa"]


def test_ext_fisher(df):
    df.ti.fisher(append=True)
    assert list(df.columns[-2:]) == ["FISHERT_9_1", "FISHERTs_9_1"]


def test_ext_inertia(df):
    df.ti.inertia(append=True)
    assert df.columns[-1] == "INERTIA_20_14"


def test_ext_kdj(df):
    df.ti.kdj(append=True)
    assert list(df.columns[-3:]) == ["K_9_3", "D_9_3", "J_9_3"]


def test_ext_kst(df):
    df.ti.kst(append=True)
    assert list(df.columns[-2:]) == ["KST_10_15_20_30_10_10_10_15", "KSTs_9"]


def test_ext_macd(df):
    df.ti.macd(append=True)
    columns = ["MACD_12_26_9", "MACDh_12_26_9", "MACDs_12_26_9"]
    assert list(df.columns[-3:]) == columns


def test_ext_mom(df):
    df.ti.mom(append=True)
    assert df.columns[-1] == "MOM_10"


def test_ext_pgo(df):
    df.ti.pgo(append=True)
    assert df.columns[-1] == "PGO_14"


def test_ext_ppo(df):
    df.ti.ppo(append=True)
    assert list(df.columns[-3:]) == ["PPO_12_26_9", "PPOh_12_26_9", "PPOs_12_26_9"]


def test_ext_psl(df):
    df.ti.psl(append=True)
    assert df.columns[-1] == "PSL_12"


def test_ext_pvo(df):
    df.ti.pvo(append=True)
    assert list(df.columns[-3:]) == ["PVO_12_26_9", "PVOh_12_26_9", "PVOs_12_26_9"]


def test_ext_qqe(df):
    df.ti.qqe(append=True)
    columns = [
        "QQE_14_5_4.236",
        "QQE_14_5_4.236_RSIMA",
        "QQEl_14_5_4.236",
        "QQEs_14_5_4.236",
    ]
    assert list(df.columns[-4:]) == columns


def test_ext_roc(df):
    df.ti.roc(append=True)
    assert df.columns[-1] == "ROC_10"


def test_ext_rsi(df):
    df.ti.rsi(append=True)
    assert df.columns[-1] == "RSI_14"


def test_ext_rsx(df):
    df.ti.rsx(append=True)
    assert df.columns[-1] == "RSX_14"


def test_ext_rmi(df):
    df.ti.rmi(append=True)
    assert df.columns[-1] == "RMI_14_5"


def test_ext_rvgi(df):
    df.ti.rvgi(append=True)
    assert list(df.columns[-2:]) == ["RVGI_14_4", "RVGIs_14_4"]


def test_ext_slope(df):
    df.ti.slope(append=True)
    assert df.columns[-1] == "SLOPE_1"


def test_ext_smc(df):
    df.ti.smc(append=True)
    columns = [
        "SMChv_14_50_20_5",
        "SMCbf_14_50_20_5",
        "SMCbi_14_50_20_5",
        "SMCbp_14_50_20_5",
        "SMCtf_14_50_20_5",
        "SMCti_14_50_20_5",
        "SMCtp_14_50_20_5",
    ]
    assert list(df.columns[-7:]) == columns


def test_ext_smi(df):
    df.ti.smi(append=True)
    columns = ["SMI_5_20_5_1.0", "SMIs_5_20_5_1.0", "SMIo_5_20_5_1.0"]
    assert list(df.columns[-3:]) == columns


def test_ext_squeeze(df):
    df.ti.squeeze(append=True)
    columns = ["SQZ_20_2.0_20_1.5", "SQZ_ON", "SQZ_OFF", "SQZ_NO"]
    assert list(df.columns[-4:]) == columns


def test_ext_squeeze_pro(df):
    df.ti.squeeze_pro(append=True)
    columns = ["SQZPRO_ON_NORMAL", "SQZPRO_ON_NARROW", "SQZPRO_OFF", "SQZPRO_NO"]
    assert list(df.columns[-4:]) == columns


def test_ext_stc(df):
    df.ti.stc(append=True)
    columns = ["STC_10_12_26_0.5", "STCmacd_10_12_26_0.5", "STCstoch_10_12_26_0.5"]
    assert list(df.columns[-3:]) == columns


def test_ext_stoch(df):
    df.ti.stoch(append=True)
    columns = ["STOCHk_14_3_3", "STOCHd_14_3_3", "STOCHh_14_3_3"]
    assert list(df.columns[-3:]) == columns


def test_ext_stochf(df):
    df.ti.stochf(append=True)
    assert list(df.columns[-2:]) == ["STOCHFk_14_3", "STOCHFd_14_3"]


def test_ext_stochrsi(df):
    df.ti.stochrsi(append=True)
    assert list(df.columns[-2:]) == ["STOCHRSIk_14_14_3_3", "STOCHRSId_14_14_3_3"]


def test_ext_tmo(df):
    df.ti.tmo(append=True)
    columns = ["TMO_14_5_3", "TMOs_14_5_3", "TMOM_14_5_3", "TMOMs_14_5_3"]
    assert list(df.columns[-4:]) == columns


def test_ext_trix(df):
    df.ti.trix(append=True)
    assert list(df.columns[-2:]) == ["TRIX_30_9", "TRIXs_30_9"]


def test_ext_tsi(df):
    df.ti.tsi(append=True)
    assert list(df.columns[-2:]) == ["TSI_13_25_13", "TSIs_13_25_13"]


def test_ext_uo(df):
    df.ti.uo(append=True)
    assert df.columns[-1] == "UO_7_14_28"


def test_ext_willr(df):
    df.ti.willr(append=True)
    assert df.columns[-1] == "WILLR_14"
