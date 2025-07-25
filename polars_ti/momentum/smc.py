# -*- coding: utf-8 -*-
from sys import float_info as sflt

from numpy import isnan, maximum, minimum, nan
from pandas import DataFrame, Series
from polars_ti._typing import DictLike, Int, IntFloat
from polars_ti.ma import ma
from polars_ti.utils import (
    v_bool,
    v_mamode,
    v_offset,
    v_pos_default,
    v_scalar,
    v_series,
    v_talib,
)


def smc(
    open_: Series,
    high: Series,
    low: Series,
    close: Series,
    abr_length: Int = None,
    close_length: Int = None,
    vol_length: Int = None,
    percent: Int = None,
    vol_ratio: IntFloat = None,
    asint: bool = None,
    mamode: str = None,
    talib: bool = None,
    offset: Int = None,
    **kwargs: DictLike,
) -> DictLike:
    """Smart Money Comcept (SMC)

    The Smart Money concept combines several techniques to identify significant
    price movements that might indicate 'smart money' actions. It uses candlestick
    patterns, moving averages, and imbalance calculations.

    Sources:
        https://www.tradingview.com/script/CnB3fSph-Smart-Money-Concepts-LuxAlgo/

    Args:
        abr_length (int): Average Bar Range (abr) window length. Default: 14
        close_length (int): The moving average length for 'close'. Default: 50
        vol_length (int): The length for calculating volatility. Default: 20
        percent (int): Percent of wick that exceeds the body. Default: 5
        vol_ratio (float): Volatility ratio to determine high volatility condition.
            Default: 1.5
        asint (bool): Keep results numerical instead of boolean.
            Default: True
        mamode (str): See ``help(ti.ma)``. Default: 'sma'
        talib (bool): If TA Lib is installed and talib is True, Returns
            the TA Lib version. Default: True
        offset (int): How many periods to offset the result. Default: 0

    Returns:
        pd.DataFrame: High Volatility, Bottom Imbalance Flag, Bottom Imbalance Value,
            Bottom Imbalance Percent, Top Imbalance Flag, Top Imbalance Value,
            Top Imbalance Percent Columns
    """
    # Validate
    abr_length = v_pos_default(abr_length, 14)
    close_length = v_pos_default(close_length, 50)
    vol_length = v_pos_default(vol_length, 20)
    if close_length < abr_length:
        abr_length, close_length = close_length, abr_length
    _length = max(abr_length, close_length, vol_length) + 1

    open_ = v_series(open_, _length)
    high = v_series(high, _length)
    low = v_series(low, _length)
    close = v_series(close, _length)

    if open_ is None or high is None or low is None or close is None:
        return

    percent = v_pos_default(percent, 5)
    body_percent = 0.01 * percent
    vol_ratio = v_scalar(vol_ratio, 1.5)
    asint = v_bool(asint)
    mamode = v_mamode(mamode, "sma")
    mode_tal = v_talib(talib)
    offset = v_offset(offset)

    # Calculate
    body_high, body_low = maximum(open_, close), minimum(open_, close)
    body = body_high - body_low + sflt.epsilon
    close_ma = ma(mamode, body, length=close_length, talib=mode_tal)

    # Calculate imbalance sizes and percentages based on Average Bar Range (abr)
    abr = high.rolling(window=abr_length).max() - low.rolling(window=abr_length).min()
    top_imbalance = low.shift(2) - high
    btm_imbalance = low - high.shift(2)
    top_imbalance_pct = 100 * top_imbalance / abr
    btm_imbalance_pct = 100 * btm_imbalance / abr
    hld = high - low + sflt.epsilon
    high_volatility = hld > vol_ratio * ma(
        mamode, hld, length=vol_length, talib=mode_tal
    )

    btm_imbalance_flag = (btm_imbalance > 0) & (btm_imbalance_pct > 1)
    top_imbalance_flag = (top_imbalance > 0) & (top_imbalance_pct > 1)

    if asint:
        high_volatility = high_volatility.astype(int)
        btm_imbalance_flag = btm_imbalance_flag.astype(int)
        top_imbalance_flag = top_imbalance_flag.astype(int)

    _props = f"_{abr_length}_{close_length}_{vol_length}_{percent}"
    data = {
        f"SMChv{_props}": high_volatility,
        f"SMCbf{_props}": btm_imbalance_flag,
        f"SMCbi{_props}": btm_imbalance,
        f"SMCbp{_props}": btm_imbalance_pct,
        f"SMCtf{_props}": top_imbalance_flag,
        f"SMCti{_props}": top_imbalance,
        f"SMCtp{_props}": top_imbalance_pct,
    }
    df = DataFrame(data, index=close.index)

    # Offset
    if offset != 0:
        df = df.shift(offset)

    # Fill
    df.ffill(inplace=True)
    df.bfill(inplace=True)

    # Name and Category
    df.name = f"SMC{_props}"
    df.category = "momentum"

    return df
