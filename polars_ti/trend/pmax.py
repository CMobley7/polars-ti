# -*- coding: utf-8 -*-
import numpy as np
from numba import njit
from pandas import DataFrame, Series

from polars_ti.ma import ma
from polars_ti.utils import v_mamode, v_offset, v_pos_default, v_series
from polars_ti.volatility.atr import atr


@njit(cache=True)
def nb_pmax(close, ub, lb):
    m = close.size
    dir_ = np.ones(m)  # 1 for Uptrend, -1 for Downtrend
    trend = np.full(m, np.nan)
    long = np.full(m, np.nan)
    short = np.full(m, np.nan)

    # State-dependent iteration
    for i in range(1, m):
        # Trend detection: price crosses bands
        if close[i] > lb[i - 1]:
            dir_[i] = 1  # Uptrend
        elif close[i] < ub[i - 1]:
            dir_[i] = -1  # Downtrend
        else:
            dir_[i] = dir_[i - 1]  # Maintain previous trend

            # Adjust bands to not move against trend
            if dir_[i] > 0 and ub[i] < ub[i - 1]:
                ub[i] = ub[i - 1]
            if dir_[i] < 0 and lb[i] > lb[i - 1]:
                lb[i] = lb[i - 1]

        # Set PMAX value based on trend direction
        if dir_[i] > 0:
            trend[i] = long[i] = ub[i]
        else:
            trend[i] = short[i] = lb[i]

    return trend, dir_, long, short


def pmax(
    high: Series,
    low: Series,
    close: Series,
    length: int | None = None,
    multiplier: int | float | None = None,
    mamode: str | None = None,
    offset: int | None = None,
    **kwargs: dict,
) -> DataFrame:
    """PMAX (Price Max)

    PMAX is a trend-following indicator that combines ATR-based volatility
    bands with a moving average. It creates adaptive trailing stop levels
    that adjust based on price action and volatility. Similar to SuperTrend
    but uses the moving average as the center instead of HL2.

    Sources:
        https://www.tradingview.com/script/sU9molfV-MOST-Moving-Stop-Loss-PMAX/
        https://kodify.net/tradingview/indicators/pmax-indicator/

    Calculation:
        Default Inputs:
            length=10, multiplier=3.0, mamode='ema'

        MA = Moving Average(close, length, mamode)
        ATR = Average True Range(high, low, close, length)

        Upper Band = MA - (multiplier × ATR)
        Lower Band = MA + (multiplier × ATR)

        In uptrend: PMAX = Upper Band (trailing stop below price)
        In downtrend: PMAX = Lower Band (trailing stop above price)

    Args:
        high (pd.Series): Series of 'high's
        low (pd.Series): Series of 'low's
        close (pd.Series): Series of 'close's
        length (int): ATR and MA period. Default: 10
        multiplier (float): ATR multiplier for bands. Default: 3.0
        mamode (str): Moving average type. See help(ti.ma). Default: 'ema'
        offset (int): How many periods to offset the result. Default: 0

    Kwargs:
        fillna (value, optional): pd.DataFrame.fillna(value)

    Returns:
        pd.DataFrame: PMAX (trend line), PMAXd (direction 1/-1),
            PMAXl (long stop), PMAXs (short stop) columns.
    """
    # Validate
    length = v_pos_default(length, 10)
    multiplier = v_pos_default(multiplier, 3.0)
    mamode = v_mamode(mamode, "ema")
    high = v_series(high, length + 1)
    low = v_series(low, length + 1)
    close = v_series(close, length + 1)

    if high is None or low is None or close is None:
        return

    offset = v_offset(offset)

    # Calculate ATR and MA
    atr_value = atr(high, low, close, length=length)
    ma_value = ma(mamode, close, length=length)

    if atr_value is None or ma_value is None:
        return

    # Calculate initial bands
    matr = multiplier * atr_value
    ub = ma_value - matr  # Upper band (support in uptrend)
    lb = ma_value + matr  # Lower band (resistance in downtrend)

    # Use Numba for state-dependent logic
    # Ensure inputs are float64 arrays to prevent truncation/dtyp issues
    np_close = close.to_numpy(dtype=np.float64)
    np_ub = ub.to_numpy(dtype=np.float64)
    np_lb = lb.to_numpy(dtype=np.float64)

    trend, dir_, long, short = nb_pmax(np_close, np_ub, np_lb)

    # Set initial values to NaN
    # Already handled by np.full(nan) but ensuring first 'length' are nan
    dir_[:length] = np.nan

    # Build result DataFrame
    _props = f"_{length}_{multiplier}"
    data = {
        f"PMAX{_props}": trend,
        f"PMAXd{_props}": dir_,
        f"PMAXl{_props}": long,
        f"PMAXs{_props}": short,
    }
    df = DataFrame(data, index=close.index)

    df.name = f"PMAX{_props}"
    df.category = "trend"

    # Offset
    if offset != 0:
        df = df.shift(offset)

    # Fill
    if "fillna" in kwargs:
        df = df.fillna(kwargs["fillna"])

    return df
