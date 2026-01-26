# -*- coding: utf-8 -*-
import numpy as np
from pandas import DataFrame, Series

from polars_ti.overlap import hl2
from polars_ti.utils import v_mamode, v_offset, v_pos_default, v_series
from polars_ti.volatility import atr


def supertrend(
    high: Series,
    low: Series,
    close: Series,
    length: int | None = None,
    atr_length: int | None = None,
    multiplier: int | float | None = None,
    atr_mamode: str | None = None,
    offset: int | None = None,
    **kwargs: dict,
) -> DataFrame:
    """Supertrend (supertrend)

    Supertrend is an overlap indicator created by Olivier Seban. It is used
    to help identify trend direction, setting stop loss, identify support
    and resistance, and/or generate buy & sell signals.

    The indicator combines trend detection and volatility using ATR. Band
    preservation logic per TradingView:
        upperBand = basicUpperBand < prev upperBand or prev close > prev upperBand
                    ? basicUpperBand : prev upperBand
        lowerBand = basicLowerBand > prev lowerBand or prev close < prev lowerBand
                    ? basicLowerBand : prev lowerBand

    Sources:
        https://www.tradingview.com/support/solutions/43000634738-supertrend/
        https://www.investopedia.com/supertrend-indicator-7976167
        http://www.freebsensetips.com/blog/detail/7/What-is-supertrend-indicator-its-calculation

    Args:
        high (pd.Series): Series of 'high's
        low (pd.Series): Series of 'low's
        close (pd.Series): Series of 'close's
        length (int) : Length for ATR calculation. Default: 7
        atr_length (int) : If None, defaults to length otherwise, provides
            variable of control. Default: length
        multiplier (float): Coefficient for upper and lower band distance to
            midrange. Default: 3.0
        atr_mamode (str) : MA type to be used for ATR calculation.
            See ``help(ti.ma)``. Default: 'rma'
        offset (int): How many periods to offset the result. Default: 0

    Kwargs:
        fillna (value, optional): pd.DataFrame.fillna(value)

    Returns:
        pd.DataFrame: SUPERT (trend), SUPERTd (direction),
            SUPERTl (long), SUPERTs (short) columns.
    """
    # Validate
    length = v_pos_default(length, 7)
    atr_length = v_pos_default(atr_length, length)
    high = v_series(high, length + 1)
    low = v_series(low, length + 1)
    close = v_series(close, length + 1)

    if high is None or low is None or close is None:
        return

    multiplier = v_pos_default(multiplier, 3.0)
    atr_mamode = v_mamode(atr_mamode, "rma")
    offset = v_offset(offset)

    # Calculate
    m = close.size
    dir_, trend = [1] * m, [0] * m
    long, short = [np.nan] * m, [np.nan] * m

    hl2_ = hl2(high, low)
    matr = multiplier * atr(high, low, close, atr_length, mamode=atr_mamode)
    lb = hl2_ - matr
    ub = hl2_ + matr

    for i in range(1, m):
        if close.iat[i] > ub.iat[i - 1]:
            dir_[i] = 1
        elif close.iat[i] < lb.iat[i - 1]:
            dir_[i] = -1
        else:
            dir_[i] = dir_[i - 1]

        # Preserve bands in trend direction (moved outside else for TV match)
        if dir_[i] > 0 and lb.iat[i] < lb.iat[i - 1]:
            lb.iat[i] = lb.iat[i - 1]
        if dir_[i] < 0 and ub.iat[i] > ub.iat[i - 1]:
            ub.iat[i] = ub.iat[i - 1]

        if dir_[i] > 0:
            trend[i] = long[i] = lb.iat[i]
        else:
            trend[i] = short[i] = ub.iat[i]

    trend[0] = np.nan
    dir_[:length] = [np.nan] * length

    _props = f"_{length}_{multiplier}"
    data = {
        f"SUPERT{_props}": trend,
        f"SUPERTd{_props}": dir_,
        f"SUPERTl{_props}": long,
        f"SUPERTs{_props}": short,
    }
    df = DataFrame(data, index=close.index)

    df.name = f"SUPERT{_props}"
    df.category = "overlap"

    # Offset
    if offset != 0:
        df = df.shift(offset)

    # Fill
    if "fillna" in kwargs:
        df = df.fillna(kwargs["fillna"])

    return df
