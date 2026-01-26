# -*- coding: utf-8 -*-
from numpy import nan as npNaN
from pandas import DataFrame, Series

from polars_ti.ma import ma
from polars_ti.utils import v_mamode, v_offset, v_pos_default, v_series


def ott(
    close: Series,
    length: int | None = None,
    multiplier: float | None = None,
    mamode: str | None = None,
    offset: int | None = None,
    **kwargs: dict,
) -> DataFrame:
    """Optimized Trend Tracker (OTT)

    Developed by Anıl Özekşi, OTT is a trend-following and trailing stop
    indicator. It helps identify trend direction - prices above OTT suggest
    uptrend, prices below suggest downtrend.

    Sources:
        https://www.tradingview.com/script/D2tBi8px-Optimized-Trend-Tracker/

    Args:
        close (pd.Series): Series of 'close's
        length (int): Moving average period. Default: 5
        multiplier (float): Percentage for band width. Default: 2.4
        mamode (str): Moving average type. Default: 'vidya'
        offset (int): How many periods to offset the result. Default: 0

    Kwargs:
        fillna (value, optional): pd.DataFrame.fillna(value)

    Returns:
        pd.DataFrame: OTT (trend line), OTTSL (signal line), OTTd (direction)
    """
    # Validate
    length = v_pos_default(length, 5)
    multiplier = v_pos_default(multiplier, 2.4)
    close = v_series(close, length)
    mamode = v_mamode(mamode, "vidya")
    offset = v_offset(offset)

    if close is None:
        return

    # Calculate
    m = close.size
    dir_ = [1] * m
    trend = [0.0] * m
    after_dir = [1] * m
    long = [npNaN] * m
    short = [npNaN] * m

    # Calculate moving average
    mavg = ma(mamode, close, length=length)

    if mavg is None:
        return

    # Calculate bands as percentage of moving average
    matr = multiplier * mavg * 0.01
    upperband = mavg + matr
    lowerband = mavg - matr

    # Make copies for modification
    upperband = upperband.copy()
    lowerband = lowerband.copy()

    for i in range(1, m):
        # Determine direction
        if mavg.iloc[i] > upperband.iloc[i - 1]:
            dir_[i] = 1
        elif mavg.iloc[i] < lowerband.iloc[i - 1]:
            dir_[i] = -1
        else:
            dir_[i] = dir_[i - 1]
            # Preserve bands in direction
            if dir_[i] > 0 and lowerband.iloc[i] < lowerband.iloc[i - 1]:
                lowerband.iloc[i] = lowerband.iloc[i - 1]
            if dir_[i] < 0 and upperband.iloc[i] > upperband.iloc[i - 1]:
                upperband.iloc[i] = upperband.iloc[i - 1]

        # Calculate OTT trend line with multiplier adjustment
        if dir_[i] > 0:
            trend[i] = lowerband.iloc[i] * (200 + multiplier) / 200
        else:
            trend[i] = upperband.iloc[i] * (200 - multiplier) / 200

    # Calculate after direction (for signal confirmation)
    for i in range(2, m):
        if mavg.iloc[i] > trend[i - 2]:
            after_dir[i] = 1
        elif mavg.iloc[i] < trend[i - 1]:
            after_dir[i] = -1
        else:
            after_dir[i] = after_dir[i - 1]

    # Build result
    _props = f"_{length}_{multiplier}"

    trend_series = Series(trend, index=close.index)
    mavg_series = mavg
    dir_series = Series(after_dir, index=close.index)

    # Offset
    if offset != 0:
        trend_series = trend_series.shift(offset)
        mavg_series = mavg_series.shift(offset)
        dir_series = dir_series.shift(offset)

    # Fill
    if "fillna" in kwargs:
        trend_series = trend_series.fillna(kwargs["fillna"])
        mavg_series = mavg_series.fillna(kwargs["fillna"])
        dir_series = dir_series.fillna(kwargs["fillna"])

    # Name and Category
    trend_series.name = f"OTT{_props}"
    mavg_series.name = f"OTTSL{_props}"
    dir_series.name = f"OTTd{_props}"

    df = DataFrame(
        {
            trend_series.name: trend_series,
            mavg_series.name: mavg_series,
            dir_series.name: dir_series,
        },
        index=close.index,
    )
    df.name = f"OTT{_props}"
    df.category = "overlap"

    return df
