# -*- coding: utf-8 -*-
from pandas import Series

from polars_ti.maps import Imports
from polars_ti.utils import v_offset, v_pos_default, v_series, v_talib


def midpoint(
    close: Series,
    length: int | None = None,
    talib: bool | None = None,
    offset: int | None = None,
    **kwargs: dict,
) -> Series:
    """Midpoint

    The Midpoint is the average of the rolling high and low of period length.

    Args:
        close (pd.Series): Series of 'close's
        length (int): It's period. Default: 2
        talib (bool): If TA Lib is installed and talib is True, Returns
            the TA Lib version. Default: True
        offset (int): How many periods to offset the result. Default: 0

    Kwargs:
        fillna (value, optional): pd.DataFrame.fillna(value)

    Returns:
        pd.Series: New feature generated.
    """
    # Validate
    length = v_pos_default(length, 2)
    if "min_periods" in kwargs and kwargs["min_periods"] is not None:
        min_periods = int(kwargs["min_periods"])
    else:
        min_periods = length
    close = v_series(close, max(length, min_periods))

    if close is None:
        return

    mode_tal = v_talib(talib)
    offset = v_offset(offset)

    # Calculate
    if Imports["talib"] and mode_tal:
        from talib import MIDPOINT

        midpoint = MIDPOINT(close, length)
    else:
        lowest = close.rolling(length, min_periods=min_periods).min()
        highest = close.rolling(length, min_periods=min_periods).max()
        midpoint = 0.5 * (lowest + highest)

    # Offset
    if offset != 0:
        midpoint = midpoint.shift(offset)

    # Fill
    if "fillna" in kwargs:
        midpoint = midpoint.fillna(kwargs["fillna"])

    # Name and Category
    midpoint.name = f"MIDPOINT_{length}"
    midpoint.category = "overlap"

    return midpoint
