# -*- coding: utf-8 -*-
from numpy import isnan
from pandas import Series

from polars_ti.maps import Imports
from polars_ti.utils import v_offset, v_pos_default, v_series, v_talib

from .ema import ema


def dema(
    close: Series,
    length: int | None = None,
    talib: bool | None = None,
    offset: int | None = None,
    **kwargs: dict,
) -> Series:
    """Double Exponential Moving Average (DEMA)

    The Double Exponential Moving Average attempts to a smoother average
    with less lag than the normal Exponential Moving Average (EMA).

    Sources:
        https://www.tradingtechnologies.com/help/x-study/technical-indicator-definitions/double-exponential-moving-average-dema/

    Args:
        close (pd.Series): Series of 'close's
        length (int): It's period. Default: 10
        talib (bool): If TA Lib is installed and talib is True, Returns
            the TA Lib version. Default: True
        offset (int): How many periods to offset the result. Default: 0

    Kwargs:
        fillna (value, optional): pd.DataFrame.fillna(value)

    Returns:
        pd.Series: New feature generated.
    """
    # Validate
    length = v_pos_default(length, 10)
    close = v_series(close, length)

    if close is None:
        return

    mode_tal = v_talib(talib)
    offset = v_offset(offset)

    # Calculate
    if Imports["talib"] and mode_tal:
        from talib import DEMA

        dema = DEMA(close, length)
    else:
        ema1 = ema(close=close, length=length, talib=mode_tal)
        ema2 = ema(close=ema1, length=length, talib=mode_tal)
        dema = 2 * ema1 - ema2

    if all(isnan(dema.to_numpy())):
        return  # Emergency Break

    # Offset
    if offset != 0:
        dema = dema.shift(offset)

    # Fill
    if "fillna" in kwargs:
        dema = dema.fillna(kwargs["fillna"])

    # Name and Category
    dema.name = f"DEMA_{length}"
    dema.category = "overlap"

    return dema
