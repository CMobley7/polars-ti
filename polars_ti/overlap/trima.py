# -*- coding: utf-8 -*-
from pandas import Series

from polars_ti.maps import Imports
from polars_ti.utils import v_offset, v_pos_default, v_series, v_talib

from .sma import sma


def trima(
    close: Series,
    length: int | None = None,
    talib: bool | None = None,
    offset: int | None = None,
    **kwargs: dict,
) -> Series:
    """Triangular Moving Average (TRIMA)

    A weighted moving average where the shape of the weights are triangular
    and the greatest weight is in the middle of the period.

    Sources:
        https://www.tradingtechnologies.com/help/x-study/technical-indicator-definitions/triangular-moving-average-trima/
        tma = sma(sma(src, ceil(length / 2)), floor(length / 2) + 1)  # Tradingview
        trima = sma(sma(x, n), n)  # Tradingview

    Args:
        close (pd.Series): Series of 'close's
        length (int): It's period. Default: 10
        talib (bool): If TA Lib is installed and talib is True, Returns
            the TA Lib version. Default: True
        offset (int): How many periods to offset the result. Default: 0

    Kwargs:
        adjust (bool): Default: True
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
        from talib import TRIMA

        trima = TRIMA(close, length)
    else:
        half_length = round(0.5 * (length + 1))
        sma1 = sma(close, length=half_length, talib=mode_tal)
        trima = sma(sma1, length=half_length, talib=mode_tal)

    # Offset
    if offset != 0:
        trima = trima.shift(offset)

    # Fill
    if "fillna" in kwargs:
        trima = trima.fillna(kwargs["fillna"])

    # Name and Category
    trima.name = f"TRIMA_{length}"
    trima.category = "overlap"

    return trima
