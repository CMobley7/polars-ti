# -*- coding: utf-8 -*-
from pandas import Series

from polars_ti.maps import Imports
from polars_ti.utils import (
    v_bool,
    v_drift,
    v_mamode,
    v_offset,
    v_pos_default,
    v_scalar,
    v_series,
    v_talib,
)
from polars_ti.volatility import atr


def natr(
    high: Series,
    low: Series,
    close: Series,
    length: int | None = None,
    scalar: int | float | None = None,
    mamode: str | None = None,
    talib: bool | None = None,
    prenan: bool | None = None,
    drift: int | None = None,
    offset: int | None = None,
    **kwargs: dict,
) -> Series:
    """Normalized Average True Range (NATR)

    Normalized Average True Range attempt to normalize the average true range.

    Sources:
        https://www.tradingtechnologies.com/help/x-study/technical-indicator-definitions/normalized-average-true-range-natr/

    Args:
        high (pd.Series): Series of 'high's
        low (pd.Series): Series of 'low's
        close (pd.Series): Series of 'close's
        length (int): The short period. Default: 20
        scalar (float): How much to magnify. Default: 100
        mamode (str): See ``help(ti.ma)``. Default: 'ema'
        talib (bool): If TA Lib is installed and talib is True, Returns
            the TA Lib version. Default: True
        prenan (bool): If True, behave like TA Lib ATR with some initial nan
            based on drift (typically 1). Default: False
        offset (int): How many periods to offset the result. Default: 0

    Kwargs:
        fillna (value, optional): pd.DataFrame.fillna(value)

    Returns:
        pd.Series: New feature
    """
    # Validate
    length = v_pos_default(length, 14)
    _length = length + 1
    high = v_series(high, _length)
    low = v_series(low, _length)
    close = v_series(close, _length)

    if high is None or low is None or close is None:
        return

    scalar = v_scalar(scalar, 100)
    mamode = v_mamode(mamode, "ema")
    mode_tal = v_talib(talib)
    prenan = v_bool(prenan, False)
    drift = v_drift(drift)
    offset = v_offset(offset)

    # Calculate
    if Imports["talib"] and mode_tal:
        from talib import NATR

        natr = NATR(high, low, close, length)
    else:
        natr = (scalar / close) * atr(
            high=high,
            low=low,
            close=close,
            length=length,
            mamode=mamode,
            drift=drift,
            talib=mode_tal,
            prenan=prenan,
            offset=offset,
            **kwargs,
        )

    # Offset
    if offset != 0:
        natr = natr.shift(offset)

    # Fill
    if "fillna" in kwargs:
        natr = natr.fillna(kwargs["fillna"])

    # Name and Category
    natr.name = f"NATR_{length}"
    natr.category = "volatility"

    return natr
