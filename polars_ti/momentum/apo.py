# -*- coding: utf-8 -*-
from pandas import Series

from polars_ti.ma import ma
from polars_ti.maps import Imports
from polars_ti.utils import tal_ma, v_mamode, v_offset, v_pos_default, v_series, v_talib


def apo(
    close: Series,
    fast: int | None = None,
    slow: int | None = None,
    mamode: str | None = None,
    talib: bool | None = None,
    offset: int | None = None,
    **kwargs: dict,
) -> Series:
    """Absolute Price Oscillator (APO)

    The Absolute Price Oscillator is an indicator used to measure a
    security's momentum.  It is simply the difference of two Exponential
    Moving Averages (EMA) of two different periods. Note: APO and MACD lines
    are equivalent.

    Sources:
        https://www.tradingtechnologies.com/xtrader-help/x-study/technical-indicator-definitions/absolute-price-oscillator-apo/

    Args:
        close (pd.Series): Series of 'close's
        fast (int): The short period. Default: 12
        slow (int): The long period. Default: 26
        mamode (str): See ``help(ti.ma)``. Default: 'sma'
        talib (bool): If TA Lib is installed and talib is True, Returns
            the TA Lib version. Default: True
        offset (int): How many periods to offset the result. Default: 0

    Kwargs:
        fillna (value, optional): pd.DataFrame.fillna(value)

    Returns:
        pd.Series: New feature generated.
    """
    # Validate
    fast = v_pos_default(fast, 12)
    slow = v_pos_default(slow, 26)
    if slow < fast:
        fast, slow = slow, fast
    close = v_series(close, max(fast, slow))

    if close is None:
        return

    mamode = v_mamode(mamode, "sma")
    mode_tal = v_talib(talib)
    offset = v_offset(offset)

    # Calculate
    if Imports["talib"] and mode_tal:
        from talib import APO

        apo = APO(close, fast, slow, tal_ma(mamode))
    else:
        fastma = ma(mamode, close, length=fast, talib=mode_tal)
        slowma = ma(mamode, close, length=slow, talib=mode_tal)
        apo = fastma - slowma

    # Offset
    if offset != 0:
        apo = apo.shift(offset)

    # Fill
    if "fillna" in kwargs:
        apo = apo.fillna(kwargs["fillna"])
    # Name and Category
    apo.name = f"APO_{fast}_{slow}"
    apo.category = "momentum"

    return apo
