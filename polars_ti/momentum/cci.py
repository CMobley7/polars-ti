# -*- coding: utf-8 -*-
from pandas import Series

from polars_ti.maps import Imports
from polars_ti.overlap import hlc3, sma
from polars_ti.statistics import mad
from polars_ti.utils import v_offset, v_pos_default, v_series, v_talib


def cci(
    high: Series,
    low: Series,
    close: Series,
    length: int | None = None,
    c: int | float | None = None,
    talib: bool | None = None,
    offset: int | None = None,
    **kwargs: dict,
) -> Series:
    """Commodity Channel Index (CCI)

    Commodity Channel Index is a momentum oscillator used to primarily
    identify overbought and oversold levels relative to a mean.

    Sources:
        https://www.tradingview.com/wiki/Commodity_Channel_Index_(CCI)

    Args:
        high (pd.Series): Series of 'high's
        low (pd.Series): Series of 'low's
        close (pd.Series): Series of 'close's
        length (int): It's period. Default: 14
        c (float): Scaling Constant. Default: 0.015
        talib (bool): If TA Lib is installed and talib is True, Returns
            the TA Lib version. Default: True
        offset (int): How many periods to offset the result. Default: 0

    Kwargs:
        fillna (value, optional): pd.DataFrame.fillna(value)

    Returns:
        pd.Series: New feature generated.
    """
    # Validate
    length = v_pos_default(length, 14)
    high = v_series(high, length)
    low = v_series(low, length)
    close = v_series(close, length)

    if high is None or low is None or close is None:
        return

    c = v_pos_default(c, 0.015)
    mode_tal = v_talib(talib)
    offset = v_offset(offset)

    # Calculate
    if Imports["talib"] and mode_tal:
        from talib import CCI

        cci = CCI(high, low, close, length)
    else:
        typical_price = hlc3(high=high, low=low, close=close, talib=mode_tal)
        mean_typical_price = sma(typical_price, length=length, talib=mode_tal)
        mad_typical_price = mad(typical_price, length=length)

        cci = (typical_price - mean_typical_price) / (c * mad_typical_price)
        # Protect against divide-by-zero when mad is near zero
        cci[mad_typical_price < 1e-8] = 0

    # Offset
    if offset != 0:
        cci = cci.shift(offset)

    # Fill
    if "fillna" in kwargs:
        cci = cci.fillna(kwargs["fillna"])

    # Name and Category
    cci.name = f"CCI_{length}_{c}"
    cci.category = "momentum"

    return cci
