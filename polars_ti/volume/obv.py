# -*- coding: utf-8 -*-
from pandas import Series

from polars_ti.maps import Imports
from polars_ti.utils import signed_series, v_offset, v_series, v_talib


def obv(
    close: Series,
    volume: Series,
    talib: bool | None = None,
    offset: int | None = None,
    **kwargs: dict,
) -> Series:
    """On Balance Volume (OBV)

    On Balance Volume is a cumulative indicator to measure buying and selling
    pressure.

    Sources:
        https://www.tradingview.com/wiki/On_Balance_Volume_(OBV)
        https://www.tradingtechnologies.com/help/x-study/technical-indicator-definitions/on-balance-volume-obv/
        https://www.motivewave.com/studies/on_balance_volume.htm

    Args:
        close (pd.Series): Series of 'close's
        volume (pd.Series): Series of 'volume's
        talib (bool): If TA Lib is installed and talib is True, Returns
            the TA Lib version. Default: True
        offset (int): How many periods to offset the result. Default: 0

    Kwargs:
        fillna (value, optional): pd.DataFrame.fillna(value)

    Returns:
        pd.Series: New feature generated.
    """
    # Validate
    _length = 1
    close = v_series(close, _length)
    volume = v_series(volume, _length)

    if close is None or volume is None:
        return

    mode_tal = v_talib(talib)
    offset = v_offset(offset)

    # Calculate
    if Imports["talib"] and mode_tal:
        from talib import OBV

        obv = OBV(close, volume)
    else:
        sv = signed_series(close, initial=1) * volume
        obv = sv.cumsum()

    # Offset
    if offset != 0:
        obv = obv.shift(offset)

    # Fill
    if "fillna" in kwargs:
        obv = obv.fillna(kwargs["fillna"])

    # Name and Category
    obv.name = f"OBV"
    obv.category = "volume"

    return obv
