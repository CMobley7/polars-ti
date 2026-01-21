# -*- coding: utf-8 -*-
from numba import njit
from pandas import Series

from polars_ti.maps import Imports
from polars_ti.utils import nb_idiff, v_offset, v_pos_default, v_series, v_talib


@njit(cache=True)
def nb_mom(x, n):
    return nb_idiff(x, n)


def mom(
    close: Series,
    length: int | None = None,
    talib: bool | None = None,
    offset: int | None = None,
    **kwargs: dict,
) -> Series:
    """Momentum (MOM)

    Momentum is an indicator used to measure a security's speed
    (or strength) of movement or simply the change in price.

    Sources:
        http://www.onlinetradingconcepts.com/TechnicalAnalysis/Momentum.html

    Args:
        close (pd.Series): Series of 'close's
        length (int): It's period. Default: 1
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
    close = v_series(close, length + 1)

    if close is None:
        return

    mode_tal = v_talib(talib)
    offset = v_offset(offset)

    # Calculate
    if Imports["talib"] and mode_tal:
        from talib import MOM

        mom = MOM(close, length)
    else:
        np_close = close.values
        _mom = nb_mom(np_close, length)
        mom = Series(_mom, index=close.index)

    # Offset
    if offset != 0:
        mom = mom.shift(offset)

    # Fill
    if "fillna" in kwargs:
        mom = mom.fillna(kwargs["fillna"])

    # Name and Category
    mom.name = f"MOM_{length}"
    mom.category = "momentum"

    return mom
