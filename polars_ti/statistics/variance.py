# -*- coding: utf-8 -*-
from pandas import Series

from polars_ti.maps import Imports
from polars_ti.utils import v_lowerbound, v_offset, v_series, v_talib


def variance(
    close: Series,
    length: int | None = None,
    ddof: int | None = None,
    talib: bool | None = None,
    offset: int | None = None,
    **kwargs: dict,
) -> Series:
    """Rolling Variance

    Calculates the Variance over a rolling period.

    Args:
        close (pd.Series): Series of 'close's
        length (int): It's period. Default: 30
        ddof (int): Delta Degrees of Freedom.
                    The divisor used in calculations is N - ddof,
                    where N represents the number of elements.
                    The 'talib' argument must be false for 'ddof' to work.
                    Default: 1
        talib (bool): If TA Lib is installed and talib is True, Returns
            the TA Lib version. Note: TA Lib does not have a 'ddof' argument.
            Default: True
        offset (int): How many periods to offset the result. Default: 0

    Kwargs:
        fillna (value, optional): pd.DataFrame.fillna(value)

    Returns:
        pd.Series: New feature generated.
    """
    # Validate
    length = v_lowerbound(length, 1, 30)
    if "min_periods" in kwargs and kwargs["min_periods"] is not None:
        min_periods = int(kwargs["min_periods"])
    else:
        min_periods = length
    close = v_series(close, max(length, min_periods))

    if close is None:
        return

    ddof = int(ddof) if isinstance(ddof, int) and 0 <= ddof < length else 1
    mode_tal = v_talib(talib)
    offset = v_offset(offset)

    # Calculate
    if Imports["talib"] and mode_tal:
        from talib import VAR

        variance = VAR(close, length)
    else:
        variance = close.rolling(length, min_periods=min_periods).var(ddof)

    # Offset
    if offset != 0:
        variance = variance.shift(offset)

    # Fill
    if "fillna" in kwargs:
        variance = variance.fillna(kwargs["fillna"])

    # Name and Category
    variance.name = f"VAR_{length}"
    variance.category = "statistics"

    return variance
