# -*- coding: utf-8 -*-
from pandas import Series

from polars_ti.utils import (
    symmetric_triangle,
    v_offset,
    v_pos_default,
    v_series,
    weights,
)


def swma(
    close: Series, length: int | None = None, offset: int | None = None, **kwargs: dict
) -> Series:
    """Symmetric Weighted Moving Average (SWMA)

    Symmetric Weighted Moving Average where weights are based on a symmetric
    triangle.  For example: n=3 -> [1, 2, 1], n=4 -> [1, 2, 2, 1], etc...
    This moving average has variable length in contrast to TradingView's
    fixed length of 4.

    Source:
        https://www.tradingview.com/study-script-reference/#fun_swma

    Args:
        close (pd.Series): Series of 'close's
        length (int): It's period. Default: 10
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

    offset = v_offset(offset)

    # Calculate
    triangle = symmetric_triangle(length, weighted=True)
    swma = close.rolling(length, min_periods=length).apply(weights(triangle), raw=True)

    # Offset
    if offset != 0:
        swma = swma.shift(offset)

    # Fill
    if "fillna" in kwargs:
        swma = swma.fillna(kwargs["fillna"])

    # Name and Category
    swma.name = f"SWMA_{length}"
    swma.category = "overlap"

    return swma
