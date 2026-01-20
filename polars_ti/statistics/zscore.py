# -*- coding: utf-8 -*-
from pandas import Series

from polars_ti.overlap import sma
from polars_ti.statistics import stdev
from polars_ti.utils import v_lowerbound, v_offset, v_series


def zscore(
    close: Series,
    length: int | None = None,
    std: int | float | None = None,
    offset: int | None = None,
    **kwargs: dict,
) -> Series:
    """Rolling Z Score

    Calculates the Z Score over a rolling period.

    Args:
        close (pd.Series): Series of 'close's
        length (int): It's period. Default: 30
        std (float): It's period. Default: 1
        offset (int): How many periods to offset the result. Default: 0

    Kwargs:
        fillna (value, optional): pd.DataFrame.fillna(value)

    Returns:
        pd.Series: New feature generated.
    """
    # Validate
    length = v_lowerbound(length, 1, 30)
    close = v_series(close, length)

    if close is None:
        return

    std = v_lowerbound(std, 1, 1.0)
    offset = v_offset(offset)

    # Calculate
    std *= stdev(close=close, length=length, **kwargs)
    mean = sma(close=close, length=length, **kwargs)
    zscore = (close - mean) / std

    # Offset
    if offset != 0:
        zscore = zscore.shift(offset)

    # Fill
    if "fillna" in kwargs:
        zscore = zscore.fillna(kwargs["fillna"])

    # Name and Category
    zscore.name = f"ZS_{length}"
    zscore.category = "statistics"

    return zscore
