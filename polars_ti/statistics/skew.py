# -*- coding: utf-8 -*-
from pandas import Series

from polars_ti.utils import v_offset, v_pos_default, v_series


def skew(
    close: Series, length: int | None = None, offset: int | None = None, **kwargs: dict
) -> Series:
    """Rolling Skew

    Calculates the Skew over a rolling period.

    WARNING: This function may leak future data when used for machine learning.
        Setting lookahead=False does not currently prevent leakage.
        See https://github.com/CMobley7/polars-ti/issues/667.

    Args:
        close (pd.Series): Series of 'close's
        length (int): It's period. Default: 30
        offset (int): How many periods to offset the result. Default: 0

    Kwargs:
        fillna (value, optional): pd.DataFrame.fillna(value)

    Returns:
        pd.Series: New feature generated.
    """
    # Validate
    length = v_pos_default(length, 30)
    if "min_periods" in kwargs and kwargs["min_periods"] is not None:
        min_periods = int(kwargs["min_periods"])
    else:
        min_periods = length
    close = v_series(close, max(length, min_periods))

    if close is None:
        return

    offset = v_offset(offset)

    # Calculate
    skew = close.rolling(length, min_periods=min_periods).skew()

    # Offset
    if offset != 0:
        skew = skew.shift(offset)

    # Fill
    if "fillna" in kwargs:
        skew = skew.fillna(kwargs["fillna"])

    # Name and Category
    skew.name = f"SKEW_{length}"
    skew.category = "statistics"

    return skew
