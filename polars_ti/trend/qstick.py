# -*- coding: utf-8 -*-
from pandas import Series

from polars_ti.ma import ma
from polars_ti.utils import non_zero_range, v_mamode, v_offset, v_pos_default, v_series


def qstick(
    open_: Series,
    close: Series,
    length: int | None = None,
    mamode: str | None = None,
    offset: int | None = None,
    **kwargs: dict,
) -> Series:
    """Q Stick

    The Q Stick indicator, developed by Tushar Chande, attempts to quantify
    and identify trends in candlestick charts.

    Sources:
        https://library.tradingtechnologies.com/trade/chrt-ti-qstick.html

    Args:
        open (pd.Series): Series of 'open's
        close (pd.Series): Series of 'close's
        length (int): It's period. Default: 1
        mamode (str): See ``help(ti.ma)``. Default: "sma"
        offset (int): How many periods to offset the result. Default: 0

    Kwargs:
        fillna (value, optional): pd.DataFrame.fillna(value)

    Returns:
        pd.Series: New feature generated.
    """
    # Validate
    length = v_pos_default(length, 10)
    open_ = v_series(open_, length)
    close = v_series(close, length)

    if open_ is None or close is None:
        return

    mamode = v_mamode(mamode, "sma")
    offset = v_offset(offset)

    # Calculate
    diff = non_zero_range(close, open_)
    qstick = ma(mamode, diff, length=length, **kwargs)

    # Offset
    if offset != 0:
        qstick = qstick.shift(offset)

    # Fill
    if "fillna" in kwargs:
        qstick = qstick.fillna(kwargs["fillna"])

    # Name and Category
    qstick.name = f"QS_{length}"
    qstick.category = "trend"

    return qstick
