# -*- coding: utf-8 -*-
from pandas import Series

from polars_ti.utils import (
    fibonacci,
    v_ascending,
    v_offset,
    v_pos_default,
    v_series,
    weights,
)


def fwma(
    close: Series,
    length: int | None = None,
    asc: bool | None = None,
    offset: int | None = None,
    **kwargs: dict,
) -> Series:
    """Fibonacci's Weighted Moving Average (FWMA)

    Fibonacci's Weighted Moving Average is similar to a Weighted Moving
    Average (WMA) where the weights are based on the Fibonacci Sequence.

    Source: Kevin Johnson

    Args:
        close (pd.Series): Series of 'close's
        length (int): It's period. Default: 10
        asc (bool): Recent values weigh more. Default: True
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

    asc = v_ascending(asc)
    offset = v_offset(offset)

    # Calculate
    fibs = fibonacci(n=length, weighted=True)
    fwma = close.rolling(length, min_periods=length).apply(weights(fibs), raw=True)

    # Offset
    if offset != 0:
        fwma = fwma.shift(offset)

    # Fill
    if "fillna" in kwargs:
        fwma = fwma.fillna(kwargs["fillna"])

    # Name and Category
    fwma.name = f"FWMA_{length}"
    fwma.category = "overlap"

    return fwma
