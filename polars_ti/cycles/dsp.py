# -*- coding: utf-8 -*-
from pandas import Series

from polars_ti.overlap.ema import ema
from polars_ti.utils import v_offset, v_pos_default, v_series


def dsp(
    close: Series,
    length: int | None = None,
    offset: int | None = None,
    **kwargs: dict,
) -> Series:
    """Detrended Synthetic Price (DSP)

    Detrended Synthetic Price removes the trend component from price data to
    reveal the cyclical component. It's useful for cycle analysis and
    identifying periodic patterns in price movement.

    Sources:
        https://www.mesasoftware.com/papers/TheInverseFisherTransform.pdf
        Cycle Analytics for Traders by John F. Ehlers

    Calculation:
        Default Inputs:
            length=14

        EMA = EMA(close, length)
        DSP = close - EMA

    Args:
        close (pd.Series): Series of 'close's
        length (int): The EMA period. Default: 14
        offset (int): How many periods to offset the result. Default: 0

    Kwargs:
        fillna (value, optional): pd.DataFrame.fillna(value)

    Returns:
        pd.Series: New feature generated.
    """
    # Validate
    length = v_pos_default(length, 14)
    close = v_series(close, length)

    if close is None:
        return

    offset = v_offset(offset)

    # Calculate
    ema_value = ema(close, length=length)
    dsp = close - ema_value

    # Offset
    if offset != 0:
        dsp = dsp.shift(offset)

    # Fill
    if "fillna" in kwargs:
        dsp = dsp.fillna(kwargs["fillna"])

    # Name and Category
    dsp.name = f"DSP_{length}"
    dsp.category = "cycles"

    return dsp
