# -*- coding: utf-8 -*-
from pandas import DataFrame, Series

from polars_ti.overlap.sma import sma
from polars_ti.utils import v_offset, v_pos_default, v_series


def rainbow(
    close: Series,
    length: int | None = None,
    num_ribbons: int | None = None,
    offset: int | None = None,
    **kwargs: dict,
) -> DataFrame:
    """Rainbow Charts

    Rainbow Charts use multiple moving averages calculated sequentially, where
    each MA is calculated on the previous MA rather than the price. This
    creates a "rainbow" effect that helps visualize trend strength and
    potential reversals.

    Sources:
        https://www.investopedia.com/articles/trading/06/rainbow.asp
        https://www.prorealcode.com/prorealtime-indicators/rainbow-oscillator/

    Calculation:
        Default Inputs:
            length=2, num_ribbons=10

        MA1 = SMA(close, length)
        MA2 = SMA(MA1, length)
        MA3 = SMA(MA2, length)
        ...
        MA[n] = SMA(MA[n-1], length)

    Args:
        close (pd.Series): Series of 'close's
        length (int): SMA period. Default: 2
        num_ribbons (int): Number of rainbow bands. Default: 10
        offset (int): How many periods to offset the result. Default: 0

    Kwargs:
        fillna (value, optional): pd.DataFrame.fillna(value)

    Returns:
        pd.DataFrame: New features generated.
    """
    # Validate
    length = v_pos_default(length, 2)
    num_ribbons = v_pos_default(num_ribbons, 10)
    close = v_series(close, length * num_ribbons)

    if close is None:
        return

    offset = v_offset(offset)

    # Calculate - Each SMA is calculated on the previous SMA
    ribbons = {}
    prev_sma = close

    for i in range(1, num_ribbons + 1):
        current_sma = sma(prev_sma, length=length)
        ribbons[f"RAINBOW_{i}"] = current_sma
        prev_sma = current_sma

    df = DataFrame(ribbons, index=close.index)

    # Offset
    if offset != 0:
        df = df.shift(offset)

    # Fill
    if "fillna" in kwargs:
        df = df.fillna(kwargs["fillna"])

    # Name and Category
    df.name = f"RAINBOW_{length}_{num_ribbons}"
    df.category = "overlap"

    return df
