# -*- coding: utf-8 -*-
from pandas import DataFrame, Series

from polars_ti.utils import v_offset, v_pos_default, v_series
from polars_ti.volume import vwma


def vwmacd(
    close: Series,
    volume: Series,
    fast: int | None = None,
    slow: int | None = None,
    signal: int | None = None,
    offset: int | None = None,
    **kwargs: dict,
) -> DataFrame:
    """Volume Weighted MACD (VWMACD)

    Volume Weighted MACD is a variation of the traditional MACD that
    incorporates volume into the calculation. It uses Volume Weighted Moving
    Averages (VWMA) instead of EMAs to give more weight to periods with
    higher volume.

    Sources:
        https://www.tradingview.com/script/NUs1Y5V7-Volume-Weighted-MACD/
        Technical Analysis Using Multiple Timeframes by Brian Shannon

    Calculation:
        Default Inputs:
            fast=12, slow=26, signal=9

        FastVWMA = VWMA(close, volume, fast)
        SlowVWMA = VWMA(close, volume, slow)

        VWMACD = FastVWMA - SlowVWMA
        Signal = VWMA(VWMACD, volume, signal)
        Histogram = VWMACD - Signal

    Args:
        close (pd.Series): Series of 'close's
        volume (pd.Series): Series of 'volume's
        fast (int): The fast period. Default: 12
        slow (int): The slow period. Default: 26
        signal (int): The signal period. Default: 9
        offset (int): How many periods to offset the result. Default: 0

    Kwargs:
        fillna (value, optional): pd.DataFrame.fillna(value)

    Returns:
        pd.DataFrame: VWMACD, Signal, and Histogram columns.
    """
    # Validate
    fast = v_pos_default(fast, 12)
    slow = v_pos_default(slow, 26)
    signal_length = v_pos_default(signal, 9)

    if slow < fast:
        fast, slow = slow, fast

    _length = max(fast, slow, signal_length)
    close = v_series(close, _length)
    volume = v_series(volume, _length)

    if close is None or volume is None:
        return

    offset = v_offset(offset)

    # Calculate
    fast_vwma = vwma(close, volume, length=fast)
    slow_vwma = vwma(close, volume, length=slow)

    vwmacd_line = fast_vwma - slow_vwma
    signal_line = vwma(vwmacd_line, volume, length=signal_length)
    histogram = vwmacd_line - signal_line

    # Offset
    if offset != 0:
        vwmacd_line = vwmacd_line.shift(offset)
        signal_line = signal_line.shift(offset)
        histogram = histogram.shift(offset)

    # Fill
    if "fillna" in kwargs:
        vwmacd_line = vwmacd_line.fillna(kwargs["fillna"])
        signal_line = signal_line.fillna(kwargs["fillna"])
        histogram = histogram.fillna(kwargs["fillna"])

    # Name and Category
    _props = f"_{fast}_{slow}_{signal_length}"
    vwmacd_line.name = f"VWMACD{_props}"
    signal_line.name = f"VWMACDs{_props}"
    histogram.name = f"VWMACDh{_props}"
    vwmacd_line.category = signal_line.category = histogram.category = "momentum"

    data = {
        vwmacd_line.name: vwmacd_line,
        histogram.name: histogram,
        signal_line.name: signal_line,
    }
    df = DataFrame(data, index=close.index)
    df.name = f"VWMACD{_props}"
    df.category = "momentum"

    return df
