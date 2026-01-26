# -*- coding: utf-8 -*-
from numpy import nan as npNaN
from pandas import DataFrame, Series

from polars_ti.utils import v_offset, v_pos_default, v_series


def avwap(
    high: Series,
    low: Series,
    close: Series,
    volume: Series,
    left_strength: int | None = None,
    right_strength: int | None = None,
    offset: int | None = None,
    **kwargs: dict,
) -> DataFrame:
    """Anchored Volume Weighted Average Price (AVWAP)

    Anchored VWAP is calculated from specific pivot points in the data,
    providing insights into price action around significant market events.
    It creates two AVWAP lines: one anchored from high pivots and one from
    low pivots.

    Sources:
        https://www.tradingview.com/support/solutions/43000502018-anchored-vwap/

    Args:
        high (pd.Series): Series of 'high's
        low (pd.Series): Series of 'low's
        close (pd.Series): Series of 'close's
        volume (pd.Series): Series of 'volume's
        left_strength (int): Bars back for pivot detection. Default: 5
        right_strength (int): Bars forward for pivot detection. Default: 5
        offset (int): How many periods to offset the result. Default: 0

    Kwargs:
        fillna (value, optional): pd.DataFrame.fillna(value)

    Returns:
        pd.DataFrame: AVWAP_HIGH, AVWAP_LOW columns
    """
    # Validate
    left_strength = v_pos_default(left_strength, 5)
    right_strength = v_pos_default(right_strength, 5)
    _length = left_strength + right_strength + 1
    high = v_series(high, _length)
    low = v_series(low, _length)
    close = v_series(close, _length)
    volume = v_series(volume, _length)

    if high is None or low is None or close is None or volume is None:
        return

    offset = v_offset(offset)

    # Create working DataFrame
    df = DataFrame(
        {
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )

    # Calculate pivot points
    pivot_highs = _find_pivots(high, left_strength, right_strength, "high")
    pivot_lows = _find_pivots(low, left_strength, right_strength, "low")

    # Initialize AVWAP columns
    avwap_high = Series(npNaN, index=close.index)
    avwap_low = Series(npNaN, index=close.index)

    # Track last pivot positions
    last_pivot_high_idx = 0
    last_pivot_low_idx = 0

    # Calculate segmented AVWAP
    for i, idx in enumerate(df.index):
        if idx in pivot_highs:
            last_pivot_high_idx = i
        if idx in pivot_lows:
            last_pivot_low_idx = i

        # Calculate AVWAP from last high pivot
        if last_pivot_high_idx <= i:
            segment = df.iloc[last_pivot_high_idx : i + 1]
            vp = (segment["volume"] * segment["close"]).sum()
            cv = segment["volume"].sum()
            avwap_high.iloc[i] = vp / cv if cv > 0 else npNaN

        # Calculate AVWAP from last low pivot
        if last_pivot_low_idx <= i:
            segment = df.iloc[last_pivot_low_idx : i + 1]
            vp = (segment["volume"] * segment["close"]).sum()
            cv = segment["volume"].sum()
            avwap_low.iloc[i] = vp / cv if cv > 0 else npNaN

    # Offset
    if offset != 0:
        avwap_high = avwap_high.shift(offset)
        avwap_low = avwap_low.shift(offset)

    # Fill
    if "fillna" in kwargs:
        avwap_high = avwap_high.fillna(kwargs["fillna"])
        avwap_low = avwap_low.fillna(kwargs["fillna"])

    # Name and Category
    _props = f"_{left_strength}_{right_strength}"
    avwap_high.name = f"AVWAPH{_props}"
    avwap_low.name = f"AVWAPL{_props}"

    data = {avwap_high.name: avwap_high, avwap_low.name: avwap_low}
    result = DataFrame(data, index=close.index)
    result.name = f"AVWAP{_props}"
    result.category = "volume"

    return result


def _find_pivots(
    data: Series,
    left_strength: int,
    right_strength: int,
    pivot_type: str,
) -> list:
    """Find pivot high/low points in a series.

    Args:
        data: Price series (high for pivot highs, low for pivot lows)
        left_strength: Bars to the left to check
        right_strength: Bars to the right to check
        pivot_type: 'high' or 'low'

    Returns:
        List of index values where pivots occur
    """
    pivots = []
    data_values = data.values
    m = len(data)

    for i in range(left_strength, m - right_strength):
        window = data_values[i - left_strength : i + right_strength + 1]
        if pivot_type == "high" and data_values[i] == window.max():
            pivots.append(data.index[i])
        elif pivot_type == "low" and data_values[i] == window.min():
            pivots.append(data.index[i])

    return pivots
