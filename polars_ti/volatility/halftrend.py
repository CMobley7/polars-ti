# -*- coding: utf-8 -*-
from numba import njit
from numpy import empty, float64, full, int32, isnan, nan, nanmax
from pandas import DataFrame, Series

from polars_ti.overlap import sma
from polars_ti.utils import v_offset, v_pos_default, v_series


@njit(cache=True)
def nb_halftrend(
    high,
    low,
    close,
    atr_arr,
    high_ma,
    low_ma,
    highest_bars,
    lowest_bars,
    atr_length,
    channel_deviation,
    smoothing,
):
    """Numba-optimized HalfTrend calculation loop.

    Returns tuple of 6 arrays: atr_high, atr_low, atr_close, direction, arr_up, arr_down
    """
    n = len(close)

    # Initialize output arrays
    atr_high_series = full(n, nan, dtype=float64)
    atr_low_series = full(n, nan, dtype=float64)
    atr_close_series = full(n, nan, dtype=float64)
    direction_series = full(n, nan, dtype=float64)
    arr_up = full(n, nan, dtype=float64)
    arr_down = full(n, nan, dtype=float64)

    # Initialize state variables
    trend = 0  # 0 = long, 1 = short
    up = low[atr_length] if atr_length < n else low[0]
    down = high[atr_length] if atr_length < n else high[0]
    max_low_price = low[atr_length] if atr_length < n else low[0]
    min_high_price = high[atr_length] if atr_length < n else high[0]

    # Initial trend direction
    if atr_length < n and close[atr_length] > low[atr_length]:
        trend = 1

    # Cap ATR to avoid extreme values
    atr_cap = nanmax(atr_arr[: min(atr_length * 2, n)]) * 0.5 if n > 0 else 1.0

    for i in range(atr_length + 1, n):
        atr_raw = atr_arr[i]
        if isnan(atr_raw):
            continue

        atr2 = min(atr_raw / 2.0, atr_cap)
        dev = channel_deviation * atr2

        high_price = highest_bars[i]
        low_price = lowest_bars[i]

        # Trend switching logic
        if trend == 0:  # Currently long
            max_low_price = max(max_low_price, low_price)
            if high_ma[i] < (max_low_price - dev) and close[i] < close[i - 1]:
                trend = 1  # Switch to short
                min_high_price = high_price
        else:  # Currently short
            min_high_price = min(min_high_price, high_price)
            if low_ma[i] > (min_high_price + dev) and close[i] > close[i - 1]:
                trend = 0  # Switch to long
                max_low_price = low_price

        # Calculate smoothed values and ATR bands
        if trend == 0:  # Long
            if isnan(up):
                up = max_low_price
            else:
                up = smoothing * max_low_price + (1 - smoothing) * up

            atr_high = up + dev
            atr_low = up - dev
            arr_up[i] = up
            atr_close_series[i] = up
            direction_series[i] = 0  # long
        else:  # Short
            if isnan(down):
                down = min_high_price
            else:
                down = smoothing * min_high_price + (1 - smoothing) * down

            atr_high = down + dev
            atr_low = down - dev
            arr_down[i] = down
            atr_close_series[i] = down
            direction_series[i] = 1  # short

        atr_high_series[i] = atr_high
        atr_low_series[i] = atr_low

    return (
        atr_high_series,
        atr_low_series,
        atr_close_series,
        direction_series,
        arr_up,
        arr_down,
    )


def halftrend(
    high: Series,
    low: Series,
    close: Series,
    atr_length: int | None = None,
    amplitude: int | None = None,
    channel_deviation: int | None = None,
    smoothing: float | None = None,
    offset: int | None = None,
    **kwargs: dict,
) -> DataFrame:
    """HalfTrend Indicator

    HalfTrend is a trend-following indicator that uses ATR-based channels
    with smoothed trend detection. It helps identify trend direction and
    provides dynamic support/resistance levels.

    The indicator works by:
    1. Calculating ATR-based deviation bands
    2. Detecting trend changes using rolling high/low extremes
    3. Smoothing the trend line with exponential weighting
    4. Plotting upper and lower ATR channels around the trend line

    Sources:
        https://www.tradingview.com/script/U1SJ8ubc-HalfTrend/
        https://www.mql5.com/en/code/viewcode/21829/198620/halftrend.mq4

    Args:
        high (pd.Series): Series of 'high's
        low (pd.Series): Series of 'low's
        close (pd.Series): Series of 'close's
        atr_length (int): ATR period. Default: 14
        amplitude (int): Rolling high/low lookback. Default: 2
        channel_deviation (int): ATR multiplier for bands. Default: 2
        smoothing (float): Smoothing factor (0 to 1). Default: 0.3
        offset (int): How many periods to offset the result. Default: 0

    Kwargs:
        fillna (value, optional): pd.DataFrame.fillna(value)

    Returns:
        pd.DataFrame: HT_atr_high, HT_atr_low, HT_close, HT_direction,
                      HT_arr_up, HT_arr_down columns
    """
    # Validate
    atr_length = v_pos_default(atr_length, 14)
    amplitude = v_pos_default(amplitude, 2)
    channel_deviation = v_pos_default(channel_deviation, 2)
    smoothing = smoothing if smoothing is not None else 0.3
    _length = max(atr_length, amplitude, channel_deviation) + 1

    high = v_series(high, _length)
    low = v_series(low, _length)
    close = v_series(close, _length)

    if high is None or low is None or close is None:
        return

    offset = v_offset(offset)

    # Calculate prerequisite series
    # Late import to avoid circular dependency
    from polars_ti.volatility.atr import atr

    atr_series = atr(high, low, close, length=atr_length)
    if atr_series is None:
        return

    high_ma = sma(high, length=amplitude)
    low_ma = sma(low, length=amplitude)
    highest_bars = high.rolling(amplitude, min_periods=1).max()
    lowest_bars = low.rolling(amplitude, min_periods=1).min()

    # Convert to numpy for Numba
    np_high = high.to_numpy().astype(float64)
    np_low = low.to_numpy().astype(float64)
    np_close = close.to_numpy().astype(float64)
    np_atr = atr_series.to_numpy().astype(float64)
    np_high_ma = high_ma.to_numpy().astype(float64)
    np_low_ma = low_ma.to_numpy().astype(float64)
    np_highest = highest_bars.to_numpy().astype(float64)
    np_lowest = lowest_bars.to_numpy().astype(float64)

    # Run Numba-optimized calculation
    results = nb_halftrend(
        np_high,
        np_low,
        np_close,
        np_atr,
        np_high_ma,
        np_low_ma,
        np_highest,
        np_lowest,
        atr_length,
        channel_deviation,
        float64(smoothing),
    )

    atr_high_arr, atr_low_arr, atr_close_arr, direction_arr, arr_up, arr_down = results

    # Convert direction to string labels
    direction_labels = Series(
        ["long" if d == 0 else "short" if d == 1 else None for d in direction_arr],
        index=close.index,
    )

    # Build DataFrame
    _props = f"_{atr_length}_{amplitude}_{channel_deviation}"
    data = {
        f"HT_atr_high{_props}": Series(atr_high_arr, index=close.index),
        f"HT_atr_low{_props}": Series(atr_low_arr, index=close.index),
        f"HT_close{_props}": Series(atr_close_arr, index=close.index),
        f"HT_direction{_props}": direction_labels,
        f"HT_arr_up{_props}": Series(arr_up, index=close.index),
        f"HT_arr_down{_props}": Series(arr_down, index=close.index),
    }
    df = DataFrame(data, index=close.index)

    # Offset
    if offset != 0:
        df = df.shift(offset)

    # Fill
    if "fillna" in kwargs:
        df = df.fillna(kwargs["fillna"])

    # Name and Category
    df.name = f"HT{_props}"
    df.category = "volatility"

    return df
