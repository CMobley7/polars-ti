# -*- coding: utf-8 -*-
from numba import njit
from numpy import empty, float64, isnan, nan
from pandas import Series

from polars_ti.ma import ma
from polars_ti.utils import v_mamode, v_offset, v_pos_default, v_series


@njit(cache=True)
def nb_trama(close, tc):
    """Numba-optimized TRAMA calculation loop.

    Args:
        close: Array of closing prices
        tc: Array of trend coefficient values (squared SMA of trend signal)

    Returns:
        Array of TRAMA values
    """
    n = len(close)
    result = empty(n, dtype=float64)
    result[0] = close[0]

    for i in range(1, n):
        curr_tc = tc[i] if not isnan(tc[i]) else 0.0
        result[i] = result[i - 1] + curr_tc * (close[i] - result[i - 1])

    return result


def trama(
    close: Series,
    length: int | None = None,
    mamode: str | None = None,
    offset: int | None = None,
    **kwargs: dict,
) -> Series:
    """Trend Regulated Adaptive Moving Average (TRAMA)

    TRAMA is an adaptive moving average that adjusts its smoothing factor
    based on trend detection. It becomes more responsive during trending
    markets and more stable during ranging markets.

    The indicator works by:
    1. Detecting new highs and lows over a rolling period
    2. Creating a trend signal when either a new high or low is made
    3. Squaring the SMA of this signal to create a "trend coefficient"
    4. Using the trend coefficient as the smoothing factor for an adaptive MA

    Sources:
        https://www.tradingview.com/script/xCZx4VrP-TRAMA-Trend-Regulated-Adaptive-Moving-Average/
        Issue #603 in pandas-ta

    Args:
        close (pd.Series): Series of 'close's
        length (int): The lookback period. Default: 10
        mamode (str): See ``help(ti.ma)``. Default: 'sma'
        offset (int): How many periods to offset the result. Default: 0

    Kwargs:
        fillna (value, optional): pd.DataFrame.fillna(value)

    Returns:
        pd.Series: TRAMA values
    """
    # Validate
    length = v_pos_default(length, 10)
    close = v_series(close, length + 1)

    if close is None:
        return

    mamode = v_mamode(mamode, "sma")
    offset = v_offset(offset)

    # Calculate
    # Find rolling highest and lowest values
    highest = close.rolling(length).max()
    lowest = close.rolling(length).min()

    # Detect new highs (diff > 0) and new lows (diff < 0)
    # Use maximum with 0 to get binary signal: 1 if new high/low, 0 otherwise
    hh = (highest.diff() > 0).astype(int)
    ll = (lowest.diff() < 0).astype(int)

    # Trend signal: 1 if either new high or new low, 0 otherwise
    trend_signal = (hh | ll).astype(float)

    # Trend coefficient: squared SMA of trend signal
    tc = ma(mamode, trend_signal, length=length, **kwargs) ** 2

    # Calculate TRAMA using Numba-optimized loop
    np_close = close.to_numpy().astype(float64)
    np_tc = tc.to_numpy().astype(float64)
    trama_arr = nb_trama(np_close, np_tc)
    trama = Series(trama_arr, index=close.index)

    # Offset
    if offset != 0:
        trama = trama.shift(offset)

    # Fill
    if "fillna" in kwargs:
        trama = trama.fillna(kwargs["fillna"])

    # Name and Category
    trama.name = f"TRAMA_{length}"
    trama.category = "trend"

    return trama
