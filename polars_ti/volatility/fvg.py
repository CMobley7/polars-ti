# -*- coding: utf-8 -*-
"""Fair Value Gap (FVG) - Volatility Indicator

An FVG occurs when a strong momentum candle creates an imbalance between
the high and low of surrounding candles, forming a price inefficiency that
the market may revisit.

Sources:
    - https://www.fluxcharts.com/articles/Trading-Concepts/Price-Action/Inversion-Fair-Value-Gaps
    - https://capital.com/fair-value-gap-in-trading
"""

from __future__ import annotations

from numba import njit
from numpy import float64, full, nan
from pandas import DataFrame, Series

from polars_ti._typing import Array, DictLike, Int, IntFloat


@njit(cache=True)
def nb_fvg(
    np_open: Array,
    np_high: Array,
    np_low: Array,
    np_close: Array,
    min_gap: float64,
) -> tuple[Array, Array, Array]:
    """Numba-optimized FVG calculation.

    Identifies Fair Value Gaps using a 3-candle pattern:
    - Bullish FVG: low[i+1] > high[i-1] during an up candle
    - Bearish FVG: high[i+1] < low[i-1] during a down candle
    """
    n = np_open.size

    fvg_high = full(n, nan)
    fvg_low = full(n, nan)
    fvg_type = full(n, nan)

    for i in range(1, n - 1):
        # Bullish candle (close > open) with upward gap
        if np_close[i] > np_open[i]:
            gap = np_low[i + 1] - np_high[i - 1]
            if gap > min_gap * np_close[i]:
                fvg_low[i] = np_high[i - 1]
                fvg_high[i] = np_low[i + 1]
                fvg_type[i] = 1.0  # bullish

        # Bearish candle (close < open) with downward gap
        elif np_close[i] < np_open[i]:
            gap = np_low[i - 1] - np_high[i + 1]
            if gap > min_gap * np_close[i]:
                fvg_low[i] = np_high[i + 1]
                fvg_high[i] = np_low[i - 1]
                fvg_type[i] = -1.0  # bearish

    return fvg_high, fvg_low, fvg_type


def fvg(
    open_: Series,
    high: Series,
    low: Series,
    close: Series,
    min_gap: IntFloat | None = None,
    offset: Int | None = None,
    **kwargs: DictLike,
) -> DataFrame | None:
    """Fair Value Gap (FVG)

    An FVG occurs when a strong momentum candle creates an imbalance between
    the high and low of surrounding candles, forming a price inefficiency that
    the market may revisit. It is a core component of Smart Money Concepts (SMC).

    Three-candle pattern:
    - Bullish FVG: Third candle's low > First candle's high
    - Bearish FVG: Third candle's high < First candle's low

    Sources:
        https://www.fluxcharts.com/articles/Trading-Concepts/Price-Action/Inversion-Fair-Value-Gaps
        https://capital.com/fair-value-gap-in-trading

    Args:
        open_ (pd.Series): Series of 'open' prices
        high (pd.Series): Series of 'high' prices
        low (pd.Series): Series of 'low' prices
        close (pd.Series): Series of 'close' prices
        min_gap (int | float | None): Minimum percentage gap size. Default: 0
        offset (int | None): How many periods to offset the result. Default: 0

    Kwargs:
        fillna (value, optional): pd.DataFrame.fillna(value)

    Returns:
        pd.DataFrame: Columns FVGh (high), FVGl (low), FVGt (type: 1=bullish, -1=bearish)
    """
    # Late imports to avoid circular dependency
    from polars_ti.utils import v_offset, v_pos_default, v_series

    # Validate
    _length = 3
    open_ = v_series(open_, _length)
    high = v_series(high, _length)
    low = v_series(low, _length)
    close = v_series(close, _length)

    if open_ is None or high is None or low is None or close is None:
        return None

    min_gap = v_pos_default(min_gap, 0)
    min_gap_pct = min_gap / 100.0
    offset = v_offset(offset)

    # Calculate using Numba kernel
    np_open = open_.to_numpy()
    np_high = high.to_numpy()
    np_low = low.to_numpy()
    np_close = close.to_numpy()

    fvg_high, fvg_low, fvg_type = nb_fvg(
        np_open=np_open,
        np_high=np_high,
        np_low=np_low,
        np_close=np_close,
        min_gap=min_gap_pct,
    )

    # Convert to Series for offset support
    fvg_high_s = Series(fvg_high, index=high.index)
    fvg_low_s = Series(fvg_low, index=low.index)
    fvg_type_s = Series(fvg_type, index=close.index)

    # Offset
    if offset != 0:
        fvg_high_s = fvg_high_s.shift(offset)
        fvg_low_s = fvg_low_s.shift(offset)
        fvg_type_s = fvg_type_s.shift(offset)

    # Fill
    if "fillna" in kwargs:
        fill_val = kwargs["fillna"]
        fvg_high_s = fvg_high_s.fillna(fill_val)
        fvg_low_s = fvg_low_s.fillna(fill_val)
        fvg_type_s = fvg_type_s.fillna(fill_val)

    _props = f"_{min_gap}"
    data = {
        f"FVGh{_props}": fvg_high_s,
        f"FVGl{_props}": fvg_low_s,
        f"FVGt{_props}": fvg_type_s,
    }
    df = DataFrame(data, index=high.index)
    df.name = f"FVG{_props}"
    df.category = "volatility"

    return df
