# -*- coding: utf-8 -*-
from numba import njit
from numpy import empty, float64, full, isnan, nan, nanmax


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
    atr_high_series = full(n, nan, dtype=float64)
    atr_low_series = full(n, nan, dtype=float64)
    atr_close_series = full(n, nan, dtype=float64)
    direction_series = full(n, nan, dtype=float64)
    arr_up = full(n, nan, dtype=float64)
    arr_down = full(n, nan, dtype=float64)

    trend = 0
    up = low[atr_length] if atr_length < n else low[0]
    down = high[atr_length] if atr_length < n else high[0]
    max_low_price = low[atr_length] if atr_length < n else low[0]
    min_high_price = high[atr_length] if atr_length < n else high[0]

    if atr_length < n and close[atr_length] > low[atr_length]:
        trend = 1

    atr_cap = nanmax(atr_arr[: min(atr_length * 2, n)]) * 0.5 if n > 0 else 1.0

    for i in range(atr_length + 1, n):
        atr_raw = atr_arr[i]
        if isnan(atr_raw):
            continue
        atr2 = min(atr_raw / 2.0, atr_cap)
        dev = channel_deviation * atr2
        high_price = highest_bars[i]
        low_price = lowest_bars[i]

        if trend == 0:
            max_low_price = max(max_low_price, low_price)
            if high_ma[i] < (max_low_price - dev) and close[i] < close[i - 1]:
                trend = 1
                min_high_price = high_price
        else:
            min_high_price = min(min_high_price, high_price)
            if low_ma[i] > (min_high_price + dev) and close[i] > close[i - 1]:
                trend = 0
                max_low_price = low_price

        if trend == 0:
            if isnan(up):
                up = max_low_price
            else:
                up = smoothing * max_low_price + (1 - smoothing) * up
            atr_high = up + dev
            atr_low = up - dev
            arr_up[i] = up
            atr_close_series[i] = up
            direction_series[i] = 0.0
        else:
            if isnan(down):
                down = min_high_price
            else:
                down = smoothing * min_high_price + (1 - smoothing) * down
            atr_high = down + dev
            atr_low = down - dev
            arr_down[i] = down
            atr_close_series[i] = down
            direction_series[i] = 1.0

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


# =============================================================================
# Polars HalfTrend Implementation (Composition: pl_atr + pl_sma + reuse nb_halftrend)
# =============================================================================
import polars as pl
import numpy as np

from polars_ti._typing import IntoExpr
from polars_ti.utils._validate import v_expr


# NOTE: Reuses nb_halftrend kernel from Pandas section above (lines 10-109)
# That kernel already takes pre-calculated ATR, MAs, and rolling max/min as inputs.


def halftrend(
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    atr_length: int = 14,
    amplitude: int = 2,
    channel_deviation: int = 2,
    smoothing: float = 0.3,
    talib: bool = True,
    offset: int = 0,
) -> pl.Expr:
    """Polars: HalfTrend Indicator

    Uses composition: pl_atr for ATR, pl_sma for MAs, native Polars for
    rolling_max/min. Reuses nb_halftrend kernel from Pandas section.

    Sources:
        https://www.tradingview.com/script/U1SJ8ubc-HalfTrend/

    Args:
        high: Column name or pl.Expr for 'high'
        low: Column name or pl.Expr for 'low'
        close: Column name or pl.Expr for 'close'
        atr_length: ATR period. Default: 14
        amplitude: Rolling high/low lookback. Default: 2
        channel_deviation: ATR multiplier. Default: 2
        smoothing: Smoothing factor. Default: 0.3
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: Struct with atr_high, atr_low, ht_close, direction, arr_up, arr_down
    """
    from polars_ti.volatility.atr import atr
    from polars_ti.overlap.sma import sma

    high_expr = v_expr(high)
    low_expr = v_expr(low)
    close_expr = v_expr(close)

    if high_expr is None or low_expr is None or close_expr is None:
        return None

    _atr_len = atr_length
    _chan_dev = channel_deviation
    _smooth = float(smoothing)
    _offset = offset

    # Use composition for pre-calculations (just like Pandas!)
    # OLD halftrend never propagated talib to its internal atr(), so the golden
    # used TA-Lib ATR in both modes; honour talib here (default True).
    atr_expr = atr(high_expr, low_expr, close_expr, length=atr_length, mamode="rma", talib=talib)
    high_ma_expr = sma(high_expr, length=amplitude)
    low_ma_expr = sma(low_expr, length=amplitude)
    highest_expr = high_expr.rolling_max(window_size=amplitude, min_samples=1)
    lowest_expr = low_expr.rolling_min(window_size=amplitude, min_samples=1)

    _props = f"_{atr_length}_{amplitude}_{channel_deviation}"
    # Field names match the OLD pandas-ta flat columns (folded by the parity
    # engine): HT_atr_high_14_2_2, HT_close_14_2_2, etc.
    _fields = {
        "atr_high": f"HT_atr_high{_props}",
        "atr_low": f"HT_atr_low{_props}",
        "ht_close": f"HT_close{_props}",
        "direction": f"HT_direction{_props}",
        "arr_up": f"HT_arr_up{_props}",
        "arr_down": f"HT_arr_down{_props}",
    }

    def compute_halftrend(struct: pl.Series) -> pl.Series:
        df = struct.struct.unnest()
        np_high = df["_high"].to_numpy().astype(np.float64)
        np_low = df["_low"].to_numpy().astype(np.float64)
        np_close = df["_close"].to_numpy().astype(np.float64)
        np_atr = df["_atr"].to_numpy().astype(np.float64)
        np_high_ma = df["_high_ma"].to_numpy().astype(np.float64)
        np_low_ma = df["_low_ma"].to_numpy().astype(np.float64)
        np_highest = df["_highest"].to_numpy().astype(np.float64)
        np_lowest = df["_lowest"].to_numpy().astype(np.float64)

        # Reuse nb_halftrend kernel from Pandas section!
        results = nb_halftrend(
            np_high,
            np_low,
            np_close,
            np_atr,
            np_high_ma,
            np_low_ma,
            np_highest,
            np_lowest,
            _atr_len,
            _chan_dev,
            _smooth,
        )
        atr_high, atr_low, ht_close, direction, arr_up, arr_down = results

        if _offset != 0:
            atr_high = np.roll(atr_high, _offset)
            atr_low = np.roll(atr_low, _offset)
            ht_close = np.roll(ht_close, _offset)
            direction = np.roll(direction, _offset)
            arr_up = np.roll(arr_up, _offset)
            arr_down = np.roll(arr_down, _offset)
            if _offset > 0:
                atr_high[:_offset] = np.nan
                atr_low[:_offset] = np.nan
                ht_close[:_offset] = np.nan
                direction[:_offset] = np.nan
                arr_up[:_offset] = np.nan
                arr_down[:_offset] = np.nan

        # Direction is emitted as string labels ("long"/"short"/None), matching
        # the OLD pandas-ta halftrend output.
        direction_labels = ["long" if d == 0 else "short" if d == 1 else None for d in direction]

        return pl.DataFrame(
            {
                _fields["atr_high"]: atr_high,
                _fields["atr_low"]: atr_low,
                _fields["ht_close"]: ht_close,
                _fields["direction"]: pl.Series(direction_labels, dtype=pl.Utf8),
                _fields["arr_up"]: arr_up,
                _fields["arr_down"]: arr_down,
            }
        ).to_struct("halftrend")

    return (
        pl.struct(
            [
                high_expr.alias("_high"),
                low_expr.alias("_low"),
                close_expr.alias("_close"),
                atr_expr.alias("_atr"),
                high_ma_expr.alias("_high_ma"),
                low_ma_expr.alias("_low_ma"),
                highest_expr.alias("_highest"),
                lowest_expr.alias("_lowest"),
            ]
        )
        .map_batches(
            compute_halftrend,
            return_dtype=pl.Struct(
                {
                    _fields["atr_high"]: pl.Float64,
                    _fields["atr_low"]: pl.Float64,
                    _fields["ht_close"]: pl.Float64,
                    _fields["direction"]: pl.Utf8,
                    _fields["arr_up"]: pl.Float64,
                    _fields["arr_down"]: pl.Float64,
                }
            ),
        )
        .alias(f"HT{_props}")
    )
