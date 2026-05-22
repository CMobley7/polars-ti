# -*- coding: utf-8 -*-
# =============================================================================
# Polars Stochastic Implementation
# =============================================================================
import polars as pl
import numpy as np
from numba import njit

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


@njit(cache=True)
def _sma_numba(values: np.ndarray, length: int) -> np.ndarray:
    """Numba-optimized Simple Moving Average handling NaNs."""
    n = len(values)
    result = np.full(n, np.nan, dtype=np.float64)

    if n < length:
        return result

    # Find first valid index
    first_valid = -1
    for i in range(n):
        if not np.isnan(values[i]):
            first_valid = i
            break

    if first_valid == -1 or n - first_valid < length:
        return result

    # Calculate first SMA
    window_sum = 0.0
    for i in range(first_valid, first_valid + length):
        window_sum += values[i]
    result[first_valid + length - 1] = window_sum / length

    # Rolling sum
    for i in range(first_valid + length, n):
        if not np.isnan(values[i]):
            window_sum = window_sum - values[i - length] + values[i]
            result[i] = window_sum / length

    return result


@njit(cache=True)
def _stoch_core(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    k: int,
    smooth_k: int,
    d: int,
) -> tuple:
    """Numba kernel for Stochastic Oscillator calculation."""
    n = len(close)

    # Calculate lowest low and highest high over k periods
    lowest_low = np.full(n, np.nan, dtype=np.float64)
    highest_high = np.full(n, np.nan, dtype=np.float64)

    for i in range(k - 1, n):
        ll = low[i]
        hh = high[i]
        for j in range(i - k + 1, i):
            if low[j] < ll:
                ll = low[j]
            if high[j] > hh:
                hh = high[j]
        lowest_low[i] = ll
        highest_high[i] = hh

    # Raw %K = 100 * (close - ll) / (hh - ll)
    stoch_raw = np.full(n, np.nan, dtype=np.float64)
    for i in range(k - 1, n):
        range_val = highest_high[i] - lowest_low[i]
        if range_val != 0:
            stoch_raw[i] = 100.0 * (close[i] - lowest_low[i]) / range_val

    # Slow %K = SMA(raw %K, smooth_k)
    if smooth_k == 1:
        stoch_k = stoch_raw.copy()
    else:
        stoch_k = _sma_numba(stoch_raw, smooth_k)

    # %D = SMA(slow %K, d)
    stoch_d = _sma_numba(stoch_k, d)

    # Histogram = %K - %D
    stoch_h = stoch_k - stoch_d

    return stoch_k, stoch_d, stoch_h


def pl_stoch(
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    k: int = 14,
    d: int = 3,
    smooth_k: int = 3,
    mamode: str = "sma",
    talib: bool = True,
    offset: int = 0,
) -> PlExpr:
    """Polars: Stochastic Oscillator (STOCH)

    The Stochastic Oscillator was developed by George Lane in the 1950's.
    It is a range-bound oscillator with two lines moving between 0 and 100.
    %K displays the current close in relation to the period's high/low range.
    %D is a Moving Average of %K.

    Sources:
        https://www.tradingview.com/wiki/Stochastic_(STOCH)
        https://www.investopedia.com/terms/s/stochasticoscillator.asp

    Args:
        high: Column name or pl.Expr for 'high' prices
        low: Column name or pl.Expr for 'low' prices
        close: Column name or pl.Expr for 'close' prices
        k: Fast %K period (lookback for high/low range). Default: 14
        d: Slow %D period (MA of smooth %K). Default: 3
        smooth_k: Slow %K smoothing period. Default: 3
        mamode: MA type for smoothing ('sma', 'ema', etc.). Default: 'sma'
        talib: If True and TA-Lib installed, use TA-Lib. Default: True
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: Struct expression with columns:
            - STOCHk_{k}_{d}_{smooth_k}: Slow %K line
            - STOCHd_{k}_{d}_{smooth_k}: %D signal line
            - STOCHh_{k}_{d}_{smooth_k}: Histogram (%K - %D)
    """
    from polars_ti.maps import Imports
    from polars_ti.utils import v_talib, tal_ma

    high_expr = v_expr(high)
    low_expr = v_expr(low)
    close_expr = v_expr(close)

    if high_expr is None or low_expr is None or close_expr is None:
        return None

    _k = k
    _d = d
    _smooth_k = smooth_k
    _mamode = mamode.lower() if isinstance(mamode, str) else "sma"
    _props = f"_{k}_{d}_{smooth_k}"
    _use_talib = Imports["talib"] and v_talib(talib) and smooth_k > 2

    if _use_talib:
        # TA-Lib path: use STOCH function
        def compute_stoch_talib(s: pl.Series) -> pl.Series:
            from talib import STOCH as TALIB_STOCH

            high_arr = s.struct.field("high").to_numpy().astype(np.float64)
            low_arr = s.struct.field("low").to_numpy().astype(np.float64)
            close_arr = s.struct.field("close").to_numpy().astype(np.float64)

            stoch_k, stoch_d = TALIB_STOCH(
                high_arr,
                low_arr,
                close_arr,
                fastk_period=_k,
                slowk_period=_smooth_k,
                slowk_matype=tal_ma(_mamode),
                slowd_period=_d,
                slowd_matype=tal_ma(_mamode),
            )
            stoch_h = stoch_k - stoch_d

            return pl.DataFrame(
                {
                    f"STOCHk{_props}": stoch_k,
                    f"STOCHd{_props}": stoch_d,
                    f"STOCHh{_props}": stoch_h,
                }
            ).to_struct("STOCH")

        result_expr = pl.struct(
            high=high_expr,
            low=low_expr,
            close=close_expr,
        ).map_batches(
            compute_stoch_talib,
            return_dtype=pl.Struct(
                [
                    pl.Field(f"STOCHk{_props}", pl.Float64),
                    pl.Field(f"STOCHd{_props}", pl.Float64),
                    pl.Field(f"STOCHh{_props}", pl.Float64),
                ]
            ),
        )
    else:
        # Pure Polars + Numba path
        def compute_stoch_numba(s: pl.Series) -> pl.Series:
            high_arr = s.struct.field("high").to_numpy().astype(np.float64)
            low_arr = s.struct.field("low").to_numpy().astype(np.float64)
            close_arr = s.struct.field("close").to_numpy().astype(np.float64)

            stoch_k, stoch_d, stoch_h = _stoch_core(high_arr, low_arr, close_arr, _k, _smooth_k, _d)

            return pl.DataFrame(
                {
                    f"STOCHk{_props}": stoch_k,
                    f"STOCHd{_props}": stoch_d,
                    f"STOCHh{_props}": stoch_h,
                }
            ).to_struct("STOCH")

        result_expr = pl.struct(
            high=high_expr,
            low=low_expr,
            close=close_expr,
        ).map_batches(
            compute_stoch_numba,
            return_dtype=pl.Struct(
                [
                    pl.Field(f"STOCHk{_props}", pl.Float64),
                    pl.Field(f"STOCHd{_props}", pl.Float64),
                    pl.Field(f"STOCHh{_props}", pl.Float64),
                ]
            ),
        )

    if offset != 0:
        result_expr = result_expr.shift(offset)

    return result_expr.alias("STOCH")
