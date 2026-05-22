# -*- coding: utf-8 -*-
# =============================================================================
# Polars StochRSI Implementation
# =============================================================================
import polars as pl
import numpy as np
from numba import njit

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


@njit(cache=True)
def _rsi_numba(values: np.ndarray, length: int) -> np.ndarray:
    """Numba RSI with Wilder's smoothing."""
    n = len(values)
    result = np.full(n, np.nan, dtype=np.float64)

    if n < length + 1:
        return result

    alpha = 1.0 / length

    deltas = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        deltas[i] = values[i] - values[i - 1]

    gain_sum = 0.0
    loss_sum = 0.0
    for i in range(1, length + 1):
        if deltas[i] > 0:
            gain_sum += deltas[i]
        else:
            loss_sum += abs(deltas[i])

    avg_gain = gain_sum / length
    avg_loss = loss_sum / length

    if avg_loss == 0:
        result[length] = 100.0
    else:
        rs = avg_gain / avg_loss
        result[length] = 100.0 - (100.0 / (1.0 + rs))

    for i in range(length + 1, n):
        delta = deltas[i]
        if delta > 0:
            gain = delta
            loss = 0.0
        else:
            gain = 0.0
            loss = abs(delta)

        avg_gain = alpha * gain + (1 - alpha) * avg_gain
        avg_loss = alpha * loss + (1 - alpha) * avg_loss

        if avg_loss == 0:
            result[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            result[i] = 100.0 - (100.0 / (1.0 + rs))

    return result


@njit(cache=True)
def _sma_numba(values: np.ndarray, length: int) -> np.ndarray:
    """Numba-optimized SMA handling NaNs."""
    n = len(values)
    result = np.full(n, np.nan, dtype=np.float64)

    if n < length:
        return result

    first_valid = -1
    for i in range(n):
        if not np.isnan(values[i]):
            first_valid = i
            break

    if first_valid == -1 or n - first_valid < length:
        return result

    window_sum = 0.0
    for i in range(first_valid, first_valid + length):
        window_sum += values[i]
    result[first_valid + length - 1] = window_sum / length

    for i in range(first_valid + length, n):
        if not np.isnan(values[i]):
            window_sum = window_sum - values[i - length] + values[i]
            result[i] = window_sum / length

    return result


@njit(cache=True)
def _stochrsi_core(
    close: np.ndarray,
    length: int,
    rsi_length: int,
    k: int,
    d: int,
) -> tuple:
    """Numba kernel for StochRSI calculation."""
    n = len(close)

    # 1. Calculate RSI
    rsi = _rsi_numba(close, rsi_length)

    # 2. Calculate rolling min/max of RSI over 'length' periods
    lowest_rsi = np.full(n, np.nan, dtype=np.float64)
    highest_rsi = np.full(n, np.nan, dtype=np.float64)

    for i in range(rsi_length + length - 1, n):
        window_start = i - length + 1
        # Check if all values in window are valid
        all_valid = True
        for j in range(window_start, i + 1):
            if np.isnan(rsi[j]):
                all_valid = False
                break

        if all_valid:
            min_val = rsi[window_start]
            max_val = rsi[window_start]
            for j in range(window_start + 1, i + 1):
                if rsi[j] < min_val:
                    min_val = rsi[j]
                if rsi[j] > max_val:
                    max_val = rsi[j]
            lowest_rsi[i] = min_val
            highest_rsi[i] = max_val

    # 3. Calculate raw StochRSI
    stochrsi_raw = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        if not np.isnan(lowest_rsi[i]) and not np.isnan(highest_rsi[i]):
            range_val = highest_rsi[i] - lowest_rsi[i]
            if range_val != 0:
                stochrsi_raw[i] = 100.0 * (rsi[i] - lowest_rsi[i]) / range_val

    # 4. %K = SMA(stochrsi, k)
    stochrsi_k = _sma_numba(stochrsi_raw, k)

    # 5. %D = SMA(%K, d)
    stochrsi_d = _sma_numba(stochrsi_k, d)

    return stochrsi_k, stochrsi_d


def pl_stochrsi(
    close: IntoExpr,
    length: int = 14,
    rsi_length: int = 14,
    k: int = 3,
    d: int = 3,
    mamode: str = "sma",
    talib: bool = True,
    offset: int = 0,
) -> PlExpr:
    """Polars: Stochastic RSI (STOCHRSI)

    Created by Tushar Chande and Stanley Kroll. It applies the Stochastic
    formula to RSI values instead of price data, resulting in an indicator
    that ranges from 0 to 100.

    Sources:
        https://www.tradingview.com/wiki/Stochastic_RSI_(STOCH_RSI)
        Stock & Commodities V.11:5 (189-199)

    Args:
        close: Column name or pl.Expr for 'close' prices
        length: StochRSI lookback period. Default: 14
        rsi_length: RSI calculation period. Default: 14
        k: Fast %K smoothing period. Default: 3
        d: Slow %D smoothing period. Default: 3
        mamode: MA type for smoothing. Default: 'sma'
        talib: If True and TA-Lib installed, use TA-Lib RSI. Default: True
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: Struct expression with columns:
            - STOCHRSIk_{length}_{rsi_length}_{k}_{d}: %K line
            - STOCHRSId_{length}_{rsi_length}_{k}_{d}: %D line
    """
    from polars_ti.maps import Imports
    from polars_ti.utils import v_talib

    close_expr = v_expr(close)
    if close_expr is None:
        return None

    _length = length
    _rsi_length = rsi_length
    _k = k
    _d = d
    _props = f"_{length}_{rsi_length}_{k}_{d}"
    _use_talib = Imports["talib"] and v_talib(talib)

    if _use_talib:
        # Use TA-Lib RSI, then apply Stochastic formula
        def compute_stochrsi_talib(s: pl.Series) -> pl.Series:
            from talib import RSI as TALIB_RSI

            arr = s.to_numpy().astype(np.float64)

            # Get RSI from TA-Lib
            rsi = TALIB_RSI(arr, timeperiod=_rsi_length)

            # Apply Stochastic to RSI
            n = len(rsi)
            lowest_rsi = np.full(n, np.nan, dtype=np.float64)
            highest_rsi = np.full(n, np.nan, dtype=np.float64)

            for i in range(_length - 1, n):
                window = rsi[i - _length + 1 : i + 1]
                if not np.any(np.isnan(window)):
                    lowest_rsi[i] = np.min(window)
                    highest_rsi[i] = np.max(window)

            # Raw StochRSI
            range_val = highest_rsi - lowest_rsi
            range_val = np.where(range_val == 0, np.nan, range_val)
            stochrsi_raw = 100.0 * (rsi - lowest_rsi) / range_val

            # %K and %D smoothing
            stochrsi_k = _sma_numba(stochrsi_raw, _k)
            stochrsi_d = _sma_numba(stochrsi_k, _d)

            return pl.DataFrame(
                {
                    f"STOCHRSIk{_props}": stochrsi_k,
                    f"STOCHRSId{_props}": stochrsi_d,
                }
            ).to_struct("STOCHRSI")

        result_expr = close_expr.map_batches(
            compute_stochrsi_talib,
            return_dtype=pl.Struct(
                [
                    pl.Field(f"STOCHRSIk{_props}", pl.Float64),
                    pl.Field(f"STOCHRSId{_props}", pl.Float64),
                ]
            ),
        )
    else:
        # Pure Numba path
        def compute_stochrsi_numba(s: pl.Series) -> pl.Series:
            arr = s.to_numpy().astype(np.float64)
            stochrsi_k, stochrsi_d = _stochrsi_core(arr, _length, _rsi_length, _k, _d)

            return pl.DataFrame(
                {
                    f"STOCHRSIk{_props}": stochrsi_k,
                    f"STOCHRSId{_props}": stochrsi_d,
                }
            ).to_struct("STOCHRSI")

        result_expr = close_expr.map_batches(
            compute_stochrsi_numba,
            return_dtype=pl.Struct(
                [
                    pl.Field(f"STOCHRSIk{_props}", pl.Float64),
                    pl.Field(f"STOCHRSId{_props}", pl.Float64),
                ]
            ),
        )

    if offset != 0:
        result_expr = result_expr.shift(offset)

    return result_expr.alias("STOCHRSI")
