# -*- coding: utf-8 -*-
from numpy import empty, float64, isnan, nan
from numba import njit


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


# =============================================================================
# Polars TRAMA Implementation (reuses nb_trama kernel)
# =============================================================================
import numpy as np
import polars as pl
from numba import njit as _njit_trama

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


@_njit_trama(cache=True)
def _nb_rolling_mean(arr, window):
    """Numba-optimized rolling mean."""
    n = len(arr)
    result = np.full(n, np.nan)
    for i in range(window - 1, n):
        s = 0.0
        for j in range(window):
            s += arr[i - j]
        result[i] = s / window
    return result


@_njit_trama(cache=True)
def _nb_rolling_max(arr, window):
    """Numba-optimized rolling max."""
    n = len(arr)
    result = np.full(n, np.nan)
    for i in range(window - 1, n):
        mx = arr[i - window + 1]
        for j in range(1, window):
            if arr[i - window + 1 + j] > mx:
                mx = arr[i - window + 1 + j]
        result[i] = mx
    return result


@_njit_trama(cache=True)
def _nb_rolling_min(arr, window):
    """Numba-optimized rolling min."""
    n = len(arr)
    result = np.full(n, np.nan)
    for i in range(window - 1, n):
        mn = arr[i - window + 1]
        for j in range(1, window):
            if arr[i - window + 1 + j] < mn:
                mn = arr[i - window + 1 + j]
        result[i] = mn
    return result


def trama(
    close: IntoExpr,
    length: int = 10,
    offset: int = 0,
) -> PlExpr:
    """Polars: Trend Regulated Adaptive Moving Average (TRAMA)

    Adaptive MA that adjusts smoothing based on trend detection.

    Args:
        close: Column name or pl.Expr for input values
        length: Period. Default: 10
        offset: Shift result. Default: 0

    Returns:
        pl.Expr: TRAMA expression
    """
    close_expr = v_expr(close)

    def _compute(s: pl.Series) -> pl.Series:
        arr = s.to_numpy().astype(np.float64)
        n = len(arr)

        # Rolling highest/lowest using Numba
        highest = _nb_rolling_max(arr, length)
        lowest = _nb_rolling_min(arr, length)

        # Detect new highs (diff > 0) and new lows (diff < 0)
        hh = np.zeros(n)
        ll = np.zeros(n)
        for i in range(1, n):
            if not np.isnan(highest[i]) and not np.isnan(highest[i - 1]):
                hh[i] = 1.0 if (highest[i] - highest[i - 1]) > 0 else 0.0
            if not np.isnan(lowest[i]) and not np.isnan(lowest[i - 1]):
                ll[i] = 1.0 if (lowest[i] - lowest[i - 1]) < 0 else 0.0

        # Trend signal: 1 if either new high or new low
        trend_signal = np.where((hh > 0) | (ll > 0), 1.0, 0.0)

        # Trend coefficient: squared SMA of trend signal
        tc = _nb_rolling_mean(trend_signal, length) ** 2

        trama_arr = nb_trama(arr, tc)
        return pl.Series(values=trama_arr, name=s.name)

    result = close_expr.map_batches(_compute, return_dtype=pl.Float64)

    if offset != 0:
        result = result.shift(offset)

    return result.alias(f"TRAMA_{length}")
