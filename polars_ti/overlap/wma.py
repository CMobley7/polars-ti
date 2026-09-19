from numba import njit

# -*- coding: utf-8 -*-

from polars_ti.utils._rolling import rolling_extreme, rolling_linear, scaled_window_linear


@njit(cache=True)
def nb_wma(x, n, asc, prenan):
    """Compute linear weighted means with compensated rolling recurrences."""
    if asc:
        _, weighted, _ = rolling_linear(x, n)
    else:
        # Reversing the input avoids subtracting independently rounded totals
        # when descending weights nearly cancel.
        _, reversed_weighted, _ = rolling_linear(x[::-1], n)
        weighted = np.full(len(x), np.nan)
        for i in range(n - 1, len(x)):
            weighted[i] = reversed_weighted[len(x) - i + n - 2]
    result = weighted * (2 / (n * n + n))
    minimum = rolling_extreme(x, n, False)[0]
    maximum = rolling_extreme(x, n, True)[0]
    for i in range(n - 1, len(x)):
        if maximum[i] == np.inf and minimum[i] != -np.inf:
            result[i] = np.inf
        elif minimum[i] == -np.inf and maximum[i] != np.inf:
            result[i] = -np.inf
        elif not np.isfinite(result[i]) and np.isfinite(minimum[i]) and np.isfinite(maximum[i]):
            _, normalized, scale = scaled_window_linear(x, i - n + 1, i + 1, asc)
            result[i] = (normalized / (n * (n + 1) / 2.0)) * scale
    if not prenan:
        result[: n - 1] = 0.0
    return result


# =============================================================================
# Polars WMA Implementation (using nb_wma kernel)
# =============================================================================
import numpy as np
import polars as pl

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr, v_pos_int


def wma(
    close: IntoExpr,
    length: int = 10,
    asc: bool = True,
    talib: bool = True,
    offset: int = 0,
) -> PlExpr:
    """Polars: Weighted Moving Average (WMA)

    The Weighted Moving Average where the weights are linearly increasing
    and the most recent data has the heaviest weight.

    Sources:
        https://en.wikipedia.org/wiki/Moving_average#Weighted_moving_average

    Args:
        close: Column name or pl.Expr for 'close' prices
        length: Rolling window period. Default: 10
        asc: Recent values weigh more. Default: True
        talib: If True and TA-Lib is installed, uses TA-Lib. Default: True
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: WMA expression for lazy evaluation
    """
    from polars_ti.maps import Imports
    from polars_ti.utils import v_talib

    close_expr = v_expr(close)
    if close_expr is None:
        return None

    _length = v_pos_int(length, "length")
    _use_talib = Imports["talib"] and v_talib(talib) and _length > 1 and asc
    _asc = asc

    def compute_wma(s: pl.Series) -> pl.Series:
        arr = s.to_numpy().astype(np.float64)

        if _use_talib:
            from talib import WMA as TALIB_WMA

            result = TALIB_WMA(arr, timeperiod=_length)
        else:
            # Use nb_wma directly - much faster than rolling_map!
            result = nb_wma(arr, _length, _asc, True)

        return pl.Series(result)

    wma_expr = close_expr.map_batches(compute_wma, return_dtype=pl.Float64)

    if offset != 0:
        wma_expr = wma_expr.shift(offset)

    return wma_expr.alias(f"WMA_{length}")
