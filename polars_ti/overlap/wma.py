# -*- coding: utf-8 -*-
from numpy import arange, dot, float64, nan, zeros_like
from numba import njit


@njit(cache=True)
def nb_wma(x, n, asc, prenan):
    m = x.size
    w = arange(1, n + 1, dtype=float64)
    result = zeros_like(x, dtype=x.dtype)

    if not asc:
        w = w[::-1]

    for i in range(n - 1, m):
        result[i] = (w * x[i - n + 1 : i + 1]).sum()
    result *= 2 / (n * n + n)

    if prenan:
        result[: n - 1] = nan

    return result


# =============================================================================
# Polars WMA Implementation (using nb_wma kernel)
# =============================================================================
import polars as pl
import numpy as np

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
