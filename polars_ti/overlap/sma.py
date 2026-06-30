# -*- coding: utf-8 -*-
from numpy import convolve, ones
from numba import njit
from polars_ti.utils._numba import nb_prepend


@njit(cache=True)
def nb_sma(x, n):
    result = convolve(ones(n) / n, x)[n - 1 : 1 - n]
    return nb_prepend(result, n - 1)


# =============================================================================
# Polars SMA Implementation
# =============================================================================
import polars as pl

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


def sma(
    close: IntoExpr,
    length: int = 10,
    talib: bool = True,
    min_periods: int | None = None,
    offset: int = 0,
) -> PlExpr:
    """Polars: Simple Moving Average (SMA)

    The Simple Moving Average is the equally weighted average over its length.

    Sources:
        https://www.tradingtechnologies.com/help/x-study/technical-indicator-definitions/simple-moving-average-sma/

    Args:
        close: Column name or pl.Expr for 'close' prices
        length: Rolling window period. Default: 10
        talib: Ignored (for API compatibility with Pandas version). Default: True
        min_periods: Minimum periods required. Default: length
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: SMA expression for lazy evaluation
    """
    from polars_ti.maps import Imports
    from polars_ti.utils import v_talib
    import numpy as np

    close_expr = v_expr(close)
    if close_expr is None:
        return None

    min_periods = min_periods if min_periods is not None else length
    _use_talib = Imports["talib"] and v_talib(talib) and length > 1
    _length = length

    if _use_talib:

        def compute_sma(s: pl.Series) -> pl.Series:
            from talib import SMA as TALIB_SMA

            arr = s.to_numpy().astype(np.float64)
            result = TALIB_SMA(arr, timeperiod=_length)
            return pl.Series(result)

        sma_expr = close_expr.map_batches(compute_sma, return_dtype=pl.Float64)
    else:
        sma_expr = close_expr.rolling_mean(window_size=length, min_samples=min_periods)

    # Apply offset
    if offset != 0:
        sma_expr = sma_expr.shift(offset)

    return sma_expr.alias(f"SMA_{length}")
