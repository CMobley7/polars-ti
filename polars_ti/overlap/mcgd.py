# -*- coding: utf-8 -*-
# =============================================================================
# Polars MCGD Implementation
# =============================================================================
import polars as pl
import numpy as np
from numba import njit

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


@njit(cache=True)
def nb_mcgd(close: np.ndarray, length: int, c: float) -> np.ndarray:
    """Numba-optimized McGinley Dynamic calculation."""
    n = len(close)
    result = np.empty(n, dtype=np.float64)
    result[0] = close[0]
    
    for i in range(1, n):
        # MCGD formula: MD = MD[i-1] + (price - MD[i-1]) / (k * n * (price / MD[i-1])^4)
        prev = result[i - 1]
        if prev != 0 and not np.isnan(prev):
            ratio = close[i] / prev
            d = c * length * (ratio ** 4)
            if d > 1e-10:
                result[i] = prev + (close[i] - prev) / d
            else:
                result[i] = close[i]
        else:
            result[i] = close[i]
    
    return result


def pl_mcgd(
    close: IntoExpr,
    length: int = 10,
    c: float = 1.0,
    offset: int = 0,
) -> PlExpr:
    """Polars: McGinley Dynamic Indicator

    Smoothing mechanism that minimizes price separation and whipsaws.

    Args:
        close: Column name or pl.Expr for 'close' prices
        length: Indicator period. Default: 10
        c: Multiplier for denominator. Default: 1.0
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: MCGD expression
    """
    close_expr = v_expr(close)
    
    def compute_mcgd(s: pl.Series) -> pl.Series:
        arr = s.to_numpy().astype(np.float64)
        result = nb_mcgd(arr, length, c)
        if offset != 0:
            result = np.roll(result, offset)
            if offset > 0:
                result[:offset] = np.nan
        return pl.Series(result)
    
    return close_expr.map_batches(compute_mcgd, return_dtype=pl.Float64).alias(f"MCGD_{length}")


