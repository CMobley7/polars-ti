# -*- coding: utf-8 -*-
# =============================================================================
# Polars CG (Center of Gravity) Implementation
# =============================================================================
import polars as pl
import numpy as np
from numba import njit

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils import v_pos_int
from polars_ti.utils._validate import v_expr


@njit(cache=True)
def nb_cg(close: np.ndarray, length: int) -> np.ndarray:
    """Numba: Center of Gravity calculation.

    CG = -sum(close[i] * weight[i]) / sum(close[i])
    where weight[i] = 1..length (1 for oldest, length for newest in window)
    """
    n = len(close)
    result = np.full(n, np.nan, dtype=np.float64)

    if length > n:
        # Window larger than the data -> all NaN. Return before allocating the
        # O(length) weight vector, which on an absurd length would exhaust memory.
        return result

    weights = np.arange(1, length + 1, dtype=np.float64)

    for i in range(length - 1, n):
        window = close[i - length + 1 : i + 1]
        weighted_sum = np.sum(window * weights)
        total_sum = np.sum(window)

        if abs(total_sum) > 1e-10:
            result[i] = -weighted_sum / total_sum
        else:
            result[i] = np.nan

    return result


def cg(
    close: IntoExpr,
    length: int = 10,
    offset: int = 0,
) -> PlExpr:
    """Polars: Center of Gravity (CG)

    The Center of Gravity Indicator by John Ehlers attempts to identify
    turning points while exhibiting zero lag and smoothing.

    Formula: CG = -sum(close * weight) / sum(close)

    Args:
        close: Column name or pl.Expr for 'close' prices
        length: Period. Default: 10
        offset: Shift result. Default: 0

    Returns:
        pl.Expr: CG expression
    """
    close_expr = v_expr(close)
    _length = v_pos_int(length, "length")
    _offset = offset

    def compute_cg(s: pl.Series) -> pl.Series:
        arr = s.to_numpy().astype(np.float64)
        result = nb_cg(arr, _length)

        if _offset != 0:
            result = np.roll(result, _offset)
            if _offset > 0:
                result[:_offset] = np.nan
            else:
                result[_offset:] = np.nan

        return pl.Series(result)

    return close_expr.map_batches(compute_cg, return_dtype=pl.Float64).alias(f"CG_{length}")
