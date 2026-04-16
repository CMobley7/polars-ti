# -*- coding: utf-8 -*-
# =============================================================================
# Polars RSX Implementation (Numba @njit kernel)
# =============================================================================
import polars as pl
import numpy as np
from numba import njit

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


@njit(cache=True)
def _rsx_numba(close: np.ndarray, length: int) -> np.ndarray:
    """Numba-optimized RSX (Relative Strength Xtra).
    
    Based on Jurik's algorithm as published at prorealcode.com.
    This mirrors the Pandas implementation exactly.
    """
    m = len(close)
    result = np.empty(m, dtype=np.float64)
    result[:] = np.nan
    
    if m < length:
        return result
    
    # Initialize state variables
    vC, v1C = 0.0, 0.0
    v4, v8, v10, v14, v18, v20 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    
    f0, f8, f10, f18, f20, f28, f30, f38 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    f40, f48, f50, f58, f60, f68, f70, f78 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    f80, f88, f90 = 0.0, 0.0, 0.0
    
    # Initial value at length-1
    result[length - 1] = 50.0
    
    for i in range(length, m):
        if f90 == 0:
            f90 = 1.0
            f0 = 0.0
            if length - 1.0 >= 5:
                f88 = length - 1.0
            else:
                f88 = 5.0
            f8 = 100.0 * close[i]
            f18 = 3.0 / (length + 2.0)
            f20 = 1.0 - f18
        else:
            if f88 <= f90:
                f90 = f88 + 1
            else:
                f90 = f90 + 1
            f10 = f8
            f8 = 100.0 * close[i]
            v8 = f8 - f10
            f28 = f20 * f28 + f18 * v8
            f30 = f18 * f28 + f20 * f30
            vC = 1.5 * f28 - 0.5 * f30
            f38 = f20 * f38 + f18 * vC
            f40 = f18 * f38 + f20 * f40
            v10 = 1.5 * f38 - 0.5 * f40
            f48 = f20 * f48 + f18 * v10
            f50 = f18 * f48 + f20 * f50
            v14 = 1.5 * f48 - 0.5 * f50
            f58 = f20 * f58 + f18 * abs(v8)
            f60 = f18 * f58 + f20 * f60
            v18 = 1.5 * f58 - 0.5 * f60
            f68 = f20 * f68 + f18 * v18
            f70 = f18 * f68 + f20 * f70
            v1C = 1.5 * f68 - 0.5 * f70
            f78 = f20 * f78 + f18 * v1C
            f80 = f18 * f78 + f20 * f80
            v20 = 1.5 * f78 - 0.5 * f80

            if f88 >= f90 and f8 != f10:
                f0 = 1.0
            if f88 == f90 and f0 == 0.0:
                f90 = 0.0

        if f88 < f90 and v20 > 0.0000000001:
            v4 = (v14 / v20 + 1.0) * 50.0
            if v4 > 100.0:
                v4 = 100.0
            if v4 < 0.0:
                v4 = 0.0
        else:
            v4 = 50.0
        result[i] = v4
    
    return result


def pl_rsx(
    close: IntoExpr,
    length: int = 14,
    offset: int = 0,
) -> PlExpr:
    """Polars: Relative Strength Xtra (RSX)

    Uses Numba @njit kernel via map_batches for the recursive calculation.

    The Relative Strength Xtra is based on the popular RSI indicator and
    inspired by the work of Jurik Research. This enhanced version of the
    RSI reduces noise and provides a clearer, only slightly delayed insight
    on momentum and velocity of price movements.

    Sources:
        http://www.jurikres.com/catalog1/ms_rsx.htm
        https://www.prorealcode.com/prorealtime-indicators/jurik-rsx/

    Args:
        close: Column name or pl.Expr for 'close' prices
        length: Period length. Default: 14
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: RSX expression for lazy evaluation
    """
    close_expr = v_expr(close)
    if close_expr is None:
        return None

    _length = length

    def compute_rsx(s: pl.Series) -> pl.Series:
        arr = s.to_numpy().astype(np.float64)
        result = _rsx_numba(arr, _length)
        return pl.Series(result)

    rsx_expr = close_expr.map_batches(compute_rsx, return_dtype=pl.Float64)

    if offset != 0:
        rsx_expr = rsx_expr.shift(offset)

    return rsx_expr.alias(f"RSX_{length}")
