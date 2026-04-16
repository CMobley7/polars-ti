# -*- coding: utf-8 -*-
from numpy import copy, cos, exp, zeros_like
from numba import njit


@njit(cache=True)
def nb_ssf3(x, n, pi, sqrt3):
    m, result = x.size, copy(x)
    a = exp(-pi / n)
    b = 2 * a * cos(-pi * sqrt3 / n)
    c = a * a

    d4 = c * c
    d3 = -c * (1 + b)
    d2 = b + c
    d1 = 1 - d2 - d3 - d4

    # result[:3] = x[:3]
    for i in range(3, m):
        result[i] = (
            d1 * x[i] + d2 * result[i - 1] + d3 * result[i - 2] + d4 * result[i - 3]
        )

    return result


# =============================================================================
# Polars SSF3 Implementation
# =============================================================================
import polars as pl
import numpy as np

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


def pl_ssf3(
    close: IntoExpr,
    length: int = 20,
    pi: float = 3.14159,
    sqrt3: float = 1.732,
    offset: int = 0,
) -> PlExpr:
    """Polars: Ehler's 3 Pole Super Smoother Filter (SSF3)

    Recursive digital filter to reduce lag and remove aliasing noise.

    Args:
        close: Column name or pl.Expr for 'close' prices
        length: Period. Default: 20
        pi: Value of PI. Default: 3.14159
        sqrt3: Value of sqrt(3). Default: 1.732
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: SSF3 expression
    """
    close_expr = v_expr(close)
    
    def compute_ssf3(s: pl.Series) -> pl.Series:
        arr = s.to_numpy().astype(np.float64)
        # Use existing Numba kernel directly
        result = nb_ssf3(arr, length, pi, sqrt3)
        if offset != 0:
            result = np.roll(result, offset)
            if offset > 0:
                result[:offset] = np.nan
        return pl.Series(result)
    
    return close_expr.map_batches(compute_ssf3, return_dtype=pl.Float64).alias(f"SSF3_{length}")

