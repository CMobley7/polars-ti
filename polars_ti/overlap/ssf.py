# -*- coding: utf-8 -*-
from numpy import copy, cos, exp, zeros_like
from numba import njit


@njit(cache=True)
def nb_ssf(x, n, pi, sqrt2):
    m, ratio, result = x.size, sqrt2 / n, copy(x)
    a = exp(-pi * ratio)
    b = 2 * a * cos(180 * ratio)
    c = a * a - b + 1

    # result[:2] = x[:2]
    for i in range(2, m):
        result[i] = 0.5 * c * (x[i] + x[i - 1]) + b * result[i - 1] - a * a * result[i - 2]

    return result


@njit(cache=True)
def nb_ssf_everget(x, n, pi, sqrt2):
    m, arg, result = x.size, pi * sqrt2 / n, copy(x)
    a = exp(-arg)
    b = 2 * a * cos(arg)

    # result[:2] = x[:2]
    for i in range(2, m):
        result[i] = 0.5 * (a * a - b + 1) * (x[i] + x[i - 1]) + b * result[i - 1] - a * a * result[i - 2]

    return result


# =============================================================================
# Polars SSF Implementation (using existing Numba kernels)
# =============================================================================
import polars as pl
import numpy as np

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


def pl_ssf(
    close: IntoExpr,
    length: int = 20,
    everget: bool = False,
    pi: float = 3.14159,
    sqrt2: float = 1.414,
    offset: int = 0,
) -> PlExpr:
    """Polars: Ehler's Super Smoother Filter (SSF)

    John F. Ehlers's solution to reduce lag and remove aliasing noise.

    Sources:
        http://traders.com/documentation/feedbk_docs/2014/01/traderstips.html

    Args:
        close: Column name or pl.Expr for 'close' prices
        length: Filter period. Default: 20
        everget: Use Everget's calculation. Default: False
        pi: Value of PI. Default: 3.14159
        sqrt2: Value of sqrt(2). Default: 1.414
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: SSF expression for lazy evaluation
    """
    close_expr = v_expr(close)
    if close_expr is None:
        return None

    _length = length
    _everget = everget
    _pi = pi
    _sqrt2 = sqrt2

    def compute_ssf(s: pl.Series) -> pl.Series:
        arr = s.to_numpy().astype(np.float64)
        # Call Numba kernels directly - NO Pandas!
        if _everget:
            result = nb_ssf_everget(arr, _length, _pi, _sqrt2)
        else:
            result = nb_ssf(arr, _length, _pi, _sqrt2)
        return pl.Series(result)

    result = close_expr.map_batches(compute_ssf, return_dtype=pl.Float64)

    if offset != 0:
        result = result.shift(offset)

    return result.alias(f"SSF{'e' if everget else ''}_{length}")
