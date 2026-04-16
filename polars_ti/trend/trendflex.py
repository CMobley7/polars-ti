# -*- coding: utf-8 -*-
from numpy import cos, exp, nan, sqrt, zeros_like
from numba import njit


@njit(cache=True)
def nb_trendflex(x, n, k, alpha, pi, sqrt2):
    m, ratio = x.size, 2 * sqrt2 / k
    a = exp(-pi * ratio)
    b = 2 * a * cos(180 * ratio)
    c = a * a - b + 1

    _f = zeros_like(x, dtype=x.dtype)
    _ms = zeros_like(x, dtype=x.dtype)
    result = zeros_like(x, dtype=x.dtype)

    for i in range(2, m):
        _f[i] = 0.5 * c * (x[i] + x[i - 1]) + b * _f[i - 1] - a * a * _f[i - 2]

    for i in range(n, m):
        _sum = 0
        for j in range(1, n):
            _sum += _f[i] - _f[i - j]
        _sum /= n

        _ms[i] = alpha * _sum * _sum + (1 - alpha) * _ms[i - 1]
        if _ms[i] != 0.0:
            result[i] = _sum / sqrt(_ms[i])

    return result


# =============================================================================
# Polars TrendFlex Implementation (reuses nb_trendflex kernel)
# =============================================================================
import numpy as np
import polars as pl

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


def pl_trendflex(
    close: IntoExpr,
    length: int = 20,
    smooth: int = 20,
    alpha: float = 0.04,
    pi: float = 3.14159,
    sqrt2: float = 1.414,
    offset: int = 0,
) -> PlExpr:
    """Polars: TrendFlex

    Lag-reduced trend oscillator by John F. Ehlers.

    Args:
        close: Column name or pl.Expr for input values
        length: Period. Default: 20
        smooth: SuperSmoother period. Default: 20
        alpha: Difference sums weight. Default: 0.04
        pi: Pi value. Default: 3.14159
        sqrt2: Sqrt(2) value. Default: 1.414
        offset: Shift result. Default: 0

    Returns:
        pl.Expr: TrendFlex expression
    """
    close_expr = v_expr(close)

    def _compute(s: pl.Series) -> pl.Series:
        arr = s.to_numpy().astype(np.float64)
        result = nb_trendflex(arr, length, smooth, alpha, pi, sqrt2)
        result[:length] = np.nan
        return pl.Series(values=result, name=s.name)

    result = close_expr.map_batches(_compute, return_dtype=pl.Float64)

    if offset != 0:
        result = result.shift(offset)

    return result.alias(f"TRENDFLEX_{length}_{smooth}_{alpha}")
