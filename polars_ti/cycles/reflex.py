# -*- coding: utf-8 -*-
from numpy import cos, exp, nan, sqrt, zeros_like
from numba import njit


@njit(cache=True)
def np_reflex(x, n, k, alpha, pi, sqrt2):
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
        slope = (_f[i - n] - _f[i]) / n

        _sum = 0
        for j in range(1, n):
            _sum += _f[i] - _f[i - j] + j * slope
        _sum /= n

        _ms[i] = alpha * _sum * _sum + (1 - alpha) * _ms[i - 1]
        if _ms[i] != 0.0:
            result[i] = _sum / sqrt(_ms[i])

    return result


# =============================================================================
# Polars REFLEX Implementation
# =============================================================================
import numpy as np
import polars as pl

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


def reflex(
    close: IntoExpr,
    length: int = 20,
    smooth: int = 20,
    alpha: float = 0.04,
    pi: float = 3.14159,
    sqrt2: float = 1.414,
    offset: int = 0,
) -> PlExpr:
    """Polars: Reflex Indicator

    John F. Ehlers' lag-reduced cycle indicator from TASC Feb 2020.
    Oscillator focused on cycle detection.

    Sources:
        http://traders.com/Documentation/FEEDbk_docs/2020/02/TradersTips.html

    Args:
        close: Column name or pl.Expr for 'close' prices.
        length: Period. Default: 20
        smooth: Period of internal SuperSmoother. Default: 20
        alpha: Alpha weight. Default: 0.04
        pi: PI value. Default: 3.14159
        sqrt2: sqrt(2) value. Default: 1.414
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: REFLEX expression.
    """
    close_expr = v_expr(close)
    if close_expr is None:
        return None

    _length, _smooth, _alpha = length, smooth, alpha
    _pi, _sqrt2, _offset = pi, sqrt2, offset

    def compute_reflex(s: pl.Series) -> pl.Series:
        np_close = s.to_numpy().astype(np.float64)
        result = np_reflex(np_close, _length, _smooth, _alpha, _pi, _sqrt2)
        result[:_length] = np.nan
        if _offset != 0:
            result = np.roll(result, _offset)
            if _offset > 0:
                result[:_offset] = np.nan
            else:
                result[_offset:] = np.nan
        return pl.Series(result)

    return close_expr.map_batches(compute_reflex, return_dtype=pl.Float64).alias(f"REFLEX_{length}_{smooth}_{alpha}")
