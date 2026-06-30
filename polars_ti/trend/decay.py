# -*- coding: utf-8 -*-
from numpy import float64, zeros_like
from numba import njit


@njit(cache=True)
def nb_exponential_decay(x, n):
    m, rate = x.size, 1.0 - (1.0 / n)

    result = zeros_like(x, dtype=float64)
    result[0] = x[0]

    for i in range(1, m):
        result[i] = max(0, x[i], result[i - 1] * rate)

    return result


@njit(cache=True)
def nb_linear_decay(x, n):
    m, rate = x.size, 1.0 / n

    result = zeros_like(x, dtype=float64)
    result[0] = x[0]

    for i in range(1, m):
        result[i] = max(0, x[i], result[i - 1] - rate)

    return result


# =============================================================================
# Polars Decay Implementation (uses existing Numba kernels)
# =============================================================================
import numpy as np
import polars as pl

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


def decay(
    close: IntoExpr,
    length: int = 1,
    mode: str = "linear",
    offset: int = 0,
) -> PlExpr:
    """Polars: Decay

    Creates a decay moving forward from prior signals. Supports linear
    and exponential modes.

    Args:
        close: Column name or pl.Expr for input values
        length: Period. Default: 1
        mode: 'linear' or 'exp'/'exponential'. Default: 'linear'
        offset: Shift result. Default: 0

    Returns:
        pl.Expr: Decay expression
    """
    close_expr = v_expr(close)
    _mode_str = mode.lower() if isinstance(mode, str) else "linear"
    is_exp = _mode_str in ("exp", "exponential")

    def _compute_decay(s: pl.Series) -> pl.Series:
        arr = s.to_numpy().astype(np.float64)
        if is_exp:
            result = nb_exponential_decay(arr, length)
        else:
            result = nb_linear_decay(arr, length)
        return pl.Series(values=result, name=s.name)

    _label = "EXP" if is_exp else "L"
    result = close_expr.map_batches(_compute_decay, return_dtype=pl.Float64)

    if offset != 0:
        result = result.shift(offset)

    return result.alias(f"{_label}DECAY_{length}")
