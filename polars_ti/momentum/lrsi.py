# -*- coding: utf-8 -*-
from numpy import zeros
from numba import njit


@njit(cache=True)
def nb_lrsi_filter(close, gamma):
    n = len(close)
    l0 = np.zeros(n)
    l1 = np.zeros(n)
    l2 = np.zeros(n)
    l3 = np.zeros(n)

    # Initialize with first value
    l0[0] = l1[0] = l2[0] = l3[0] = close[0]

    for i in range(1, n):
        l0[i] = (1 - gamma) * close[i] + gamma * l0[i - 1]
        l1[i] = -gamma * l0[i] + l0[i - 1] + gamma * l1[i - 1]
        l2[i] = -gamma * l1[i] + l1[i - 1] + gamma * l2[i - 1]
        l3[i] = -gamma * l2[i] + l2[i - 1] + gamma * l3[i - 1]

    return l0, l1, l2, l3


# =============================================================================
# Polars LRSI Implementation (reuses nb_lrsi_filter kernel)
# =============================================================================
import polars as pl
import numpy as np

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


def pl_lrsi(
    close: IntoExpr,
    length: int = 14,
    gamma: float = 0.5,
    offset: int = 0,
) -> PlExpr:
    """Polars: Laguerre RSI (LRSI)

    Modified RSI using Laguerre polynomials for reduced lag.

    Args:
        close: Column name or pl.Expr for input values
        length: Period (used for naming). Default: 14
        gamma: Laguerre filter coefficient (0 to 1). Default: 0.5
        offset: Shift result. Default: 0

    Returns:
        pl.Expr: LRSI expression
    """
    close_expr = v_expr(close)

    def _compute(s: pl.Series) -> pl.Series:
        arr = s.to_numpy().astype(np.float64)
        l0, l1, l2, l3 = nb_lrsi_filter(arr, gamma)

        cu = np.zeros(len(arr))
        cd = np.zeros(len(arr))

        cu += np.maximum(l0 - l1, 0)
        cd += np.maximum(l1 - l0, 0)
        cu += np.maximum(l1 - l2, 0)
        cd += np.maximum(l2 - l1, 0)
        cu += np.maximum(l2 - l3, 0)
        cd += np.maximum(l3 - l2, 0)

        denom = cu + cd
        denom = np.where(denom == 0, 1, denom)
        result = 100 * cu / denom
        return pl.Series(values=result, name=s.name)

    result = close_expr.map_batches(_compute, return_dtype=pl.Float64)

    if offset != 0:
        result = result.shift(offset)

    return result.alias(f"LRSI_{length}")
