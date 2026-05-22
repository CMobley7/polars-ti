# -*- coding: utf-8 -*-
from numba import njit
from polars_ti.utils._numba import nb_idiff, nb_shift


@njit(cache=True)
def nb_roc(x, n, k):
    return k * nb_idiff(x, n) / nb_shift(x, n)


# =============================================================================
# Polars ROC (Rate of Change) Implementation
# =============================================================================
import polars as pl
import numpy as np

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


def pl_roc(
    close: IntoExpr,
    length: int = 10,
    scalar: float = 100.0,
    talib: bool = True,
    offset: int = 0,
) -> PlExpr:
    """Polars: Rate of Change (ROC)

    Measures percent change in price.
    ROC = scalar * (close - close[n]) / close[n]

    Args:
        close: Column name or pl.Expr for 'close' prices
        length: Lookback period. Default: 10
        scalar: Magnification factor. Default: 100
        talib: If True and TA-Lib installed, use TA-Lib. Default: True
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: ROC expression
    """
    from polars_ti.maps import Imports
    from polars_ti.utils import v_talib

    close_expr = v_expr(close)
    if close_expr is None:
        return None

    _use_talib = Imports["talib"] and v_talib(talib) and length > 1
    _length = length
    _scalar = scalar

    if _use_talib:

        def compute_roc(s: pl.Series) -> pl.Series:
            from talib import ROC as TALIB_ROC

            arr = s.to_numpy().astype(np.float64)
            result = TALIB_ROC(arr, timeperiod=_length)
            return pl.Series(result)

        roc_expr = close_expr.map_batches(compute_roc, return_dtype=pl.Float64)
    else:
        # Pure Polars: ROC = scalar * (close - close.shift(n)) / close.shift(n)
        shifted = close_expr.shift(length)
        roc_expr = _scalar * (close_expr - shifted) / shifted

    if offset != 0:
        roc_expr = roc_expr.shift(offset)

    return roc_expr.alias(f"ROC_{length}")
