# -*- coding: utf-8 -*-
# =============================================================================
# Polars HMA Implementation
# =============================================================================
import polars as pl
import numpy as np

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr
from polars_ti.overlap.wma import nb_wma


def hma(
    close: IntoExpr,
    length: int = 10,
    offset: int = 0,
) -> PlExpr:
    """Polars: Hull Moving Average (HMA)

    HMA = WMA(2*WMA(half) - WMA(full), sqrt(length))

    Sources:
        https://alanhull.com/hull-moving-average

    Args:
        close: Column name or pl.Expr for 'close' prices
        length: Rolling window period. Default: 10
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: HMA expression
    """
    close_expr = v_expr(close)
    if close_expr is None:
        return None

    half_length = int(length / 2)
    sqrt_length = int(length**0.5)
    _length = length

    def compute_hma(s: pl.Series) -> pl.Series:
        arr = s.to_numpy().astype(np.float64)
        # Reuse nb_wma from wma.py (no duplicate kernels!)
        wmaf = nb_wma(arr, half_length, True, True)
        wmas = nb_wma(arr, _length, True, True)
        intermediate = 2.0 * wmaf - wmas
        result = nb_wma(intermediate, sqrt_length, True, True)
        return pl.Series(result)

    result = close_expr.map_batches(compute_hma, return_dtype=pl.Float64)

    if offset != 0:
        result = result.shift(offset)

    return result.alias(f"HMA_{length}")
