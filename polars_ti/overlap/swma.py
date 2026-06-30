# -*- coding: utf-8 -*-
# =============================================================================
# Polars SWMA Implementation
# =============================================================================
import polars as pl

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr
from polars_ti.utils._math import symmetric_triangle


def swma(
    close: IntoExpr,
    length: int = 10,
    offset: int = 0,
) -> PlExpr:
    """Polars: Symmetric Weighted Moving Average (SWMA)

    Symmetric Weighted Moving Average where weights are based on a symmetric
    triangle. For example: n=3 -> [1, 2, 1], n=4 -> [1, 2, 2, 1], etc...

    Source:
        https://www.tradingview.com/study-script-reference/#fun_swma

    Args:
        close: Column name or pl.Expr for 'close' prices
        length: Rolling window period. Default: 10
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: SWMA expression for lazy evaluation
    """
    close_expr = v_expr(close)
    if close_expr is None:
        return None

    # Get symmetric triangle weights
    triangle_weights = symmetric_triangle(length, weighted=True)
    weights_list = triangle_weights.tolist()

    _length = length
    _weights = weights_list

    def triangle_weighted_mean(s: pl.Series) -> float:
        vals = s.to_numpy()
        if len(vals) < _length:
            return float("nan")
        return (vals * _weights[-len(vals) :]).sum()

    swma_expr = close_expr.rolling_map(function=triangle_weighted_mean, window_size=length, min_samples=length)

    # Apply offset
    if offset != 0:
        swma_expr = swma_expr.shift(offset)

    return swma_expr.alias(f"SWMA_{length}")
