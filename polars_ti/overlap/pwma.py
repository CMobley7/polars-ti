# -*- coding: utf-8 -*-
# =============================================================================
# Polars PWMA Implementation
# =============================================================================
import polars as pl

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr
from polars_ti.utils._math import pascals_triangle


def pl_pwma(
    close: IntoExpr,
    length: int = 10,
    asc: bool = True,
    offset: int = 0,
) -> PlExpr:
    """Polars: Pascal's Weighted Moving Average (PWMA)

    Pascal's Weighted Moving Average is similar to a symmetric triangular
    window except PWMA's weights are based on Pascal's Triangle.

    Source: Kevin Johnson

    Args:
        close: Column name or pl.Expr for 'close' prices
        length: Rolling window period. Default: 10
        asc: Recent values weigh more. Default: True
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: PWMA expression for lazy evaluation
    """
    close_expr = v_expr(close)
    if close_expr is None:
        return None

    # Get Pascal's Triangle weights (row n-1 for length n)
    pascal_weights = pascals_triangle(n=length - 1, weighted=True)
    if not asc:
        pascal_weights = pascal_weights[::-1]
    weights_list = pascal_weights.tolist()

    _length = length
    _weights = weights_list

    def pascal_weighted_mean(s: pl.Series) -> float:
        vals = s.to_numpy()
        if len(vals) < _length:
            return float('nan')
        return (vals * _weights[-len(vals):]).sum()

    pwma_expr = close_expr.rolling_map(
        function=pascal_weighted_mean,
        window_size=length,
        min_samples=length
    )

    # Apply offset
    if offset != 0:
        pwma_expr = pwma_expr.shift(offset)

    return pwma_expr.alias(f"PWMA_{length}")
