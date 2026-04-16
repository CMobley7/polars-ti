# -*- coding: utf-8 -*-
# =============================================================================
# Polars FWMA Implementation
# =============================================================================
import polars as pl

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr
from polars_ti.utils._math import fibonacci


def pl_fwma(
    close: IntoExpr,
    length: int = 10,
    asc: bool = True,
    offset: int = 0,
) -> PlExpr:
    """Polars: Fibonacci's Weighted Moving Average (FWMA)

    Fibonacci's Weighted Moving Average is similar to a Weighted Moving
    Average (WMA) where the weights are based on the Fibonacci Sequence.

    Source: Kevin Johnson

    Args:
        close: Column name or pl.Expr for 'close' prices
        length: Rolling window period. Default: 10
        asc: Recent values weigh more. Default: True
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: FWMA expression for lazy evaluation
    """
    close_expr = v_expr(close)
    if close_expr is None:
        return None

    # Get Fibonacci weights
    fibs = fibonacci(n=length, weighted=True)
    if not asc:
        fibs = fibs[::-1]
    weights_sum = fibs.sum()
    fib_list = fibs.tolist()

    _length = length
    _weights = fib_list
    _total = weights_sum

    def fib_weighted_mean(s: pl.Series) -> float:
        vals = s.to_numpy()
        if len(vals) < _length:
            return float('nan')
        return (vals * _weights[-len(vals):]).sum() / _total

    fwma_expr = close_expr.rolling_map(
        function=fib_weighted_mean,
        window_size=length,
        min_samples=length
    )

    # Apply offset
    if offset != 0:
        fwma_expr = fwma_expr.shift(offset)

    return fwma_expr.alias(f"FWMA_{length}")
