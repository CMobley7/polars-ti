# -*- coding: utf-8 -*-
# =============================================================================
# Polars FWMA Implementation
# =============================================================================
import polars as pl

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr
from polars_ti.utils._math import fibonacci


def fwma(
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

    _length = length
    _weights: list[float] | None = None
    _total: float | None = None

    def fib_weighted_mean(s: pl.Series) -> float:
        nonlocal _weights, _total
        vals = s.to_numpy()
        if len(vals) < _length:
            return float("nan")
        if _weights is None:
            # Build the length-sized weight vector lazily — only once a full window
            # exists — so an absurd length (>> data) returns all-null instead of
            # eagerly allocating an O(length) array (a hang/OOM on e.g. length=1e9).
            fibs = fibonacci(n=_length, weighted=True)
            if not asc:
                fibs = fibs[::-1]
            _total = fibs.sum()
            _weights = fibs.tolist()
        return (vals * _weights[-len(vals) :]).sum() / _total

    fwma_expr = close_expr.rolling_map(function=fib_weighted_mean, window_size=length, min_samples=length)

    # Apply offset
    if offset != 0:
        fwma_expr = fwma_expr.shift(offset)

    return fwma_expr.alias(f"FWMA_{length}")
