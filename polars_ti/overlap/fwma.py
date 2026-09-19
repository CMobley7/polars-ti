# -*- coding: utf-8 -*-
# =============================================================================
# Polars FWMA Implementation
# =============================================================================
import polars as pl

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._fir import fir_series
from polars_ti.utils._math import fibonacci
from polars_ti.utils._validate import v_expr


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

    def compute(series: pl.Series) -> pl.Series:
        """Apply the filter once per batch, preserving missing-window semantics."""
        # Reject undersized batches before allocating length-sized weights;
        # an oversized requested window must not allocate proportional memory.
        if len(series) < length:
            return pl.Series([None] * len(series), dtype=pl.Float64)
        weights = fibonacci(n=length, weighted=True)
        if not asc:
            weights = weights[::-1]
        total = weights.sum()
        weights = weights[-length:]
        return fir_series(series, weights) / total

    fwma_expr = close_expr.map_batches(compute, return_dtype=pl.Float64)

    # Apply offset
    if offset != 0:
        fwma_expr = fwma_expr.shift(offset)

    return fwma_expr.alias(f"FWMA_{length}")
