import numpy as np

# -*- coding: utf-8 -*-
# =============================================================================
# Polars PWMA Implementation
# =============================================================================
import polars as pl

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._fir import fir_series
from polars_ti.utils._math import pascals_triangle
from polars_ti.utils._validate import v_expr


def pwma(
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

    def compute(series: pl.Series) -> pl.Series:
        """Apply the filter once per batch, preserving missing-window semantics."""
        # Reject undersized batches before allocating length-sized weights;
        # an oversized requested window must not allocate proportional memory.
        if len(series) < length:
            return pl.Series([None] * len(series), dtype=pl.Float64)
        weights = np.asarray(pascals_triangle(n=length - 1, weighted=True), dtype=np.float64)
        if not asc:
            weights = weights[::-1]
        return fir_series(series, weights)

    pwma_expr = close_expr.map_batches(compute, return_dtype=pl.Float64)

    # Apply offset
    if offset != 0:
        pwma_expr = pwma_expr.shift(offset)

    return pwma_expr.alias(f"PWMA_{length}")
