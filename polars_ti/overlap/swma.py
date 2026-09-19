import numpy as np

# -*- coding: utf-8 -*-
# =============================================================================
# Polars SWMA Implementation
# =============================================================================
import polars as pl

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._fir import fir_series
from polars_ti.utils._math import symmetric_triangle
from polars_ti.utils._validate import v_expr


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

    def compute(series: pl.Series) -> pl.Series:
        """Apply the filter once per batch, preserving missing-window semantics."""
        if len(series) < length:
            return pl.Series([None] * len(series), dtype=pl.Float64)
        weights = np.asarray(symmetric_triangle(length, weighted=True), dtype=np.float64)
        return fir_series(series, weights)

    swma_expr = close_expr.map_batches(compute, return_dtype=pl.Float64)

    # Apply offset
    if offset != 0:
        swma_expr = swma_expr.shift(offset)

    return swma_expr.alias(f"SWMA_{length}")
