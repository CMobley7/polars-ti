# -*- coding: utf-8 -*-
# =============================================================================
# Polars MEDIAN Implementation
# =============================================================================
import polars as pl

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


def pl_median(
    close: IntoExpr,
    length: int = 30,
    offset: int = 0,
) -> pl.Expr:
    """Polars: Rolling Median

    Calculates the Median over a rolling period using native Polars.

    Args:
        close: Column name or pl.Expr for 'close' prices
        length: Rolling window period. Default: 30
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: Median expression
    """
    close_expr = v_expr(close)
    if close_expr is None:
        return None

    # Use native Polars rolling_median
    result = close_expr.rolling_median(window_size=length, min_samples=length)

    # Apply offset if needed
    if offset != 0:
        result = result.shift(offset)

    return result.alias(f"MEDIAN_{length}")

