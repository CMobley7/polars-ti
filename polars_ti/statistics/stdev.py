# -*- coding: utf-8 -*-
# =============================================================================
# Polars STDEV Implementation
# =============================================================================
import polars as pl

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


def pl_stdev(
    close: IntoExpr,
    length: int = 30,
    ddof: int = 1,
    offset: int = 0,
) -> pl.Expr:
    """Polars: Rolling Standard Deviation

    Calculates Standard Deviation over a rolling period using native Polars.

    Args:
        close: Column name or pl.Expr for 'close' prices
        length: Rolling window period. Default: 30
        ddof: Delta Degrees of Freedom. Default: 1
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: Standard deviation expression
    """
    close_expr = v_expr(close)
    if close_expr is None:
        return None

    # Native Polars rolling_std
    result = close_expr.rolling_std(window_size=length, min_samples=length, ddof=ddof)

    if offset != 0:
        result = result.shift(offset)

    return result.alias(f"STDEV_{length}")
