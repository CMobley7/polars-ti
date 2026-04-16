# -*- coding: utf-8 -*-
# =============================================================================
# Polars RAINBOW Implementation (pl_sma composition)
# =============================================================================
import polars as pl
import numpy as np

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr
from polars_ti.overlap.sma import pl_sma


def pl_rainbow(
    close: IntoExpr,
    length: int = 2,
    num_ribbons: int = 10,
    offset: int = 0,
) -> pl.Expr:
    """Polars: Rainbow Charts

    Sequential SMAs where each is calculated on the previous SMA.
    Returns a struct with RAINBOW_1 through RAINBOW_{num_ribbons} fields.

    Args:
        close: Column name or pl.Expr for 'close'
        length: SMA period. Default: 2
        num_ribbons: Number of rainbow bands. Default: 10
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: Struct with RAINBOW_1 through RAINBOW_{num_ribbons} fields
    """
    close_expr = v_expr(close)
    if close_expr is None:
        return None
    
    # Build chained SMA expressions - just like Pandas version!
    ribbon_exprs = []
    prev_expr = close_expr
    
    for i in range(1, num_ribbons + 1):
        # Each SMA is calculated on the previous SMA
        sma_expr = pl_sma(prev_expr, length=length, talib=False)
        
        # Apply offset if needed
        if offset != 0:
            sma_expr = sma_expr.shift(offset)
        
        ribbon_exprs.append(sma_expr.alias(f"RAINBOW_{i}"))
        prev_expr = sma_expr
    
    return pl.struct(ribbon_exprs).alias(f"RAINBOW_{length}_{num_ribbons}")


