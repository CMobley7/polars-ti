# -*- coding: utf-8 -*-
# =============================================================================
# Polars MMAR Implementation
# =============================================================================
import polars as pl
import numpy as np

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr
from polars_ti.overlap.ema import pl_ema


def pl_mmar(
    close: IntoExpr,
    length: int = 10,
    step: int = 5,
    num_ribbons: int = 6,
    offset: int = 0,
) -> pl.Expr:
    """Polars: Madrid Moving Average Ribbon (MMAR)

    Returns a struct with multiple MMAR_period fields.

    Args:
        close: Column name or pl.Expr for 'close'
        length: Initial EMA period. Default: 10
        step: Period increment. Default: 5
        num_ribbons: Number of ribbons. Default: 6
        offset: Offset periods. Default: 0

    Returns:
        pl.Expr: Struct with MMAR_period fields
    """
    # Build list of EMA expressions with incrementing periods
    ema_exprs = []
    for i in range(num_ribbons):
        period = length + (i * step)
        ema_expr = pl_ema(close, length=period, offset=offset, talib=False).alias(f"MMAR_{period}")
        ema_exprs.append(ema_expr)
    
    return pl.struct(ema_exprs).alias(f"MMAR_{length}_{step}_{num_ribbons}")


