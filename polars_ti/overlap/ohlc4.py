# -*- coding: utf-8 -*-
# =============================================================================
# Polars OHLC4 Implementation
# =============================================================================
import polars as pl

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


def pl_ohlc4(open_: IntoExpr, high: IntoExpr, low: IntoExpr, close: IntoExpr) -> PlExpr:
    """Polars: OHLC4 - Average of Open, High, Low, Close

    Args:
        open_: Column name or pl.Expr for 'open' prices
        high: Column name or pl.Expr for 'high' prices
        low: Column name or pl.Expr for 'low' prices
        close: Column name or pl.Expr for 'close' prices

    Returns:
        pl.Expr: OHLC4 expression
    """
    open_expr = v_expr(open_)
    high_expr = v_expr(high)
    low_expr = v_expr(low)
    close_expr = v_expr(close)
    
    return ((open_expr + high_expr + low_expr + close_expr) / 4).alias("OHLC4")

