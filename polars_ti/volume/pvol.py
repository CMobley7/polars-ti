# -*- coding: utf-8 -*-
# =============================================================================
# Polars PVOL (Price-Volume) Implementation
# =============================================================================
import polars as pl

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


def pl_pvol(
    close: IntoExpr,
    volume: IntoExpr,
    signed: bool = False,
    offset: int = 0,
) -> PlExpr:
    """Polars: Price-Volume (PVOL)

    Returns the product of price and volume.

    Args:
        close: Column name or pl.Expr for 'close' prices
        volume: Column name or pl.Expr for 'volume'
        signed: If True, apply sign of close diff. Default: False
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: PVOL expression
    """
    close_expr = v_expr(close)
    volume_expr = v_expr(volume)
    
    if close_expr is None or volume_expr is None:
        return None
    
    pvol_expr = close_expr * volume_expr
    
    if signed:
        close_diff = close_expr.diff()
        sign = pl.when(close_diff > 0).then(1).when(close_diff < 0).then(-1).otherwise(0)
        sign = sign.fill_null(1)
        pvol_expr = pvol_expr * sign
    
    if offset != 0:
        pvol_expr = pvol_expr.shift(offset)
    
    return pvol_expr.alias("PVOL")

