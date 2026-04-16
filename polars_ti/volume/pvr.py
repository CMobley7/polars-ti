# -*- coding: utf-8 -*-
# =============================================================================
# Polars PVR (Price Volume Rank) Implementation
# =============================================================================
import polars as pl

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


def pl_pvr(
    close: IntoExpr,
    volume: IntoExpr,
    drift: int = 1,
) -> PlExpr:
    """Polars: Price Volume Rank (PVR)

    Returns categorical rank 1-4 based on price and volume change directions.

    Args:
        close: Column name or pl.Expr for 'close' prices
        volume: Column name or pl.Expr for 'volume'
        drift: Difference period. Default: 1

    Returns:
        pl.Expr: PVR expression (values 1-4)
    """
    close_expr = v_expr(close)
    volume_expr = v_expr(volume)
    
    if close_expr is None or volume_expr is None:
        return None
    
    close_diff = close_expr.diff(drift)
    volume_diff = volume_expr.diff(drift)
    
    # PVR categories:
    # 1: close up, volume up
    # 2: close up, volume down
    # 3: close down, volume up
    # 4: close down, volume down
    pvr_expr = (
        pl.when((close_diff >= 0) & (volume_diff >= 0)).then(1)
        .when((close_diff >= 0) & (volume_diff < 0)).then(2)
        .when((close_diff < 0) & (volume_diff >= 0)).then(3)
        .when((close_diff < 0) & (volume_diff < 0)).then(4)
        .otherwise(None)
    )
    
    return pvr_expr.alias("PVR")

