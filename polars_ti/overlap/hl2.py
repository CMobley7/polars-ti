# -*- coding: utf-8 -*-
# =============================================================================
# Polars HL2 Implementation
# =============================================================================
import polars as pl

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


def pl_hl2(high: IntoExpr, low: IntoExpr) -> PlExpr:
    """Polars: HL2 - Midpoint of High and Low

    Args:
        high: Column name or pl.Expr for 'high' prices
        low: Column name or pl.Expr for 'low' prices

    Returns:
        pl.Expr: HL2 expression
    """
    high_expr = v_expr(high)
    low_expr = v_expr(low)
    if high_expr is None or low_expr is None:
        return None

    return ((high_expr + low_expr) / 2).alias("HL2")
