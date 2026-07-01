# -*- coding: utf-8 -*-
# =============================================================================
# Polars MARKETFI (Market Facilitation Index) Implementation
# =============================================================================
import polars as pl

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._math import non_zero_range
from polars_ti.utils._validate import v_expr


def marketfi(
    high: IntoExpr,
    low: IntoExpr,
    volume: IntoExpr,
    offset: int = 0,
) -> PlExpr:
    """Polars: Market Facilitation Index (MARKETFI)

    Developed by Dr. Bill Williams.  Measures efficiency of price movement
    per unit of volume.

    Formula:
        MARKETFI = (High - Low) / Volume

    Sources:
        Bill Williams, "Trading Chaos", 1995
        https://www.investopedia.com/terms/m/marketfacilitationindex.asp

    Args:
        high: Column name or pl.Expr for 'high' prices.
        low: Column name or pl.Expr for 'low' prices.
        volume: Column name or pl.Expr for volume.
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: MARKETFI expression.
    """
    high_expr = v_expr(high)
    low_expr = v_expr(low)
    volume_expr = v_expr(volume)
    if any(e is None for e in [high_expr, low_expr, volume_expr]):
        return None

    _offset = offset

    result = non_zero_range(high_expr, low_expr) / volume_expr

    if _offset != 0:
        result = result.shift(_offset)

    return result.alias("MARKETFI")
