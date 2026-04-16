# -*- coding: utf-8 -*-
# =============================================================================
# Polars TTM Trend Implementation
# =============================================================================
import polars as pl

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


def pl_ttm_trend(
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    length: int = 6,
    offset: int = 0,
) -> PlExpr:
    """Polars: TTM Trend

    Checks if close is above/below the rolling average of HL2 over length bars.
    Returns 1 for uptrend, -1 for downtrend.

    Formula:
        trend_avg = rolling_mean(HL2, length)
        TTM_TRND = 1 if close > trend_avg else -1

    Args:
        high: Column name or pl.Expr for 'high' prices
        low: Column name or pl.Expr for 'low' prices
        close: Column name or pl.Expr for 'close' prices
        length: Period. Default: 6
        offset: Shift result. Default: 0

    Returns:
        pl.Expr: TTM Trend struct expression with TTM_TRND column
    """
    from polars_ti.overlap.hl2 import pl_hl2

    high_expr = v_expr(high)
    low_expr = v_expr(low)
    close_expr = v_expr(close)

    hl2_expr = pl_hl2(high_expr, low_expr)
    trend_avg = hl2_expr.rolling_mean(window_size=length)

    ttm = pl.when(close_expr > trend_avg).then(1).otherwise(-1)

    if offset != 0:
        ttm = ttm.shift(offset)

    return ttm.alias(f"TTM_TRND_{length}")
