# -*- coding: utf-8 -*-
# =============================================================================
# Polars True Range Implementation (Pure Native Polars)
# =============================================================================
import polars as pl

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


def pl_true_range(
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    drift: int = 1,
    offset: int = 0,
) -> pl.Expr:
    """Polars: True Range

    Uses pure native Polars expressions.

    An method to expand a classical range (high minus low) to include
    possible gap scenarios.

    Formula: TR = max(High - Low, |High - Prev Close|, |Prev Close - Low|)

    Sources:
        https://www.macroption.com/true-range/

    Args:
        high: Column name or pl.Expr for 'high'
        low: Column name or pl.Expr for 'low'
        close: Column name or pl.Expr for 'close'
        drift: Shift period for previous close. Default: 1
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: True Range expression
    """
    high_expr = v_expr(high)
    low_expr = v_expr(low)
    close_expr = v_expr(close)

    if high_expr is None or low_expr is None or close_expr is None:
        return None

    # Previous close
    prev_close = close_expr.shift(drift)

    # Three component ranges
    hl_range = high_expr - low_expr
    hc_range = (high_expr - prev_close).abs()
    lc_range = (prev_close - low_expr).abs()

    # True Range = max of all three
    result = pl.max_horizontal(hl_range, hc_range, lc_range)

    if offset != 0:
        result = result.shift(offset)

    return result.alias(f"TRUERANGE_{drift}")
