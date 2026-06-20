# -*- coding: utf-8 -*-
# =============================================================================
# Polars PDIST Implementation (Pure Expressions)
# =============================================================================
import polars as pl

from polars_ti._typing import IntoExpr
from polars_ti.utils._math import non_zero_range
from polars_ti.utils._validate import v_expr


def pdist(
    open_: IntoExpr,
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    drift: int = 1,
    offset: int = 0,
) -> pl.Expr:
    """Polars: Price Distance (PDIST)

    Pure Polars expressions - measures the "distance" covered by price movements.

    Sources:
        https://www.prorealcode.com/prorealtime-indicators/pricedistance/

    Args:
        open_: Column name or pl.Expr for 'open'
        high: Column name or pl.Expr for 'high'
        low: Column name or pl.Expr for 'low'
        close: Column name or pl.Expr for 'close'
        drift: The difference period. Default: 1
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: PDIST expression
    """
    open_expr = v_expr(open_)
    high_expr = v_expr(high)
    low_expr = v_expr(low)
    close_expr = v_expr(close)

    if open_expr is None or high_expr is None or low_expr is None or close_expr is None:
        return None

    # PDIST = 2 * (high - low) + |open - close.shift| - |close - open|
    # Using non_zero protection like Pandas
    hl_range = non_zero_range(high_expr, low_expr)
    oc_shift_range = (open_expr - close_expr.shift(drift)).abs()
    co_range = (close_expr - open_expr).abs()

    result = pl.lit(2.0) * hl_range + oc_shift_range - co_range

    # Apply offset
    if offset != 0:
        result = result.shift(offset)

    return result.alias("PDIST")
