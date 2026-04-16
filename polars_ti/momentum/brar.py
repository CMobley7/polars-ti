# -*- coding: utf-8 -*-
# =============================================================================
# Polars BRAR Implementation  
# =============================================================================
import polars as pl

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._math import pl_non_zero_range
from polars_ti.utils._validate import v_expr


def pl_brar(
    open_: IntoExpr,
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    length: int = 26,
    scalar: float = 100.0,
    drift: int = 1,
    offset: int = 0,
) -> list[PlExpr]:
    """Polars: BRAR (BR and AR)

    Returns list with AR and BR expressions.

    Args:
        open_: Column name or pl.Expr for 'open'
        high: Column name or pl.Expr for 'high'
        low: Column name or pl.Expr for 'low'
        close: Column name or pl.Expr for 'close'
        length: Period. Default: 26
        scalar: Magnification factor. Default: 100
        drift: Difference period. Default: 1
        offset: Shift result. Default: 0

    Returns:
        list[pl.Expr]: [AR, BR] expressions
    """
    open_expr = v_expr(open_)
    high_expr = v_expr(high)
    low_expr = v_expr(low)
    close_expr = v_expr(close)
    
    # AR = scalar * rolling_sum(high - open) / rolling_sum(open - low)
    # Use pl_non_zero_range to match Pandas implementation
    high_open = pl_non_zero_range(high_expr, open_expr)
    open_low = pl_non_zero_range(open_expr, low_expr)
    
    ar_num = high_open.rolling_sum(window_size=length, min_samples=length)
    ar_den = open_low.rolling_sum(window_size=length, min_samples=length)
    ar_expr = scalar * ar_num / ar_den
    
    # BR = scalar * rolling_sum(max(0, high - close.shift)) / rolling_sum(max(0, close.shift - low))
    close_shifted = close_expr.shift(drift)
    # Use pl_non_zero_range for base ranges, then clip to 0
    hcy = pl_non_zero_range(high_expr, close_shifted).clip(lower_bound=0.0)
    cyl = pl_non_zero_range(close_shifted, low_expr).clip(lower_bound=0.0)
    
    br_num = hcy.rolling_sum(window_size=length, min_samples=length)
    br_den = cyl.rolling_sum(window_size=length, min_samples=length)
    br_expr = scalar * br_num / br_den
    
    if offset != 0:
        ar_expr = ar_expr.shift(offset)
        br_expr = br_expr.shift(offset)
    
    return [
        ar_expr.alias(f"AR_{length}"),
        br_expr.alias(f"BR_{length}"),
    ]


