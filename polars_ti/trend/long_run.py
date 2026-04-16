# -*- coding: utf-8 -*-
# =============================================================================
# Polars Long Run Implementation (uses pl_increasing/pl_decreasing)
# =============================================================================
import polars as pl

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr
from polars_ti.trend.increasing import pl_increasing
from polars_ti.trend.decreasing import pl_decreasing


def pl_long_run(
    fast: IntoExpr,
    slow: IntoExpr,
    length: int = 2,
    offset: int = 0,
) -> PlExpr:
    """Polars: Long Run

    Long Run returns 1 when:
    - (fast increasing AND slow decreasing) OR
    - (fast increasing AND slow increasing)

    Uses pl_increasing and pl_decreasing composition.

    Args:
        fast: Column name or pl.Expr for 'fast' signal
        slow: Column name or pl.Expr for 'slow' signal
        length: Period length. Default: 2
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: Long run expression (1 = long trend, 0 = not)
    """
    fast_expr = v_expr(fast)
    slow_expr = v_expr(slow)
    
    if fast_expr is None or slow_expr is None:
        return None
    
    # pb = potential bottom: fast increasing AND slow decreasing
    # bi = both increasing: fast increasing AND slow increasing
    fast_inc = pl_increasing(fast_expr, length=length, asint=False)
    slow_dec = pl_decreasing(slow_expr, length=length, asint=False)
    slow_inc = pl_increasing(slow_expr, length=length, asint=False)
    
    pb = fast_inc & slow_dec
    bi = fast_inc & slow_inc
    result = (pb | bi).cast(pl.Int64)
    
    if offset != 0:
        result = result.shift(offset)
    
    return result.alias(f"LR_{length}")

