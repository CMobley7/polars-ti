# -*- coding: utf-8 -*-
# =============================================================================
# Polars Short Run Implementation (uses pl_increasing/pl_decreasing)
# =============================================================================
import polars as pl

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr
from polars_ti.trend.increasing import increasing
from polars_ti.trend.decreasing import decreasing


def short_run(
    fast: IntoExpr,
    slow: IntoExpr,
    length: int = 2,
    offset: int = 0,
) -> PlExpr:
    """Polars: Short Run

    Short Run returns 1 when:
    - (fast decreasing AND slow increasing) OR
    - (fast decreasing AND slow decreasing)

    Uses pl_increasing and pl_decreasing composition.

    Args:
        fast: Column name or pl.Expr for 'fast' signal
        slow: Column name or pl.Expr for 'slow' signal
        length: Period length. Default: 2
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: Short run expression (1 = short trend, 0 = not)
    """
    fast_expr = v_expr(fast)
    slow_expr = v_expr(slow)

    if fast_expr is None or slow_expr is None:
        return None

    # pt = potential top: fast decreasing AND slow increasing
    # bd = both decreasing: fast decreasing AND slow decreasing
    fast_dec = decreasing(fast_expr, length=length, asint=False)
    slow_inc = increasing(slow_expr, length=length, asint=False)
    slow_dec = decreasing(slow_expr, length=length, asint=False)

    pt = fast_dec & slow_inc
    bd = fast_dec & slow_dec
    result = (pt | bd).cast(pl.Int64)

    if offset != 0:
        result = result.shift(offset)

    return result.alias(f"SR_{length}")
