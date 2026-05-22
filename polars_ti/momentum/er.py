# -*- coding: utf-8 -*-
# =============================================================================
# Polars ER (Efficiency Ratio) Implementation
# =============================================================================
import polars as pl

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


def pl_er(
    close: IntoExpr,
    length: int = 10,
    drift: int = 1,
    offset: int = 0,
) -> PlExpr:
    """Polars: Efficiency Ratio (ER)

    By Perry J. Kaufman. Measures trend efficiency by comparing net price
    change to total volatility over N periods.
    Formula: ER = |change over N| / sum(|changes|)

    Args:
        close: Column name or pl.Expr for 'close' prices
        length: Period. Default: 10
        drift: Difference period for volatility. Default: 1
        offset: Shift result. Default: 0

    Returns:
        pl.Expr: ER expression (values 0-1, higher = more efficient trend)
    """
    close_expr = v_expr(close)

    # Net change over length periods
    abs_diff = (close_expr - close_expr.shift(length)).abs()

    # Sum of absolute period-to-period changes
    abs_volatility = (close_expr - close_expr.shift(drift)).abs()
    abs_volatility_rsum = abs_volatility.rolling_sum(window_size=length)

    # ER = net change / total volatility
    er_expr = abs_diff / abs_volatility_rsum

    if offset != 0:
        er_expr = er_expr.shift(offset)

    return er_expr.alias(f"ER_{length}")
