# -*- coding: utf-8 -*-
# =============================================================================
# Polars BIAS Implementation
# =============================================================================
import polars as pl

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr
from polars_ti.ma import ma


def bias(
    close: IntoExpr,
    length: int = 26,
    mamode: str = "sma",
    offset: int = 0,
) -> PlExpr:
    """Polars: Bias (BIAS)

    Rate of change between price and a moving average.
    BIAS = (close / MA) - 1

    Args:
        close: Column name or pl.Expr for 'close' prices
        length: MA period. Default: 26
        mamode: Moving average type. Default: 'sma'
        offset: Shift result. Default: 0

    Returns:
        pl.Expr: BIAS expression
    """
    close_expr = v_expr(close)

    # Use pl_ma for code reuse
    ma_expr = ma(mamode, close_expr, length=length, talib=False)
    ma_name = f"{mamode.upper()}_{length}"

    # BIAS = (close / MA) - 1 - Pure Polars expression
    bias_expr = (close_expr / ma_expr) - 1.0

    if offset != 0:
        bias_expr = bias_expr.shift(offset)

    return bias_expr.alias(f"BIAS_{ma_name}")
