# -*- coding: utf-8 -*-
# =============================================================================
# Polars RWI Implementation
# =============================================================================
import polars as pl

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


def rwi(
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    length: int = 14,
    offset: int = 0,
) -> PlExpr:
    """Polars: Random Walk Index (RWI)

    Differentiates whether price follows a trend or a random walk.

    Formula:
        RWI_High = (high - low.shift(length)) / (ATR * sqrt(length))
        RWI_Low  = (high.shift(length) - low) / (ATR * sqrt(length))

    Args:
        high: Column name or pl.Expr for 'high' prices
        low: Column name or pl.Expr for 'low' prices
        close: Column name or pl.Expr for 'close' prices
        length: Period. Default: 14
        offset: Shift result. Default: 0

    Returns:
        pl.Expr: Struct with RWIh and RWIl columns
    """
    from polars_ti.volatility.atr import atr

    high_expr = v_expr(high)
    low_expr = v_expr(low)
    close_expr = v_expr(close)

    atr_expr = atr(high_expr, low_expr, close_expr, length=length, talib=False)
    denom = atr_expr * (length**0.5)

    rwi_high = (high_expr - low_expr.shift(length)) / denom
    rwi_low = (high_expr.shift(length) - low_expr) / denom

    if offset != 0:
        rwi_high = rwi_high.shift(offset)
        rwi_low = rwi_low.shift(offset)

    return pl.struct(
        rwi_high.alias(f"RWIh_{length}"),
        rwi_low.alias(f"RWIl_{length}"),
    ).alias(f"RWI_{length}")
