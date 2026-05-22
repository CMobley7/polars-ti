# -*- coding: utf-8 -*-
# =============================================================================
# Polars CKSP Implementation
# =============================================================================
import polars as pl

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


def pl_cksp(
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    p: int = 10,
    x: float = 1.0,
    q: int = 9,
    mamode: str = "rma",
    offset: int = 0,
) -> PlExpr:
    """Polars: Chande Kroll Stop (CKSP)

    Trend-following indicator using ATR-based stops.

    Formula:
        long_stop = rolling_max(high, p) - x * ATR(p)  -> rolling_max(q)
        short_stop = rolling_min(low, p) + x * ATR(p)  -> rolling_min(q)

    Args:
        high: Column name or pl.Expr for 'high' prices
        low: Column name or pl.Expr for 'low' prices
        close: Column name or pl.Expr for 'close' prices
        p: ATR period. Default: 10
        x: ATR multiplier. Default: 1
        q: Second stop period. Default: 9
        mamode: MA mode for ATR. Default: 'rma'
        offset: Shift result. Default: 0

    Returns:
        pl.Expr: Struct with CKSPl and CKSPs columns
    """
    from polars_ti.volatility.atr import pl_atr

    high_expr = v_expr(high)
    low_expr = v_expr(low)
    close_expr = v_expr(close)

    atr_expr = pl_atr(high_expr, low_expr, close_expr, length=p, talib=False)

    long_stop_ = high_expr.rolling_max(window_size=p) - x * atr_expr
    long_stop = long_stop_.rolling_max(window_size=q)

    short_stop_ = low_expr.rolling_min(window_size=p) + x * atr_expr
    short_stop = short_stop_.rolling_min(window_size=q)

    if offset != 0:
        long_stop = long_stop.shift(offset)
        short_stop = short_stop.shift(offset)

    _props = f"_{p}_{x}_{q}"
    return pl.struct(
        long_stop.alias(f"CKSPl{_props}"),
        short_stop.alias(f"CKSPs{_props}"),
    ).alias(f"CKSP{_props}")
