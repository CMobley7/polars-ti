# -*- coding: utf-8 -*-
# =============================================================================
# Polars CDL_Z Implementation
# =============================================================================
import polars as pl

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


def pl_zscore(col: IntoExpr, length: int = 30, ddof: int = 1) -> PlExpr:
    """Polars: Rolling Z-Score calculation."""
    expr = v_expr(col)
    mean = expr.rolling_mean(window_size=length, min_samples=length)
    std = expr.rolling_std(window_size=length, ddof=ddof, min_samples=length)
    return (expr - mean) / std


def pl_cdl_z(
    open_: IntoExpr,
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    length: int = 30,
    full: bool = False,
    ddof: int = 1,
    offset: int = 0,
) -> list[PlExpr]:
    """Polars: Candle Type Z - Rolling Z-Score normalized OHLC

    Normalizes OHLC Candles with a rolling Z Score.

    Source: Kevin Johnson

    Args:
        open_: Column name or pl.Expr for 'open' prices
        high: Column name or pl.Expr for 'high' prices
        low: Column name or pl.Expr for 'low' prices
        close: Column name or pl.Expr for 'close' prices
        length: Rolling window period. Default: 30
        full: If True, use full series length for z-score. Default: False
        ddof: Degrees of freedom for std. Default: 1
        offset: Shift result by N periods. Default: 0

    Returns:
        list[pl.Expr]: List of Z-score expressions for OHLC
    """
    props = f"_{length}_{ddof}"

    # Build base expressions
    exprs = [
        pl_zscore(open_, length, ddof).alias(f"open_Z{props}"),
        pl_zscore(high, length, ddof).alias(f"high_Z{props}"),
        pl_zscore(low, length, ddof).alias(f"low_Z{props}"),
        pl_zscore(close, length, ddof).alias(f"close_Z{props}"),
    ]

    # Apply offset if needed
    if offset != 0:
        exprs = [e.shift(offset) for e in exprs]

    return exprs
