# -*- coding: utf-8 -*-
# =============================================================================
# Polars VWMA Implementation (Composition: pl_sma)
# =============================================================================
import polars as pl

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


def vwma(
    close: IntoExpr,
    volume: IntoExpr,
    length: int = 10,
    offset: int = 0,
) -> pl.Expr:
    """Polars: Volume Weighted Moving Average (VWMA)

    Uses composition: pl_sma for calculating moving averages.

    Formula: VWMA = SMA(Close * Volume, n) / SMA(Volume, n)

    Note: Since SMA = SUM/n, the n's cancel out giving VWMA = SUM(pv)/SUM(v)

    Sources:
        https://www.motivewave.com/studies/volume_weighted_moving_average.htm

    Args:
        close: Column name or pl.Expr for 'close'
        volume: Column name or pl.Expr for 'volume'
        length: Rolling window period. Default: 10
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: VWMA expression
    """
    from polars_ti.overlap.sma import sma

    close_expr = v_expr(close)
    volume_expr = v_expr(volume)

    if close_expr is None or volume_expr is None:
        return None

    # VWMA = SMA(close * volume) / SMA(volume) - matches Pandas exactly
    pv = close_expr * volume_expr
    result = sma(pv, length=length) / sma(volume_expr, length=length)

    if offset != 0:
        result = result.shift(offset)

    return result.alias(f"VWMA_{length}")
