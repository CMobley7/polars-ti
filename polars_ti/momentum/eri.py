# -*- coding: utf-8 -*-
# =============================================================================
# Polars ERI (Elder Ray Index) Implementation
# =============================================================================
import polars as pl

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


def pl_eri(
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    length: int = 13,
    offset: int = 0,
) -> list[PlExpr]:
    """Polars: Elder Ray Index (ERI)

    Elder's Bull and Bear Power. Bull Power = High - EMA, Bear Power = Low - EMA.
    Measures buyers' and sellers' strength relative to average.

    Args:
        high: Column name or pl.Expr for 'high' prices
        low: Column name or pl.Expr for 'low' prices
        close: Column name or pl.Expr for 'close' prices
        length: EMA period. Default: 13
        offset: Shift result. Default: 0

    Returns:
        list[pl.Expr]: [BULLP_13, BEARP_13] expressions
    """
    from polars_ti.overlap.ema import pl_ema

    high_expr = v_expr(high)
    low_expr = v_expr(low)
    close_expr = v_expr(close)

    # EMA of close
    ema_expr = pl_ema(close_expr, length=length, offset=0)

    # Bull Power = High - EMA, Bear Power = Low - EMA
    bull_expr = high_expr - ema_expr
    bear_expr = low_expr - ema_expr

    if offset != 0:
        bull_expr = bull_expr.shift(offset)
        bear_expr = bear_expr.shift(offset)

    return [bull_expr.alias(f"BULLP_{length}"), bear_expr.alias(f"BEARP_{length}")]
