# -*- coding: utf-8 -*-
# =============================================================================
# Polars PGO Implementation
# =============================================================================
import polars as pl

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


def pgo(
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    length: int = 14,
    talib: bool = True,
    offset: int = 0,
) -> PlExpr:
    """Polars: Pretty Good Oscillator (PGO)

    Measures the distance of close from its N-day SMA, expressed
    in terms of an average true range.

    Formula: PGO = (close - SMA(close, length)) / EMA(ATR(length), length)

    Args:
        high: Column name or pl.Expr for 'high' prices
        low: Column name or pl.Expr for 'low' prices
        close: Column name or pl.Expr for 'close' prices
        length: Period. Default: 14
        talib: Use TA-Lib SMA/ATR/EMA when available (matches OLD). Default: True
        offset: Shift result. Default: 0

    Returns:
        pl.Expr: PGO expression
    """
    from polars_ti.overlap.sma import sma
    from polars_ti.overlap.ema import ema
    from polars_ti.volatility.atr import atr

    close_expr = v_expr(close)
    high_expr = v_expr(high)
    low_expr = v_expr(low)

    sma_close = sma(close_expr, length=length, talib=talib)
    atr_expr = atr(high_expr, low_expr, close_expr, length=length, talib=talib)
    ema_atr = ema(atr_expr, length=length, talib=talib)

    pgo_expr = (close_expr - sma_close) / ema_atr

    if offset != 0:
        pgo_expr = pgo_expr.shift(offset)

    return pgo_expr.alias(f"PGO_{length}")
