# -*- coding: utf-8 -*-
# =============================================================================
# Polars DSP Implementation
# =============================================================================
import polars as pl

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.overlap.ema import ema
from polars_ti.utils._validate import v_expr


def dsp(close: IntoExpr, length: int = 14, offset: int = 0) -> PlExpr:
    """Polars: Detrended Synthetic Price (DSP)

    Removes the trend component from price data to reveal cycles.
    DSP = close - EMA(close, length)

    Sources:
        Cycle Analytics for Traders by John F. Ehlers

    Args:
        close: Column name or pl.Expr for 'close' prices
        length: The EMA period. Default: 14
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: DSP expression
    """
    close_expr = v_expr(close)

    # Use pl_ema for proper EMA initialization matching Pandas
    ema_expr = ema(close, length=length)

    # DSP = close - EMA(close)
    dsp_expr = close_expr - ema_expr

    # Apply offset
    if offset != 0:
        dsp_expr = dsp_expr.shift(offset)

    return dsp_expr.alias(f"DSP_{length}")
