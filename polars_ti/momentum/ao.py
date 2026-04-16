# -*- coding: utf-8 -*-
# =============================================================================
# Polars AO (Awesome Oscillator) Implementation
# =============================================================================
import polars as pl

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


def pl_ao(
    high: IntoExpr,
    low: IntoExpr,
    fast: int = 5,
    slow: int = 34,
    offset: int = 0,
) -> PlExpr:
    """Polars: Awesome Oscillator (AO)

    Measures momentum using the difference between fast and slow SMAs
    of the median price (HL2).

    Formula: AO = SMA(HL2, fast) - SMA(HL2, slow)

    Args:
        high: Column name or pl.Expr for 'high' prices
        low: Column name or pl.Expr for 'low' prices
        fast: Short period. Default: 5
        slow: Long period. Default: 34
        offset: Shift result. Default: 0

    Returns:
        pl.Expr: AO expression
    """
    from polars_ti.overlap.hl2 import pl_hl2
    from polars_ti.overlap.sma import pl_sma
    
    if slow < fast:
        fast, slow = slow, fast
    
    high_expr = v_expr(high)
    low_expr = v_expr(low)

    # Use pl_hl2 for median price
    hl2_expr = pl_hl2(high_expr, low_expr)

    # Use pl_sma for fast and slow SMAs
    fast_sma = pl_sma(hl2_expr, length=fast, talib=False, offset=0)
    slow_sma = pl_sma(hl2_expr, length=slow, talib=False, offset=0)

    # AO = fast SMA - slow SMA
    ao_expr = fast_sma - slow_sma
    
    if offset != 0:
        ao_expr = ao_expr.shift(offset)

    return ao_expr.alias(f"AO_{fast}_{slow}")


