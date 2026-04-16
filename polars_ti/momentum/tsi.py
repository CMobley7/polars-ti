# -*- coding: utf-8 -*-
# =============================================================================
# Polars TSI (True Strength Index) Implementation
# =============================================================================
import polars as pl
import numpy as np

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


def pl_tsi(
    close: IntoExpr,
    fast: int = 13,
    slow: int = 25,
    signal: int = 13,
    scalar: float = 100.0,
    mamode: str = "ema",
    drift: int = 1,
    offset: int = 0,
) -> list[PlExpr]:
    """Polars: True Strength Index (TSI)

    The True Strength Index is a momentum indicator used to identify
    short-term swings while in the direction of the trend.

    Args:
        close: Column name or pl.Expr for 'close' prices
        fast: Fast EMA period. Default: 13
        slow: Slow EMA period. Default: 25
        signal: Signal MA period. Default: 13
        scalar: Multiplication factor. Default: 100
        mamode: MA type for signal line. Default: 'ema'
        drift: Periods for diff. Default: 1
        offset: Shift result by N periods. Default: 0

    Returns:
        list[pl.Expr]: [TSI, TSI_signal]
    """
    from polars_ti.overlap.ema import pl_ema
    from polars_ti.ma import pl_ma
    
    close_expr = v_expr(close)
    if close_expr is None:
        return None
    
    if slow < fast:
        fast, slow = slow, fast
    
    _props = f"_{fast}_{slow}_{signal}"
    
    # TSI = scalar * double_smooth(diff) / double_smooth(abs(diff))
    diff = close_expr.diff(drift)
    
    # Double smooth the diff
    diff_slow = pl_ema(diff, length=slow)
    diff_fast_slow = pl_ema(diff_slow, length=fast)
    
    # Double smooth the abs(diff)
    abs_diff = diff.abs()
    abs_slow = pl_ema(abs_diff, length=slow)
    abs_fast_slow = pl_ema(abs_slow, length=fast)
    
    # TSI = scalar * double_smooth(diff) / double_smooth(|diff|)
    tsi_expr = scalar * diff_fast_slow / abs_fast_slow
    
    # Signal = MA(TSI, signal)
    tsi_signal_expr = pl_ma(name=mamode, source=tsi_expr, length=signal)
    
    if offset != 0:
        tsi_expr = tsi_expr.shift(offset)
        tsi_signal_expr = tsi_signal_expr.shift(offset)
    
    return [
        tsi_expr.alias(f"TSI{_props}"),
        tsi_signal_expr.alias(f"TSIs{_props}"),
    ]

