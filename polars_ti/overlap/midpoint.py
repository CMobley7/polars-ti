# -*- coding: utf-8 -*-
# =============================================================================
# Polars MIDPOINT Implementation
# =============================================================================
import polars as pl

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


def pl_midpoint(
    close: IntoExpr,
    length: int = 2,
    talib: bool = True,
    offset: int = 0,
) -> PlExpr:
    """Polars: Midpoint

    The Midpoint is the average of the rolling high and low of period length.

    Args:
        close: Column name or pl.Expr for 'close' prices
        length: Rolling window period. Default: 2
        talib: If True and TA-Lib is installed, uses TA-Lib. Default: True
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: MIDPOINT expression for lazy evaluation
    """
    from polars_ti.maps import Imports
    from polars_ti.utils import v_talib
    import numpy as np
    
    close_expr = v_expr(close)
    if close_expr is None:
        return None

    _use_talib = Imports["talib"] and v_talib(talib) and length > 1
    _length = length

    if _use_talib:
        def compute_midpoint(s: pl.Series) -> pl.Series:
            from talib import MIDPOINT as TALIB_MIDPOINT
            arr = s.to_numpy().astype(np.float64)
            result = TALIB_MIDPOINT(arr, timeperiod=_length)
            return pl.Series(result)
        midpoint_expr = close_expr.map_batches(compute_midpoint, return_dtype=pl.Float64)
    else:
        # Native Polars expression: (rolling_min + rolling_max) / 2
        lowest = close_expr.rolling_min(window_size=length, min_samples=length)
        highest = close_expr.rolling_max(window_size=length, min_samples=length)
        midpoint_expr = 0.5 * (lowest + highest)

    # Apply offset
    if offset != 0:
        midpoint_expr = midpoint_expr.shift(offset)

    return midpoint_expr.alias(f"MIDPOINT_{length}")
