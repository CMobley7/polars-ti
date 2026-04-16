# -*- coding: utf-8 -*-
from numba import njit


@njit(cache=True)
def nb_mom(x, n):
    return nb_idiff(x, n)


# =============================================================================
# Polars MOM (Momentum) Implementation
# =============================================================================
import polars as pl
import numpy as np

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


def pl_mom(
    close: IntoExpr,
    length: int = 10,
    talib: bool = True,
    offset: int = 0,
) -> PlExpr:
    """Polars: Momentum (MOM)

    Measures speed of price movement.
    MOM = close - close[n periods ago]

    Args:
        close: Column name or pl.Expr for 'close' prices
        length: Lookback period. Default: 10
        talib: If True and TA-Lib installed, use TA-Lib. Default: True
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: MOM expression
    """
    from polars_ti.maps import Imports
    from polars_ti.utils import v_talib
    
    close_expr = v_expr(close)
    if close_expr is None:
        return None
    
    _use_talib = Imports["talib"] and v_talib(talib) and length > 1
    _length = length
    
    if _use_talib:
        def compute_mom(s: pl.Series) -> pl.Series:
            from talib import MOM as TALIB_MOM
            arr = s.to_numpy().astype(np.float64)
            result = TALIB_MOM(arr, timeperiod=_length)
            return pl.Series(result)
        mom_expr = close_expr.map_batches(compute_mom, return_dtype=pl.Float64)
    else:
        # Pure Polars: MOM = close - close.shift(length)
        mom_expr = close_expr - close_expr.shift(length)
    
    if offset != 0:
        mom_expr = mom_expr.shift(offset)
    
    return mom_expr.alias(f"MOM_{length}")


