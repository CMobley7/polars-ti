# -*- coding: utf-8 -*-
# =============================================================================
# Polars TEMA Implementation (using _ema_numba kernel directly)
# =============================================================================
import polars as pl
import numpy as np

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr
from polars_ti.overlap.ema import _ema_numba


def pl_tema(
    close: IntoExpr,
    length: int = 10,
    talib: bool = True,
    presma: bool = True,
    offset: int = 0,
) -> PlExpr:
    """Polars: Triple Exponential Moving Average (TEMA)

    A less laggy Exponential Moving Average.
    TEMA = 3 * (EMA1 - EMA2) + EMA3

    Sources:
        https://www.tradingtechnologies.com/help/x-study/technical-indicator-definitions/triple-exponential-moving-average-tema/

    Args:
        close: Column name or pl.Expr for 'close' prices
        length: Smoothing period. Default: 10
        talib: If True and TA-Lib available, uses TA-Lib. Default: True
        presma: If True, uses SMA for initial EMA value. Default: True
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: TEMA expression for lazy evaluation
    """
    from polars_ti.maps import Imports
    from polars_ti.utils import v_talib
    
    close_expr = v_expr(close)
    if close_expr is None:
        return None

    _use_talib = Imports["talib"] and v_talib(talib) and length > 1
    _length = length
    _presma = presma

    def compute_tema(s: pl.Series) -> pl.Series:
        arr = s.to_numpy().astype(np.float64)
        
        if _use_talib:
            from talib import TEMA as TALIB_TEMA
            result = TALIB_TEMA(arr, timeperiod=_length)
        else:
            # Call _ema_numba directly - NO DataFrame creation!
            ema1 = _ema_numba(arr, _length, presma=_presma, adjust=False)
            ema2 = _ema_numba(ema1, _length, presma=False, adjust=False)
            ema3 = _ema_numba(ema2, _length, presma=False, adjust=False)
            
            # TEMA = 3 * (EMA1 - EMA2) + EMA3
            result = 3 * (ema1 - ema2) + ema3
        
        return pl.Series(result)

    tema_expr = close_expr.map_batches(compute_tema, return_dtype=pl.Float64)

    if offset != 0:
        tema_expr = tema_expr.shift(offset)

    return tema_expr.alias(f"TEMA_{length}")


