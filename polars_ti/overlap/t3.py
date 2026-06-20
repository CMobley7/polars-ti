# -*- coding: utf-8 -*-
# =============================================================================
# Polars T3 Implementation (using _ema_numba kernel directly)
# =============================================================================
import polars as pl
import numpy as np

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr
from polars_ti.overlap.ema import _ema_numba


def t3(
    close: IntoExpr,
    length: int = 10,
    a: float = 0.7,
    talib: bool = True,
    presma: bool = True,
    offset: int = 0,
) -> PlExpr:
    """Polars: Tim Tillson's T3 Moving Average (T3)

    Tim Tillson's T3 Moving Average is considered a smoother and more
    responsive moving average relative to other moving averages.

    Sources:
        http://www.binarytribune.com/forex-trading-indicators/t3-moving-average-indicator/

    Args:
        close: Column name or pl.Expr for 'close' prices
        length: Smoothing period. Default: 10
        a: Volume factor (0 < a < 1). Default: 0.7
        talib: If True and TA-Lib is installed, uses TA-Lib. Default: True
        presma: If True, uses SMA for initial EMA value. Default: True
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: T3 expression for lazy evaluation
    """
    from polars_ti.maps import Imports
    from polars_ti.utils import v_talib

    close_expr = v_expr(close)
    if close_expr is None:
        return None

    # Validate 'a' parameter
    if not (0 < a < 1):
        a = 0.7

    _use_talib = Imports["talib"] and v_talib(talib) and length > 1
    _length = length
    _a = a
    _presma = presma

    # Calculate coefficients
    c1 = -a * a**2
    c2 = 3 * a**2 + 3 * a**3
    c3 = -6 * a**2 - 3 * a - 3 * a**3
    c4 = a**3 + 3 * a**2 + 3 * a + 1

    def compute_t3(s: pl.Series) -> pl.Series:
        arr = s.to_numpy().astype(np.float64)

        if _use_talib:
            from talib import T3 as TALIB_T3

            result = TALIB_T3(arr, timeperiod=_length, vfactor=_a)
        else:
            # Call _ema_numba directly - NO DataFrame creation!
            e1 = _ema_numba(arr, _length, presma=_presma, adjust=False)
            e2 = _ema_numba(e1, _length, presma=False, adjust=False)
            e3 = _ema_numba(e2, _length, presma=False, adjust=False)
            e4 = _ema_numba(e3, _length, presma=False, adjust=False)
            e5 = _ema_numba(e4, _length, presma=False, adjust=False)
            e6 = _ema_numba(e5, _length, presma=False, adjust=False)

            # T3 = c1*e6 + c2*e5 + c3*e4 + c4*e3
            result = c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3

        return pl.Series(result)

    result = close_expr.map_batches(compute_t3, return_dtype=pl.Float64)

    if offset != 0:
        result = result.shift(offset)

    return result.alias(f"T3_{length}_{a}")
