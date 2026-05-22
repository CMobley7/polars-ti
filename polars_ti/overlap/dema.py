# -*- coding: utf-8 -*-
# =============================================================================
# Polars DEMA Implementation
# =============================================================================
import polars as pl
import numpy as np

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr
from polars_ti.overlap.ema import _ema_numba


def pl_dema(
    close: IntoExpr,
    length: int = 10,
    talib: bool = True,
    offset: int = 0,
) -> PlExpr:
    """Polars: Double Exponential Moving Average (DEMA)

    DEMA = 2 * EMA(close) - EMA(EMA(close))

    Sources:
        https://www.tradingtechnologies.com/help/x-study/technical-indicator-definitions/double-exponential-moving-average-dema/

    Args:
        close: Column name or pl.Expr for 'close' prices
        length: Smoothing period. Default: 10
        talib: If True and TA-Lib available, uses TA-Lib. Default: True
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: DEMA expression
    """
    from polars_ti.maps import Imports
    from polars_ti.utils import v_talib

    close_expr = v_expr(close)
    if close_expr is None:
        return None

    _use_talib = Imports["talib"] and v_talib(talib) and length > 1
    _length = length

    def compute_dema(s: pl.Series) -> pl.Series:
        arr = s.to_numpy().astype(np.float64)

        if _use_talib:
            from talib import DEMA as TALIB_DEMA

            result = TALIB_DEMA(arr, timeperiod=_length)
        else:
            # Direct Numba calls - no DataFrame creation!
            ema1 = _ema_numba(arr, _length, presma=True, adjust=False)
            ema2 = _ema_numba(ema1, _length, presma=True, adjust=False)
            result = 2 * ema1 - ema2

        return pl.Series(result)

    dema_expr = close_expr.map_batches(compute_dema, return_dtype=pl.Float64)

    # Apply offset
    if offset != 0:
        dema_expr = dema_expr.shift(offset)

    return dema_expr.alias(f"DEMA_{length}")
