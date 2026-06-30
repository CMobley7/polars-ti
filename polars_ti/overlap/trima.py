# -*- coding: utf-8 -*-
# =============================================================================
# Polars TRIMA Implementation (pure Polars rolling_mean)
# =============================================================================
from math import ceil, floor

import polars as pl
import numpy as np

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr
from polars_ti.overlap.sma import nb_sma


def trima(
    close: IntoExpr,
    length: int = 10,
    talib: bool = True,
    offset: int = 0,
) -> PlExpr:
    """Polars: Triangular Moving Average (TRIMA)

    A weighted moving average where the shape of the weights are triangular
    and the greatest weight is in the middle of the period.

    Sources:
        https://www.tradingtechnologies.com/help/x-study/technical-indicator-definitions/triangular-moving-average-trima/

    Args:
        close: Column name or pl.Expr for 'close' prices
        length: Rolling window period. Default: 10
        talib: If True and TA-Lib is installed, uses TA-Lib. Default: True
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: TRIMA expression for lazy evaluation
    """
    from polars_ti.maps import Imports
    from polars_ti.utils import v_talib

    close_expr = v_expr(close)
    if close_expr is None:
        return None

    _use_talib = Imports["talib"] and v_talib(talib) and length > 1
    _length = length
    # Native TRIMA uses asymmetric ceil/floor windows to match the documented
    # TradingView / TA-Lib formula (classic fork commit 41c91db):
    #   tma = sma(sma(src, ceil(length/2)), floor(length/2)+1)
    # The old banker's-rounding round(0.5*(length+1)) gave symmetric windows
    # that diverged from TA-Lib for even lengths.
    _first_window = ceil(length / 2)
    _second_window = floor(length / 2) + 1

    def compute_trima(s: pl.Series) -> pl.Series:
        arr = s.to_numpy().astype(np.float64)

        if _use_talib:
            from talib import TRIMA as TALIB_TRIMA

            result = TALIB_TRIMA(arr, timeperiod=_length)
        else:
            # Call nb_sma directly - NO Pandas!
            sma1 = nb_sma(arr, _first_window)
            result = nb_sma(sma1, _second_window)

        return pl.Series(result)

    result = close_expr.map_batches(compute_trima, return_dtype=pl.Float64)

    if offset != 0:
        result = result.shift(offset)

    return result.alias(f"TRIMA_{length}")
