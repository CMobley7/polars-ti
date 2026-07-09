# -*- coding: utf-8 -*-
# =============================================================================
# Polars STDEV Implementation
# =============================================================================
import polars as pl
import numpy as np

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


def stdev(
    close: IntoExpr,
    length: int = 30,
    ddof: int = 0,
    talib: bool = True,
    offset: int = 0,
) -> pl.Expr:
    """Polars: Rolling Standard Deviation

    Calculates Standard Deviation over a rolling period using native Polars.

    Args:
        close: Column name or pl.Expr for 'close' prices
        length: Rolling window period. Default: 30
        ddof: Delta Degrees of Freedom (population std, TA-Lib/TradingView convention). Default: 0
        talib: If True and TA-Lib installed, use TA-Lib. Default: True
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: Standard deviation expression
    """
    close_expr = v_expr(close)
    if close_expr is None:
        return None

    from polars_ti.maps import Imports
    from polars_ti.utils import v_talib

    if Imports["talib"] and v_talib(talib) and ddof == 0:

        def compute_stdev(s: pl.Series) -> pl.Series:
            from talib import STDDEV

            arr = s.to_numpy().astype(np.float64)
            return pl.Series(STDDEV(arr, timeperiod=length))

        result = close_expr.map_batches(compute_stdev, return_dtype=pl.Float64)
    else:
        result = close_expr.rolling_std(window_size=length, min_samples=length, ddof=ddof)

    if offset != 0:
        result = result.shift(offset)

    return result.alias(f"STDEV_{length}")
