# -*- coding: utf-8 -*-
# =============================================================================
# Polars ROCP (Rate of Change Percentage) Implementation
# =============================================================================
import polars as pl
import numpy as np

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


def rocp(
    close: IntoExpr,
    length: int = 10,
    talib: bool = True,
    offset: int = 0,
) -> PlExpr:
    """Polars: Rate of Change Percentage (ROCP)

    Measures the percent change in price over a period, expressed as a ratio
    (not multiplied by 100).

    ROCP = (close - close[n]) / close[n]

    Args:
        close: Column name or pl.Expr for 'close' prices
        length: Lookback period. Default: 10
        talib: If True and TA-Lib installed, use TA-Lib. Default: True
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: ROCP expression
    """
    from polars_ti.maps import Imports
    from polars_ti.utils import v_talib

    close_expr = v_expr(close)
    if close_expr is None:
        return None

    _use_talib = Imports["talib"] and v_talib(talib) and length > 1
    _length = length

    if _use_talib:

        def compute_rocp(s: pl.Series) -> pl.Series:
            from talib import ROCP as TALIB_ROCP

            arr = s.to_numpy().astype(np.float64)
            return pl.Series(TALIB_ROCP(arr, timeperiod=_length))

        rocp_expr = close_expr.map_batches(compute_rocp, return_dtype=pl.Float64)
    else:
        shifted = close_expr.shift(length)
        rocp_expr = (close_expr - shifted) / shifted

    if offset != 0:
        rocp_expr = rocp_expr.shift(offset)

    return rocp_expr.alias(f"ROCP_{length}")
