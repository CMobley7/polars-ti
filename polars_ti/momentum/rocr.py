# -*- coding: utf-8 -*-
# =============================================================================
# Polars ROCR (Rate of Change Ratio) Implementation
# =============================================================================
import polars as pl
import numpy as np

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


def rocr(
    close: IntoExpr,
    length: int = 10,
    talib: bool = True,
    offset: int = 0,
) -> PlExpr:
    """Polars: Rate of Change Ratio (ROCR)

    Measures the ratio of the current price to the price n periods ago.

    ROCR = close / close[n]

    Args:
        close: Column name or pl.Expr for 'close' prices
        length: Lookback period. Default: 10
        talib: If True and TA-Lib installed, use TA-Lib. Default: True
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: ROCR expression
    """
    from polars_ti.maps import Imports
    from polars_ti.utils import v_talib

    close_expr = v_expr(close)
    if close_expr is None:
        return None

    _use_talib = Imports["talib"] and v_talib(talib) and length > 1
    _length = length

    if _use_talib:

        def compute_rocr(s: pl.Series) -> pl.Series:
            from talib import ROCR as TALIB_ROCR

            arr = s.to_numpy().astype(np.float64)
            return pl.Series(TALIB_ROCR(arr, timeperiod=_length))

        rocr_expr = close_expr.map_batches(compute_rocr, return_dtype=pl.Float64)
    else:
        rocr_expr = close_expr / close_expr.shift(length)

    if offset != 0:
        rocr_expr = rocr_expr.shift(offset)

    return rocr_expr.alias(f"ROCR_{length}")
