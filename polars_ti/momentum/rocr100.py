# -*- coding: utf-8 -*-
# =============================================================================
# Polars ROCR100 (Rate of Change Ratio * 100) Implementation
# =============================================================================
import polars as pl
import numpy as np

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


def rocr100(
    close: IntoExpr,
    length: int = 10,
    talib: bool = True,
    offset: int = 0,
) -> PlExpr:
    """Polars: Rate of Change Ratio * 100 (ROCR100)

    Measures the ratio of the current price to the price n periods ago, scaled
    by 100. A value of 100 means no change.

    ROCR100 = 100 * (close / close[n])

    Args:
        close: Column name or pl.Expr for 'close' prices
        length: Lookback period. Default: 10
        talib: If True and TA-Lib installed, use TA-Lib. Default: True
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: ROCR100 expression
    """
    from polars_ti.maps import Imports
    from polars_ti.utils import v_talib

    close_expr = v_expr(close)
    if close_expr is None:
        return None

    _use_talib = Imports["talib"] and v_talib(talib) and length > 1
    _length = length

    if _use_talib:

        def compute_rocr100(s: pl.Series) -> pl.Series:
            from talib import ROCR100 as TALIB_ROCR100

            arr = s.to_numpy().astype(np.float64)
            return pl.Series(TALIB_ROCR100(arr, timeperiod=_length))

        rocr100_expr = close_expr.map_batches(compute_rocr100, return_dtype=pl.Float64)
    else:
        rocr100_expr = 100.0 * (close_expr / close_expr.shift(length))

    if offset != 0:
        rocr100_expr = rocr100_expr.shift(offset)

    return rocr100_expr.alias(f"ROCR100_{length}")
