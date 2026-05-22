# -*- coding: utf-8 -*-
# =============================================================================
# Polars HLC3 Implementation
# =============================================================================
import polars as pl
import numpy as np

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


def pl_hlc3(
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    talib: bool = True,
    offset: int = 0,
) -> PlExpr:
    """Polars: HLC3 - Typical Price (Average of High, Low, Close)

    Args:
        high: Column name or pl.Expr for 'high' prices
        low: Column name or pl.Expr for 'low' prices
        close: Column name or pl.Expr for 'close' prices
        talib: If True and TA-Lib available, uses TA-Lib TYPPRICE. Default: True
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: HLC3 expression
    """
    from polars_ti.maps import Imports
    from polars_ti.utils import v_talib

    high_expr = v_expr(high)
    low_expr = v_expr(low)
    close_expr = v_expr(close)
    if any(x is None for x in [high_expr, low_expr, close_expr]):
        return None

    _use_talib = Imports["talib"] and v_talib(talib)

    if _use_talib:

        def compute_hlc3(struct: pl.Series) -> pl.Series:
            from talib import TYPPRICE

            df = struct.struct.unnest()
            h = df["_high"].to_numpy().astype(np.float64)
            l = df["_low"].to_numpy().astype(np.float64)
            c = df["_close"].to_numpy().astype(np.float64)
            result = TYPPRICE(h, l, c)
            return pl.Series(result)

        result = pl.struct(
            [
                high_expr.alias("_high"),
                low_expr.alias("_low"),
                close_expr.alias("_close"),
            ]
        ).map_batches(compute_hlc3, return_dtype=pl.Float64)
    else:
        result = (high_expr + low_expr + close_expr) / pl.lit(3.0)

    # Apply offset
    if offset != 0:
        result = result.shift(offset)

    return result.alias("HLC3")
