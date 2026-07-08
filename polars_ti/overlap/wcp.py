# -*- coding: utf-8 -*-
# =============================================================================
# Polars WCP Implementation
# =============================================================================
import polars as pl
import numpy as np

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr
from polars_ti.maps import Imports


def wcp(
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    talib: bool = True,
    offset: int = 0,
) -> PlExpr:
    """Polars: Weighted Closing Price (WCP)

    Args:
        high: Column name or pl.Expr for 'high' prices
        low: Column name or pl.Expr for 'low' prices
        close: Column name or pl.Expr for 'close' prices
        talib: If True and TA-Lib installed, use TA-Lib. Default: True
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: WCP expression
    """
    high_expr = v_expr(high)
    low_expr = v_expr(low)
    close_expr = v_expr(close)

    if Imports["talib"] and talib:

        def compute_wcp(struct: pl.Series) -> pl.Series:
            from talib import WCLPRICE

            h = struct.struct.field("h").to_numpy()
            l = struct.struct.field("l").to_numpy()
            c = struct.struct.field("c").to_numpy()
            result = WCLPRICE(h, l, c)
            if offset != 0:
                result = np.roll(result, offset)
                if offset > 0:
                    result[:offset] = np.nan
                else:
                    result[offset:] = np.nan
            return pl.Series(result)

        return (
            pl.struct(
                [
                    high_expr.alias("h"),
                    low_expr.alias("l"),
                    close_expr.alias("c"),
                ]
            )
            .map_batches(compute_wcp)
            .alias("WCP")
        )
    else:
        result = (high_expr + low_expr + 2 * close_expr) / 4
        if offset != 0:
            result = result.shift(offset)
        return result.alias("WCP")
