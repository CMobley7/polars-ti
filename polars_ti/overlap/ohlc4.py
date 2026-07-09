# -*- coding: utf-8 -*-
# =============================================================================
# Polars OHLC4 Implementation
# =============================================================================
import numpy as np
import polars as pl

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.maps import Imports
from polars_ti.utils import v_talib
from polars_ti.utils._validate import v_expr


def ohlc4(
    open_: IntoExpr,
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    talib: bool = True,
) -> PlExpr:
    """Polars: OHLC4 - Average of Open, High, Low, Close

    Equivalent to TA-Lib's ``AVGPRICE``. The native
    ``(open + high + low + close) / 4`` expression is numerically identical to
    ``talib.AVGPRICE``; the ``talib`` switch only selects which implementation
    evaluates it, so the output is unchanged either way.

    Args:
        open_: Column name or pl.Expr for 'open' prices
        high: Column name or pl.Expr for 'high' prices
        low: Column name or pl.Expr for 'low' prices
        close: Column name or pl.Expr for 'close' prices
        talib: If True and TA-Lib is installed, route via talib.AVGPRICE.
            Default: True

    Returns:
        pl.Expr: OHLC4 expression
    """
    open_expr = v_expr(open_)
    high_expr = v_expr(high)
    low_expr = v_expr(low)
    close_expr = v_expr(close)

    if Imports["talib"] and v_talib(talib):

        def _compute(s: pl.Series) -> pl.Series:
            from talib import AVGPRICE

            data = s.struct.unnest()
            return pl.Series(
                AVGPRICE(
                    data["_open"].to_numpy().astype(np.float64),
                    data["_high"].to_numpy().astype(np.float64),
                    data["_low"].to_numpy().astype(np.float64),
                    data["_close"].to_numpy().astype(np.float64),
                )
            )

        return (
            pl.struct(
                open_expr.alias("_open"),
                high_expr.alias("_high"),
                low_expr.alias("_low"),
                close_expr.alias("_close"),
            )
            .map_batches(_compute, return_dtype=pl.Float64)
            .alias("OHLC4")
        )

    return ((open_expr + high_expr + low_expr + close_expr) / 4).alias("OHLC4")
