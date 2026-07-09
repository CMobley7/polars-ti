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
    offset: int = 0,
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
        offset: Shift the result by N periods. Default: 0

    Returns:
        pl.Expr: OHLC4 expression
    """
    open_expr = v_expr(open_)
    high_expr = v_expr(high)
    low_expr = v_expr(low)
    close_expr = v_expr(close)

    # Consistency with hl2: a None input expr yields None rather than raising.
    if any(e is None for e in [open_expr, high_expr, low_expr, close_expr]):
        return None

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

        ohlc4_expr = pl.struct(
            open_expr.alias("_open"),
            high_expr.alias("_high"),
            low_expr.alias("_low"),
            close_expr.alias("_close"),
        ).map_batches(_compute, return_dtype=pl.Float64)
    else:
        ohlc4_expr = (open_expr + high_expr + low_expr + close_expr) / 4

    if offset != 0:
        ohlc4_expr = ohlc4_expr.shift(offset)

    return ohlc4_expr.alias("OHLC4")
