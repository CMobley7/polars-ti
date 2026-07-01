# -*- coding: utf-8 -*-
# =============================================================================
# Polars AVGPRICE Implementation
# =============================================================================
import polars as pl
import numpy as np

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


def avgprice(
    open_: IntoExpr,
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    talib: bool = True,
    offset: int = 0,
) -> PlExpr:
    """Polars: Average Price (AVGPRICE)

    Per-bar (Open + High + Low + Close) / 4.  Equivalent to ta.ohlc4.
    TA-Lib name: AVGPRICE.

    Args:
        open_: Column name or pl.Expr for 'open' prices.
        high: Column name or pl.Expr for 'high' prices.
        low: Column name or pl.Expr for 'low' prices.
        close: Column name or pl.Expr for 'close' prices.
        talib: If True and TA-Lib installed, use TA-Lib AVGPRICE. Default: True
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: AVGPRICE expression.
    """
    from polars_ti.maps import Imports
    from polars_ti.utils import v_talib

    open_expr = v_expr(open_)
    high_expr = v_expr(high)
    low_expr = v_expr(low)
    close_expr = v_expr(close)
    if any(e is None for e in [open_expr, high_expr, low_expr, close_expr]):
        return None

    _use_talib = Imports["talib"] and v_talib(talib)
    _offset = offset

    if _use_talib:

        def _compute(struct: pl.Series) -> pl.Series:
            from talib import AVGPRICE

            df = struct.struct.unnest()
            result = AVGPRICE(
                df["_open"].to_numpy().astype(np.float64),
                df["_high"].to_numpy().astype(np.float64),
                df["_low"].to_numpy().astype(np.float64),
                df["_close"].to_numpy().astype(np.float64),
            )
            return pl.Series(result)

        result = pl.struct(
            [
                open_expr.alias("_open"),
                high_expr.alias("_high"),
                low_expr.alias("_low"),
                close_expr.alias("_close"),
            ]
        ).map_batches(_compute, return_dtype=pl.Float64)
    else:
        result = (open_expr + high_expr + low_expr + close_expr) / pl.lit(4.0)

    if _offset != 0:
        result = result.shift(_offset)

    return result.alias("AVGPRICE")
