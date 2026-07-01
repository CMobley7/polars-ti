# -*- coding: utf-8 -*-
# =============================================================================
# Polars TYPPRICE Implementation
# =============================================================================
import polars as pl
import numpy as np

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


def typprice(
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    talib: bool = True,
    offset: int = 0,
) -> PlExpr:
    """Polars: Typical Price (TYPPRICE)

    Per-bar (High + Low + Close) / 3.  Equivalent to ta.hlc3.
    TA-Lib name: TYPPRICE.

    Args:
        high: Column name or pl.Expr for 'high' prices.
        low: Column name or pl.Expr for 'low' prices.
        close: Column name or pl.Expr for 'close' prices.
        talib: If True and TA-Lib installed, use TA-Lib TYPPRICE. Default: True
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: TYPPRICE expression.
    """
    from polars_ti.maps import Imports
    from polars_ti.utils import v_talib

    high_expr = v_expr(high)
    low_expr = v_expr(low)
    close_expr = v_expr(close)
    if any(e is None for e in [high_expr, low_expr, close_expr]):
        return None

    _use_talib = Imports["talib"] and v_talib(talib)
    _offset = offset

    if _use_talib:

        def _compute(struct: pl.Series) -> pl.Series:
            from talib import TYPPRICE

            df = struct.struct.unnest()
            result = TYPPRICE(
                df["_high"].to_numpy().astype(np.float64),
                df["_low"].to_numpy().astype(np.float64),
                df["_close"].to_numpy().astype(np.float64),
            )
            return pl.Series(result)

        result = pl.struct(
            [
                high_expr.alias("_high"),
                low_expr.alias("_low"),
                close_expr.alias("_close"),
            ]
        ).map_batches(_compute, return_dtype=pl.Float64)
    else:
        result = (high_expr + low_expr + close_expr) / pl.lit(3.0)

    if _offset != 0:
        result = result.shift(_offset)

    return result.alias("TYPPRICE")
