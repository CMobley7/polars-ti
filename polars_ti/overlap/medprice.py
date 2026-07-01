# -*- coding: utf-8 -*-
# =============================================================================
# Polars MEDPRICE Implementation
# =============================================================================
import polars as pl
import numpy as np

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


def medprice(
    high: IntoExpr,
    low: IntoExpr,
    talib: bool = True,
    offset: int = 0,
) -> PlExpr:
    """Polars: Median Price (MEDPRICE)

    Per-bar (High + Low) / 2.  Equivalent to ta.hl2.
    TA-Lib name: MEDPRICE.

    Note: distinct from ta.midprice which is a rolling max/min indicator.

    Args:
        high: Column name or pl.Expr for 'high' prices.
        low: Column name or pl.Expr for 'low' prices.
        talib: If True and TA-Lib installed, use TA-Lib MEDPRICE. Default: True
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: MEDPRICE expression.
    """
    from polars_ti.maps import Imports
    from polars_ti.utils import v_talib

    high_expr = v_expr(high)
    low_expr = v_expr(low)
    if any(e is None for e in [high_expr, low_expr]):
        return None

    _use_talib = Imports["talib"] and v_talib(talib)
    _offset = offset

    if _use_talib:

        def _compute(struct: pl.Series) -> pl.Series:
            from talib import MEDPRICE

            df = struct.struct.unnest()
            result = MEDPRICE(
                df["_high"].to_numpy().astype(np.float64),
                df["_low"].to_numpy().astype(np.float64),
            )
            return pl.Series(result)

        result = pl.struct(
            [
                high_expr.alias("_high"),
                low_expr.alias("_low"),
            ]
        ).map_batches(_compute, return_dtype=pl.Float64)
    else:
        result = (high_expr + low_expr) / pl.lit(2.0)

    if _offset != 0:
        result = result.shift(_offset)

    return result.alias("MEDPRICE")
