# -*- coding: utf-8 -*-
# =============================================================================
# Polars HL2 Implementation
# =============================================================================
import numpy as np
import polars as pl

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.maps import Imports
from polars_ti.utils import v_talib
from polars_ti.utils._validate import v_expr


def hl2(high: IntoExpr, low: IntoExpr, talib: bool = True, offset: int = 0) -> PlExpr:
    """Polars: HL2 - Midpoint of High and Low

    Equivalent to TA-Lib's ``MEDPRICE``. The native ``(high + low) / 2``
    expression is numerically identical to ``talib.MEDPRICE``; the ``talib``
    switch only selects which implementation evaluates it, so the output is
    unchanged either way.

    Args:
        high: Column name or pl.Expr for 'high' prices
        low: Column name or pl.Expr for 'low' prices
        talib: If True and TA-Lib is installed, route via talib.MEDPRICE.
            Default: True
        offset: Shift the result by N periods. Default: 0

    Returns:
        pl.Expr: HL2 expression
    """
    high_expr = v_expr(high)
    low_expr = v_expr(low)
    if high_expr is None or low_expr is None:
        return None

    if Imports["talib"] and v_talib(talib):

        def _compute(s: pl.Series) -> pl.Series:
            from talib import MEDPRICE

            data = s.struct.unnest()
            return pl.Series(
                MEDPRICE(
                    data["_high"].to_numpy().astype(np.float64),
                    data["_low"].to_numpy().astype(np.float64),
                )
            )

        hl2_expr = pl.struct(high_expr.alias("_high"), low_expr.alias("_low")).map_batches(
            _compute, return_dtype=pl.Float64
        )
    else:
        hl2_expr = (high_expr + low_expr) / 2

    if offset != 0:
        hl2_expr = hl2_expr.shift(offset)

    return hl2_expr.alias("HL2")
