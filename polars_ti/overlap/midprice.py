# -*- coding: utf-8 -*-
# =============================================================================
# Polars MIDPRICE Implementation
# =============================================================================
import polars as pl
import numpy as np

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr
from polars_ti.maps import Imports


def pl_midprice(
    high: IntoExpr,
    low: IntoExpr,
    length: int = 2,
    talib: bool = True,
    offset: int = 0,
) -> PlExpr:
    """Polars: Midprice (average of rolling high and low)

    Args:
        high: Column name or pl.Expr for 'high' prices
        low: Column name or pl.Expr for 'low' prices
        length: Rolling window period. Default: 2
        talib: If True and TA-Lib installed, use TA-Lib. Default: True
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: MIDPRICE expression
    """
    high_expr = v_expr(high)
    low_expr = v_expr(low)
    
    if Imports["talib"] and talib:
        def compute_midprice(struct: pl.Series) -> pl.Series:
            from talib import MIDPRICE
            h = struct.struct.field("h").to_numpy()
            l = struct.struct.field("l").to_numpy()
            result = MIDPRICE(h, l, length)
            if offset != 0:
                result = np.roll(result, offset)
                if offset > 0:
                    result[:offset] = np.nan
            return pl.Series(result)
        
        return pl.struct([
            high_expr.alias("h"),
            low_expr.alias("l"),
        ]).map_batches(compute_midprice).alias(f"MIDPRICE_{length}")
    else:
        lowest_low = low_expr.rolling_min(length)
        highest_high = high_expr.rolling_max(length)
        result = (lowest_low + highest_high) / 2
        if offset != 0:
            result = result.shift(offset)
        return result.alias(f"MIDPRICE_{length}")

