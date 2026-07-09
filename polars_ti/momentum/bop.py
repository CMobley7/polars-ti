# -*- coding: utf-8 -*-
# =============================================================================
# Polars BOP (Balance of Power) Implementation
# =============================================================================
import polars as pl

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._math import non_zero_range
from polars_ti.utils._validate import v_expr


def bop(
    open_: IntoExpr,
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    scalar: float = 1.0,
    talib: bool = True,
    offset: int = 0,
) -> PlExpr:
    """Polars: Balance of Power (BOP)

    Measures market strength of buyers vs sellers.
    BOP = scalar * (close - open) / (high - low)

    Args:
        open_: Column name or pl.Expr for 'open' prices
        high: Column name or pl.Expr for 'high' prices
        low: Column name or pl.Expr for 'low' prices
        close: Column name or pl.Expr for 'close' prices
        scalar: Magnification factor. Default: 1.0
        talib: If True and TA-Lib installed, use TA-Lib. Default: True
        offset: Shift result. Default: 0

    Returns:
        pl.Expr: BOP expression
    """
    import numpy as np
    from polars_ti.maps import Imports
    from polars_ti.utils import v_talib

    open_expr = v_expr(open_)
    high_expr = v_expr(high)
    low_expr = v_expr(low)
    close_expr = v_expr(close)

    _use_talib = Imports["talib"] and v_talib(talib)

    if _use_talib:

        def compute_bop(df: pl.DataFrame) -> pl.Series:
            from talib import BOP as TALIB_BOP

            o = df["open"].to_numpy().astype(np.float64)
            h = df["high"].to_numpy().astype(np.float64)
            l = df["low"].to_numpy().astype(np.float64)
            c = df["close"].to_numpy().astype(np.float64)
            result = TALIB_BOP(o, h, l, c)
            # TA-Lib BOP is fixed at scalar=1.0; honor a non-default scalar with a
            # linear rescale (BOP = scalar * (close - open) / (high - low)).
            if scalar != 1.0:
                result = result * scalar
            return pl.Series("BOP", result)

        bop_expr = pl.struct(
            [
                open_expr.alias("open"),
                high_expr.alias("high"),
                low_expr.alias("low"),
                close_expr.alias("close"),
            ]
        ).map_batches(lambda s: compute_bop(s.struct.unnest()), return_dtype=pl.Float64)
    else:
        # Use shared utility for zero-protected ranges (matches pandas-ta)
        high_low_safe = non_zero_range(high_expr, low_expr)
        close_open_safe = non_zero_range(close_expr, open_expr)
        bop_expr = scalar * close_open_safe / high_low_safe

    if offset != 0:
        bop_expr = bop_expr.shift(offset)

    return bop_expr.alias("BOP")
