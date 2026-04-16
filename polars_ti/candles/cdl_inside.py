# -*- coding: utf-8 -*-
from numpy import roll, where
from numba import njit


@njit(cache=True)
def np_cdl_inside(high, low):
    hdiff = where(high - roll(high, 1) < 0, 1, 0)
    ldiff = where(low - roll(low, 1) > 0, 1, 0)
    return hdiff & ldiff


# =============================================================================
# Polars CDL_INSIDE Implementation
# =============================================================================
import polars as pl

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


def pl_cdl_inside(
    open_: IntoExpr,
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    asbool: bool = False,
    scalar: float = 100.0,
    offset: int = 0,
) -> PlExpr:
    """Polars: Candle Type - Inside Bar

    An Inside Bar is a bar that is engulfed by the prior high and low.
    Current bar: high < prev_high AND low > prev_low

    Set asbool=True if you want boolean result. Default returns:
    0 if not inside bar, scalar if inside bar.

    Sources:
        https://www.tradingview.com/script/IyIGN1WO-Inside-Bar/

    Args:
        open_: Column name or pl.Expr for 'open' prices
        high: Column name or pl.Expr for 'high' prices
        low: Column name or pl.Expr for 'low' prices
        close: Column name or pl.Expr for 'close' prices
        asbool: Returns boolean result instead of scaled int. Default: False
        scalar: Result multiplier. Default: 100.0
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: CDL_INSIDE expression
    """
    high_expr = v_expr(high)
    low_expr = v_expr(low)

    # Inside Bar: current high < previous high AND current low > previous low
    prev_high = high_expr.shift(1)
    prev_low = low_expr.shift(1)

    is_inside = (high_expr < prev_high) & (low_expr > prev_low)

    if asbool:
        result = is_inside
    else:
        result = pl.when(is_inside).then(scalar).otherwise(0.0).cast(pl.Int64)

    # Apply offset
    if offset != 0:
        result = result.shift(offset)

    return result.alias("CDL_INSIDE")

