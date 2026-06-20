# -*- coding: utf-8 -*-
# =============================================================================
# Polars CHOP Implementation
# =============================================================================
import polars as pl

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


def chop(
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    length: int = 14,
    atr_length: int = 1,
    ln: bool = False,
    scalar: float = 100.0,
    drift: int = 1,
    offset: int = 0,
) -> PlExpr:
    """Polars: Choppiness Index (CHOP)

    Determines if the market is choppy (sideways) or trending.
    Values near 100 = choppy, near 0 = trending.

    Formula: CHOP = scalar * (log(sum(ATR)) - log(HH-LL)) / log(length)

    Args:
        high: Column name or pl.Expr for 'high' prices
        low: Column name or pl.Expr for 'low' prices
        close: Column name or pl.Expr for 'close' prices
        length: Period. Default: 14
        atr_length: ATR period. Default: 1
        ln: Use natural log instead of log10. Default: False
        scalar: Magnification factor. Default: 100
        drift: Difference period. Default: 1
        offset: Shift result. Default: 0

    Returns:
        pl.Expr: CHOP expression
    """
    from polars_ti.volatility.atr import atr

    high_expr = v_expr(high)
    low_expr = v_expr(low)
    close_expr = v_expr(close)

    diff = high_expr.rolling_max(window_size=length) - low_expr.rolling_min(window_size=length)
    atr_expr = atr(high_expr, low_expr, close_expr, length=atr_length, talib=False)
    atr_sum = atr_expr.rolling_sum(window_size=length)

    if ln:
        chop_expr = scalar * (atr_sum.log() - diff.log()) / pl.lit(length).cast(pl.Float64).log()
    else:
        chop_expr = (
            scalar
            * (atr_sum.log() / pl.lit(10.0).log() - diff.log() / pl.lit(10.0).log())
            / (pl.lit(length).cast(pl.Float64).log() / pl.lit(10.0).log())
        )

    if offset != 0:
        chop_expr = chop_expr.shift(offset)

    _label = f"CHOP{'ln' if ln else ''}_{length}_{atr_length}_{int(scalar)}"
    return chop_expr.alias(_label)
