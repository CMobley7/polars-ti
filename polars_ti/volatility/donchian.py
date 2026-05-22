# -*- coding: utf-8 -*-
# =============================================================================
# Polars Donchian Channels Implementation (Pure Native Polars)
# =============================================================================
import polars as pl

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


def pl_donchian(
    high: IntoExpr,
    low: IntoExpr,
    lower_length: int = 20,
    upper_length: int = 20,
    offset: int = 0,
) -> pl.Expr:
    """Polars: Donchian Channels (DC)

    Uses pure native Polars expressions.

    Sources:
        https://www.tradingview.com/wiki/Donchian_Channels_(DC)

    Args:
        high: Column name or pl.Expr for 'high'
        low: Column name or pl.Expr for 'low'
        lower_length: Lower band rolling period. Default: 20
        upper_length: Upper band rolling period. Default: 20
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: Struct with DCL, DCM, DCU columns
    """
    high_expr = v_expr(high)
    low_expr = v_expr(low)

    if high_expr is None or low_expr is None:
        return None

    # Calculate
    lower = low_expr.rolling_min(window_size=lower_length, min_samples=lower_length)
    upper = high_expr.rolling_max(window_size=upper_length, min_samples=upper_length)
    mid = (lower + upper) * 0.5

    if offset != 0:
        lower = lower.shift(offset)
        mid = mid.shift(offset)
        upper = upper.shift(offset)

    _props = f"_{lower_length}_{upper_length}"

    return pl.struct(
        [
            lower.alias(f"DCL{_props}"),
            mid.alias(f"DCM{_props}"),
            upper.alias(f"DCU{_props}"),
        ]
    ).alias(f"DC{_props}")
