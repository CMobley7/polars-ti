# -*- coding: utf-8 -*-
# =============================================================================
# Polars CUBE Implementation (Pure Native Polars)
# =============================================================================
import polars as pl

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


def pl_cube(
    close: IntoExpr,
    pwr: float = 3.0,
    signal_offset: int = -1,
    offset: int = 0,
) -> pl.Expr:
    """Polars: Cube Transform

    Compresses signals near zero for normalized oscillators like
    the Inverse Fisher Transform. Values close to -1 and 1 are
    nearly unchanged, while those near zero are reduced.

    Uses pure native Polars expressions.

    Sources:
        Book: Cycle Analytics for Traders, 2014, by John Ehlers

    Args:
        close: Column name or pl.Expr for 'close' prices
        pwr: Exponent for soft limiter. Default: 3.0
        signal_offset: Offset for signal line. Default: -1
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: Cube transform expression
    """
    close_expr = v_expr(close)
    if close_expr is None:
        return None

    # Pure native Polars: simple power operation
    result = close_expr.pow(pwr)

    # Apply offsets
    if offset != 0:
        result = result.shift(offset)
    if signal_offset != 0:
        result = result.shift(signal_offset)

    return result.alias(f"CUBE_{pwr}_{signal_offset}")

