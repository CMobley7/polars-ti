# -*- coding: utf-8 -*-
# =============================================================================
# Polars CUBE Implementation (Pure Native Polars)
# =============================================================================
import polars as pl

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


def cube(
    close: IntoExpr,
    pwr: float = 3.0,
    signal_offset: int = -1,
    offset: int = 0,
) -> list[PlExpr]:
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
        list[pl.Expr]: ``[CUBE_{pwr}_{signal_offset}, CUBEs_{pwr}_{signal_offset}]``.
    """
    close_expr = v_expr(close)
    if close_expr is None:
        return None

    # Pure native Polars: simple power operation
    result = close_expr.pow(pwr)

    # OLD applies BOTH offset and signal_offset to BOTH lines, so the main and
    # signal series are identical (preserved for parity).
    ct = result
    signal = result
    if offset != 0:
        ct = ct.shift(offset)
        signal = signal.shift(offset)
    if signal_offset != 0:
        ct = ct.shift(signal_offset)
        signal = signal.shift(signal_offset)

    _props = f"_{pwr}_{signal_offset}"
    return [ct.alias(f"CUBE{_props}"), signal.alias(f"CUBEs{_props}")]
