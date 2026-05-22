# -*- coding: utf-8 -*-
# =============================================================================
# Polars REMAP Implementation (Pure Native Polars)
# =============================================================================
import polars as pl

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


def pl_remap(
    close: IntoExpr,
    from_min: float = 0.0,
    from_max: float = 100.0,
    to_min: float = -1.0,
    to_max: float = 1.0,
    offset: int = 0,
) -> pl.Expr:
    """Polars: ReMap (Linear Normalization)

    Maps input from one range to another (static normalizer).

    Uses pure native Polars expressions.

    Formula: y = to_min + (trange/frange) * (x - from_min)

    Examples:
        RSI -> IFISHER: from_min=0, from_max=100, to_min=-1, to_max=1

    Args:
        close: Column name or pl.Expr for input values
        from_min: Input minimum. Default: 0.0
        from_max: Input maximum. Default: 100.0
        to_min: Output minimum. Default: -1.0
        to_max: Output maximum. Default: 1.0
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: Remapped values expression
    """
    close_expr = v_expr(close)
    if close_expr is None:
        return None

    frange = from_max - from_min
    trange = to_max - to_min

    if frange <= 0 or trange <= 0:
        return None

    # Pure native Polars: linear remapping
    result = pl.lit(to_min) + (pl.lit(trange / frange) * (close_expr - pl.lit(from_min)))

    if offset != 0:
        result = result.shift(offset)

    return result.alias(f"REMAP_{from_min}_{from_max}_{to_min}_{to_max}")
