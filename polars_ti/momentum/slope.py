# -*- coding: utf-8 -*-
# =============================================================================
# Polars SLOPE Implementation
# =============================================================================
import polars as pl
import numpy as np

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


def slope(
    close: IntoExpr,
    length: int = 1,
    as_angle: bool = False,
    to_degrees: bool = False,
    offset: int = 0,
) -> PlExpr:
    """Polars: Slope

    Returns the slope of a series of length n. Can convert the slope to angle.
    Default: slope.

    Source: Algebra

    Args:
        close: Column name or pl.Expr for 'close' prices
        length: Lookback period. Default: 1
        as_angle: If True, converts slope to an angle in radians. Default: False
        to_degrees: If as_angle=True, converts angle to degrees. Default: False
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: Slope expression for lazy evaluation
    """
    close_expr = v_expr(close)
    if close_expr is None:
        return None

    # slope = (close - close.shift(length)) / length
    # This is equivalent to nb_idiff(close, length) / length
    slope_expr = (close_expr - close_expr.shift(length)) / length

    if as_angle:
        # Convert to angle using arctan
        slope_expr = slope_expr.arctan()
        if to_degrees:
            # Convert radians to degrees
            slope_expr = slope_expr * (180.0 / np.pi)

    if offset != 0:
        slope_expr = slope_expr.shift(offset)

    # Name based on mode
    if as_angle:
        suffix = "d" if to_degrees else "r"
        alias = f"ANGLE{suffix}_{length}"
    else:
        alias = f"SLOPE_{length}"

    return slope_expr.alias(alias)
