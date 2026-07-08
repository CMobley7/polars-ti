# -*- coding: utf-8 -*-
# =============================================================================
# Polars STDERR (Standard Error) Implementation
# =============================================================================
import polars as pl
import numpy as np

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


def stderr(
    close: IntoExpr,
    length: int = 14,
    ddof: int = 1,
    offset: int = 0,
) -> PlExpr:
    """Polars: Standard Error (STDERR)

    The standard deviation of a rolling window divided by the square root
    of the sample size.  Estimates the precision of the sample mean.

    Formula:
        STDERR = StdDev(close, length, ddof) / sqrt(length)

    Sources:
        https://en.wikipedia.org/wiki/Standard_error

    Args:
        close: Column name or pl.Expr for 'close' prices.
        length: Rolling window period. Default: 14
        ddof: Degrees of freedom for std. Default: 1
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: STDERR expression.
    """
    close_expr = v_expr(close)
    if close_expr is None:
        return None

    _length = length
    # Clamp ddof to a valid range; ddof>=length would otherwise yield all-NaN.
    _ddof = ddof if 0 <= ddof < length else 1
    _offset = offset

    result = close_expr.rolling_std(window_size=_length, ddof=_ddof) / np.sqrt(_length)

    if _offset != 0:
        result = result.shift(_offset)

    return result.alias(f"STDERR_{length}")
