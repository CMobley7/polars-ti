import numpy as np

# -*- coding: utf-8 -*-
# =============================================================================
# Polars MAD Implementation (Numba @njit kernel)
# =============================================================================
import polars as pl
from numba import njit

from polars_ti._typing import IntoExpr
from polars_ti.utils import v_pos_int
from polars_ti.utils._order_stats import rolling_mad
from polars_ti.utils._validate import v_expr


@njit(cache=True)
def nb_mad(close: np.ndarray, length: int) -> np.ndarray:
    """Return mean absolute deviation from stable rolling order statistics."""
    return rolling_mad(close, length)


def mad(
    close: IntoExpr,
    length: int = 30,
    offset: int = 0,
) -> pl.Expr:
    """Polars: Rolling Mean Absolute Deviation

    Calculates the Mean Absolute Deviation over a rolling period.
    Uses Numba @njit kernel for high performance.

    Args:
        close: Column name or pl.Expr for 'close' prices
        length: Rolling window period. Default: 30
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: MAD expression
    """
    close_expr = v_expr(close)
    if close_expr is None:
        return None

    length = v_pos_int(length, "length")
    _length = length

    def compute_mad(s: pl.Series) -> pl.Series:
        """Compute MAD using Numba kernel."""
        arr = s.to_numpy().astype(np.float64)
        result = nb_mad(arr, _length)
        return pl.Series(result)

    result = close_expr.map_batches(compute_mad, return_dtype=pl.Float64)

    if offset != 0:
        result = result.shift(offset)

    return result.alias(f"MAD_{length}")
