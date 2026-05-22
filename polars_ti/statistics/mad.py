# -*- coding: utf-8 -*-
# =============================================================================
# Polars MAD Implementation (Numba @njit kernel)
# =============================================================================
import polars as pl
import numpy as np
from numba import njit

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


@njit(cache=True)
def nb_mad(close: np.ndarray, length: int) -> np.ndarray:
    """Numba-optimized Mean Absolute Deviation calculation.

    MAD = mean(|x - mean(x)|) for each rolling window
    """
    n = len(close)
    result = np.full(n, np.nan)

    for i in range(length - 1, n):
        window = close[i - length + 1 : i + 1]

        # Compute mean
        mean = 0.0
        for j in range(length):
            mean += window[j]
        mean /= length

        # Compute mean absolute deviation
        mad = 0.0
        for j in range(length):
            mad += np.abs(window[j] - mean)
        mad /= length

        result[i] = mad

    return result


def pl_mad(
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
