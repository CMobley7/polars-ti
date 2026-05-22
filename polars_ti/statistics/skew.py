# -*- coding: utf-8 -*-
# =============================================================================
# Polars SKEW Implementation (Numba @njit kernel)
# =============================================================================
import polars as pl
import numpy as np
from numba import njit

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


@njit(cache=True)
def nb_skew(close: np.ndarray, length: int) -> np.ndarray:
    """Numba-optimized Fisher's skewness calculation.

    Fisher's definition (matching Pandas):
    skewness = m3 / m2^(3/2)

    With bias correction for sample skewness.
    """
    n = len(close)
    result = np.full(n, np.nan)

    for i in range(length - 1, n):
        window = close[i - length + 1 : i + 1]
        window_n = length

        # Compute mean
        mean = 0.0
        for j in range(window_n):
            mean += window[j]
        mean /= window_n

        # Compute centered moments
        m2 = 0.0
        m3 = 0.0
        for j in range(window_n):
            diff = window[j] - mean
            diff2 = diff * diff
            m2 += diff2
            m3 += diff2 * diff

        m2 /= window_n
        m3 /= window_n

        # Fisher's skewness with bias correction (matching Pandas)
        # g1 = m3 / m2^(3/2)
        # Then applies bias correction: G1 = sqrt(n*(n-1))/(n-2) * g1
        if window_n > 2 and m2 > 0:
            m2_32 = m2**1.5
            g1 = m3 / m2_32
            # Bias correction factor
            adj = np.sqrt(window_n * (window_n - 1.0)) / (window_n - 2.0)
            result[i] = adj * g1

    return result


def pl_skew(
    close: IntoExpr,
    length: int = 30,
    offset: int = 0,
) -> pl.Expr:
    """Polars: Rolling Skew

    Calculates Fisher's skewness over a rolling period.
    Uses Numba @njit kernel for high performance.

    Args:
        close: Column name or pl.Expr for 'close' prices
        length: Rolling window period. Default: 30
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: Skew expression
    """
    close_expr = v_expr(close)
    if close_expr is None:
        return None

    _length = length

    def compute_skew(s: pl.Series) -> pl.Series:
        """Compute skew using Numba kernel."""
        arr = s.to_numpy().astype(np.float64)
        result = nb_skew(arr, _length)
        return pl.Series(result)

    result = close_expr.map_batches(compute_skew, return_dtype=pl.Float64)

    if offset != 0:
        result = result.shift(offset)

    return result.alias(f"SKEW_{length}")
