# -*- coding: utf-8 -*-
# =============================================================================
# Polars KURTOSIS Implementation (Numba @njit kernel)
# =============================================================================
import polars as pl
import numpy as np
from numba import njit

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


@njit(cache=True)
def nb_kurtosis(close: np.ndarray, length: int) -> np.ndarray:
    """Numba-optimized Fisher's excess kurtosis calculation.

    Fisher's definition (matching Pandas):
    kurtosis = (m4 / m2^2) - 3 (excess kurtosis)

    With bias correction for sample kurtosis.
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

        # Compute centered moments (m2 = variance without ddof, m4 = fourth moment)
        m2 = 0.0
        m4 = 0.0
        for j in range(window_n):
            diff = window[j] - mean
            diff2 = diff * diff
            m2 += diff2
            m4 += diff2 * diff2

        m2 /= window_n
        m4 /= window_n

        # Fisher's excess kurtosis with bias correction (matching Pandas)
        # Pandas uses the formula: g2 = m4/m2^2 - 3
        # Then applies bias correction: G2 = ((n+1)*g2 + 6) * (n-1) / ((n-2)*(n-3))
        if window_n > 3 and m2 > 0:
            g2 = m4 / (m2 * m2) - 3.0
            # Bias correction factor
            adj = (window_n - 1.0) / ((window_n - 2.0) * (window_n - 3.0))
            G2 = ((window_n + 1.0) * g2 + 6.0) * adj
            result[i] = G2

    return result


def pl_kurtosis(
    close: IntoExpr,
    length: int = 30,
    offset: int = 0,
) -> pl.Expr:
    """Polars: Rolling Kurtosis

    Calculates Fisher's excess kurtosis over a rolling period.
    Uses Numba @njit kernel for high performance.

    Args:
        close: Column name or pl.Expr for 'close' prices
        length: Rolling window period. Default: 30
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: Kurtosis expression
    """
    close_expr = v_expr(close)
    if close_expr is None:
        return None

    _length = length

    def compute_kurtosis(s: pl.Series) -> pl.Series:
        """Compute kurtosis using Numba kernel."""
        arr = s.to_numpy().astype(np.float64)
        result = nb_kurtosis(arr, _length)
        return pl.Series(result)

    result = close_expr.map_batches(compute_kurtosis, return_dtype=pl.Float64)

    if offset != 0:
        result = result.shift(offset)

    return result.alias(f"KURT_{length}")
