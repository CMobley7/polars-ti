# -*- coding: utf-8 -*-
# =============================================================================
# Polars ZSCORE Implementation (Numba @njit kernel)
# =============================================================================
import polars as pl
import numpy as np
from numba import njit

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


@njit(cache=True)
def nb_zscore(close: np.ndarray, length: int, std_mult: float) -> np.ndarray:
    """Numba-optimized Z-Score calculation.

    Z = (close - rolling_mean) / (std_mult * rolling_std)

    Uses ddof=0 for std matching TA-Lib's behavior.
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

        # Compute std with ddof=0 (TA-Lib style)
        var = 0.0
        for j in range(length):
            diff = window[j] - mean
            var += diff * diff
        var /= length
        std = np.sqrt(var)

        if std > 0:
            result[i] = (close[i] - mean) / (std_mult * std)

    return result


def pl_zscore(
    close: IntoExpr,
    length: int = 30,
    std: float = 1.0,
    offset: int = 0,
) -> pl.Expr:
    """Polars: Rolling Z Score

    Calculates Z Score over a rolling period.
    Uses Numba @njit kernel for high performance.

    Z = (close - rolling_mean) / (std * rolling_std)

    Args:
        close: Column name or pl.Expr for 'close' prices
        length: Rolling window period. Default: 30
        std: Standard deviation multiplier. Default: 1.0
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: Z Score expression
    """
    close_expr = v_expr(close)
    if close_expr is None:
        return None

    _length = length
    _std = std

    def compute_zscore(s: pl.Series) -> pl.Series:
        """Compute zscore using Numba kernel."""
        arr = s.to_numpy().astype(np.float64)
        result = nb_zscore(arr, _length, _std)
        return pl.Series(result)

    result = close_expr.map_batches(compute_zscore, return_dtype=pl.Float64)

    if offset != 0:
        result = result.shift(offset)

    return result.alias(f"ZS_{length}")
