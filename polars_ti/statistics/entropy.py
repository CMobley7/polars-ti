# -*- coding: utf-8 -*-
# =============================================================================
# Polars ENTROPY Implementation (Numba @njit kernel)
# =============================================================================
import polars as pl
import numpy as np
from numba import njit

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


@njit(cache=True)
def nb_entropy(close: np.ndarray, length: int, base: float) -> np.ndarray:
    """Numba-optimized Shannon entropy calculation.

    Matches Pandas logic:
    1. p = close / close.rolling(length).sum()
    2. entropy = (-p * log(p) / log(base)).rolling(length).sum()
    """
    n = len(close)
    result = np.full(n, np.nan)
    log_base = np.log(base)

    # Compute rolling sum for p calculation
    rolling_sum = np.full(n, np.nan)
    for i in range(length - 1, n):
        rolling_sum[i] = np.sum(close[i - length + 1 : i + 1])

    # Compute p = close / rolling_sum
    p = np.full(n, np.nan)
    for i in range(length - 1, n):
        if rolling_sum[i] > 0:
            p[i] = close[i] / rolling_sum[i]

    # Compute term = -p * log(p) / log(base)
    term = np.full(n, np.nan)
    for i in range(n):
        if not np.isnan(p[i]) and p[i] > 0:
            term[i] = -p[i] * np.log(p[i]) / log_base

    # Compute rolling sum of term (second rolling)
    for i in range(2 * length - 2, n):
        window = term[i - length + 1 : i + 1]
        valid_count = 0
        window_sum = 0.0
        for j in range(length):
            if not np.isnan(window[j]):
                window_sum += window[j]
                valid_count += 1
        if valid_count == length:
            result[i] = window_sum

    return result


def pl_entropy(
    close: IntoExpr,
    length: int = 10,
    base: float = 2.0,
    offset: int = 0,
) -> pl.Expr:
    """Polars: Entropy (ENTP)

    Shannon entropy measures the unpredictability of data.
    Uses Numba @njit kernel for high performance.

    Sources:
        https://en.wikipedia.org/wiki/Entropy_(information_theory)

    Args:
        close: Column name or pl.Expr for 'close' prices
        length: Rolling window period. Default: 10
        base: Logarithmic base. Default: 2.0
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: Entropy expression
    """
    close_expr = v_expr(close)
    if close_expr is None:
        return None

    _length = length
    _base = base

    def compute_entropy(s: pl.Series) -> pl.Series:
        """Compute entropy using Numba kernel."""
        arr = s.to_numpy().astype(np.float64)
        result = nb_entropy(arr, _length, _base)
        return pl.Series(result)

    result = close_expr.map_batches(compute_entropy, return_dtype=pl.Float64)

    if offset != 0:
        result = result.shift(offset)

    return result.alias(f"ENTP_{length}")
