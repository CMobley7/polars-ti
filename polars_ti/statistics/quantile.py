# -*- coding: utf-8 -*-
# =============================================================================
# Polars QUANTILE Implementation (Numba @njit kernel)
# =============================================================================
import polars as pl
import numpy as np
from numba import njit

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


@njit(cache=True)
def nb_quantile(close: np.ndarray, length: int, q: float) -> np.ndarray:
    """Numba-optimized rolling quantile with linear interpolation.

    Matches Pandas rolling.quantile() with default linear interpolation.
    """
    n = len(close)
    result = np.full(n, np.nan)

    for i in range(length - 1, n):
        window = close[i - length + 1 : i + 1].copy()

        # Sort the window
        for j in range(length):
            for k in range(j + 1, length):
                if window[j] > window[k]:
                    tmp = window[j]
                    window[j] = window[k]
                    window[k] = tmp

        # Linear interpolation (matching Pandas default)
        # idx = q * (n - 1), then interpolate
        idx = q * (length - 1)
        lower_idx = int(idx)
        upper_idx = lower_idx + 1

        if upper_idx >= length:
            result[i] = window[length - 1]
        else:
            frac = idx - lower_idx
            result[i] = window[lower_idx] * (1 - frac) + window[upper_idx] * frac

    return result


def quantile(
    close: IntoExpr,
    length: int = 30,
    q: float = 0.5,
    offset: int = 0,
) -> pl.Expr:
    """Polars: Rolling Quantile

    Calculates the Quantile over a rolling period.
    Uses Numba @njit kernel for high performance.

    Args:
        close: Column name or pl.Expr for 'close' prices
        length: Rolling window period. Default: 30
        q: Quantile value (0 to 1). Default: 0.5
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: Quantile expression
    """
    close_expr = v_expr(close)
    if close_expr is None:
        return None

    _q = float(q) if isinstance(q, (int, float)) and 0 < q < 1 else 0.5
    _length = length

    def compute_quantile(s: pl.Series) -> pl.Series:
        """Compute quantile using Numba kernel."""
        arr = s.to_numpy().astype(np.float64)
        result = nb_quantile(arr, _length, _q)
        return pl.Series(result)

    result = close_expr.map_batches(compute_quantile, return_dtype=pl.Float64)

    if offset != 0:
        result = result.shift(offset)

    return result.alias(f"QTL_{length}_{_q}")
