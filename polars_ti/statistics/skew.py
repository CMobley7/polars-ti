import numpy as np

# -*- coding: utf-8 -*-
# =============================================================================
# Polars SKEW Implementation (Numba @njit kernel)
# =============================================================================
import polars as pl
from numba import njit

from polars_ti._typing import IntoExpr
from polars_ti.utils._rolling import needs_scaling, rolling_extreme, rolling_moments, scaled_window_moments
from polars_ti.utils._validate import v_expr


@njit(cache=True)
def nb_skew(close: np.ndarray, length: int) -> np.ndarray:
    """Compute bias-corrected Fisher-Pearson sample skewness; constant windows remain undefined."""
    _, second, third, fourth = rolling_moments(close, length, 3)
    result = np.full(len(close), np.nan)
    magnitude = rolling_extreme(np.abs(close), length, True)[0]
    for i in range(length - 1, len(close)):
        if needs_scaling(magnitude[i]):
            _, normalized2, normalized3, normalized4, _, last = scaled_window_moments(close, i - length + 1, i + 1)
            if length > 2 and normalized2 > 0:
                result[i] = np.sqrt(length * (length - 1.0)) / (length - 2.0) * normalized3 / normalized2**1.5
            continue
        if length > 2 and second[i] > 0:
            result[i] = np.sqrt(length * (length - 1.0)) / (length - 2.0) * third[i] / second[i] ** 1.5
    return result


def skew(
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
