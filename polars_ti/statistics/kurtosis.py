import numpy as np

# -*- coding: utf-8 -*-
# =============================================================================
# Polars KURTOSIS Implementation (Numba @njit kernel)
# =============================================================================
import polars as pl
from numba import njit

from polars_ti._typing import IntoExpr
from polars_ti.utils._rolling import needs_scaling, rolling_extreme, rolling_moments, scaled_window_moments
from polars_ti.utils._validate import v_expr


@njit(cache=True)
def nb_kurtosis(close: np.ndarray, length: int) -> np.ndarray:
    """Compute bias-corrected Fisher excess kurtosis; constant windows remain undefined."""
    _, second, third, fourth = rolling_moments(close, length, 4)
    result = np.full(len(close), np.nan)
    magnitude = rolling_extreme(np.abs(close), length, True)[0]
    for i in range(length - 1, len(close)):
        if needs_scaling(magnitude[i]):
            _, normalized2, normalized3, normalized4, _, last = scaled_window_moments(close, i - length + 1, i + 1)
            if length > 3 and normalized2 > 0:
                excess = normalized4 / (normalized2 * normalized2) - 3.0
                result[i] = ((length + 1.0) * excess + 6.0) * ((length - 1.0) / ((length - 2.0) * (length - 3.0)))
            continue
        if length > 3 and second[i] > 0:
            excess = fourth[i] / (second[i] * second[i]) - 3.0
            result[i] = ((length + 1.0) * excess + 6.0) * ((length - 1.0) / ((length - 2.0) * (length - 3.0)))
    return result


def kurtosis(
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
