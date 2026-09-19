import numpy as np

# -*- coding: utf-8 -*-
# =============================================================================
# Polars ZSCORE Implementation (Numba @njit kernel)
# =============================================================================
import polars as pl
from numba import njit

from polars_ti._typing import IntoExpr
from polars_ti.utils._rolling import needs_scaling, rolling_extreme, rolling_moments, scaled_window_moments
from polars_ti.utils._validate import v_expr


@njit(cache=True)
def nb_zscore(close: np.ndarray, length: int, std_mult: float) -> np.ndarray:
    """Use population deviation (ddof=0) for Z scores; constant windows are undefined."""
    mean, variance, _, _ = rolling_moments(close, length)
    result = np.full(len(close), np.nan)
    magnitude = rolling_extreme(np.abs(close), length, True)[0]
    for i in range(length - 1, len(close)):
        if needs_scaling(magnitude[i]):
            _, normalized2, normalized3, normalized4, _, last = scaled_window_moments(close, i - length + 1, i + 1)
            if normalized2 > 0:
                result[i] = last / (std_mult * np.sqrt(normalized2))
            continue
        if variance[i] > 0:
            result[i] = (close[i] - mean[i]) / (std_mult * np.sqrt(variance[i]))
    return result


def zscore(
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
