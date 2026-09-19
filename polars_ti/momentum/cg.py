import numpy as np

# -*- coding: utf-8 -*-
# =============================================================================
# Polars CG (Center of Gravity) Implementation
# =============================================================================
import polars as pl
from numba import njit

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils import v_pos_int
from polars_ti.utils._rolling import rolling_linear, scaled_window_linear
from polars_ti.utils._validate import v_expr


@njit(cache=True)
def nb_cg(close: np.ndarray, length: int) -> np.ndarray:
    """Compute center of gravity from compensated linear weighted sums.

    Weights increase from 1 for the oldest value to length for the newest;
    reversing that orientation changes the oscillator rather than its rounding.
    """
    total, weighted, _ = rolling_linear(close, length)
    result = np.full(len(close), np.nan)
    for i in range(length - 1, len(close)):
        if not np.isfinite(total[i]) or not np.isfinite(weighted[i]):
            normalized_total, normalized_weighted, scale = scaled_window_linear(close, i - length + 1, i + 1)
            if abs(normalized_total) > 1e-10 / scale:
                result[i] = -normalized_weighted / normalized_total
        elif abs(total[i]) > 1e-10:
            result[i] = -weighted[i] / total[i]
    return result


def cg(
    close: IntoExpr,
    length: int = 10,
    offset: int = 0,
) -> PlExpr:
    """Polars: Center of Gravity (CG)

    The Center of Gravity Indicator by John Ehlers attempts to identify
    turning points while exhibiting zero lag and smoothing.

    Formula: CG = -sum(close * weight) / sum(close)

    Args:
        close: Column name or pl.Expr for 'close' prices
        length: Period. Default: 10
        offset: Shift result. Default: 0

    Returns:
        pl.Expr: CG expression
    """
    close_expr = v_expr(close)
    _length = v_pos_int(length, "length")
    _offset = offset

    def compute_cg(s: pl.Series) -> pl.Series:
        arr = s.to_numpy().astype(np.float64)
        result = nb_cg(arr, _length)

        if _offset != 0:
            result = np.roll(result, _offset)
            if _offset > 0:
                result[:_offset] = np.nan
            else:
                result[_offset:] = np.nan

        return pl.Series(result)

    return close_expr.map_batches(compute_cg, return_dtype=pl.Float64).alias(f"CG_{length}")
