# -*- coding: utf-8 -*-
# =============================================================================
# Polars Fisher Transform Implementation
# =============================================================================
import polars as pl
import numpy as np
from numba import njit

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


@njit(cache=True)
def nb_fisher(hl2_arr: np.ndarray, lowest: np.ndarray, hlr: np.ndarray, length: int) -> np.ndarray:
    """Numba: Fisher Transform with recursive EMA-like smoothing."""
    n = len(hl2_arr)
    result = np.full(n, np.nan)

    # Initialize: result[length-1] = 0 (matches Pandas)
    result[length - 1] = 0.0

    v = 0.0
    # Loop from i=length to n-1 (matches Pandas: for i in range(length, m))
    for i in range(length, n):
        position = ((hl2_arr[i] - lowest[i]) / hlr[i]) - 0.5
        v = 0.66 * position + 0.67 * v
        if v < -0.99:
            v = -0.999
        if v > 0.99:
            v = 0.999
        result[i] = 0.5 * (np.log((1 + v) / (1 - v)) + result[i - 1])

    return result


def fisher(
    high: IntoExpr,
    low: IntoExpr,
    length: int = 9,
    signal: int = 1,
    offset: int = 0,
) -> list[PlExpr]:
    """Polars: Fisher Transform (FISHT)

    Identifies price reversals by normalizing prices over N periods.

    Args:
        high: Column name or pl.Expr for 'high' prices
        low: Column name or pl.Expr for 'low' prices
        length: Fisher period. Default: 9
        signal: Signal period (shift). Default: 1
        offset: Shift result. Default: 0

    Returns:
        list[pl.Expr]: [FISHERT, FISHERTs] expressions
    """
    from polars_ti.overlap.hl2 import hl2

    high_expr = v_expr(high)
    low_expr = v_expr(low)
    _length = length

    def compute_fisher(s: pl.Series) -> pl.Series:
        df = s.struct.unnest()
        high_arr = df["high"].to_numpy().astype(np.float64)
        low_arr = df["low"].to_numpy().astype(np.float64)

        # HL2
        hl2_arr = (high_arr + low_arr) / 2.0

        # Rolling max/min of HL2
        n = len(hl2_arr)
        highest = np.full(n, np.nan)
        lowest = np.full(n, np.nan)
        for i in range(_length - 1, n):
            window = hl2_arr[i - _length + 1 : i + 1]
            highest[i] = np.max(window)
            lowest[i] = np.min(window)

        # High-low range with floor
        hlr = highest - lowest
        hlr = np.maximum(hlr, 0.001)

        result = nb_fisher(hl2_arr, lowest, hlr, _length)
        return pl.Series(result)

    struct_expr = pl.struct(high=high_expr, low=low_expr)
    fisher_expr = struct_expr.map_batches(compute_fisher, return_dtype=pl.Float64)

    signal_expr = fisher_expr.shift(signal)

    if offset != 0:
        fisher_expr = fisher_expr.shift(offset)
        signal_expr = signal_expr.shift(offset)

    _props = f"_{length}_{signal}"
    return [
        fisher_expr.alias(f"FISHERT{_props}"),
        signal_expr.alias(f"FISHERTs{_props}"),
    ]
