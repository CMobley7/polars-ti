# -*- coding: utf-8 -*-
# =============================================================================
# Polars RMA Implementation (Numba @njit with presma support)
# =============================================================================
import polars as pl
import numpy as np
from numba import njit

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


@njit(cache=True)
def _rma_numba(close: np.ndarray, length: int, presma: bool = True) -> np.ndarray:
    """Numba-optimized RMA with presma support.

    RMA (Wilder's Moving Average) = EMA with alpha = 1/length (vs 2/(n+1) for EMA).

    With presma=True (TA-Lib style):
        - First `length-1` values are NaN
        - Value at index `length-1` is SMA of first `length` values
        - From `length` onwards: RMA = alpha * close + (1 - alpha) * prev_RMA

    With presma=False:
        - First value is the first close value itself
        - Standard exponential smoothing from the start
    """
    n = len(close)
    result = np.empty(n, dtype=np.float64)
    result[:] = np.nan

    alpha = 1.0 / length

    if presma:
        # Calculate SMA for initial value (first 'length' values)
        sma_sum = 0.0
        valid_count = 0
        for i in range(length):
            if i < n and not np.isnan(close[i]):
                sma_sum += close[i]
                valid_count += 1

        if valid_count == length:
            sma_val = sma_sum / length
            result[length - 1] = sma_val

            # Continue with RMA from there
            for i in range(length, n):
                if not np.isnan(close[i]):
                    result[i] = alpha * close[i] + (1 - alpha) * result[i - 1]
                else:
                    result[i] = result[i - 1]
    else:
        # Standard RMA without SMA initialization
        first_valid = -1
        for i in range(n):
            if not np.isnan(close[i]):
                first_valid = i
                break

        if first_valid >= 0:
            result[first_valid] = close[first_valid]
            for i in range(first_valid + 1, n):
                if not np.isnan(close[i]):
                    result[i] = alpha * close[i] + (1 - alpha) * result[i - 1]
                else:
                    result[i] = result[i - 1]

    return result


def rma(
    close: IntoExpr,
    length: int = 10,
    presma: bool = False,
    offset: int = 0,
) -> PlExpr:
    """Polars: Wilder's Moving Average (RMA)

    Uses Numba @njit kernel via map_batches.

    Wilder's Moving Average is an EMA with alpha = 1 / length (vs 2/(n+1) for EMA).
    This is the smoothing used by ATR, RSI, and other Wilder indicators.

    Sources:
        https://tlc.thinkorswim.com/center/reference/Tech-Indicators/studies-library/V-Z/WildersSmoothing
        https://www.incrediblecharts.com/indicators/wilder_moving_average.php

    Args:
        close: Column name or pl.Expr for 'close' prices
        length: Smoothing period. Default: 10
        presma: If True, uses SMA for initial value like TA-Lib. Default: False
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: RMA expression for lazy evaluation
    """
    close_expr = v_expr(close)
    if close_expr is None:
        return None

    _length = length
    _presma = presma

    def compute_rma(s: pl.Series) -> pl.Series:
        arr = s.to_numpy().astype(np.float64)
        result = _rma_numba(arr, _length, _presma)
        return pl.Series(result)

    rma_expr = close_expr.map_batches(compute_rma, return_dtype=pl.Float64)

    # Apply offset
    if offset != 0:
        rma_expr = rma_expr.shift(offset)

    return rma_expr.alias(f"RMA_{length}")
