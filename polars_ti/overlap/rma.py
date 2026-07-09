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

    # First finite index (leading NaNs — e.g. true_range[0] or a diff series —
    # must never poison the whole column).
    fv = -1
    for i in range(n):
        if not np.isnan(close[i]):
            fv = i
            break

    if presma and fv >= 0 and fv + length <= n:
        # SMA seed over the first ``length`` FINITE values (contiguous from the
        # first finite index), placed at ``fv+length-1`` — exactly matching
        # TA-Lib's Wilder warmup (e.g. ATR seeds the SMA of TR[1..length] at
        # index ``length`` because TR[0] is undefined). For a fully finite input
        # (fv=0) this is the mean of close[0:length] at index length-1, identical
        # to before; on a leading-NaN input it seeds one bar later than the old
        # NaN-skipping window (which jumped the gun by a bar). Mirrors _ema_numba.
        seeded = np.empty(n, dtype=np.float64)
        for i in range(n):
            seeded[i] = close[i]
        sma_sum = 0.0
        for i in range(fv, fv + length):
            sma_sum += close[i]
        for i in range(fv + length - 1):
            seeded[i] = np.nan
        seeded[fv + length - 1] = sma_sum / length
        close = seeded

    # ewm(alpha, adjust=False): seed from first finite value, carry forward on NaN.
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
