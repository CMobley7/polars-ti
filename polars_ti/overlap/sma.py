import numpy as np
from numba import njit

# -*- coding: utf-8 -*-

from polars_ti.utils._rolling import binary_scale, compensated_add, rolling_extreme, rolling_sum


@njit(cache=True)
def nb_sma(x, n):
    """Return compensated means with a scaled fallback for overflowing sums."""
    result = rolling_sum(x, n) / n
    minimum = rolling_extreme(x, n, False)[0]
    maximum = rolling_extreme(x, n, True)[0]
    for i in range(n - 1, len(x)):
        if not np.isfinite(result[i]) and np.isfinite(minimum[i]) and np.isfinite(maximum[i]):
            scale = max(abs(minimum[i]), abs(maximum[i]))
            if scale == 0.0:
                result[i] = 0.0
                continue
            scale = binary_scale(scale)
            total = correction = 0.0
            for j in range(i - n + 1, i + 1):
                total, correction = compensated_add(total, correction, x[j] / scale)
            result[i] = ((total + correction) / n) * scale
    return result


# =============================================================================
# Polars SMA Implementation
# =============================================================================
import polars as pl

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


def sma(
    close: IntoExpr,
    length: int = 10,
    talib: bool = True,
    min_periods: int | None = None,
    offset: int = 0,
) -> PlExpr:
    """Polars: Simple Moving Average (SMA)

    The Simple Moving Average is the equally weighted average over its length.

    Sources:
        https://www.tradingtechnologies.com/help/x-study/technical-indicator-definitions/simple-moving-average-sma/

    Args:
        close: Column name or pl.Expr for 'close' prices
        length: Rolling window period. Default: 10
        talib: Use TA-Lib when available for full windows longer than one. Default: True
        min_periods: Minimum periods required. Default: length
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: SMA expression for lazy evaluation
    """
    import numpy as np

    from polars_ti.maps import Imports
    from polars_ti.utils import v_talib

    close_expr = v_expr(close)
    if close_expr is None:
        return None

    min_periods = min_periods if min_periods is not None else length
    # TA-Lib SMA is fixed at min_periods == length (full window); honor a smaller
    # min_periods by falling through to the native rolling_mean (which applies it).
    _use_talib = Imports["talib"] and v_talib(talib) and length > 1 and min_periods == length
    _length = length

    if _use_talib:

        def compute_sma(s: pl.Series) -> pl.Series:
            from talib import SMA as TALIB_SMA

            arr = s.to_numpy().astype(np.float64)
            result = TALIB_SMA(arr, timeperiod=_length)
            return pl.Series(result)

        sma_expr = close_expr.map_batches(compute_sma, return_dtype=pl.Float64)
    elif min_periods == length:

        def compute_native(s: pl.Series) -> pl.Series:
            """Evaluate the full window with stable means and explicit null masks."""
            result = pl.Series(nb_sma(s.to_numpy().astype(np.float64), _length))
            missing = s.is_null().cast(pl.Int64).rolling_sum(_length, min_samples=_length)
            return result.set(missing.is_null() | (missing > 0), None)

        sma_expr = close_expr.map_batches(compute_native, return_dtype=pl.Float64)
    else:
        sma_expr = close_expr.rolling_mean(window_size=length, min_samples=min_periods)

    # Apply offset
    if offset != 0:
        sma_expr = sma_expr.shift(offset)

    return sma_expr.alias(f"SMA_{length}")
