# -*- coding: utf-8 -*-
# =============================================================================
# Polars JMA Implementation
# =============================================================================
import polars as pl
import numpy as np
from numba import njit

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


@njit(cache=True)
def nb_jma(close: np.ndarray, length: int, phase: float) -> np.ndarray:
    """Numba-optimized Jurik Moving Average calculation."""
    m = len(close)
    jma = np.empty(m, dtype=np.float64)
    volty = np.zeros(m, dtype=np.float64)
    v_sum = np.zeros(m, dtype=np.float64)

    # Initialize
    kv = det0 = det1 = ma2 = 0.0
    jma[0] = ma1 = uBand = lBand = close[0]

    # Static variables
    sum_length = 10
    length_half = 0.5 * (length - 1)

    if phase < -100:
        pr = 0.5
    elif phase > 100:
        pr = 2.5
    else:
        pr = 1.5 + phase * 0.01

    length1 = max(np.log(np.sqrt(length_half)) / np.log(2.0) + 2.0, 0.0)
    pow1 = max(length1 - 2.0, 0.5)
    length2 = length1 * np.sqrt(length_half)
    bet = length2 / (length2 + 1)
    beta = 0.45 * (length - 1) / (0.45 * (length - 1) + 2.0)

    for i in range(1, m):
        price = close[i]

        # Price volatility
        del1 = price - uBand
        del2 = price - lBand
        if abs(del1) != abs(del2):
            volty[i] = max(abs(del1), abs(del2))
        else:
            volty[i] = 0.0

        # Relative price volatility factor
        start = max(i - sum_length, 0)
        v_sum[i] = v_sum[i - 1] + (volty[i] - volty[start]) / sum_length

        # Average volatility
        avg_start = max(i - 65, 0)
        avg_volty = np.mean(v_sum[avg_start : i + 1])

        if avg_volty == 0:
            d_volty = 0.0
        else:
            d_volty = volty[i] / avg_volty

        r_volty = max(1.0, min(length1 ** (1 / pow1), d_volty))

        # Jurik volatility bands
        pow2 = r_volty**pow1
        kv = bet ** np.sqrt(pow2)

        if del1 > 0:
            uBand = price
        else:
            uBand = price - kv * del1

        if del2 < 0:
            lBand = price
        else:
            lBand = price - kv * del2

        # Jurik Dynamic Factor
        power = r_volty**pow1
        alpha = beta**power

        # 1st stage - preliminary smoothing by adaptive EMA
        ma1 = (1 - alpha) * price + alpha * ma1

        # 2nd stage - one more preliminary smoothing by Kalman filter
        det0 = (1 - beta) * (price - ma1) + beta * det0
        ma2 = ma1 + pr * det0

        # 3rd stage - final smoothing by unique Jurik adaptive filter
        det1 = (ma2 - jma[i - 1]) * (1 - alpha) * (1 - alpha) + alpha * alpha * det1
        jma[i] = jma[i - 1] + det1

    # Set warmup period to NaN
    jma[: length - 1] = np.nan

    return jma


def pl_jma(
    close: IntoExpr,
    length: int = 7,
    phase: float = 0.0,
    offset: int = 0,
) -> PlExpr:
    """Polars: Jurik Moving Average (JMA)

    Adaptive filter with extremely low lag and smoothing.

    Args:
        close: Column name or pl.Expr for 'close' prices
        length: Period of calculation. Default: 7
        phase: How heavy/light the average is [-100, 100]. Default: 0
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: JMA expression
    """
    close_expr = v_expr(close)

    def compute_jma(s: pl.Series) -> pl.Series:
        arr = s.to_numpy().astype(np.float64)
        result = nb_jma(arr, length, phase)
        if offset != 0:
            result = np.roll(result, offset)
            if offset > 0:
                result[:offset] = np.nan
        return pl.Series(result)

    return close_expr.map_batches(compute_jma, return_dtype=pl.Float64).alias(f"JMA_{length}_{phase}")
